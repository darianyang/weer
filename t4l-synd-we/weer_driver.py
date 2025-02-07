import logging

import numpy as np
import operator
from westpa.core.we_driver import WEDriver

import absurder

log = logging.getLogger(__name__)

class WEERDriver(WEDriver):
    '''
    Binless resampler with split/merge decisions based on ABSURDer reweighting.
    '''

    def _split_decision(self, bin, to_split, split_into):
        '''
        This removes an extra walker
        '''
        # remove the walker being split
        bin.remove(to_split)
        # get the n split walker children
        new_segments_list = self._split_walker(to_split, split_into, bin)
        # add all new split walkers back into bin, maintaining history
        bin.update(new_segments_list)

        # other implementation, where walkers are split multiple times
        # if len(to_split) > 1:
        #     for segment in to_split:
        #         bin.remove(segment)
        #         new_segments_list = self._split_walker(segment, split_into, bin)
        #         bin.update(new_segments_list)
        # else:
        #     to_split = to_split[0]
        #     bin.remove(to_split)
        #     new_segments_list = self._split_walker(to_split, split_into, bin)
        #     bin.update(new_segments_list)


    def _merge_decision(self, bin, to_merge, cumul_weight=None):
        '''
        This adds an extra walker
        '''
        # removes every walker in to_merge 
        # TODO: I think I need to remove the current walker, which isn't in the to_merge list
        bin.difference_update(to_merge)
        #new_segment, parent = self._merge_walkers(to_merge, None, bin)
        new_segment, parent = self._merge_walkers(to_merge, cumul_weight, bin)
        # add in new merged walker
        bin.add(new_segment)

    def extract_nmr_data(self, nmr_file="data-NH/600MHz-R1R2NOE.dat"):
        """
        Reference data and errors for reweighting.

        Returned data shape: 
            n_rates x n_vectors

        Also return error for nmr data with dimensions: 
            n_rates x n_vectors

        Parameters
        ----------
        nmr_file : str, optional
            Path to the NMR relaxation data file.

        Updates
        -------
        exp_residues : np.ndarray
            Residue list array.

        Returns
        -------
        nmr_rates : np.ndarray
            NMR relaxation rates.
        nmr_err : np.ndarray
            NMR relaxation rate errors
        """
        # load nmrfile: Res | R1 | R1_err | R2 | R2_err | NOE | NOE_err
        nmr_data = np.loadtxt(nmr_file)
        # residue list array
        self.exp_residues = nmr_data[:, 0]
        # shape: n_rates x n_vectors x 1 single traj
        self.nmr_rates = nmr_data[:, (1, 3, 5)].T
        # shape: n_rates x n_vectors
        self.nmr_err = nmr_data[:, (2, 4, 6)].T
        return self.nmr_rates, self.nmr_err

    def _adjust_count(self, ibin):
        '''
        TODO: adjust to sort/adjust by variance, not weight.
        '''
        bin = self.next_iter_binning[ibin]
        target_count = self.bin_target_counts[ibin]
        weight_getter = operator.attrgetter('weight')

        #print("PRINT ATTRS: ", dir(bin))
        #for b in bin:
        #    print(weight_getter(b))

        # split
        while len(bin) < target_count:
            log.debug('adjusting counts by splitting')
            # always split the highest variance walker into two
            segments = sorted(bin, key=weight_getter)
            bin.remove(segments[-1])
            new_segments_list = self._split_walker(segments[-1], 2, bin)
            bin.update(new_segments_list)

        # merge
        while len(bin) > target_count:
            log.debug('adjusting counts by merging')
            # always merge the two lowest variance walkers
            segments = sorted(bin, key=weight_getter)
            bin.difference_update(segments[:2])
            merged_segment, parent = self._merge_walkers(segments[:2], cumul_weight=None, bin=bin)
            bin.add(merged_segment)

    def _run_we(self):
        '''
        Run recycle/split/merge. Do not call this function directly; instead, use
        populate_initial(), rebin_current(), or construct_next().
        '''
        self._recycle_walkers()

        # sanity check
        self._check_pre()

        # before the next iter setup, run WEER to update weights
        # TODO: could also do it after the setup?
        # grab segments and weights and pcoords
        #seg_list = list(self.next_iter_segments)
        segments = sorted(self.next_iter_segments, key=operator.attrgetter('weight'))
        #segments = np.asarray(sorted())
        #weights = np.array(list(map(operator.attrgetter('weight'), segments)))
        weights = np.asarray([seg.weight for seg in segments])
        #print("seg list: ", seg_list)
        #print("segments: ", segments[0])
        print("weights: ", weights, "weights sum: ", np.sum(weights))

        # get all pcoords, not just the final frame
        # curr_segments = np.array(sorted(self.current_iter_segments, 
        #                                 key=operator.attrgetter('weight')), dtype=np.object_)
        # curr_pcoords = np.array(list(map(operator.attrgetter('pcoord'), curr_segments)))
        curr_segments = sorted(self.current_iter_segments, key=operator.attrgetter('weight'))
        curr_pcoords = np.asarray([seg.pcoord for seg in curr_segments])
        curr_data = np.asarray([seg.data for seg in curr_segments])
        print("curr data len: ", len(curr_data))
        # check for empty dict in first segment (e.g. during init)
        if not curr_data[0]:
            print("segment data is empty")
        else:
            # otherwise, can save data: n_segs(rows) x residue_id(cols)
            r1 = np.array([data['R1'] for data in curr_data])
            r2 = np.array([data['R2'] for data in curr_data])
            noe = np.array([data['NOE'] for data in curr_data])
            residues = np.array([data['residues'] for data in curr_data])
            # the residue arrays for each segment should be the same, make sure this is true
            assert all(np.array_equal(residues[0], res) for res in residues)
            # then should be fine to grab the first one
            residues = residues[0]

            # extract NMR data (also updates self.exp_residues)
            nmr_rates, nmr_err = self.extract_nmr_data()

            # Convert self.exp_residues to a set for faster membership testing
            exp_residues_set = set(self.exp_residues)

            # Create a boolean mask for filtering
            filtered_indices = np.array([resid in exp_residues_set for resid in residues])

            # Apply the mask to filter r1, r2, and noe arrays
            r1 = r1[:, filtered_indices]
            r2 = r2[:, filtered_indices]
            noe = noe[:, filtered_indices]

            # desired shape for ABSURDer: n_rates x n_vectors x n_trajs (blocks)
            absurder_input = np.stack((r1, r2, noe), axis=0).transpose(0, 2, 1)
            print("absurder input shape: ", absurder_input.shape)

            # run reweighting
            theta = 100 # initial test value (TODO: optimize)
            rw = absurder.ABSURDer(nmr_rates, absurder_input, nmr_err, thetas=np.array([theta]))
            # reweight according to the data corresponding to the selected index
            # in this test case, use R2 (TODO: use all rates eventually?)
            rw.reweight(1)
            # save the optimized weights
            absurder_weights = rw.res[theta]
            np.savetxt(f"w_opt_{theta}.txt", absurder_weights)
            print("ABSURDer weights: ", absurder_weights)
            

        # TODO: grab data for multiple iterations (could also use this to get aux data?)
        #       maybe, would need to be careful of the seg ordering
        # I thought it would be useful to use multiple iterations of data
        # but actually this isn't really feasible since I am only optimizing 
        # a single set of weights corresponding to one iteration
        n_curr_iter = self.rc.data_manager.current_iteration
        print("current iteration: ", n_curr_iter)

        # reports = segments.copy()
        # #report_weights = np.array(list(map(operator.attrgetter('weight'), reports)))
        # report_weights = np.asarray([seg.weight for seg in reports])
        # #print("OPT SEG WEIGHTS: ", np.array(list(map(operator.attrgetter('weight'), segments))))
        # #print("OPT SEG WEIGHTS UNSORTED: ", report_weights)
        # #print("OPT SEG WEIGHTS SORTED: ", np.asarray(sorted(report_weights)))

        # TODO: could update to be able to run using bins

        # # Regardless of current particle count, always split overweight particles and merge underweight particles
        # # Then and only then adjust for correct particle count
        # total_number_of_subgroups = 0
        # total_number_of_particles = 0
        # for (ibin, bin) in enumerate(self.next_iter_binning):
        #     if len(bin) == 0:
        #         continue

        #     # Splits the bin into subgroups as defined by the called function
        #     target_count = self.bin_target_counts[ibin]
        #     subgroups = self.subgroup_function(self, ibin, **self.subgroup_function_kwargs)
        #     total_number_of_subgroups += len(subgroups)
        #     # grab segments and weights
        #     segments = np.array(sorted(bin, key=operator.attrgetter('weight')), dtype=np.object_)
        #     weights = np.array(list(map(operator.attrgetter('weight'), segments)))
        #     #print("BIN WEIGHT: ", weights)

        #     # Calculate ideal weight and clear the bin
        #     ideal_weight = weights.sum() / target_count
        #     bin.clear()
        #     # Determines to see whether we have more sub bins than we have target walkers in a bin (or equal to), and then uses
        #     # different logic to deal with those cases.  Should devolve to the Huber/Kim algorithm in the case of few subgroups.
        #     if len(subgroups) >= target_count:
        #         for i in subgroups:
        #             # Merges all members of set i.  Checks to see whether there are any to merge.
        #             if len(i) > 1:
        #                 (segment, parent) = self._merge_walkers(
        #                     list(i),
        #                     np.add.accumulate(np.array(list(map(operator.attrgetter('weight'), i)))),
        #                     i,
        #                 )
        #                 i.clear()
        #                 i.add(segment)
        #             # Add all members of the set i to the bin.  This keeps the bins in sync for the adjustment step.
        #             bin.update(i)

        #         if len(subgroups) > target_count:
        #             self._adjust_count(bin, subgroups, target_count)

        #     if len(subgroups) < target_count:
        #         for i in subgroups:
        #             self._split_by_weight(i, target_count, ideal_weight)
        #             self._merge_by_weight(i, target_count, ideal_weight)
        #             # Same logic here.
        #             bin.update(i)
        #         if self.do_adjust_counts:
        #             # A modified adjustment routine is necessary to ensure we don't unnecessarily destroy trajectory pathways.
        #             self._adjust_count(bin, subgroups, target_count)
        #     if self.do_thresholds:
        #         for i in subgroups:
        #             self._split_by_threshold(bin, i)
        #             self._merge_by_threshold(bin, i)
        #         for iseg in bin:
        #             if iseg.weight > self.largest_allowed_weight or iseg.weight < self.smallest_allowed_weight:
        #                 log.warning(
        #                     f'Unable to fulfill threshold conditions for {iseg}. The given threshold range is likely too small.'
        #                 )
        #     total_number_of_particles += len(bin)
        # log.debug('Total number of subgroups: {!r}'.format(total_number_of_subgroups))

        # dummy resampling block
        # TODO: wevo is really only using one bin
        # ibin only needed right now for temp split merge option (TODO)
        for ibin, bin in enumerate(self.next_iter_binning):
            
            # TODO: is this needed? skips empty bins probably
            if len(bin) == 0:
                continue

            else:
                #resample = REAP(pcoords, weights, n_clusters=20) # 80/4 = 20 (25% of walkers)
                #split, merge = resample.reap()
                # special case for initialization
                if n_curr_iter == 0:
                    split = [0,0,0,0]
                    merge = [[],[],[],[]]
                else:
                    split = [1,0,0,0]
                    merge = [[],[],[3],[]]

                # TODO: testing filters
                #split = [1 if i > 1 else i for i in split]

                print(f"WEER split: {split}\nWEER merge: {merge}")
                print("Split indices: ", [i for i, e in enumerate(split) if e != 0])
                print("split sum: ", np.sum(split))
                print("merge sum: ", np.sum([np.count_nonzero(i) for i in merge]))

                # count each operation and segment as a check
                segs = 0
                splitting = 0
                merging = 0

                # go through each seg and split merge
                for i, seg in enumerate(segments):
                    
                    # split or merge on a segment-by-segment basis
                    if split[i] != 0:
                        # need an extra walker since split operation reduces total walkers by 1
                        # I think revo doesn't count the current seg
                        self._split_decision(bin, seg, split[i] + 1)
                        #splitting += split[i] + 1
                        splitting += split[i]
                    if len(merge[i]) != 0:
                        # list of all segs objects in the current merge list element
                        to_merge = [segment for num, segment in enumerate(segments) if num in merge[i]]
                        # adding current segment to to_merge list
                        # I think revo doesn't count the current seg
                        to_merge.append(seg)
                        
                        # cumul_weight should be the total weights of all the segments being merged
                        # cumul_weight is calculated automatically if not given
                        self._merge_decision(bin, to_merge)
                        merging += len(to_merge)
                    
                    segs += 1

                if self.do_adjust_counts:
                    self._adjust_count(ibin)

                print("Bin attrs post WEER: ", self.next_iter_binning[ibin])
                print(f"Total = {segs}, splitting = {splitting}, merging = {merging}")

        self._check_post()

        self.new_weights = self.new_weights or []

        log.debug('used initial states: {!r}'.format(self.used_initial_states))
        log.debug('available initial states: {!r}'.format(self.avail_initial_states))