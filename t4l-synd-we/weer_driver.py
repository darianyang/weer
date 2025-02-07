import logging

import numpy as np
import operator
from westpa.core.we_driver import WEDriver

#from weer import WEER
import absurder

log = logging.getLogger(__name__)

class WEERDriver(WEDriver):
    '''
    This introduces a weight resampling procedure before or after (TODO)
    the split/merge resampling.
    '''

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
        print("weights: ", weights)

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
            rw.reweight(1)
            # save the optimized weights
            np.savetxt(f"w_opt_{theta}.txt", rw.res[theta])
            

        # TODO: grab data for multiple iterations (could also use this to get aux data?)
        #       maybe, would need to be careful of the seg ordering
        # I thought it would be useful to use multiple iterations of data
        # but actually this isn't really feasible since I am only optimizing 
        # a single set of weights corresponding to one iteration
        # n_curr_iter = self.rc.data_manager.current_iteration
        # print("current iter:", n_curr_iter)
        # print(self.rc.data_manager.get_iter_group(n_curr_iter)['pcoord'][:].shape)
        # import sys ; sys.exit(0)

        # # only use WEER on specific iterations
        # n_curr_iter = self.rc.data_manager.current_iteration
        # #if n_curr_iter in [50, 100, 150, 200, 250, 300, 400, 500]:
        # if n_curr_iter in [i for i in range(50, 1001, 50)]:
        #     # initialize WEER
        #     true_dist = np.loadtxt("true_1d_odld.txt")
        #     reweight = WEER(curr_pcoords, weights, true_dist)
        #     # opt for updated weights
        #     weights = reweight.run_weer()
        #     # set new weights to each segment
        #     # for i, seg in enumerate(segments):
        #     #     seg.weight = weights[i]
        #     for seg, weight in zip(segments, weights):
        #         seg.weight = weight

        reports = segments.copy()
        #report_weights = np.array(list(map(operator.attrgetter('weight'), reports)))
        report_weights = np.asarray([seg.weight for seg in reports])
        #print("OPT SEG WEIGHTS: ", np.array(list(map(operator.attrgetter('weight'), segments))))
        #print("OPT SEG WEIGHTS UNSORTED: ", report_weights)
        #print("OPT SEG WEIGHTS SORTED: ", np.asarray(sorted(report_weights)))

        # TODO: NEXT run using bins

        # Regardless of current particle count, always split overweight particles and merge underweight particles
        # Then and only then adjust for correct particle count
        total_number_of_subgroups = 0
        total_number_of_particles = 0
        for (ibin, bin) in enumerate(self.next_iter_binning):
            if len(bin) == 0:
                continue

            # Splits the bin into subgroups as defined by the called function
            target_count = self.bin_target_counts[ibin]
            subgroups = self.subgroup_function(self, ibin, **self.subgroup_function_kwargs)
            total_number_of_subgroups += len(subgroups)
            # grab segments and weights
            segments = np.array(sorted(bin, key=operator.attrgetter('weight')), dtype=np.object_)
            weights = np.array(list(map(operator.attrgetter('weight'), segments)))
            #print("BIN WEIGHT: ", weights)

            # Calculate ideal weight and clear the bin
            ideal_weight = weights.sum() / target_count
            bin.clear()
            # Determines to see whether we have more sub bins than we have target walkers in a bin (or equal to), and then uses
            # different logic to deal with those cases.  Should devolve to the Huber/Kim algorithm in the case of few subgroups.
            if len(subgroups) >= target_count:
                for i in subgroups:
                    # Merges all members of set i.  Checks to see whether there are any to merge.
                    if len(i) > 1:
                        (segment, parent) = self._merge_walkers(
                            list(i),
                            np.add.accumulate(np.array(list(map(operator.attrgetter('weight'), i)))),
                            i,
                        )
                        i.clear()
                        i.add(segment)
                    # Add all members of the set i to the bin.  This keeps the bins in sync for the adjustment step.
                    bin.update(i)

                if len(subgroups) > target_count:
                    self._adjust_count(bin, subgroups, target_count)

            if len(subgroups) < target_count:
                for i in subgroups:
                    self._split_by_weight(i, target_count, ideal_weight)
                    self._merge_by_weight(i, target_count, ideal_weight)
                    # Same logic here.
                    bin.update(i)
                if self.do_adjust_counts:
                    # A modified adjustment routine is necessary to ensure we don't unnecessarily destroy trajectory pathways.
                    self._adjust_count(bin, subgroups, target_count)
            if self.do_thresholds:
                for i in subgroups:
                    self._split_by_threshold(bin, i)
                    self._merge_by_threshold(bin, i)
                for iseg in bin:
                    if iseg.weight > self.largest_allowed_weight or iseg.weight < self.smallest_allowed_weight:
                        log.warning(
                            f'Unable to fulfill threshold conditions for {iseg}. The given threshold range is likely too small.'
                        )
            total_number_of_particles += len(bin)
        log.debug('Total number of subgroups: {!r}'.format(total_number_of_subgroups))

        self._check_post()

        self.new_weights = self.new_weights or []

        log.debug('used initial states: {!r}'.format(self.used_initial_states))
        log.debug('available initial states: {!r}'.format(self.avail_initial_states))