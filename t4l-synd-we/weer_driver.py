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
        Adjust the number of walkers in a bin to match the target count.
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

    def _all_to_all_distance(self, pcoords):
        '''
        Calculate the pairwise all-to-all distances between segments.
        TODO: sort function to turn distance matrix to paired list?
        TODO: can this handle ndarrays where n > 1 ?
            I think so, if pcoord = [[0, 0, n],
                                     [0, 0, n]
                                     [0, 0, n]
                                     [...]]

        Parameters
        ----------
        pcoords : 2d array
            Segment coordinate values.
                                     
        Returns
        -------
        dist_matrix : 2d array
            Distance matrix between each segment coordinate value.
        '''
        # initialize an all-to-all matrix, with 0.0 for self distances (diagonal)
        dist_matrix = np.zeros((len(pcoords), len(pcoords)))

        # build distance matrix from pcoord value distances
        for i, pcoord_i in enumerate(pcoords):
            for j, pcoord_j in enumerate(pcoords):
                # calculate Euclidean distance between two points 
                # points can be n-dimensional
                dist = np.linalg.norm(pcoord_i - pcoord_j)
                dist_matrix[i,j] = dist
        
        return dist_matrix

    def generate_split_merge_decisions(self, segments, absurder_weights, n_splits, n_merges):
        '''
        Generate split and merge decisions based on ABSURDer weights.

        Parameters
        ----------
        segments : list
            List of segment objects.
        absurder_weights : np.ndarray
            ABSURDer weights for each segment.
        n_splits : int
            Number of segments to split.
        n_merges : int
            Number of segments to merge.    

        Returns
        -------
        split : list
            List of split decisions.
        merge : list of lists
            List of list of merge decisions.
        '''
        # Sort segments by absurder weights (smallest to largest)
        sorted_indices = np.argsort(absurder_weights)
        
        # Initialize split and merge lists
        split = [0] * len(segments)
        merge = [[] for _ in range(len(segments))]
        
        # Split the sorted segments into top and bottom parts
        # top third for splitting and bottom two-thirds for merging
        thirds_index = len(sorted_indices) // 3
        top_split_indices = sorted_indices[thirds_index * 2:]
        bottom_merge_indices = sorted_indices[:thirds_index * 2]
        # print(f"thirds index: {thirds_index}")
        # print(f"sorted indices: {sorted_indices}")
        # print(f"top (split) indices: {top_split_indices}")
        # print(f"bottom (merge) indices: {bottom_merge_indices}")    
    
        # SPLITTING: Mark the top n_splits segments for splitting
        split_segments = set()
        for i in range(n_splits):
            # TODO: eventually consider splitting a segment multiple times
            # go from the end of the top half indices to the beginning (reverse order)
            # i.e. split the largest ABSURDer weight segment first
            split_index = top_split_indices[-(i % len(top_split_indices) + 1)]
            #print(f"...SPLIT segment: {split_index}")
            # increment the split count for the segment
            split[split_index] += 1
            # mark segment being split if not already marked
            if split_index not in split_segments:
                split_segments.add(split_index)

        # MERGING: Mark the bottom n_merges segments for merging
        merged_segments = set()
        for i in range(n_merges):
            # get the segment to merge
            merge_index = bottom_merge_indices[i % len(bottom_merge_indices)]
            # if the merge_index is already being merged or split, try the next one
            # TODO: note that this while loop can also be problematic if there are no eligible merges
            while merge_index in merged_segments or merge_index in split_segments:
                #merge_index = bottom_merge_indices[i % len(bottom_merge_indices)] # this can cause an infinite loop
                merge_index = bottom_merge_indices[i]
                #print(f"......accessing eligibility of merge index: {merge_index}")
                i += 1

            # TODO: eventually consider multiple merges into one segment
            # find the nearest neighbor to merge with (skip index 0, which is the segment itself (dist 0))
            merge_partner = np.argsort(self.dist_matrix[merge_index])[1]
            # checked and the argsort looks good, merging dists look correct
            #print(f"merge index: {merge_index}, merge partner: {merge_partner}")
            #print(f"dists: {self.dist_matrix[merge_index]}")
            #print(f"sorted dists: {np.argsort(self.dist_matrix[merge_index])}")

            # if the merge_partner is same segment as merge index or is already being merged or split: 
            # find the next (eligible) nearest neighbor
            #print(f"...MERGE: current merged segments = {merged_segments}")
            if merge_partner == merge_index or merge_partner in merged_segments or merge_partner in split_segments:
                #print("......looking for eligible merge partner")
                
                # iterate through the distances to find the next nearest neighbor (starting with index 2)
                # TODO: using a while loop like above might simplify this a bit
                for j in range(2, len(self.dist_matrix[merge_index])):
                    # get the next nearest neighbor
                    merge_partner = np.argsort(self.dist_matrix[merge_index])[j]
                    # if the merge pair members are not already being merged or split
                    # then a suitable merge partner has been found, break the loop
                    if merge_partner not in merged_segments and merge_partner not in split_segments:
                        #print(f"......eligible merge found: {merge_index} and {merge_partner}")
                        break

                # if no eligible merge partner was found, (TODO: deal with this case)
                if merge_partner in merged_segments or merge_partner in split_segments:
                    print(f"..........FAILURE: no eligible merge found for: {merge_index} and {merge_partner}")
            #     else:
            #         print(f"..........attempting merge: {merge_index} and {merge_partner}")
                
            # else:
            #     print(f"......found good initial merge: {merge_index} and {merge_partner}")

            # add the eligible merge pair to the merge list and merged_segment set
            merge[merge_index].append(merge_partner)
            merged_segments.add(merge_index)
            merged_segments.add(merge_partner)
        
        return split, merge

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
        print("WE weights: \n", weights, "\nweights sum: ", np.sum(weights))

        # get final frame pcoords
        pcoords = np.array(list(map(operator.attrgetter('pcoord'), segments)))
        # pcoord for the last frame of previous iteration
        # or the first frame of current iteration
        pcoords = pcoords[:,0,:]
        print("pcoords shape: ", pcoords.shape)
        # calculate all-to-all distances
        self.dist_matrix = self._all_to_all_distance(pcoords)
        # print("dist matrix shape: ", self.dist_matrix.shape)
        # print("dist matrix: \n", self.dist_matrix)

        # get all pcoords, not just the final frame
        # curr_segments = np.array(sorted(self.current_iter_segments, 
        #                                 key=operator.attrgetter('weight')), dtype=np.object_)
        # curr_pcoords = np.array(list(map(operator.attrgetter('pcoord'), curr_segments)))
        curr_segments = sorted(self.current_iter_segments, key=operator.attrgetter('weight'))
        #curr_pcoords = np.asarray([seg.pcoord for seg in curr_segments])
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
            #print("absurder input shape: ", absurder_input.shape)

            # run reweighting
            # TODO: could also input multiple and find the best one
            theta = 100 # initial test value (TODO: optimize)
            # initialize ABSURDer object
            # TODO: testing use with initial WE weights, not sure if this will be for better or worse
            #       I should test with and without
            rw = absurder.ABSURDer(nmr_rates, absurder_input, nmr_err, thetas=np.array([theta]), w0=weights)
            # reweight according to the data corresponding to the selected index
            # in this test case, use R2 (TODO: use all rates eventually? Would just use -1)
            rw.reweight(1)
            # save the optimized weights
            absurder_weights = rw.res[theta]
            # TODO: save weights per iteration
            #np.savetxt(f"w_opt_{theta}.txt", absurder_weights)
            print("ABSURDer weights: \n", absurder_weights)
            print(f"ABSURDer phi_eff: {rw.phi_eff(absurder_weights)}")
            # TODO: eventually could include in resampler script or pull from cfg:
            #       theta value, use WE initial weights bool, which rate to use, etc.
            

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

        # dummy resampling block
        # TODO: really only using one bin
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
                    # this defaults to empty with 1 bstate
                    # tested with multiple bstates, seems to work
                    split = [0] * len(segments)
                    merge = [[] for _ in range(len(segments))]
                else:
                    # TODO: theres also the question of if I should use the top weight or the 
                    #       top weight change from the previous iteration (and if I should 
                    #       initialize the rw weights with the WE traj weights)

                    # TODO: testing with fake decision lists
                    # split = [2,0,0,0]
                    # merge = [[],[],[3,1],[]]

                    # the number of splits/merges is (max) 1/3 of the total number of segments
                    n_split_merge = len(curr_segments) // 3
                    # currently set to use same n_split and n_merge amounts
                    split, merge = self.generate_split_merge_decisions(segments, absurder_weights, 
                                                                       n_split_merge, n_split_merge)


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