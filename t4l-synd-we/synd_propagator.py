from westpa.core.propagators import WESTPropagator
import numpy as np
import scipy.sparse as sparse
import westpa
import pickle
from westpa.core.states import InitialState, BasisState
#import mdtraj as md
from copy import deepcopy

import synd.core
from synd.models.discrete.markov import MarkovGenerator

import MDAnalysis as mda
import MDAnalysis.analysis.rms

from relax import NH_Relaxation

def get_segment_index(segment):

    data_manager = westpa.rc.data_manager

    # If this segment was just created, it doesn't have a state index, so get it from its bstate
    if segment.parent_id < 0:

        segment_state_index = get_segment_ibstate_discrete_index(segment)

    else:

        # TODO: Would be nice to avoid explicitly reading the H5 file
        iter_group = data_manager.get_iter_group(segment.n_iter)
        auxdata = iter_group['auxdata/state_indices']
        segment_state_index = auxdata[segment.seg_id][-1]

    return int(segment_state_index)


def get_segment_parent_index(segment):
    """
    For a given segment, identify the discrete index of its parent.

    For parents that are plain segments, get it from the auxdata.

    For parents that are initial/basis states, get it from the state definition's auxref.
    """

    # Note that this is invoked at the beginning of propagation, to obtain initial positions for each segment,
    #   and also in augmentation.
    # At the beginning of propagation, nothing's been written to the H5 file yet, so we can't use westpa.analysis
    #   to obtain the parent.
    # In augmentation, it's fine in theory to just look in the H5 file (i.e. with westpa.analysis), but it comes
    #   at a massive performance hit.

    data_manager = westpa.rc.get_data_manager()

    # If the parent id is >= 0, then the parent was a segment, and we can get its index directly.
    #   Otherwise, we have to get it from the ibstate auxdata
    parent_was_ibstate = segment.parent_id < 0

    if not parent_was_ibstate:

        if 'parent_final_state_index' in segment.data:
            parent_state_index = segment.data['parent_final_state_index']
            return parent_state_index

        else:
            prev_segments = data_manager.get_segments(
                n_iter=segment.n_iter - 1, seg_ids=[segment.parent_id], load_pcoords=False
            )
            parent_state_index = get_segment_index(prev_segments[0])

    # Otherwise, that means the segment was a bstate/istate
    else:
        parent_state_index = get_segment_ibstate_discrete_index(segment)

    return int(parent_state_index)


def get_segment_ibstate_discrete_index(segment):
    """
    Given an ibstate, returns the discrete index of the SynD state that generated it
    """
    sim_manager = westpa.rc.get_sim_manager()
    data_manager = westpa.rc.get_data_manager()

    istate = data_manager.get_segment_initial_states([segment])[0]

    if istate.istate_type is InitialState.ISTATE_TYPE_BASIS:

        # Without gen_istate=True, there *is* no istate corresponding to the -(basis state ID + 1).
        #   Instead, we have to get the basis state that this istate *is* (remember, with gen_istate=False initial
        #   states are mapped directly to basis states)
        # bstate_id = -(segment.parent_id + 1)
        bstate_id = istate.basis_state_id
        parent_state_index = sim_manager.current_iter_bstates[bstate_id].auxref

    elif istate.istate_type is InitialState.ISTATE_TYPE_GENERATED:
        bstate_id = istate.basis_state_id
        parent_state_index = sim_manager.current_iter_bstates[bstate_id].auxref

    elif istate.istate_type is InitialState.ISTATE_TYPE_START:
        parent_state_index = istate.basis_auxref

    else:
        raise Exception(f"Couldn't get parent state for istate {istate}")

    # It's possible the parent ib state of this segment is an H5-defined state.
    # If that's the case, then we need to look up the discrete index for that state.
    if type(parent_state_index) is not int and 'hdf:' in parent_state_index:
        # Make a dummy bstate, so we can get cached values
        dummy_bstate = BasisState(label='_', probability=0, auxref=parent_state_index)
        cached_state, (_, _, seg_id) = dummy_bstate.get_h5_cached_segment_value(
            key='auxdata/state_indices'
        )

        parent_state_index = cached_state

    return parent_state_index


def copy_segment_data():
    """
    In order to avoid any disk IO, we store each segment's discrete state as an attribute on the Segment object.
    Between iterations, we copy the parent's state index to the segment.

    Note that in this function, we're augmenting segments FOR THE NEXT ITERATION with the discrete state indices
    of segments that were run IN THE CURRENT ITERATION.
    That means, the "parent" segments here are the segments that just finished running, or the segments associated
     with cur_iter_* properties.
    In other words, this means that cur_iter_istates are NOT the istates for `next_iter_segments`!

    This runs during finalize_iteration.
    """

    sim_manager = westpa.rc.get_sim_manager()

    for segment in sim_manager.we_driver.next_iter_segments:

        segment.data["parent_final_state_index"] = get_segment_parent_index(
            segment
        )


class SynMDPropagator(WESTPropagator):
    """
    A WESTPA propagator for SynD models.
    """
    def __init__(self, rc: westpa.rc = None):
        """
        The keys loaded from WESTPA configuration are:

        - :code:`west.system.system_options.pcoord_len`: The number of steps to propagate
        - :code:`west.data.reference_pdb`: The reference PDB file for RMSD calculations

        EITHER

        - :code:`west.propagation.parameters.pcoord_map`: The path to either a pickled dictionary, mapping discrete states to progress coordinates, or to an arbitrary pickled callable that takes a discrete state and returns a progress coordinate.

        - :code:`west.propagation.parameters.transition_matrix`: The path to a transition matrix to construct the SynD propagator from.

        OR

        - :code:`west.propagation.parameters.synd_model`: The path to a saved SynD model.

        Parameters
        ----------
        rc :
            A :code:`westpa.rc` configuration object, containing :code:`west.propagation.parameters.transition_matrix`/:code:`pcoord_map`

        TODO
        ----
        Instead of creating the synD model from a transition matrix and states, just load in a SynD model.
        Then users can provide arbitrary models, discrete or not.
        However, may need to make some changes to generalize the latent space representation in the auxdata (won't be
        an integer any more).
        """

        super(SynMDPropagator, self).__init__(rc)

        rc_parameters = rc.config.get(['west', 'propagation', 'parameters'])
        # this is only needed for the h5 plugin to save coords in h5 files
        #self.topology = md.load(rc_parameters['topology'])

        if 'synd_model' in rc_parameters.keys():
            model_path = rc_parameters['synd_model']
            self.synd_model = synd.core.load_model(model_path)
        else:
            pcoord_map_path = rc_parameters['pcoord_map']
            with open(pcoord_map_path, 'rb') as inf:
                pcoord_map = pickle.load(inf)
            if type(pcoord_map) is dict:
                backmapper = pcoord_map.get
            else:
                backmapper = pcoord_map

            transition_matrix = rc_parameters['transition_matrix']
            try:
                self.transition_matrix = sparse.load_npz(transition_matrix)
            except ValueError:  # .npz file doesn't contain a sparse matrix
                with np.load(transition_matrix) as npzfile:
                    self.transition_matrix = npzfile[npzfile.files[0]]

            self.synd_model = MarkovGenerator(
                transition_matrix=self.transition_matrix,
                backmapper=backmapper,
                seed=None
            )

        # Our dynamics are propagated in the discrete space, which is recorded only in auxdata. After completing an
        #   iteration, we write the final discrete indices to the initial point auxdata of the next segments.
        # All discrete information is stored exclusively in auxdata, so that as far as the WE is concerned, it's all
        #   continuous.
        sim_manager = rc.get_sim_manager()
        sim_manager.register_callback(
            sim_manager.finalize_iteration, copy_segment_data, 1
        )

        # TODO: Would be nice to decouple this from pcoord len, so you can run dynamics at a higher resolution
        #  but only save every N
        n_steps = rc.config.get(['west', 'system', 'system_options', 'pcoord_len'])
        print(f"SynD propagator inferring {n_steps} steps per iteration from west.system.system_options.pcoord_len")

        # AKA the number of steps to take
        self.coord_len = n_steps
        self.coord_dtype = int

        self.reference_pdb = rc.config.get(['west','data', 'reference_pdb'])

    @staticmethod
    def calc_heavy_atom_rmsd(u, reference_pdb, selection="protein and not name H*"):
        """
        Calculate the RMSD of heavy atoms (or a custom selection) for a trajectory compared to a reference structure.

        Parameters
        ----------
        u : MDAnalysis.Universe
            Universe object for the trajectory.
        reference_pdb : str
            Path to the reference PDB file.
        selection : str
            Atom selection string for selecting heavy atoms. Default is "protein and not name H*".

        Returns
        -------
        numpy.ndarray
            RMSD values for each frame, shape (n_frames,).
        """
        # Load the universe for the trajectory and reference
        #u = mda.Universe(reference_pdb, traj)
        ref = mda.Universe(reference_pdb)

        # Select heavy atoms in both reference and trajectory
        ref_atoms = ref.select_atoms(selection)
        traj_atoms = u.select_atoms(selection)

        # Check that both selections have the same number of atoms
        assert ref_atoms.n_atoms == traj_atoms.n_atoms, \
            f"Atom selection mismatch: reference has {ref_atoms.n_atoms}, trajectory has {traj_atoms.n_atoms}"

        # Perform the RMSD calculation
        rmsd_analysis = MDAnalysis.analysis.rms.RMSD(traj_atoms, ref_atoms)
        rmsd_analysis.run()

        # Extract RMSD values (in Ångstroms) for all frames
        # Column 2 contains RMSD values: frame | time | RMSD
        rmsd_values = rmsd_analysis.results.rmsd[:, 2]

        return rmsd_values

    # def coords_to_pcoord(self, segment):
    #     """
    #     Given a WE segment, backmap to coords and return the progress coordinate.
    #     """
    #     syn_u = self.segment_to_syn_u(segment)
    #     pcoord = self.calc_heavy_atom_rmsd(syn_u, self.reference_pdb)
    #     return pcoord

    def backmap_to_rmsd(self, state_index):
        """
        Given a state index, return the RMSD of the backmapped coordinates.
        """
        with open('backmapper_rmsd.pkl','rb') as file:
            backmapper_rmsd = pickle.load(file)
        return backmapper_rmsd[state_index]

    def get_pcoord(self, state):
        """
        Get the progress coordinate of the given basis or initial state.
        """
        state_index = int(state.auxref)
        # calc the rmsd using the coordinates (slow)
        #state.pcoord = self.coords_to_pcoord(self.synd_model.backmap(state_index))
        # calc the rmsd using the state index (fast)
        state.pcoord = self.backmap_to_rmsd(state_index)

    def segment_to_syn_u(self, segment):
        """
        Convert a segment to an MDAnalysis Universe object.
        """
        backmapped_traj = [self.synd_model.backmap(x) for x in segment.data["state_indices"]]
        syn_u = mda.Universe(self.reference_pdb)
        syn_u.load_new(backmapped_traj, format="memory")
        return syn_u

    def calc_relax(self, segment):
        """
        Calculate relaxation rates for a given segment.

        Parameters
        ----------
        segment : westpa.core.segment.Segment
            The segment to calculate relaxation rates for.
        
        Returns
        -------
        dict
            Dictionary containing the relaxation rates and the residues they correspond to.
            Format: {"R1": np.ndarray, "R2": np.ndarray, "NOE": np.ndarray, "residues": np.ndarray}
        """
        syn_u = self.segment_to_syn_u(segment)
        relaxation = NH_Relaxation(self.reference_pdb, syn_u,
                                   max_lag=100, traj_step=10, acf_plot=False,
                                   n_exps=5, tau_c=10e-9, b0=600)
        R1, R2, NOE, S2 = relaxation.run()
        return {"R1" : R1, "R2" : R2, "NOE" : NOE, "S2" : S2,
                "residues" : relaxation.residue_indices}

    def propagate(self, segments):

        # Populate the segment initial positions
        n_segs = len(segments)

        initial_points = np.empty(n_segs, dtype=self.coord_dtype)

        for iseg, segment in enumerate(segments):
            initial_points[iseg] = get_segment_parent_index(segment)

        new_trajectories = self.synd_model.generate_trajectory(
            initial_states=initial_points,
            n_steps=self.coord_len
        )

        for iseg, segment in enumerate(segments):
            segment.data["state_indices"] = new_trajectories[iseg, :]

            # use separate backmapper to get rmsd
            segment.pcoord = np.array([
                self.backmap_to_rmsd(x) for x in segment.data["state_indices"]
                ]).reshape(self.coord_len, -1)

            # use backmapper to get coords and do rmsd calc
            # segment.pcoord = np.array([
            #     self.coords_to_pcoord(self.synd_model.backmap(x)) for x in segment.data["state_indices"]
            #     ]).reshape(self.coord_len, -1)
            # updated version
            # segment.pcoord = self.calc_heavy_atom_rmsd(self.segment_to_syn_u(segment), 
            #                                            self.reference_pdb).reshape(self.coord_len, -1)
            
            # use pure synd backmapper (coords return)
            # segment.pcoord = np.array([
            #     self.synd_model.backmap(x) for x in segment.data["state_indices"]
            # ]).reshape(self.coord_len, -1)

            # calc aux data
            relaxation_data = self.calc_relax(segment)
            segment.data["R1"] = relaxation_data["R1"]
            segment.data["R2"] = relaxation_data["R2"]
            segment.data["NOE"] = relaxation_data["NOE"]
            segment.data["S2"] = relaxation_data["S2"]
            segment.data["residues"] = relaxation_data["residues"]

            # # For H5 plugin
            # if westpa.rc.get_data_manager().store_h5:

            #     # TODO: how to handle restart data?
            #     #       I think we just don't for SynD, but let's silence that warning.
            #     segment.data['iterh5/restart'] = None

            #     full_coordinate_trajectory = np.array([
            #         self.synd_model.backmap(x, 'full_coordinates')
            #         for x in segment.data["state_indices"]
            #     ])

            #     # To mimic the behavior of a saved MD trajectory, we omit the first point.
            #     # I don't love this, but it's consistent with the OpenMM propagator.
            #     # TODO: Change this -- it's inconsistent with how pcoords are saved, and isn't quite right
            #     #   for the haMSM plugin.
            #     self.topology.xyz = full_coordinate_trajectory[1:]
            #     self.topology.time = np.arange(self.coord_len-1) + \
            #                          (self.coord_len-1) * westpa.rc.get_sim_manager().n_iter

            #     # TODO: Avoid this copy! Probably super slow
            #     segment.data['iterh5/trajectory'] = deepcopy(self.topology)

            segment.status = segment.SEG_STATUS_COMPLETE

        return segments
