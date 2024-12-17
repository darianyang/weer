import MDAnalysis as mda
from MDAnalysis import transformations as trans
import os

def load_u(input_top, input_xtc, step_size=1, image=False):
    """
    Load and image an md trajectory.

    Parameters
    ----------
    input_top : str
        Path to the topology file.
    input_xtc : str
        Path to the trajectory file.
    step_size : int, optional
        Step size for frames in picoseconds (default is 1 ps).
    image : bool, optional
        Whether to image the trajectory (default is False).

    Returns
    -------
    u : MDAnalysis Universe
        Universe object with the trajectory loaded.
    """
    # Load the trajectory
    print("Loading trajectory...")
    u = mda.Universe(input_top, input_xtc, in_memory=True, in_memory_step=step_size)
    print("Trajectory loaded.")
    
    if image:
        print("Guessing bonds...")
        # guess bonds
        u.atoms.guess_bonds()
        print("Bonds guessed.")

        print("Imaging...")
        # Apply imaging to account for periodic boundary conditions and center solute
        protein = u.select_atoms("protein")
        workflow = [trans.unwrap(protein),
                    trans.center_in_box(protein, center='geometry', wrap=True)]
        u.trajectory.add_transformations(*workflow)
        print("Imaging complete.")

    return u

def split_trajectory(u, output_dir, segment_ns=10, step_size=1):
    """
    Split a trajectory into smaller segments.

    Parameters
    ----------
    u : MDAnalysis Universe
        Universe object with the trajectory loaded.
    output_dir : str
        Directory to save the split trajectory files.
    segment_ns : int, optional
        Length of each segment in nanoseconds (default is 10 ns).
    step_size : int, optional
        Step size for frames in picoseconds (default is 1 ps).

    Notes
    -----
    Assumes the trajectory is saved every 1 ps (1 frame per 1 ps).
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Calculate the number of frames per segment
    frames_per_segment = segment_ns * 1000  # 10 ns * 1000 ps/ns
    total_frames = len(u.trajectory)
    
    # Determine the number of segments
    num_segments = total_frames // frames_per_segment
    print(f"Splitting trajectory with {total_frames} total frames into {num_segments} segments of {segment_ns} ns each with step size {step_size} ps.")

    for i in range(num_segments):
        # Define start and stop frames
        start_frame = i * frames_per_segment
        stop_frame = (i + 1) * frames_per_segment
        
        # Slice the trajectory
        with mda.Writer(os.path.join(output_dir, f"segment_{i+1:03d}.xtc"), n_atoms=u.atoms.n_atoms) as writer:
            for ts in u.trajectory[start_frame:stop_frame:step_size]:
                writer.write(u.atoms)
        
        print(f"Segment {i+1:03d} saved: Frames {start_frame} to {stop_frame - 1}")
    
    print("All segments have been saved.")

def reduce_trajectory(u, output_xtc, step_size=10):
    """
    Reduce the number of frames in a trajectory by sampling at the specified interval.

    Parameters
    ----------
    u : MDAnalysis Universe
        Universe object with the trajectory loaded.
    output_xtc : str
        Path to save the reduced trajectory file.
    step_size : int, optional
        Interval for sampling frames (default is 10).
    """
    # Create the writer for the reduced trajectory
    with mda.Writer(output_xtc, n_atoms=u.atoms.n_atoms) as writer:
        for ts in u.trajectory[::step_size]:
            writer.write(u.atoms)
    
    print(f"Reduced trajectory saved to {output_xtc} with interval {step_size}.")

# Example usage
input_pdb = "sim1_dry.pdb" 
input_xtc = "sim1_imaged2.xtc"  
output_dir = "t4l-10ps-imaged2"

# load universe
u = load_u(input_pdb, input_xtc, step_size=10)

# split the trajectory into 10 ps segments
split_trajectory(u, output_dir, step_size=1, segment_ns=1)

# Reduce the trajectory by a factor of 100 (for ref data)
# output_xtc_reduced = "sim1-100ps-imaged2.xtc"
# reduce_trajectory(u, output_xtc_reduced, step_size=10)
