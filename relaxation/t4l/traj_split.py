import MDAnalysis as mda
import os

def split_trajectory(input_pdb, input_xtc, output_dir, segment_ns=10, step_size=1):
    """
    Split a trajectory into smaller segments.

    Parameters
    ----------
    input_pdb : str
        Path to the PDB topology file.
    input_xtc : str
        Path to the trajectory file.
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
    
    # Load the trajectory
    u = mda.Universe(input_pdb, input_xtc, step_size=step_size)
    
    # Calculate the number of frames per segment
    frames_per_segment = segment_ns * 1000  # 10 ns * 1000 ps/ns
    total_frames = len(u.trajectory)
    
    # Determine the number of segments
    num_segments = total_frames // frames_per_segment
    print(f"Splitting trajectory into {num_segments} segments of {segment_ns} ns each with step size {step_size} ps.")
    
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

def reduce_trajectory(input_pdb, input_xtc, output_xtc, interval=10):
    """
    Reduce the number of frames in a trajectory by sampling at the specified interval.

    Parameters
    ----------
    input_pdb : str
        Path to the PDB topology file.
    input_xtc : str
        Path to the trajectory file.
    output_xtc : str
        Path to save the reduced trajectory file.
    interval : int, optional
        Interval for sampling frames (default is 10).
    """
    # Load the trajectory
    u = mda.Universe(input_pdb, input_xtc)
    
    # Create the writer for the reduced trajectory
    with mda.Writer(output_xtc, n_atoms=u.atoms.n_atoms) as writer:
        for ts in u.trajectory[::interval]:
            writer.write(u.atoms)
    
    print(f"Reduced trajectory saved to {output_xtc} with interval {interval}.")

# Example usage
input_pdb = "sim1_dry.pdb" 
input_xtc = "sim1.xtc"  
output_dir = "t4l-10ps"

#split_trajectory(input_pdb, input_xtc, output_dir, step_size=10)

# Reduce the trajectory by a factor of 100
output_xtc_reduced = "sim1-100ps.xtc"
reduce_trajectory(input_pdb, input_xtc, output_xtc_reduced, interval=100)
