import MDAnalysis as mda
import os

def split_trajectory(input_tpr, input_xtc, output_dir, segment_ns=10):
    """
    Split a trajectory into smaller segments.

    Parameters
    ----------
    input_tpr : str
        Path to the GROMACS .tpr topology file.
    input_xtc : str
        Path to the GROMACS .xtc trajectory file.
    output_dir : str
        Directory to save the split trajectory files.
    segment_ns : int, optional
        Length of each segment in nanoseconds (default is 10 ns).

    Notes
    -----
    Assumes the trajectory is saved every 1 ps (1 frame per 1 ps).
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Load the trajectory
    u = mda.Universe(input_tpr, input_xtc)
    
    # Calculate the number of frames per segment
    frames_per_segment = segment_ns * 1000  # 10 ns * 1000 ps/ns = 10,000 frames
    total_frames = len(u.trajectory)
    
    # Determine the number of segments
    num_segments = total_frames // frames_per_segment
    print(f"Splitting trajectory into {num_segments} segments of {segment_ns} ns each.")
    
    for i in range(num_segments):
        # Define start and stop frames
        start_frame = i * frames_per_segment
        stop_frame = (i + 1) * frames_per_segment
        
        # Slice the trajectory
        with mda.Writer(os.path.join(output_dir, f"segment_{i+1:03d}.xtc"), n_atoms=u.atoms.n_atoms) as writer:
            for ts in u.trajectory[start_frame:stop_frame]:
                writer.write(u.atoms)
        
        print(f"Segment {i+1:03d} saved: Frames {start_frame} to {stop_frame - 1}")
    
    print("All segments have been saved.")

# Example usage
input_tpr = "sim1.tpr"  # Replace with your .tpr file
input_xtc = "sim1.xtc"  # Replace with your .xtc file
output_dir = "t4l"      # Directory to save split trajectories

split_trajectory(input_tpr, input_xtc, output_dir)
