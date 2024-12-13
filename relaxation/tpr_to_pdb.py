import MDAnalysis as mda

def convert_tpr_to_pdb(tpr_file, xtc_file, output_pdb):
    """
    Convert a GROMACS .tpr file and the first frame of the associated .xtc trajectory
    into a .pdb file for cpptraj compatibility.

    Parameters
    ----------
    tpr_file : str
        Path to the GROMACS .tpr topology file.
    xtc_file : str
        Path to the GROMACS .xtc trajectory file.
    output_pdb : str
        Path to the output .pdb file.
    """
    # Load the GROMACS topology and trajectory into MDAnalysis
    u = mda.Universe(tpr_file, xtc_file)
    
    # Select the first frame
    first_frame = u.trajectory[0]
    
    # Write the first frame to a PDB file
    u.atoms.write(output_pdb)
    print(f"Converted {tpr_file} and the first frame of {xtc_file} to {output_pdb}")

# Example usage
tpr_file = "your_topology.tpr"  # Replace with your .tpr file path
xtc_file = "your_trajectory.xtc"  # Replace with your .xtc file path
output_pdb = "output.pdb"  # Output PDB file for cpptraj

convert_tpr_to_pdb(tpr_file, xtc_file, output_pdb)

