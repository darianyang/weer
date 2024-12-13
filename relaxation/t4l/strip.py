import parmed as pmd

def strip_water_and_ions(topology_file, output_topology):
    """
    Strip water and ions from a GROMACS .tpr topology file using ParmEd.

    Parameters
    ----------
    topology_file : str
        Path to the input GROMACS .tpr topology file.
    output_topology : str
        Path to save the stripped GROMACS .tpr file.
    """
    # Load the topology file
    topology = pmd.load_file(topology_file)

    # Strip water and ions (common names: SOL, HOH, WAT, NA, CL, MG, etc.)
    stripped_topology = topology["!:SOL & !:HOH & !:WAT & !:NA & !:CL & !:K & !:MG & !:CA"]

    # Save the stripped topology file
    stripped_topology.save(output_topology, format='gromacs')
    print(f"Stripped topology saved to {output_topology}")


# Example usage
topology_file = "sim1.tpr" 
output_topology = "sim1_dry.tpr" 

strip_water_and_ions(topology_file, output_topology)

