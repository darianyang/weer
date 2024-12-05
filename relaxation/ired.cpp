# Load the topology and trajectory
parm alanine-dipeptide.pdb
trajin alanine-dipeptide-0-250ns.xtc

# Define the NH vectors for analysis
vector vALA @7 ired @8

# iRED matrix 
matrix ired name matired order 2

# Diagonalize IRED matrix
diagmatrix matired vecs 1 out ired.vec name ired.vec

# Calculate R1 and R2 relaxation rates and NOE
ired relax NHdist 1.02 freq 600.13 tstep 10 tcorr 5000.0 out ired.out noefile ired.noe order 2 modes ired.vec orderparamfile ired.s2
