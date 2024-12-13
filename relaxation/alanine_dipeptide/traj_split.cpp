parm alanine-dipeptide.pdb
trajin alanine-dipeptide-0-250ns.xtc 

# Output the first 50 ns (0 to 50000 frames)
trajout traj_0_50ns.xtc start 0 stop 50000

# Output the second 50 ns (50001 to 100000 frames)
trajout traj_50_100ns.xtc start 50001 stop 100000

# Output the third 50 ns (100001 to 150000 frames)
trajout traj_100_150ns.xtc start 100001 stop 150000

# Output the fourth 50 ns (150001 to 200000 frames)
trajout traj_150_200ns.xtc start 150001 stop 200000

# Output the fifth 50 ns (200001 to 250000 frames)
trajout traj_200_250ns.xtc start 200001 stop 250000

run
quit
