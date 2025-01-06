#Options for SGI IRIS
#FCOM=f77
#FF= -O1 -n32 -DSGI -DUSE_GETSEED -DSGI_FS0 -DUSE_DATE -DUSE_SGITIME
#LF= -lcomplib.sgimath
#CCOM= cc -n32 -O1

#Options for SUN
#FCOM=f77
#FF= -O1
#LF= -llapack -lblas

#Options for CONVEX
#FCOM=fc
#FF= -O1 -fpp
#LF= -lveclib -llapack

#Options for linux using Portland Group Compiler
#FCOM=pgf77
#FF= -DUSE_GETSEED -DUSE_SGITIME -DUSE_DATE
#LF= -llapack -lblas

#Options for linux using the gfortran compiler
#FCOM=gfortran
#FF= -DUSE_GETSEED -DUSE_TIME8
#LF= -llapack -lblas

#Options for 32-bit linux using the gfortran compiler
#FCOM=gfortran
#FF= -m32 -DUSE_GETSEED -DUSE_TIME8
#LF= -m32 -llapack -lblas

#Options for linux using the g77 compiler
#FCOM=g77
#FF= -DUSE_GETSEED -DUSE_TIME8
#LF= -llapack -lblas

#Options for Mac G5 CPU OSX using GNU g77
#FCOM=g77
#FF= -O3 -xf77-cpp-input -mcpu=970 -mpowerpc64 -mtune=970 -mpowerpc-gpopt -force_cpusubtype_ALL
#LF= -Wl,-F/System/Library/Frameworks -Wl,-framework -Wl,vecLib

#Options for Mac G4 CPU OSX using GNU g77
#FCOM=g77
#FF= -O3 -xf77-cpp-input -mcpu=7450 -mtune=7450
#LF= -Wl,-F/System/Library/Frameworks -Wl,-framework -Wl,vecLib

#Flags for Mac OS X using gfortran
FCOM= /usr/local/bin/gfortran
FF=-Xpreprocessor -DUSE_GETSEED -DUSE_TIME8 -fno-automatic -fno-second-underscore
LF=-static-libgfortran -Wl,-framework -Wl,Accelerate

pdbinertia: pdbinertia.o 
	$(FCOM) pdbinertia.o $(LF) -o pdbinertia 

pdbinertia.o: pdbinertia.F
	$(FCOM) -O2 $(FF) -c pdbinertia.F


