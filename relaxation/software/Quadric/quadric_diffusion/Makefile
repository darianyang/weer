obj1 = axial.o  pdbread.o pdbwrite.o svbksb.o indexx.o
obj2 = mrqmin.o  mrqcof.o covsrt.o gasdev.o getseed.o

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

#Options for linux using the g77 compiler
#FCOM=g77
#FF= -DUSE_GETSEED -DUSE_TIME8
#LF= -llapack -lblas

#Options for linux using the gfortran compiler
#FCOM=gfortran
#FF= -DUSE_GETSEED -DUSE_TIME8
#LF= -llapack -lblas

#Options for 32-bit linux using the gfortran compiler
#FCOM=gfortran
#FF= -m32 -DUSE_GETSEED -DUSE_TIME8
#LF= -m32 -llapack -lblas

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
 FF=-Xpreprocessor -DUSE_GETSEED -DUSE_TIME8 -fno-automatic -fno-second-underscore -ff2c
 LF=-static-libgfortran -Wl,-framework -Wl,Accelerate 

target: quadric_diffusion r2r1_tm

quadric_diffusion: $(obj1) $(obj2) quadric_diffusion.o
	$(FCOM)  $(obj1) $(obj2) quadric_diffusion.o -o quadric_diffusion $(LF) 

r2r1_tm: $(obj2) zbrent.o  r2r1_tm.o
	$(FCOM)  $(obj2) zbrent.o r2r1_tm.o -o r2r1_tm $(LF) 

quadric_diffusion.o: quadric_diffusion.F
	$(FCOM)  $(FF) -c quadric_diffusion.F 

r2r1_tm.o: r2r1_tm.F
	$(FCOM)  $(FF) -c r2r1_tm.F 

zbrent.o: zbrent.F
	$(FCOM)  $(FF)  -c zbrent.F  

axial.o: axial.F
	$(FCOM)  $(FF)  -c axial.F  

pdbread.o: pdbread.F
	$(FCOM)  $(FF) -c pdbread.F 

pdbwrite.o: pdbwrite.F
	$(FCOM)  $(FF) -c pdbwrite.F 

svbksb.o: svbksb.F
	$(FCOM)  $(FF) -c svbksb.F 

indexx.o: indexx.F
	$(FCOM)  $(FF) -c indexx.F 

gasdev.o: gasdev.F
	$(FCOM)  $(FF) -c gasdev.F 

getseed.o: getseed.F
	$(FCOM)  $(FF) -c getseed.F 

mrqmin.o: mrqmin.F
	$(FCOM)  $(FF)  -c mrqmin.F  

mrqcof.o: mrqcof.F
	$(FCOM)  $(FF)  -c mrqcof.F  

covsrt.o: covsrt.F
	$(FCOM)  $(FF)  -c covsrt.F  

