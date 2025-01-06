       real function gasdev(idum)

c      returns a normally distributed random number.  
c      uses the Numerical Recipes randum number generator to generate
c      uniform random deviates

       real r, v1, v2, fac, gset, ran1
       integer idum,iset

       save iset,gset

       data iset/0/

       if (iset.eq.0) then
1        v1=2.*ran1(idum)-1.
         v2=2.*ran1(idum)-1.
         r=v1**2+v2**2
         if (r.ge.1.0) goto 1
         fac=sqrt(-2.*log(r)/r)
         gset=v1*fac
         gasdev=v2*fac
         iset=1
       else
         gasdev=gset
         iset=0
       endif
       return
       end


      FUNCTION ran1(idum)
      INTEGER idum,IA,IM,IQ,IR,NTAB,NDIV
      REAL ran1,AM,EPS,RNMX
      PARAMETER (IA=16807,IM=2147483647,AM=1./IM,IQ=127773,IR=2836,
     1           NTAB=32,NDIV=1+(IM-1)/NTAB,EPS=1.2e-7,RNMX=1.-EPS)
      INTEGER j,k,iv(NTAB),iy
      SAVE iv,iy
      DATA iv /NTAB*0/, iy /0/
      if (idum.le.0.or.iy.eq.0) then
        idum=max(-idum,1)
        do 11 j=NTAB+8,1,-1
          k=idum/IQ
          idum=IA*(idum-k*IQ)-IR*k
          if (idum.lt.0) idum=idum+IM
          if (j.le.NTAB) iv(j)=idum
11      continue
        iy=iv(1)
      endif
      k=idum/IQ
      idum=IA*(idum-k*IQ)-IR*k
      if (idum.lt.0) idum=idum+IM
      j=1+iy/NDIV
      iy=iv(j)
      iv(j)=idum
      ran1=min(AM*iy,RNMX)
      return
      END

