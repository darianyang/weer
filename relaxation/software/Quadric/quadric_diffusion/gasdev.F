c     gasdev.f				ver. 4.0  1Apr98  agp
c     random number generators
c     the routines are from the text Numerical Recipes

c     Copyright (C) 1998  Arthur G. Palmer
c
c          Arthur G. Palmer
c          Department of Biochemistry and Molecular Biophysics
c          Columbia University
c          630 West 168th Street
c          New York, NY 10032
c          email: agp6@columbia.edu

c     This program is free software; you can redistribute it and/or
c     modify it under the terms of the GNU General Public License
c     as published by the Free Software Foundation; either version 2
c     of the License, or any later version.

c     This program is distributed in the hope that it will be useful,
c     but WITHOUT ANY WARRANTY; without even the implied warranty of
c     MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
c     GNU General Public License for more details.

c     You should have received a copy of the GNU General Public License
c     along with this program; if not, write to the
c           Free Software Foundation, Inc.
c           59 Temple Place - Suite 330
c           Boston, MA 02111-1307, USA.

c     This program uses some software routines copyrighted by 
c     Numerical Recipes Software. You should obtain a license from 
c     Numerical Recipes if you do not have one already. You can obtain 
c     an academic workstation license by sending your name, address, 
c     email address, workstation hostname, workstation
c     internet address, workstation brand and model number, and a check
c     for $50.00 to
c           Numerical Recipes Software
c           P.O. Box 243
c           Cambridge, MA 02238
c     Be certain to state that you want the FORTRAN version of the
c     software.

      function gasdev(idum)
      integer idum
      real gasdev
cu    uses ran1
      integer iset
      real fac,gset,rsq,v1,v2,ran1
      save iset,gset
      data iset/0/
      if (iset.eq.0) then
1       v1=2.*ran1(idum)-1.
        v2=2.*ran1(idum)-1.
        rsq=v1**2+v2**2
        if(rsq.ge.1..or.rsq.eq.0.)goto 1
        fac=sqrt(-2.*log(rsq)/rsq)
        gset=v1*fac
        gasdev=v2*fac
        iset=1
      else
        gasdev=gset
        iset=0
      endif
      return
      end


      function ran1(idum)
      integer idum,ia,im,iq,ir,ntab,ndiv
      real ran1,am,eps,rnmx
      parameter (ia=16807,im=2147483647,am=1./im,iq=127773,ir=2836,
     1           ntab=32,ndiv=1+(im-1)/ntab,eps=1.2e-7,rnmx=1.-eps)
      integer j,k,iv(ntab),iy
      save iv,iy
      data iv /ntab*0/, iy /0/
      if (idum.le.0.or.iy.eq.0) then
        idum=max(-idum,1)
        do 11 j=ntab+8,1,-1
          k=idum/iq
          idum=ia*(idum-k*iq)-ir*k
          if (idum.lt.0) idum=idum+im
          if (j.le.ntab) iv(j)=idum
11      continue
        iy=iv(1)
      endif
      k=idum/iq
      idum=ia*(idum-k*iq)-ir*k
      if (idum.lt.0) idum=idum+im
      j=1+iy/ndiv
      iy=iv(j)
      iv(j)=idum
      ran1=min(am*iy,rnmx)
      return
      end
