c     getseed.f				ver. 4.0  1Apr98  AGP
c     use system clock to get a random number seed

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

      subroutine getseed(idum)

c     NOTE: not all systems may support the 'time' subroutine

c     This is an SGI/PGF specific call
#if defined(USE_SGITIME)

      integer ihour,imin,isec,idum

      character*8 tstr

      call time(tstr)
      ihour =10*(ichar(tstr(1:1))-48) + ichar(tstr(2:2)) - 48
      imin = 10*(ichar(tstr(4:4))-48) + ichar(tstr(5:5)) - 48
      isec = 10*(ichar(tstr(7:7))-48) + ichar(tstr(8:8)) - 48
      idum = 3600*ihour + 60*imin + isec
      idum=-abs(idum)
#endif

c     This is a fairly standard call
#if defined(USE_TIME)
      integer idum

      idum=-abs(time())
#endif

c     This is a G77 call
#if defined(USE_TIME8)
      integer idum

      idum=-abs(time8())
#endif

      return
      end


