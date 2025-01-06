c     charconv.f			ver. 4.0  1Apr98  AGP
c     functions to convert a string to a integer or floating point number

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

      subroutine rline(iunit,ibuf,ierr)

c     ----- routine to read a line from a file -----

      integer iunit,ierr,ilen,i
      character*1 IBLK,ISHRP,ITAB
      character*(*) ibuf

      data IBLK,ISHRP/' ','#'/

      ITAB=char(9)
      ierr=0
 
111   read(iunit,('(a80)'),end=800,iostat=ierr) ibuf
      if (ierr.ne.0) return

      ilen=len(ibuf)
      if (ilen.eq.0 .or. ibuf(1:1).eq.ISHRP) goto 111
      do 112 i=1,ilen
        if (ibuf(i:i).ne.IBLK .and. ibuf(i:i).ne.ITAB) goto 113
112   continue
      goto 111
      
800   ierr=9
113   return
 
      end


      character*(*) function atoa(ibuf,indx)

      integer indx,ip,ilen,istart,iend,i
      character*1 IBLK,ITAB
      character*(*) ibuf

      data IBLK/' '/

      ITAB=char(9)

      ip = 0
      ilen = len(ibuf) 

      if (ilen.eq.0) then
         atoa = ''
         return
      endif

      do 10 i=1,indx

c       check for leading blanks
240     continue
        ip = ip+1
        if (ip.gt.ilen) then
           atoa = ''
           return
        endif 
        if (ibuf(ip:ip).eq.IBLK .or. ibuf(ip:ip).eq.ITAB) goto 240

c       reading field
        istart = ip
250     continue
        ip = ip+1
        if (ip.le.ilen .and. ibuf(ip:ip).ne.IBLK .and. 
     1      ibuf(ip:ip).ne.ITAB) goto 250
        ip=ip-1
        iend=ip

10    continue
      atoa = ibuf(istart:iend)
      return
 
      end


      integer function atoi(ibuf,indx)

      integer indx,ip,ilen,istart,iend,i
      character*1 IBLK,ITAB
      character*(*) ibuf

      data IBLK/' '/

      ITAB=char(9)

      ip = 0
      ilen = len(ibuf) 

      if (ilen.eq.0) then
         atoi = 0
         return
      endif

      do 10 i=1,indx

c       check for leading blanks
240     continue
        ip = ip+1
        if (ip.gt.ilen) then
           atoi = 0
           return
        endif 
        if (ibuf(ip:ip).eq.IBLK .or. ibuf(ip:ip).eq.ITAB) goto 240

c       reading field
        istart = ip
250     continue
        ip = ip+1
        if (ip.le.ilen .and. ibuf(ip:ip).ne.IBLK .and. 
     1      ibuf(ip:ip).ne.ITAB) goto 250
        ip=ip-1
        iend=ip

10    continue
      read (ibuf(istart:iend),*) atoi
       
      return
      end


      function atof(ibuf,indx)

      double precision atof

      integer indx,ip,ilen,istart,iend,i
      character*1 IBLK,ITAB
      character*(*) ibuf

      data IBLK/' '/

      ITAB=char(9)

      ip = 0
      ilen = len(ibuf) 

      if (ilen.eq.0) then
         atof = 0
         return
      endif

      do 10 i=1,indx

c       check for leading blanks
240     continue
        ip = ip+1
        if (ip.gt.ilen) then
           atof = 0
           return
        endif 
        if (ibuf(ip:ip).eq.IBLK .or. ibuf(ip:ip).eq.ITAB) goto 240

c       reading field
        istart = ip
250     continue
        ip = ip+1
        if (ip.le.ilen .and. ibuf(ip:ip).ne.IBLK .and. 
     1      ibuf(ip:ip).ne.ITAB) goto 250
        ip=ip-1
        iend=ip

10    continue
      read (ibuf(istart:iend),*) atof

      return
 
      end

