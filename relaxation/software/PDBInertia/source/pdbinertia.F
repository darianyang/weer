      program pdbinertia

c     ver. 1.2     8Jun10 AGP

c     usage: pdbinertia -rt pdb.in pdb.out 

c     read cartesian coords from .pdb files, translate to
c     center of mass and rotate to principle axes of interia tensor

      parameter (nmax=64000)

      real*4 atcrd(3,nmax),atmass(nmax),
     1       xcen,ycen,zcen,pvec(3,3),pmom(3),mtot

      integer index(nmax),ios,oarg,iarg

      character atname(nmax)*5,pdbin*80,pdbout*80,line*80

      logical rflag, tflag

c     read command line
      rflag=.FALSE.
      tflag=.FALSE.

      if (iargc().lt. 1) then
        write(0,*) 'Usage: pdbinertia [-rt] pdb.in [pdb.out]'
        stop
      endif

      iarg=1
      call getarg(1,line)
      if (line(1:1).eq.'-') then
         iarg=2

         if (iargc().lt. 3) then
            write(0,*) 'Usage: pdbinertia [-rt] pdb.in [pdb.out]'
            stop
         endif

         if (line.eq.'-r') then
              rflag=.TRUE.
              tflag=.TRUE.
         elseif (line.eq.'-t') then
              tflag=.TRUE.
         elseif (line.eq.'-rt' .or. line.eq.'-tr') then
              rflag=.TRUE.
              tflag=.TRUE.
         else
           write(0,*) 'Error: unknown input flag'
           write(0,*) 'Usage: pdbinertia [-rt] pdb.in [pdb.out]'
           stop
         endif
      endif
              
      call getarg(iarg,pdbin)
      if(pdbin(1:1).eq.'>' .or. pdbin(1:1).eq.'<' 
     1         .or. pdbin(1:1).eq.'|') then
        write(0,*) 'Usage: pdbinertia [-rt] pdb.in [pdb.out]'
        stop
      endif
c     open the input file
      open(unit=11,file=pdbin,form='formatted',status='old',
     1     iostat=ios)
      if (ios.ne.0) then
         write(*,'(/,a,a)') ' Error opening file: ', pdbin
         stop
      endif

      if (rflag .or. tflag) then
        call getarg(3,pdbout)
        if(pdbout(1:1).eq.'>' .or. pdbout(1:1).eq.'<' 
     1         .or. pdbout(1:1).eq.'|') then
          write(0,*) 'Usage: pdbinertia [-rt] pdb.in [pdb.out]'
          stop
        endif
c       open the output file
        open(unit=12,file=pdbout,form='formatted',status='new',
     1     iostat=ios)
        if (ios.ne.0) then
           write(*,'(/,a,a)') ' Error opening file: ', pdbout
           stop
        endif
      endif

c     read input file
      natom=0
      iskip=0

10    continue
      read(11,'(a80)',END=20) line
      if (line(1:4).ne.'ATOM' .and. line(1:6).ne.'HETATM') goto 10

      natom=natom+1
      read(line,9011) index(natom),atname(natom),
     1   atcrd(1,natom),atcrd(2,natom),atcrd(3,natom)

c     here the atom type is read. Add new atom types here
c     Current atom types: H C N O S P Fe

      if (atname(natom)(1:1).eq.'H' .or. 
     1       atname(natom)(2:2).eq.'H') then
           atmass(natom)=1.00797
      elseif (atname(natom)(1:1).eq.'C' .or. (atname(natom)(1:1).eq.' '
     1       .and.atname(natom)(2:2).eq.'C')) then
           atmass(natom)=12.01115
      elseif (atname(natom)(1:1).eq.'N' .or. (atname(natom)(1:1).eq.' '
     1       .and.atname(natom)(2:2).eq.'N')) then
           atmass(natom)=14.0067
      elseif (atname(natom)(1:1).eq.'O' .or. (atname(natom)(1:1).eq.' '
     1       .and.atname(natom)(2:2).eq.'O')) then
           atmass(natom)=15.9994
      elseif (atname(natom)(1:1).eq.'S' .or. (atname(natom)(1:1).eq.' '
     1       .and.atname(natom)(2:2).eq.'S')) then
           atmass(natom)=32.06
      elseif (atname(natom)(1:1).eq.'P' .or. (atname(natom)(1:1).eq.' '
     1       .and.atname(natom)(2:2).eq.'P')) then
           atmass(natom)=30.974
      elseif (atname(natom)(1:2).eq.'FE' .or. (atname(natom)(1:1).eq.' '
     1       .and.atname(natom)(2:3).eq.'FE')) then
           atmass(natom)=55.847
      else
c          skip unknown atom types
           write(0,*) 'Unknown atom type ',atname(natom)
           iskip=iskip+1
           natom=natom-1
      endif

c     print*,index(natom),atname(natom),atmass(natom),atcrd(1,natom),
c    1        atcrd(2,natom),atcrd(3,natom)
      goto 10

20    continue

c     translate and rotate coordinates
      call cenmas(natom,atmass,atcrd,xcen,ycen,zcen,mtot)
      call trans(natom,atcrd,xcen,ycen,zcen)
      call inertia(natom,atmass,atcrd,pmom,pvec)
      if (rflag) call rotate(natom,atcrd,pvec)

      print*
      ilen=len(pdbin)
      write(6,'(a,a)') 'Input pdb file:  ',pdbin(1:ilen)
      if (rflag .or. tflag) then
        ilen=len(pdbout)
        write(6,'(a,a)') 'Output pdb file: ',pdbout(1:ilen)
      endif
      write(6,*) '# atoms read',natom,'     # atoms skipped',iskip
      print*
      write(6,'(1x,a4,16x,f15.4)') 'mass',mtot
      write(6,'(1x,a14,6x,f15.4,2x,f15.4,2x,f15.4)') 
     1     'center of mass',xcen,ycen,zcen
      write(6,'(1x,a17,3x,f15.4,2x,f15.4,2x,f15.4)') 
     1     'principle moments',pmom(1),pmom(2),pmom(3)
      write(6,'(1x,a17,3x,f15.4,2x,f15.4,2x,f15.4)') 
     1     'relative moments ',pmom(1)/pmom(1),pmom(2)/pmom(1),
     1         pmom(3)/pmom(1)
      print*
      write(6,'(27x,a15)') 'rotation matrix'
      do 11 i=1,3
        write(6,'(5x,f15.4,2x,f15.4,2x,f15.4)') (pvec(i,j),j=1,3)
11    continue

c     now output rotated pdb file if desired
      if (rflag .or. tflag) then
        i=0
        rewind(11)
50      read(11,'(a80)',end=60) line
        ilen=len(line)
        if (line(1:4).eq.'ATOM' .or. line(1:6).eq.'HETATM') then
          i=i+1
          read(line,'(6x,i5)') ii
          if (ii .eq. index(i)) then
            write(12,'(a31,f7.3,1x,f7.3,1x,f7.3,a)')
     1        line(1:31),atcrd(1,i),atcrd(2,i),atcrd(3,i),line(55:ilen)
          else
            i=i-1
            write(12,'(a80)') line
          endif
        else
            write(12,'(a80)') line
        endif
        goto 50       
        
60      close(12)
      endif

      close(11)

9011  format(6x,i5,1x,a5,13x,3f8.3)

      end


      subroutine cenmas(natom,atmass,atcrd,xcen,ycen,zcen,mass)

c     calculate center of mass of molecule 

      parameter (nmax=64000)
      real*4 atmass(nmax),atcrd(3,nmax),xcen,ycen,zcen,mass
      integer natom

      mass=0
      xcen=0
      ycen=0
      zcen=0

      do 10 i=1,natom
       mass=mass+atmass(i)
       xcen=xcen+atmass(i)*atcrd(1,i)
       ycen=ycen+atmass(i)*atcrd(2,i)
       zcen=zcen+atmass(i)*atcrd(3,i)
10    continue

      xcen=xcen/mass
      ycen=ycen/mass
      zcen=zcen/mass

      return
      end


      subroutine trans(natom,atcrd,xcen,ycen,zcen)

c     translate molecule to center of mass

      parameter (nmax=64000)
      real*4 atcrd(3,nmax),xcen,ycen,zcen
      integer natom

      do 10 i=1,natom
        atcrd(1,i)=atcrd(1,i)-xcen
        atcrd(2,i)=atcrd(2,i)-ycen
        atcrd(3,i)=atcrd(3,i)-zcen
10    continue

      return
      end


      subroutine inertia(natom,atmass,atcrd,pmom,tensor)

c     calculate inertia tensor 

      parameter (nmax=64000,ld=3)
      real*4 atmass(nmax),atcrd(3,nmax),pmom(3),
     1       tsym(3,3),temp,xrot,yrot,zrot,det,work(27),
     1       tensor(3,3),r2

      integer natom,lwork

      do 10 i=1,3
        do 20 j=1,3
          tensor(i,j)=0.0
20      continue
10    continue

      do 30 i=1,natom
         r2=atcrd(1,i)**2+atcrd(2,i)**2+atcrd(3,i)**2
         do 40 j=1,3
           do 50 k=j,3
              if (k.eq.j) then
               tensor(j,j)=tensor(j,j)+atmass(i)*(r2-atcrd(j,i)**2) 
              else
               tensor(j,k)=tensor(j,k)-atmass(i)*atcrd(j,i)*atcrd(k,i)
              endif
50         continue
40       continue
30    continue

c     now diagonalize tensor
      call ssyev('V','U',3,tensor,3,pmom,work,27,ierr)
      if (ierr.ne.0) then
	 print*,'error ',ierr
	 stop
      endif

c     normalize column vectors to form rotation matrix
      do 70 i=1,3
        temp=0.0
        do 80 j=1,3 
          temp=temp+tensor(j,i)**2
80      continue
        temp=sqrt(temp)
        do 90 j=1,3
          tensor(j,i)=tensor(j,i)/temp
90      continue
70    continue

c     resort from largest to smallest
        temp=pmom(1)
        pmom(1)=pmom(3)
        pmom(3)=temp

      do 300 i=1,3
        temp = tensor(i,1)
        tensor(i,1)=tensor(i,3)
        tensor(i,3)=temp

300   continue
          
c     transpose matrix
      do 200 i=1,3
        do 210 j=i+1,3
          temp=tensor(i,j)
          tensor(i,j)=tensor(j,i)
          tensor(j,i)=temp
210     continue
200   continue

c     force determinant to be +1 for right-handed triple
      det=tensor(1,1)*(tensor(2,2)*tensor(3,3)-tensor(3,2)*tensor(2,3))
     1   -tensor(1,2)*(tensor(2,1)*tensor(3,3)-tensor(2,3)*tensor(3,1))
     1   +tensor(1,3)*(tensor(2,1)*tensor(3,2)-tensor(2,2)*tensor(3,1))

      if (det.lt.0) then
        tensor(1,1)=-1*tensor(1,1)
        tensor(1,2)=-1*tensor(1,2)
        tensor(1,3)=-1*tensor(1,3)
      endif

      return
      end


      subroutine rotate(natom,atcrd,pvec)

c     rotate molecule using rotation matrix provided 

      parameter (nmax=64000)
      real*4 atcrd(3,nmax),pvec(3,3),xrot,yrot,zrot
      integer natom

c     now rotate coordinates
      do 100 i=1,natom
        xrot=0.0
        yrot=0.0
        zrot=0.0
        do 110 j=1,3
          xrot=xrot+pvec(1,j)*atcrd(j,i)
          yrot=yrot+pvec(2,j)*atcrd(j,i)
          zrot=zrot+pvec(3,j)*atcrd(j,i)
110     continue
        atcrd(1,i)=xrot
        atcrd(2,i)=yrot
        atcrd(3,i)=zrot
100   continue

      return
      end


