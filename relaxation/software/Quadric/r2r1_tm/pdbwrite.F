      subroutine pdbwrite(pdbfile,outfile,rotmat)

      real rotmat(3,3),vec(3),rotvec(3)
      character*80 pdbfile,outfile,line
      integer      ios

      open(unit=11,file=pdbfile,form='formatted',status='old',
     1     iostat=ios)
      if (ios.ne.0) then
         write(*,'(/,a,a)') ' Error opening file: ', pdbfile
         stop
      endif

      open(unit=12,file=outfile,form='formatted',status='new',
     1     iostat=ios)
      if (ios.ne.0) then
         write(*,'(/,a,a)') ' Error opening file: ', outfile
         stop
      endif

50    read(11,'(a80)',end=60) line
      ilen=len(line)
      if (line(1:4).eq.'ATOM') then
        read(line,'(31x,3(f7.3,1x))') vec(1),vec(2),vec(3)
        rotvec(1)=rotmat(1,1)*vec(1) +
     1    rotmat(1,2)*vec(2)+rotmat(1,3)*vec(3)
        rotvec(2)=rotmat(2,1)*vec(1) +
     1    rotmat(2,2)*vec(2)+rotmat(2,3)*vec(3)
        rotvec(3)=rotmat(3,1)*vec(1) +
     1    rotmat(3,2)*vec(2)+rotmat(3,3)*vec(3)
        write(12,'(a31,f7.3,1x,f7.3,1x,f7.3,a)')
     1    line(1:31),rotvec(1),rotvec(2),rotvec(3),line(55:ilen)
      else
        write(12,'(a80)') line
      endif
      goto 50       

60    close(11)
      close(12)

      return
      end

