      subroutine pdbread(pdbfile,resnum,nd,nr,dcos)

      parameter(nrmax=500,idim=3)
      real dcos(idim,nrmax),cn1,cn2,cn3,ch1,ch2,ch3,
     $     xx,yy,zz,rmag
      character*80 pdbfile,line,atoa
      integer      ios,nr,tmpres,resnum(nrmax)

      open(unit=11,file=pdbfile,form='formatted',status='old',
     $     iostat=ios)
      if (ios.ne.0) then
         write(*,'(/,a,a)') ' Error opening file: ', pdbfile
         stop
      endif

      j=1
50    if (j.gt.nd) goto 60
      read(11,'(a80)',end=60) line
      if (line(1:4).eq.'ATOM') then
        if (atoa(line(13:16),1).eq.'N'.or.atoa(line(13:16),1).eq.
     1      'H'.or. atoa(line(13:16),1).eq.'HN') then
          read(line(24:26),'(i3)') tmpres
          if (tmpres .gt. resnum(j)) then
              write(0,'(a,i4,a)') 'Error in R2/R1 data: residue ', 
     1          resnum(j),' is not present in pdb file'
              stop
          endif
          if (tmpres.eq.resnum(j).and.atoa(line(13:16),1).eq.'N') then
            read(line,'(31x,3(f7.3,1x))') cn1,cn2,cn3
            goto 50
          elseif (tmpres.eq.resnum(j).and.(atoa(line(13:16),1).eq.'H'
     1           .or. atoa(line(13:16),1).eq.'HN')) then
            read(line,'(31x,3(f7.3,1x))') ch1,ch2,ch3
            xx=cn1-ch1
            yy=cn2-ch2
            zz=cn3-ch3
            rmag=sqrt(xx**2+yy**2+zz**2)
            dcos(1,j)=xx/rmag
            dcos(2,j)=yy/rmag
            dcos(3,j)=zz/rmag
            j=j+1
            goto 50       
          else
            goto 50
          endif
        else
          goto 50
        endif
      else
        goto 50
      endif

60    close(11)

      nr=j-1
70    continue

c     write(*,'(/,a,i3)') ' Number of residues in database: ', nr

      return
      end

