      subroutine pdbread(pdbfile,resnum,xname,nd,nr,dcos)

      parameter(nrmax=256,idim=3)
      real cn1,cn2,cn3,ch1,ch2,ch3,
     1     xx,yy,zz,rmag,dcos(idim,nrmax)
      character*80 pdbfile,line
      character*2 xname,hname
      integer      ios,nr,tmpres,resnum(nrmax)

      if (xname.eq.'N') hname = 'H'
      if (xname.eq.'N1') hname = 'H1'
      if (xname.eq.'N3') hname = 'H3'
      if (xname.eq.'CA') hname = 'HA'

      open(unit=11,file=pdbfile,form='formatted',status='old',
     1     iostat=ios)
      if (ios.ne.0) then
         write(*,'(/,a,a)') ' Error opening file: ', pdbfile
         stop
      endif

      j=1
50    if (j.gt.nd) goto 60
      read(11,'(a80)',end=60) line

      if (line(1:4).eq.'ATOM' .and. 
     1  (line(14:15).eq.xname .or.line(14:15).eq. hname)) then
          read(line(24:26),'(i3)') tmpres

          if (tmpres .gt. resnum(j)) then
              write(0,'(a,i4,a)') 'Error in tm data: residue ', 
     1          resnum(j),' is not present in pdb file'
              stop
          endif
          if (tmpres.eq.resnum(j).and.line(14:15).eq.xname) then
            read(line,'(31x,3(f7.3,1x))') cn1,cn2,cn3
            goto 50
          elseif (tmpres.eq.resnum(j).and.line(14:15).eq.hname) then
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

60    close(11)

      nr=j-1
70    continue

c     write(*,'(/,a,i3)') ' Number of residues in database: ', nr

      return
      end

