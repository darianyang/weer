      program r2r1_diffusion 
c                                          ver 1.11 20Jun00 AGP
c     Fit diffusion constants to R2/R1 data

      parameter (nmax=500,mmax=6,imax=500)
      parameter (mfiso=1,mfax=4)

      real r2r1(nmax), sig(nmax), yax(nmax),yiso(nmax),
     1     a(mmax), aiso(mmax), atrue(mmax), siga(mmax), 
     1     sigaiso(mmax), sigat(mmax),asim(mmax), 
     1     resiso(2), chiiso(imax),fsim(imax),
     1     resax(mmax,2), chiax(imax),
     1     orient(3,nmax),wh,resind(nmax),rotmat(3,3)

      integer iseed,idum,resnum(nmax)

      real gasdev
      external gasdev, getseed
      character*80 line,mdffile,pdbfile,outfile,cfile

      common /rvec/ orient
      common /const/ wh

      if (iargc().lt. 1) then
        write(0,*) 'Usage: r2r1_diffusion input.file'
        stop
      endif

      call getarg(1,cfile)
      open(unit=13,file=cfile,form='formatted',status='old',
     1     iostat=ios)
      if (ios.ne.0) then
         write(0,'(/,a,a)') ' Error opening file: ', cfile
         stop
      endif

100   read(13,'(a80)') line
      if (line(1:1).eq.'#') goto 100
      read(line,*) ndata,wh,nsim

200   read(13,'(a80)') line
      if (line(1:1).eq.'#') goto 200
      read(line,*) atrue(1),atrue(2),atrue(3),atrue(4)
      aiso(1)=atrue(1)
      call getseed(idum)
      iseed=idum
      if (idum.gt.0) idum=-idum

250   read(13,'(a80)') line
      if (line(1:1).eq.'#') goto 250
      read(line,*) ratlo,rathi,nsteps

c     get R2/R1 ratios and uncertainties
300   read(13,'(a80)') mdffile
      if (mdffile(1:1).eq.'#') goto 300

      open(unit=11,file=mdffile,form='formatted',status='old',
     $     iostat=ios)
      if (ios.ne.0) then
         write(0,'(/,a,a)') ' Error opening file: ', mdffile
         stop
      endif

      j=0
45    read(11,'(a80)',end=49) line
      if (line(1:1).eq.'#' .or. line.eq.'') then
         goto 45
      else
         j=j+1
         read(line,*) resnum(j),r2r1(j),sig(j)
         resind(j)=j+0.0
         goto 45
      endif

49    close(11)
      if (j.ne.ndata) then
         print*,'Error: input R2/R1 data has incorrect entries',
     1    j 
         stop
      endif

c     get coordinates
400   read(13,'(a80)') pdbfile
      if (pdbfile(1:1).eq.'#') goto 400
550   read(13,'(a80)') outfile
      if (outfile(1:1).eq.'#') goto 550

      close(13)

      call pdbread(pdbfile,resnum,ndata,nr,orient)
      if (nr.ne.ndata) then
         print*,'Error: input R2/R1 data does not match pdb file',
     1     ndata,nr
         stop
      endif

      print*
      print*
      print*,'     Least-Squares R2/R1 Diffusion Analysis'
      print*

      write(6,'(10x,a)')
     1  '******************** Input Data ********************'
      write(6,'(a,8x,a,10x,a,11x,a,14x,a,14x,a)')
     1      ' Residue','R2/R1','dR2/R1','x','y','z'
      chiaxb=0.0
      do 10 i=1,ndata
       write(6,'(i5,5x,f12.5,3x,f12.5,3x,f12.5,3x,f12.5,3x,f12.5)') 
     1   resnum(i),r2r1(i),sig(i),orient(1,i),
     1   orient(2,i),orient(3,i)
        call jwaxial(resind(i),atrue,ytemp,mfax)
        chiaxb=chiaxb+(r2r1(i)-ytemp)**2/sig(i)**2
10    continue

      print*
      write(6,'(10x,a)')
     1     '***** Grid Search for Initial Parameter Values *****'
      write(6,'(a,8x,a,8x,a,3x,a,5x,a,8x,a)')
     1   '   X2new','X2old','Diso/10^7','Dpar/Dper','Theta','Phi'
      write(6,'(f10.2,3x,f10.2,3x,f10.5,3x,f10.5,3x,f10.5,3x,f10.5)') 
     1   chiaxb,chiaxb,atrue(1)*1.0e-7,atrue(2),atrue(3),atrue(4)

c     now grid search for initial guesses
      dinput=atrue(1)
      do 450, jj=-5,5
        a(1)=dinput*(1+jj/40.0)
        do 500, i=0,nsteps
          a(2)=ratlo + i*(rathi-ratlo)/nsteps
          do 501, j=1,9
            a(3)=0.174533*j
            do 502, k=1,36
              a(4)=0.174533*k
              chitry=0
              do 503, ii=1,ndata
                 call jwaxial(resind(ii),a,ytemp,mfax)
                 chitry=chitry+(r2r1(ii)-ytemp)**2/sig(ii)**2
503           continue
              if (chitry.lt.chiaxb) then
               write(6,'(f10.2,3x,f10.2,3x,f10.5,3x,
     1            f10.5,3x,f10.5,3x,f10.5)') 
     1            chitry,chiaxb,a(1)*1.0e-7,a(2),a(3),a(4)
                chiaxb=chitry
                atrue(1)=a(1)
                atrue(2)=a(2)
                atrue(3)=a(3)
                atrue(4)=a(4)
              endif
502         continue
501       continue
500     continue
450   continue

      call isofit(resind,r2r1,sig,ndata,aiso,sigaiso,mfiso,chiisob)
      call axialfit(resind,r2r1,sig,ndata,atrue,sigat,mfax,chiaxb)

      if (nsim.eq.0) goto 99

       write(0,*)
       write(0,'(a,$)') 'Simulations '
       do 1000 index=1,nsim
         if (index/10 .eq. index/10.0)
     1      write(0,'(a,$)') '. '
         chitry0=0.0
         do 1003 j=1,ndata
              call jwiso(aiso(1),yiso(j))
              yiso(j)=yiso(j)+gasdev(idum)*sig(j)
              call jwaxial(resind(j),atrue,yax(j),mfax)
              rtemp=gasdev(idum)
              yax(j)=yax(j)+rtemp*sig(j)
              chitry0=chitry0+rtemp**2
1003     continue

         a(1)=aiso(1)
         call isofit(resind,yiso,sig,ndata,a,siga,mfiso,chiiso(index))
         resiso(1)=resiso(1)+a(1)
         resiso(2)=resiso(2)+a(1)**2

c        now grid search for initial guesses
         asim(1)=atrue(1)
         asim(2)=atrue(2)
         asim(3)=atrue(3)
         asim(4)=atrue(4)

         a(1)=aiso(1)
         do 520, i=0,5
          a(2)=0.9 + i*0.05
          do 521, j=0,9
            a(3)=0.174533*j
            do 522, k=0,10
              a(4)=0.314159*k
              chitry=0
              do 523, ii=1,ndata
                 call jwaxial(resind(ii),a,ytemp,mfax)
                 chitry=chitry+(yiso(ii)-ytemp)**2/sig(ii)**2
523           continue
              if (chitry.lt.chitry0) then
                chitry0=chitry
                asim(2)=a(2)
                asim(3)=a(3)
                asim(4)=a(4)
              endif
522         continue
521       continue
520      continue
         call axialfit(resind,yiso,sig,ndata,asim,siga,mfax,chisq)
         fsim(index)=
     1      (chiiso(index)-chisq)*(ndata-mfax)/(chisq*(mfax-mfiso))

         a(1)=atrue(1)
         a(2)=atrue(2)
         a(3)=atrue(3)
         a(4)=atrue(4)

         call axialfit(resind,yax,sig,ndata,a,siga,mfax,chiax(index))

         do 1004 i=1,mfax
            resax(i,1)=resax(i,1)+a(i)
            resax(i,2)=resax(i,2)+a(i)**2
1004     continue

1000   continue

       resiso(1)=resiso(1)/nsim
       resiso(2)=resiso(2)/nsim - resiso(1)**2
       if (resiso(2).gt.0) then
            resiso(2)=sqrt(resiso(2))
       else
            resiso(2)=0.0
       endif
       do 1005 i=1,mfax
          resax(i,1)=resax(i,1)/nsim
          resax(i,2)=resax(i,2)/nsim - resax(i,1)**2
          if (resax(i,2).gt.0) then
            resax(i,2)=sqrt(resax(i,2))
          else
            resax(i,2)=0.0
          endif
1005   continue

       call qcksrt(nsim,chiiso)
       call qcksrt(nsim,chiax)
       call qcksrt(nsim,fsim)

99    continue
      print*
      print*
      write(6,'(10x,a)')
     1       '**************** Isotropic Results *****************'
      if (nsim.gt.0) then
        print*,'Parameters          Actual                 Simulation'
        write(6,199) 'Diso (1/s)',aiso(1),' +/- ',sigaiso(1),
     1         resiso(1),' +/- ', resiso(2)
      else
        print*,'Parameters          Actual'
        write(6,299) 'Diso (1/s)',aiso(1),' +/- ',sigaiso(1)
      endif
      print*
      print*,'X2 ',chiisob,'     X2 (red) ', chiisob/(ndata-mfiso)

      print*
      write(6,'(10x,a)')
     1       '****************** Axial Results *******************'
      if (nsim.gt.0) then
        print*,'Parameters          Actual                 Simulation'
        write(6,199) 'Diso (1/s)',atrue(1),' +/- ',sigat(1),
     1         resax(1,1),' +/- ', resax(1,2)
        write(6,199) 'Dpar/Dper ',atrue(2),' +/- ',sigat(2),
     1         resax(2,1),' +/- ', resax(2,2)
        write(6,199) 'Theta     ',atrue(3),' +/- ',sigat(3),
     1         resax(3,1),' +/- ', resax(3,2)
        write(6,199) 'Phi       ',atrue(4),' +/- ',sigat(4),
     1         resax(4,1),' +/- ', resax(4,2)
      else
        print*,'Parameters          Actual'
        write(6,299) 'Diso (1/s)',atrue(1),' +/- ',sigat(1)
        write(6,299) 'Dpar/Dper ',atrue(2),' +/- ',sigat(2)
        write(6,299) 'Theta     ',atrue(3),' +/- ',sigat(3)
        write(6,299) 'Phi       ',atrue(4),' +/- ',sigat(4)
      endif
      print*
      print*,'X2 ',chiaxb,'     X2 (red) ',chiaxb/(ndata-mfax)
      print*
      fstat=(chiisob-chiaxb)*(ndata-mfax)/(chiaxb*(mfax-mfiso))
      print*,'F   ',fstat,'     #points  ',ndata
      print*

      if (nsim.gt.0) then
        write(6,'(10x,a)')
     1       '*********** Statistical Distributions **************'
        print*,'#simulations ',nsim,'     seed ',iseed
        print*
        print*,'Percentile         X2(iso)     X2(axial)        F'
        do 35 i=1,19
          print*,i*0.05,chiiso(i*nsim/20),chiax(i*nsim/20),
     1         fsim(i*nsim/20)
35      continue
      endif

      print*
      write(6,'(10x,a)')
     1       '**************** Residual Analysis *****************'
      write(6,'(a,5x,a,6x,a,4x,a,2x,a,5x,a,3x,a,2x,a)')
     1      'Res#','Y20','R2/R1','dR2/R1','R2/R1(iso)','X2(iso)',
     1      'R2/R1(axial)','X2(axial)'

      do 40 i=1,ndata
        call jwaxial(resind(i),atrue,ytemp,mfax)
        call jwiso(aiso(1),ytempi)
        zz=sin(atrue(3))*(orient(1,i)*cos(atrue(4))+
     1     orient(2,i)*sin(atrue(4)))  + orient(3,i)*cos(atrue(3))
        write(6,'(i4,2x,f8.4,2x,f8.4,1x,f8.4,2x,f8.4,1x,f12.4,
     1            2x,f8.4,1x,f12.4)') 
     1   resnum(i),0.5*(3*zz*zz-1),r2r1(i),sig(i),
     1   ytempi,(r2r1(i)-ytempi)**2/sig(i)**2,
     1   ytemp,(r2r1(i)-ytemp)**2/sig(i)**2
40    continue

c     now write rotated pdbfile
      rotmat(1,1) = cos(atrue(3))*cos(atrue(4))
      rotmat(1,2) = cos(atrue(3))*sin(atrue(4))
      rotmat(1,3) = -sin(atrue(3)) 

      rotmat(2,1) = -sin(atrue(4)) 
      rotmat(2,2) = cos(atrue(4)) 
      rotmat(2,3) = 0

      rotmat(3,1) = sin(atrue(3))*cos(atrue(4))
      rotmat(3,2) = sin(atrue(3))*sin(atrue(4)) 
      rotmat(3,3) = cos(atrue(3))

      call pdbwrite(pdbfile,outfile,rotmat)

199   format(a10,e12.5,a5,e12.5,5x,e12.5,a5,e12.5)
299   format(a10,e12.5,a5,e12.5)
      end

