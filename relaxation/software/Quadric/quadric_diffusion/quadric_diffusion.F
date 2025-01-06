      program local_diffusion
c                                          ver 1.14 15Mar09 AGP
c     Fit diffusion constants to tm data using quadric 
c     linearization approach

      parameter (nmax=256,mmax=6,imax=500,namax=3)
      parameter (mfax=4,mfani=6,nsim=300,lwork=1024)
      parameter (TOL=1.0e-6)

      real tm(nmax), sig(nmax),dt(3,3),afunc(mmax),tdt(3,3), 
     1     a(mmax), atrue(mmax), sigat(mmax),dyda(mmax), 
     1     orient(3,nmax),resind(nmax),rotmat(3,3),work(lwork),
     1     u(nmax,mmax),vt(mmax,mmax),w(mmax),eval(mmax),
     1     ypred(nmax),tmsim(nmax),adiso(imax),adpdp(imax),
     1     adeldp(imax),atheta(imax),aphi(imax),apsi(imax),
     1     ressim(imax),sigsim(imax),asim(mmax),sigasim(mmax),
     1     tmpdt(3,3)

      integer iseed,idum,resnum(nmax),ind(3),nres(namax),info

      real gasdev
      external gasdev, getseed

      common /rvec/ orient

      character*80 line,mdffile(namax),pdbfile,outf1,
     1             outf2,cfile
      character*2 atname(namax)

      if (iargc().lt. 1) then
        write(0,*) 'Usage: quadric_diffusion input.file'
        stop
      endif

      call getarg(1,cfile)
      open(unit=13,file=cfile,form='formatted',status='old',
     1     iostat=ios)
      if (ios.ne.0) then
         write(0,'(/,a,a)') ' Error opening file: ', cfile
         stop
      endif

250   read(13,'(a80)') line
      if (line(1:1).eq.'#') goto 250
      read(line,*) ratlo,rathi,nsteps

600   read(13,'(a80)') line
      if (line(1:1).eq.'#') goto 600
      read(line,*) nats,(atname(i),i=1,nats)

c     get tm and uncertainties
      ndata=0
      do 1000 nn=1,nats
300     read(13,'(a80)') mdffile(nn)
        if (mdffile(nn)(1:1).eq.'#') goto 300

        open(unit=11,file=mdffile(nn),form='formatted',
     1      status='old',iostat=ios)
        if (ios.ne.0) then
           write(0,'(/,a,a)') ' Error opening file: ', mdffile(nn)
           stop
        endif

        j=0
45      read(11,'(a80)',end=49) line
        if (line(1:1).eq.'#' .or. line.eq.'') then
           goto 45
        else
           j=j+1
           read(line,*) resnum(ndata+j),tm(ndata+j),sig(ndata+j)
           resind(ndata+j)=j+ndata
           sig(ndata+j)=sig(ndata+j)/(6.0*tm(ndata+j)*tm(ndata+j))
           tm(ndata+j)=1/(6.0*tm(ndata+j))
           goto 45
        endif

49      close(11)
        nres(nn)=j
        ndata=ndata+nres(nn)
1000  continue

c     get coordinates
400   read(13,'(a80)') pdbfile
      if (pdbfile(1:1).eq.'#') goto 400

c     get output file names
550   read(13,'(a80)') outf1
      if (outf1(1:1).eq.'#') goto 550
551   read(13,'(a80)') outf2
      if (outf2(1:1).eq.'#') goto 551

      close(13)

      j=1
      do 1100 nn=1,nats
       if (nn.gt.1) j=j+nres(nn-1)
       call pdbread(pdbfile,resnum(j),atname(nn),nres(nn),
     1              nr,orient(1,j))

       if (nr.ne.nres(nn)) then
          print*,'Error: input tm data does not match pdb file',
     1      mdffile(nn),nres(nn),nr
         stop
        endif
1100  continue

      print*
      print*
      write(6,'(15x,a)') 'Quadric Diffusion Analysis'
      print*

      write(6,'(10x,a)')
     1  '******************** Input Data ********************'
      write(6,'(a,t10,a,t20,a,t30,a,t40,a,t50,a)')
     1      'Residue','D/10^7','dD/10^7','x','y','z'
      j=1
      jj=nres(j)
      do 10 i=1,ndata
       if (i.gt.jj) then
           j=j+1
           jj=jj+nres(j)   
       endif
       write(6,'(i3,1x,a2,1x,f8.4,2x,f8.4,2x,f8.4,2x,f8.4,
     1            2x,f8.4)')
     1   resnum(i),atname(j),tm(i)*100,sig(i)*100,orient(1,i),
     1   orient(2,i),orient(3,i)
10    continue

      print*
      print*
      write(6,'(10x,a)')
     1       '**************** Isotropic Results *****************'
      aiso=0.0
      sigiso=0.0
      chiiso=0.0

      do 100 i=1,ndata
         aiso=aiso+tm(i)/sig(i)**2
         sigiso=sigiso+1/sig(i)**2
100   continue
      aiso=aiso/sigiso
      sigiso=sqrt(1/sigiso)

      do 110 i=1,ndata
         chiiso=chiiso+(tm(i)-aiso)**2/sig(i)**2
110   continue

      print*,'Parameters          Actual'
      write(6,299) 'Diso (10^-7/s)',aiso*100,' +/- ',sigiso*100
      print*
      print*,'X2 ',chiiso,'     X2 (red) ', chiiso/(ndata-1)

      print*
      write(6,'(10x,a)')
     1       '**************** Residual Analysis *****************'
      write(6,'(a,t10,a,t20,a,t30,a,t41,a)')
     1      'Residue','D/10^7','dD/10^7','Diso/10^7','X2(iso)'

      j=1
      jj=nres(j)
      do 41 i=1,ndata
       if (i.gt.jj) then
           j=j+1
           jj=jj+nres(j)   
       endif
        write(6,'(i3,1x,a2,1x,f8.4,2x,f8.4,2x,f8.4,2x,f8.2)')
     1   resnum(i),atname(j),tm(i)*100,sig(i)*100,aiso*100,
     1   (tm(i)-aiso)**2/sig(i)**2
41    continue


      print*
      print*
      write(6,'(10x,a)')
     1       '****************** Axial Results *******************'
      print*

c     now grid search for initial guesses
c     write(6,'(10x,a)')
c    1     '***** Grid Search for Initial Parameter Values *****'
c     write(6,'(a,8x,a,8x,a,3x,a,5x,a,8x,a)')
c    1   '   X2new','X2old','Diso','Dpar/Dper','Theta','Phi'

      chiaxb=1.0e10
      do 450, jj=-10,10
        a(1)=(1+jj/50.0)*aiso
        do 500, i=0,nsteps
          a(2)=ratlo + i*(rathi-ratlo)/nsteps
          do 501, j=0,9
            a(3)=0.174533*j
            do 502, k=0,36
              a(4)=0.174533*k
              chitry=0
              do 503, ii=1,ndata
                 call jwaxial(resind(ii),a,ytemp,mfax)
                 chitry=chitry+(tm(ii)-ytemp)**2/sig(ii)**2
503           continue
              if (chitry.lt.chiaxb) then
c              write(6,'(f10.2,3x,f10.2,3x,f10.5,3x,
c    1            f12.5,3x,f12.5,3x,f12.5)')
c    1            chitry,chiaxb,a(1)*100,a(2),a(3),a(4)
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

      call axialfit(resind,tm,sig,ndata,atrue,sigat,mfax,chiaxb)

      print*
      print*,'Parameters          Actual'
      write(6,299) 'Diso (10^-7/s)',atrue(1)*100,' +/- ',sigat(1)*100
      write(6,299) 'Dpar/Dper     ',atrue(2),' +/- ',sigat(2)
      write(6,299) 'Theta         ',atrue(3),' +/- ',sigat(3)
      write(6,299) 'Phi           ',atrue(4),' +/- ',sigat(4)

c     now do jacknife simulations
      disoj = 0
      disoj2 = 0
      dratioj = 0
      dratioj2 = 0
      thetaj = 0
      thetaj2 = 0
      phij = 0
      phij2 = 0
      do 740 i = 1,ndata
        j = 1
        do 730 k = 1, ndata
          tmsim(j) = tm(k)
          sigsim(j) = sig(k)
          ressim(j) = resind(k)
          if (k .ne. i) j=j+1
730     continue
        asim(1)=atrue(1)
        asim(2)=atrue(2)
        asim(3)=atrue(3)
        asim(4)=atrue(4)
        call axialfit(ressim,tmsim,sigsim,ndata-1,asim,sigasim,
     1        mfax,chisim)
        disoj = disoj + ndata*atrue(1) - (ndata-1)*asim(1)
        disoj2 = disoj2 + (ndata*atrue(1) - (ndata-1)*asim(1))**2
        dratioj = dratioj + ndata*atrue(2) - (ndata-1)*asim(2)
        dratioj2 = dratioj2 + (ndata*atrue(2) - (ndata-1)*asim(2))**2
        thetaj = thetaj + ndata*atrue(3) - (ndata-1)*asim(3)
        thetaj2 = thetaj2 + (ndata*atrue(3) - (ndata-1)*asim(3))**2
        phij =  phij + ndata*atrue(4) - (ndata-1)*asim(4)
        phij2 = phij2 + (ndata*atrue(4) - (ndata-1)*asim(4))**2
740   continue
      disoj2=sqrt((disoj2-disoj*disoj/ndata)/(ndata*(ndata-1)))
      disoj = disoj/ndata
      dratioj2=sqrt((dratioj2-dratioj*dratioj/ndata)/(ndata*(ndata-1)))
      dratioj = dratioj/ndata
      thetaj2=sqrt((thetaj2-thetaj*thetaj/ndata)/(ndata*(ndata-1)))
      thetaj = thetaj/ndata
      phij2=sqrt((phij2-phij*phij/ndata)/(ndata*(ndata-1)))
      phij = phij/ndata

      print*
      print*,'Parameters          Jacknife'
      write(6,299) 'Diso (10^-7/s)',disoj*100,' +/- ',disoj2*100
      write(6,299) 'Dpar/Dper     ',dratioj,' +/- ',dratioj2
      write(6,299) 'Theta         ',thetaj,' +/- ',thetaj2
      write(6,299) 'Phi           ',phij,' +/- ',phij2

      print*
      print*,'X2 ',chiaxb,'     X2 (red) ',chiaxb/(ndata-mfax)
      print*
      fstat=(chiiso-chiaxb)*(ndata-mfax)/(chiaxb*(mfax-1))
      print*,'F   ',fstat,'     #points  ',ndata
      print*

      print*
      write(6,'(10x,a)')
     1       '**************** Residual Analysis *****************'
      write(6,'(a,t10,a,t20,a,t30,a,t40,a,t52,a)')
     1      'Residue','Y20','D/10^7','dD/10^7','Dax/10^7','X2(ax)'

      j=1
      jj=nres(j)
      do 40 i=1,ndata
        if (i.gt.jj) then
           j=j+1
           jj=jj+nres(j)   
        endif
        call jwaxial(resind(i),atrue,ytemp,mfax)
        zz=sin(atrue(3))*(orient(1,i)*cos(atrue(4))+
     1     orient(2,i)*sin(atrue(4)))  + orient(3,i)*cos(atrue(3))

        write(6,'(i3,1x,a2,1x,f8.4,2x,f8.4,2x,f8.4,2x,f8.4,
     1            2x,f8.2)')
     1   resnum(i),atname(j),0.5*(3*zz*zz-1),tm(i)*100,sig(i)*100,
     1   ytemp*100,(tm(i)-ytemp)**2/sig(i)**2

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

      call pdbwrite(pdbfile,outf1,rotmat)

      print*
      write(6,'(10x,a)')
     1       '************* Anisotropic Results ******************'
      print*

c     setup the design matrix
      do 12 i=1,ndata
        j=resind(i)
        u(i,1)=orient(1,j)**2/sig(i)
        u(i,2)=orient(2,j)**2/sig(i)
        u(i,3)=orient(3,j)**2/sig(i)
        u(i,4)=2*orient(1,j)*orient(2,j)/sig(i)
        u(i,5)=2*orient(1,j)*orient(3,j)/sig(i)
        u(i,6)=2*orient(2,j)*orient(3,j)/sig(i)
        tmsim(i)=tm(i)/sig(i)
12    continue

c     compute the singular value decomposition
      info=0
      call sgesvd('O','S',ndata,mfani,u,nmax,w,u,nmax,
     1            vt,mmax,work,lwork,info)
      if (info.ne.0) goto 99

c     now check for small singular values
      thresh=TOL*w(1)
      do 14 j=1,mfani
        if(w(j).lt.thresh) w(j)=0.
14    continue

c     solve the least squares problem
      call svbksb(u,w,vt,ndata,mfani,nmax,mmax,tmsim,a)

c     compute predicted values and chi-square
      chiani=0.0
      do 16 i=1,ndata
        j=resind(i)
        ypred(i)=a(1)*orient(1,j)**2
     1          +a(2)*orient(2,j)**2
     1          +a(3)*orient(3,j)**2
     1          +a(4)*2*orient(1,j)*orient(2,j)
     1          +a(5)*2*orient(1,j)*orient(3,j)
     1          +a(6)*2*orient(2,j)*orient(3,j)
        chiani=chiani+(tm(i)-ypred(i))**2/sig(i)**2
16    continue

c     sort the coefficients into a square matrix
      dt(1,1)=a(1)
      dt(2,2)=a(2)
      dt(3,3)=a(3)
      dt(1,2)=a(4)
      dt(1,3)=a(5)
      dt(2,3)=a(6)
      dt(2,1)=dt(1,2)
      dt(3,1)=dt(1,3)
      dt(3,2)=dt(2,3)

c     diagonalize the matrix
      call ssyev('V','U',3,dt,3,eval,work,lwork,ierr)
      if (ierr.ne.0) then
         print*,'error ',ierr
         stop
      endif

      atrue(1)=-eval(1)+eval(2)+eval(3)
      atrue(2)=eval(1)-eval(2)+eval(3)
      atrue(3)=eval(1)+eval(2)-eval(3)

      call indexx(3,atrue,ind)

c     permute and transpose the columns of dt
      do 555 i=1,3
         tmpdt(1,i)=dt(1,i)
         tmpdt(2,i)=dt(2,i)
         tmpdt(3,i)=dt(3,i)
555   continue
      do 556 i=1,3
         dt(i,1)=tmpdt(1,ind(i))
         dt(i,2)=tmpdt(2,ind(i))
         dt(i,3)=tmpdt(3,ind(i))
556   continue

c     force determinant to be +1 for right-handed triple
      det=dt(1,1)*(dt(2,2)*dt(3,3)-dt(3,2)*dt(2,3))
     1   -dt(1,2)*(dt(2,1)*dt(3,3)-dt(2,3)*dt(3,1))
     1   +dt(1,3)*(dt(2,1)*dt(3,2)-dt(2,2)*dt(3,1))

      if (det.lt.0) then
        dt(1,1)=-1*dt(1,1)
        dt(1,2)=-1*dt(1,2)
        dt(1,3)=-1*dt(1,3)
      endif

      diso=(atrue(ind(1))+atrue(ind(2))+atrue(ind(3)))/3
      dpdp=2*atrue(ind(3))/(atrue(ind(1))+atrue(ind(2)))
      deldp=atrue(ind(1))/atrue(ind(2))

      theta=acos(dt(3,3))
      phi=atan2(dt(3,2),dt(3,1))
      tmp=sin(phi)*dt(1,1)-cos(phi)*dt(1,2)
      psi=asin(-tmp)

      call pdbwrite(pdbfile,outf2,dt)

c     now generate simulated data for uncertainties

c     To print the Monte Carlo results, uncomment the next line
c     print*,
c    1'i  Diso    2Dzz/(Dxx+Dyy)   Dxx/Dyy   Theta   Phi   Psi'

      call getseed(iseed)
      do 170 index=1,nsim
         do 180 i=1,ndata
           tmsim(i)=ypred(i)/sig(i)+gasdev(iseed)
180      continue
         call svbksb(u,w,vt,ndata,mfani,nmax,mmax,tmsim,a)

c        sort the coefficients into a square matrix
         tdt(1,1)=a(1)
         tdt(2,2)=a(2)
         tdt(3,3)=a(3)
         tdt(1,2)=a(4)
         tdt(1,3)=a(5)
         tdt(2,3)=a(6)
         tdt(2,1)=tdt(1,2)
         tdt(3,1)=tdt(1,3)
         tdt(3,2)=tdt(2,3)

c        diagonalize the matrix
         call ssyev('V','U',3,tdt,3,eval,work,lwork,ierr)
         if (ierr.ne.0) then
            print*,'error ',ierr
            stop
         endif

c        permute and transpose the columns of tdt
         do 557 i=1,3
            tmpdt(1,i)=tdt(1,i)
            tmpdt(2,i)=tdt(2,i)
            tmpdt(3,i)=tdt(3,i)
557      continue
         do 558 i=1,3
            tdt(i,1)=tmpdt(1,ind(i))
            tdt(i,2)=tmpdt(2,ind(i))
            tdt(i,3)=tmpdt(3,ind(i))
558      continue

c        force determinant to be +1 for right-handed triple
         det=tdt(1,1)*(tdt(2,2)*tdt(3,3)-tdt(3,2)*tdt(2,3))
     1      -tdt(1,2)*(tdt(2,1)*tdt(3,3)-tdt(2,3)*tdt(3,1))
     1      +tdt(1,3)*(tdt(2,1)*tdt(3,2)-tdt(2,2)*tdt(3,1))

         if (det.lt.0) then
           tdt(1,1)=-1*tdt(1,1)
           tdt(1,2)=-1*tdt(1,2)
           tdt(1,3)=-1*tdt(1,3)
         endif

         atrue(1)=-eval(1)+eval(2)+eval(3)
         atrue(2)=eval(1)-eval(2)+eval(3)
         atrue(3)=eval(1)+eval(2)-eval(3)

         tdiso=(atrue(ind(1))+atrue(ind(2))+atrue(ind(3)))/3
         tdpdp=2*atrue(ind(3))/(atrue(ind(1))+atrue(ind(2)))
         tdeldp=atrue(ind(1))/atrue(ind(2))

         ttheta=acos(tdt(3,3))
         tphi=atan2(tdt(3,2),tdt(3,1))
         tmp=sin(tphi)*tdt(1,1)-cos(tphi)*tdt(1,2)
         tpsi=asin(-tmp)

         adiso(1)=adiso(1)+tdiso
         adiso(2)=adiso(2)+tdiso*tdiso

         adpdp(1)=adpdp(1)+tdpdp
         adpdp(2)=adpdp(2)+tdpdp*tdpdp

         adeldp(1)=adeldp(1)+tdeldp
         adeldp(2)=adeldp(2)+tdeldp*tdeldp

         atheta(1)=atheta(1)+ttheta
         atheta(2)=atheta(2)+ttheta*ttheta

c        crude fix for phi wrappping around 2*pi
         if (phi .gt. 2.5 .and. tphi .lt. 0) tphi = tphi + 6.283185
         if (phi .lt. -2.5 .and. tphi .gt. 0) tphi = tphi - 6.283185
         aphi(1)=aphi(1)+tphi
         aphi(2)=aphi(2)+tphi*tphi

c        crude fix for psi wrappping around 2*pi
         if (psi .gt. 2.5 .and. tpsi .lt. 0) tpsi = tpsi + 6.283185
         if (psi .lt. -2.5 .and. tpsi .gt. 0) tpsi = tpsi - 6.283185
         apsi(1)=apsi(1)+tpsi
         apsi(2)=apsi(2)+tpsi*tpsi

c        To print the Monte Carlo results, uncomment the next line
c        write(6,'(i5,6f8.3)'),index,tdiso*100,tdpdp,
c    1        tdeldp,ttheta,tphi,tpsi
170      continue
     
      adiso(1)=adiso(1)/nsim
      adiso(2)=adiso(2)/nsim-adiso(1)**2
      if (adiso(2).gt.0) then
        adiso(2)=sqrt(adiso(2))
      else
        adiso(2)=0.0
      endif

      adpdp(1)=adpdp(1)/nsim
      adpdp(2)=adpdp(2)/nsim-adpdp(1)**2
      if (adpdp(2).gt.0) then
        adpdp(2)=sqrt(adpdp(2))
      else
        adpdp(2)=0.0
      endif

      adeldp(1)=adeldp(1)/nsim
      adeldp(2)=adeldp(2)/nsim-adeldp(1)**2
      if (adeldp(2).gt.0) then
        adeldp(2)=sqrt(adeldp(2))
      else
        adeldp(2)=0.0
      endif

      atheta(1)=atheta(1)/nsim
      atheta(2)=atheta(2)/nsim-atheta(1)**2
      if (atheta(2).gt.0) then
        atheta(2)=sqrt(atheta(2))
      else
        atheta(2)=0.0
      endif

      aphi(1)=aphi(1)/nsim
      aphi(2)=aphi(2)/nsim-aphi(1)**2
      if (aphi(2).gt.0) then
        aphi(2)=sqrt(aphi(2))
      else
        aphi(2)=0.0
      endif

      apsi(1)=apsi(1)/nsim
      apsi(2)=apsi(2)/nsim-apsi(1)**2
      if (apsi(2).gt.0) then
        apsi(2)=sqrt(apsi(2))
      else
        apsi(2)=0.0
      endif

      print*
      print*,'Parameters          Actual                 Simulated'
      write(6,199) 'Diso (10^-7/s)',diso*100, adiso(1)*100,' +/- ',
     1    adiso(2)*100
      write(6,199) '2Dzz/(Dxx+Dyy)',dpdp,adpdp(1),' +/- ',adpdp(2)
      write(6,199) 'Dxx/Dyy       ',deldp,adeldp(1),' +/- ',adeldp(2)
      write(6,199) 'Theta         ',theta,atheta(1),' +/- ',atheta(2)
      write(6,199) 'Phi           ',phi,aphi(1),' +/- ',aphi(2)
      write(6,199) 'Psi           ',psi,apsi(1),' +/- ',apsi(2)
      print*
      print*,'X2 ',chiani,'     X2 (red) ',chiani/(ndata-mfani)
      print*
      fstat=(chiaxb-chiani)*(ndata-mfani)/(chiani*(mfani-mfax))
      print*,'F   ',fstat,'     #points  ',ndata
      print*

      print*
      write(6,'(10x,a)')
     1       '**************** Residual Analysis *****************'
      write(6,'(a,t10,a,t20,a,t30,a,t40,a,t50,a,t61,a)')
     1'Residue','Y20','Y22','D/10^7','dD/10^7','Dani/10^7','X2(ani)'

      j=1
      do 1300 nn=1,nats
       if (nn.gt.1) j=j+nres(nn-1)
       call pdbread(outf2,resnum(j),atname(nn),nres(nn),
     1              nr,orient(1,j))
       if (nr.ne.nres(nn)) then
          print*,'Error: input tm data does not match pdb file',
     1      mdffile(nn),nres(nn),nr
         stop
        endif
1300  continue

      j=1
      jj=nres(j)
      do 46 i=1,ndata
        if (i.gt.jj) then
           j=j+1
           jj=jj+nres(j)   
        endif
        y20=0.5*(3*orient(3,i)*orient(3,i)-1)
        y22=sqrt(3.0/2.0)*
     1     (orient(1,i)*orient(1,i)-orient(2,i)*orient(2,i))

        write(6,'(i3,1x,a2,1x,f8.4,2x,f8.4,2x,f8.4,2x,f8.4,
     1            2x,f8.4,2x,f8.2)')
     1   resnum(i),atname(j),y20,y22,tm(i)*100,sig(i)*100,
     1   ypred(i)*100,(tm(i)-ypred(i))**2/sig(i)**2
46    continue

      goto 999
c     ***************** format statements ***************************
299   format(a14,f12.5,a5,f12.5)
199   format(a14,f12.5,5x,f12.5,a5,f12.5)

c     ****************** error returns *****************************
99    write(0,*) 'error in SVD'

999   continue
      end

