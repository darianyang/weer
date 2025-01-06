c     r2r1_tm.f				ver. 1.2  10Jun15  AGP
c     program to fit tm to R2/R1 ratio with model free theory

      program  r2r1_tm

      parameter (iupper=500)

      real tm,tm0,errabs,ftm,r2r1,lb,ub,tminit,
     1     sdat,stddev,r2r10,tmave,gasdev,csa,rxh,
     1     gamh,gamc,gamn,gamx,wh1,wx1,wh2,wx2,ran1

      logical succes

      character nucleus*3,mdffile*80,line*80

      integer iseed,iarg,iargc,nsim

      logical onefl

      data errabs/1.0d-12/
      data gamh,gamc,gamn/26.7519d7,6.7283d7,-2.71d7/
      data planc,u0/1.0545887d-34,1.0d-7/
      data rnh,rch,csan,csac /1.02d-10,1.09d-10,-160d-6,25d-6/

      common /const/wh1,wx1,wh2,wx2,dd,csa1,csa2

      external ftm

      onefl=.FALSE.
      if (iargc().gt.0) then
        call getarg(1,line)
        if (line.eq.'-1') onefl=.TRUE.
      endif

      write(0,'(/,a)') 'Estimation of tm using R2/R1 ratio'

      write(0,'(/,a,$)') 'Input nucleus (13C or 15N): '
      read(5,'(a3)') nucleus

      if (nucleus.eq.'13C' .or. nucleus.eq.'13c') then
         gamx=gamc
         rxh=rch
         csa=csac
      elseif (nucleus.eq.'15N' .or. nucleus.eq.'15n') then
         gamx=gamn
         rxh=rnh
         csa=csan
      else
        write(6,'(/,a,a)') 'Unknown nucleus: ',nucleus
        stop
      endif
      write(0,'(/a,f8.4,a,f8.2,a)') 
     1 'The default bond length and csa are: ', rxh*1.0d10,
     1 ' A and ', csa*1.0d6, ' ppm'
      write(0,'(/a,$)') 'Enter new values or <RETURN> to accept: '
35    read(5,'(a80)') line
      if (line(1:1).eq.'#') goto 35
      if (line(1:1).ne.'') then
             read(line,*) rxh,csa
             rxh=rxh*1.0d-10
             csa=csa*1.0d-6
      endif

      write(0,*)
      write(0,'(a,$)') 
     1     'Input the proton spectrometer field used for R1 (MHz): '
      read(5,*) wh1
      write(0,*)
      write(0,'(a,$)')
     1     'Input the proton spectrometer field used for R2 (MHz): '
      read(5,*) wh2
      write(0,*)
      write(0,'(a,$)') 'Input initial guess for tm (ns): '
      read(5,*) tm0
      write(0,*)

      if (onefl) then
        nres = 1
        write(0,'(a,$)') 'Input R2/R1, std. error in R2/R1: '
        read(5,*) r2r10,stddev
        write(0,*)
      else
        nres = 0
        write(0,*)
        write(0,'(a,$)') 'Input file name for R2/R1 data: '
        read(5,'(a)') mdffile
        write(0,*)
      endif

      wx1=2*3.141592654*1.0d6*wh1*gamx/gamh
      wh1=2*3.141592654*1.0d6*wh1
      wx2=2*3.141592654*1.0d6*wh2*gamx/gamh
      wh2=2*3.141592654*1.0d6*wh2
      dd = gamx*gamh*planc*u0/rxh**3
      csa1 = wx1*csa/sqrt(3.0d0)
      csa2 = wx2*csa/sqrt(3.0d0)

      tminit=tm0*1.0d-9
      call getseed(iseed)

      if (.not.onefl) then
        open(unit=11,file=mdffile,form='formatted',status='old',
     $     iostat=ios)
        if (ios.ne.0) then
           write(0,'(/,a,a)') ' Error opening file: ', mdffile
           stop
        endif
      endif

      write(6,'(a)') '# R2/R1 Analysis:  '
      write(6,'(a)') '#'
      write(6,'(a,2x,a)') '# nucleus: ',nucleus
      write(6,'(a,2x,f8.4)') '# bond length (A): ',rxh*1.0d10
      write(6,'(a,2x,f8.2)') '# CSA (ppm): ',csa*1.0d6
      write(6,'(a,2x,2f10.4)') '# Fields for R1 and R2 (MHz): ',
     1         wh1/(2*3.141592654*1.0d6),wh2/(2*3.141592654*1.0d6)
      write(6,'(a)') '#'
      if (.not.onefl) write(6,'(a)') '#residue      tm     dtm' 

50    if (.not.onefl) then
45        read(11,'(a80)',end=49) line
          if (line(1:1).eq.'#' .or. line.eq.'') goto 45
          read(line,*) ires,r2r10,stddev
          nres=nres+1
      endif

      tmave=0.0
      sdat=0.0
      tm0=0.0
      lb=tminit/1.6
      ub=tminit*1.6
      nsim=0

      call zbrac(r2r10,lb,ub,succes)
      if (succes) then
          call zbrent(tm0,r2r10,lb,ub,errabs)

c     now do simulation of uncertainties
      do 1001 i=1,iupper
             temp=gasdev(iseed)
             r2r1=r2r10+stddev*temp
             lb=tm0/1.6
             ub=tm0*1.6
             call zbrac(r2r1,lb,ub,succes)
             if (succes) then
                 call zbrent(tm,r2r1,lb,ub,errabs)
                 nsim=nsim+1
                 tmave=tmave+tm
                 sdat=sdat+tm**2
             endif
1001  continue

c         now output results
          tmave=tmave/nsim
          sdat=sdat/nsim - tmave**2
          sdat=sqrt(sdat)
          if (onefl) then
            write(6,'(a,2x,f8.4,a,f8.4)') '# tm (ns): ',tm0*1.0d9,
     1        ' +/- ',sdat*1.0d9
          else
            write(6,'(i3,2x,f8.4,4x,f8.4)') ires,tm0*1.0d9,
     1        sdat*1.0d9
          endif
       endif

      if (.not.onefl) goto 50

49    if (.not.onefl) close(11)

      end

      real function ftm(tm)

      real tm,jwmf,wh1,wx1,wh2,wx2,denom,r2r1,numer

      external jwmf

      common /const/ wh1,wx1,wh2,wx2,dd,csa1,csa2

      if (tm.eq.0.0) then
	ftm=1.0
      else
        denom=jwmf(wh1-wx1,tm) + 3*jwmf(wx1,tm) + 6*jwmf(wh1+wx1,tm)
     1          + 4*(csa1/dd)**2*jwmf(wx1,tm)
        numer=4*jwmf(0.0,tm) + jwmf(wh2-wx2,tm) + 3*jwmf(wx2,tm) 
     1          + 6*jwmf(wh2,tm) + 6*jwmf(wh2+wx2,tm)
     1          + 8*(csa2/dd)**2/6*(4*jwmf(0.0,tm)+3*jwmf(wx2,tm))
        ftm=(numer/denom)/2
      endif

      return
      end


      real function jwmf(w,tm)

      real w,tm

      jwmf=(2.0d0/5.0d0)*tm/(w**2*tm**2+1)

      return
      end


