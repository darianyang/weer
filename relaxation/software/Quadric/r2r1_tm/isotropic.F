       subroutine isofit(x,y,sig,ndata,a,siga,mfit,chisq)

       parameter (nmax=500,mmax=6,maxc=200)
       integer lista(mmax)
       real x(nmax), y(nmax), sig(nmax), a(mmax), conver, siga(mmax),
     1     covar(mmax,mmax), alpha(mmax,mmax), chisq, ochisq, alamda,
     1     delta

       parameter (conver=0.001)

       external fiso

       ma=1
       do 12 i=1,mfit
           lista(i)=i
12     continue

       alamda=-1.0
       icnt=0
       chisq=1.0e20
15     ochisq=chisq
       call mrqmin(x,y,sig,ndata,a,ma,lista,mfit,covar,alpha,mmax,chisq,
     1            fiso,alamda)
       icnt=icnt+1
       delta=ochisq-chisq
c      print*,icnt,chisq,delta
       if ((delta.gt.conver .and. icnt.le.maxc) .or. delta.lt.0) goto 15

c      calculate uncertainties in parameters
       alamda=0.0
       call mrqmin(x,y,sig,ndata,a,ma,lista,mfit,covar,alpha,mmax,chisq,
     1            fiso,alamda)
       do 33 i=1,mfit
         siga(i)=sqrt(covar(i,i))
33     continue

       return
       end


      subroutine jwiso(a,y)

      real a, y, diso,jwmfi,nuh

      real*8 gamh,gamx,wh,wx,rxh,csa,planc,u0,dxh,dcsax

      common /const/ nuh

      data planc,u0/1.0545887d-34,1.0d-7/
      data gamh,gamx,rxh,csa
     1 /26.7519d7,-2.71d7,1.02d-10,-160d-6/

      external jwmfi
      
      wh=2*3.14159*nuh*1.0d6
      wx=wh*gamx/gamh
      dxh = gamx*gamh*planc*u0/rxh**3
      dcsax = wx*csa/sqrt(3.0d0)

c     first calculate r2/r1
      r1x= jwmfi(wh-wx,a)+3*jwmfi(wx,a)+6*jwmfi(wx+wh,a)
      r1x = dxh**2*r1x/4

      r1xa= dcsax**2*jwmfi(wx,a)

      r2x= 4*jwmfi(0.0d0,a)+jwmfi(wh-wx,a)+3*jwmfi(wx,a)+
     1     6*jwmfi(wh,a)+6*jwmfi(wx+wh,a)
      r2x = dxh**2*r2x/8

      r2xa = 4*jwmfi(0.0d0,a)+3*jwmfi(wx,a)
      r2xa = dcsax**2*r2xa/6

      y=(r2x+r2xa)/(r1x+r1xa)

c     print*,y
      return
      end


      subroutine fiso(x,a,y,dyda,na)

      real x, a(na), y, dyda(na),diso,jwmfi,nuh

      real*8 gamh,gamx,wh,wx,rxh,csa,planc,u0,dxh,dcsax

      common /const/ nuh

      data planc,u0/1.0545887d-34,1.0d-7/
      data gamh,gamx,rxh,csa
     1 /26.7519d7,-2.71d7,1.02d-10,-160d-6/

      external jwmfi
      
      wh=2*3.14159*nuh*1.0d6
      wx=wh*gamx/gamh
      dxh = gamx*gamh*planc*u0/rxh**3
      dcsax = wx*csa/sqrt(3.0d0)

c     first calculate r2/r1
      r1x= jwmfi(wh-wx,a(1))+3*jwmfi(wx,a(1))+6*jwmfi(wx+wh,a(1))
      r1x = dxh**2*r1x/4

      r1xa= dcsax**2*jwmfi(wx,a(1))

      r2x= 4*jwmfi(0.0d0,a(1))+jwmfi(wh-wx,a(1))+3*jwmfi(wx,a(1))+
     1     6*jwmfi(wh,a(1))+6*jwmfi(wx+wh,a(1))
      r2x = dxh**2*r2x/8

      r2xa = 4*jwmfi(0.0d0,a(1))+3*jwmfi(wx,a(1))
      r2xa = dcsax**2*r2xa/6

      y=(r2x+r2xa)/(r1x+r1xa)

c     now calculate d(r2/r1)/ddiso
      dr1x= djw(wh-wx,a(1))+3*djw(wx,a(1))+6*djw(wx+wh,a(1))
      dr1x = dxh**2*dr1x/4

      dr1xa= dcsax**2*djw(wx,a(1))

      dr2x= 4*djw(0.0d0,a(1))+djw(wh-wx,a(1))+3*djw(wx,a(1))+
     1     6*djw(wh,a(1))+6*djw(wx+wh,a(1))
      dr2x = dxh**2*dr2x/8

      dr2xa = 4*djw(0.0d0,a(1))+3*djw(wx,a(1))
      dr2xa = dcsax**2*dr2xa/6

      dyda(1) = ((dr2x+dr2xa)*(r1x+r1xa)-
     1          (r2x+r2xa)*(dr1x+dr1xa))/
     1          (r1x+r1xa)**2

c     print*,y,dyda
      return
      end

      real function jwmfi(w,diso)

      real tm,diso
      real*8  w

      tm=1/(6*diso)
      jwmfi=(2.0d0/5.0d0)*tm/(w**2*tm**2+1)

      return
      end

      real function djw(w,diso)

      real tm,diso
      real*8  w

      tm=1/(6*diso)
      djw=(-12.0*tm**2/5.0)*(1-w**2*tm**2)/(w**2*tm**2+1)**2
      return
      end

