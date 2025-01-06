       subroutine axialfit(x,y,sig,ndata,a,siga,mfit,chisq)

       parameter (nmax=500,mmax=6,maxc=200)
       integer lista(mmax)
       real x(nmax), y(nmax), sig(nmax), a(mmax), conver, siga(mmax),
     1     covar(mmax,mmax), alpha(mmax,mmax), chisq, ochisq, alamda,
     1     delta

       parameter (conver=0.001)

       external faxial

       ma=4
       do 12 i=1,mfit
           lista(i)=i
12     continue

       alamda=-1.0
       icnt=0
       chisq=1.0e20
15     ochisq=chisq
       call mrqmin(x,y,sig,ndata,a,ma,lista,mfit,covar,alpha,mmax,chisq,
     1            faxial,alamda)
       icnt=icnt+1
       delta=ochisq-chisq
c      print*,icnt,chisq,delta
       if ((delta.gt.conver .and. icnt.le.maxc) .or. delta.lt.0) goto 15

c      calculate uncertainties in parameters
       alamda=0.0
       call mrqmin(x,y,sig,ndata,a,ma,lista,mfit,covar,alpha,mmax,chisq,
     1            faxial,alamda)
       do 33 i=1,mfit
         siga(i)=sqrt(covar(i,i))
33     continue

       return
       end


      subroutine jwaxial(x,a,y,na)

      real x,a(na),y,r(3),orient(3,256),jwmf,nuh
      integer ix

      real*8 gamh,gamx,wh,wx,rxh,csa,planc,u0,dxh,dcsax

      common /rvec/ orient
      common /const/ nuh

      data planc,u0/1.0545887d-34,1.0d-7/
      data gamh,gamx,rxh,csa
     1 /26.7519d7,-2.71d7,1.02d-10,-160d-6/

      wh=2*3.14159*nuh*1.0d6
      wx=wh*gamx/gamh
      dxh = gamx*gamh*planc*u0/rxh**3
      dcsax = wx*csa/sqrt(3.0d0)

      ix=x
      r(1)=orient(1,ix)
      r(2)=orient(2,ix)
      r(3)=orient(3,ix)

c     first calculate r2/r1
      r1x= jwmf(wh-wx,a,r)+3*jwmf(wx,a,r)+6*jwmf(wx+wh,a,r)
      r1x = dxh**2*r1x/4

      r1xa= dcsax**2*jwmf(wx,a,r)

      r2x= 4*jwmf(0.0d0,a,r)+jwmf(wh-wx,a,r)+3*jwmf(wx,a,r)+
     1     6*jwmf(wh,a,r)+6*jwmf(wx+wh,a,r)
      r2x = dxh**2*r2x/8

      r2xa = 4*jwmf(0.0d0,a,r)+3*jwmf(wx,a,r)
      r2xa = dcsax**2*r2xa/6

      y=(r2x+r2xa)/(r1x+r1xa)

      return
      end


      subroutine faxial(x,a,y,dyda,na)

      real x,a(na),y,dyda(na),r(3),orient(3,256),
     1     jwmf,djwi,djwr,djwt,djwp,nuh

      integer ix

      real*8 gamh,gamx,wh,wx,rxh,csa,planc,u0,dxh,dcsax

      common /rvec/ orient
      common /const/ nuh

      data planc,u0/1.0545887d-34,1.0d-7/
      data gamh,gamx,rxh,csa
     1 /26.7519d7,-2.71d7,1.02d-10,-160d-6/

      wh=2*3.14159*nuh*1.0d6
      wx=wh*gamx/gamh
      dxh = gamx*gamh*planc*u0/rxh**3
      dcsax = wx*csa/sqrt(3.0d0)

      ix=x
      r(1)=orient(1,ix)
      r(2)=orient(2,ix)
      r(3)=orient(3,ix)

c     first calculate r2/r1
      r1x= jwmf(wh-wx,a,r)+3*jwmf(wx,a,r)+6*jwmf(wx+wh,a,r)
      r1x = dxh**2*r1x/4

      r1xa= dcsax**2*jwmf(wx,a,r)

      r2x= 4*jwmf(0.0d0,a,r)+jwmf(wh-wx,a,r)+3*jwmf(wx,a,r)+
     1     6*jwmf(wh,a,r)+6*jwmf(wx+wh,a,r)
      r2x = dxh**2*r2x/8

      r2xa = 4*jwmf(0.0d0,a,r)+3*jwmf(wx,a,r)
      r2xa = dcsax**2*r2xa/6

      y=(r2x+r2xa)/(r1x+r1xa)

c     now calculate d(r2/r1)/ddiso
      dr1x= djwi(wh-wx,a,r)+3*djwi(wx,a,r)+6*djwi(wx+wh,a,r)
      dr1x = dxh**2*dr1x/4

      dr1xa= dcsax**2*djwi(wx,a,r)

      dr2x= 4*djwi(0.0d0,a,r)+djwi(wh-wx,a,r)+3*djwi(wx,a,r)+
     1     6*djwi(wh,a,r)+6*djwi(wx+wh,a,r)
      dr2x = dxh**2*dr2x/8

      dr2xa = 4*djwi(0.0d0,a,r)+3*djwi(wx,a,r)
      dr2xa = dcsax**2*dr2xa/6

      dyda(1) = ((dr2x+dr2xa)*(r1x+r1xa)-
     1          (r2x+r2xa)*(dr1x+dr1xa))/
     1          (r1x+r1xa)**2

c     now calculate d(r2/r1)/drat
      dr1x= djwr(wh-wx,a,r)+3*djwr(wx,a,r)+6*djwr(wx+wh,a,r)
      dr1x = dxh**2*dr1x/4

      dr1xa= dcsax**2*djwr(wx,a,r)

      dr2x= 4*djwr(0.0d0,a,r)+djwr(wh-wx,a,r)+3*djwr(wx,a,r)+
     1     6*djwr(wh,a,r)+6*djwr(wx+wh,a,r)
      dr2x = dxh**2*dr2x/8

      dr2xa = 4*djwr(0.0d0,a,r)+3*djwr(wx,a,r)
      dr2xa = dcsax**2*dr2xa/6

      dyda(2) = ((dr2x+dr2xa)*(r1x+r1xa)-
     1          (r2x+r2xa)*(dr1x+dr1xa))/
     1          (r1x+r1xa)**2

c     now calculate d(r2/r1)/dtheta
      dr1x= djwt(wh-wx,a,r)+3*djwt(wx,a,r)+6*djwt(wx+wh,a,r)
      dr1x = dxh**2*dr1x/4

      dr1xa= dcsax**2*djwt(wx,a,r)

      dr2x= 4*djwt(0.0d0,a,r)+djwt(wh-wx,a,r)+3*djwt(wx,a,r)+
     1     6*djwt(wh,a,r)+6*djwt(wx+wh,a,r)
      dr2x = dxh**2*dr2x/8

      dr2xa = 4*djwt(0.0d0,a,r)+3*djwt(wx,a,r)
      dr2xa = dcsax**2*dr2xa/6

      dyda(3) = ((dr2x+dr2xa)*(r1x+r1xa)-
     1          (r2x+r2xa)*(dr1x+dr1xa))/
     1          (r1x+r1xa)**2

c     now calculate d(r2/r1)/dphi
      dr1x= djwp(wh-wx,a,r)+3*djwp(wx,a,r)+6*djwp(wx+wh,a,r)
      dr1x = dxh**2*dr1x/4

      dr1xa= dcsax**2*djwp(wx,a,r)

      dr2x= 4*djwp(0.0d0,a,r)+djwp(wh-wx,a,r)+3*djwp(wx,a,r)+
     1     6*djwp(wh,a,r)+6*djwp(wx+wh,a,r)
      dr2x = dxh**2*dr2x/8

      dr2xa = 4*djwp(0.0d0,a,r)+3*djwp(wx,a,r)
      dr2xa = dcsax**2*dr2xa/6

      dyda(4) = ((dr2x+dr2xa)*(r1x+r1xa)-
     1          (r2x+r2xa)*(dr1x+dr1xa))/
     1          (r1x+r1xa)**2

c     print*,y,dyda
      return
      end

      real function jwmf(w,a,r)

      real tm(3),a(4),r(3),coef(3),z
      real*8  w

      z=sin(a(3))*(r(1)*cos(a(4))+r(2)*sin(a(4)))
     1     +r(3)*cos(a(3))

      coef(1)=0.25*(3*z**2-1)**2
      coef(2)=3*z**2*(1-z**2)
      coef(3)=0.75*(1-z**2)**2

      tm(1) = (2+a(2))/(18*a(1)) 
      tm(2) = (4+2*a(2))/(6*a(1)*(5+a(2))) 
      tm(3) = (2+a(2))/(6*a(1)*(1+2*a(2))) 

      jwmf=0
      do 10, i=1,3
        jwmf=jwmf+coef(i)*tm(i)/(w**2*tm(i)**2+1)
10    continue
      jwmf=(2.0/5.0)*jwmf

      return
      end

      real function djwi(w,a,r)

      real tm(3),a(4),r(3),coef(3),z
      real*8  w

      z=sin(a(3))*(r(1)*cos(a(4))+r(2)*sin(a(4)))
     1     +r(3)*cos(a(3))

      coef(1)=0.25*(3*z**2-1)**2
      coef(2)=3*z**2*(1-z**2)
      coef(3)=0.75*(1-z**2)**2

      tm(1) = (2+a(2))/(18*a(1))
      tm(2) = (4+2*a(2))/(6*a(1)*(5+a(2)))
      tm(3) = (2+a(2))/(6*a(1)*(1+2*a(2)))

      djwi=0
      do 10, i=1,3
        temp=(1-w**2*tm(i)**2)/(w**2*tm(i)**2+1)**2
        temp=-(tm(i)/a(1))*temp
        djwi=djwi+coef(i)*temp
10    continue
      djwi=(2.0/5.0)*djwi

      return
      end

      real function djwr(w,a,r)

      real tm(3),a(4),r(3),coef(3),z,dfi(3)
      real*8  w

      z=sin(a(3))*(r(1)*cos(a(4))+r(2)*sin(a(4)))
     1     +r(3)*cos(a(3))

      coef(1)=0.25*(3*z**2-1)**2
      coef(2)=3*z**2*(1-z**2)
      coef(3)=0.75*(1-z**2)**2

      tm(1) = (2+a(2))/(18*a(1))
      tm(2) = (4+2*a(2))/(6*a(1)*(5+a(2)))
      tm(3) = (2+a(2))/(6*a(1)*(1+2*a(2)))

      dfi(1)=1.0/3.0
      dfi(2)= 6/(5+a(2))**2
      dfi(3)= -3/(1+2*a(2))**2

      djwr=0
      do 10, i=1,3
        temp=(1-w**2*tm(i)**2)/(w**2*tm(i)**2+1)**2
        temp=dfi(i)/(6*a(1))*temp
        djwr=djwr+coef(i)*temp
10    continue
      djwr=(2.0/5.0)*djwr

      return
      end

      real function djwt(w,a,r)

      real tm(3),a(4),r(3),coef(3),z,dz,dcoef(3)
      real*8  w

      z=sin(a(3))*(r(1)*cos(a(4))+r(2)*sin(a(4)))
     1     +r(3)*cos(a(3))

      dz=cos(a(3))*(r(1)*cos(a(4))+r(2)*sin(a(4)))
     1     -r(3)*sin(a(3))

      coef(1)=0.25*(3*z**2-1)**2
      coef(2)=3*z**2*(1-z**2)
      coef(3)=0.75*(1-z**2)**2

      dcoef(1)=-3*z*(1-3*z**2)
      dcoef(2)=6*z*(1-2*z**2)
      dcoef(3)=-3*z*(1-z**2)

      tm(1) = (2+a(2))/(18*a(1))
      tm(2) = (4+2*a(2))/(6*a(1)*(5+a(2)))
      tm(3) = (2+a(2))/(6*a(1)*(1+2*a(2)))

      djwt=0
      do 10, i=1,3
        temp=tm(i)/(w**2*tm(i)**2+1)
        djwt=djwt+dz*dcoef(i)*temp
10    continue
      djwt=(2.0/5.0)*djwt

      return
      end

      real function djwp(w,a,r)

      real tm(3),a(4),r(3),coef(3),z,dz,dcoef(3)
      real*8  w

      z=sin(a(3))*(r(1)*cos(a(4))+r(2)*sin(a(4)))
     1     +r(3)*cos(a(3))

      dz=sin(a(3))*(-r(1)*sin(a(4))+r(2)*cos(a(4)))

      coef(1)=0.25*(3*z**2-1)**2
      coef(2)=3*z**2*(1-z**2)
      coef(3)=0.75*(1-z**2)**2

      dcoef(1)=-3*z*(1-3*z**2)
      dcoef(2)=6*z*(1-2*z**2)
      dcoef(3)=-3*z*(1-z**2)

      tm(1) = (2+a(2))/(18*a(1))
      tm(2) = (4+2*a(2))/(6*a(1)*(5+a(2)))
      tm(3) = (2+a(2))/(6*a(1)*(1+2*a(2)))

      djwp=0
      do 10, i=1,3
        temp=tm(i)/(w**2*tm(i)**2+1)
        djwp=djwp+dz*dcoef(i)*temp
10    continue
      djwp=(2.0/5.0)*djwp

      return
      end

