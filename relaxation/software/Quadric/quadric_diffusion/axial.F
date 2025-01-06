       subroutine axialfit(x,y,sig,ndata,a,siga,mfit,chisq)

       parameter (nmax=256,mmax=6,maxc=200)
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


      subroutine faxial(x,a,y,dyda,na)

      real x,a(na),y,dyda(na),r(3),orient(3,256)

      common /rvec/ orient

      i=int(x)
      r(1)=orient(1,i)
      r(2)=orient(2,i)
      r(3)=orient(3,i)

      z=sin(a(3))*(r(1)*cos(a(4))+r(2)*sin(a(4)))
     1     + r(3)*cos(a(3))

      dz3=cos(a(3))*(r(1)*cos(a(4))+r(2)*sin(a(4)))
     1     -r(3)*sin(a(3))

      dz4=sin(a(3))*(-r(1)*sin(a(4))+r(2)*cos(a(4)))
 
      yy=0.5*(3*z**2-1)

      y=a(1)*(1+yy*(1-a(2))/(2+a(2)))

      dyda(1)=(1+yy*(1-a(2))/(2+a(2)))

      dyda(2)=-3*a(1)*yy/(2+a(2))**2
 
      dyda(3)=3*z*a(1)*(1-a(2))/(2+a(2))*dz3

      dyda(4)=3*z*a(1)*(1-a(2))/(2+a(2))*dz4
      
      return
      end


      subroutine jwaxial(x,a,y,na)

      real x,a(na),y,r(3),orient(3,256)

      common /rvec/ orient

      i=int(x)
      r(1)=orient(1,i)
      r(2)=orient(2,i)
      r(3)=orient(3,i)

      z=sin(a(3))*(r(1)*cos(a(4))+r(2)*sin(a(4)))
     1     + r(3)*cos(a(3))


      yy=0.5*(3*z**2-1)

      y=a(1)*(1+yy*(1-a(2))/(2+a(2)))

      return
      end


