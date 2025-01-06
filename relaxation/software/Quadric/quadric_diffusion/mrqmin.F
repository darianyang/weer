      subroutine mrqmin(x,y,sig,ndata,a,ma,lista,mfit,covar,alpha,nca,
     1                  chisq,funcs,alamda)

c     Routine to perform one iteration of levenberg-marquardt non-linear
c     least squares fitting by minimization of chi-square.  On the first
c     call, set alamda < 0 to initialize, on the last call set alamda =
c     0 in order to return final statistics

c     Parameter list
c      x       independent variable data vector
c      y       dependent variable data vector
c      sig     standard deviations in y data
c      ndata   number of data elements
c      a       vector of coefficients for nonlinear function
c      ma      length of vector a
c      lista   pointer array to vector a, such that first mfit elements
c                 point to the coefficients to be optimized
c      mfit    number of coefficients to be optimized (<=ma); rest are
c                 fixed
c      covar   workspace, returns covariance array on last iteration
c      alpha   workspace, returns curvature array on last iteration
c      nca     dimension of covar and alpha, >=mfit
c      chisq   chi square for current fit
c      funcs   subroutine funcs(x,a,yfit,dyda,ma) which evaluates the
c                function yfit and its derivative dyda at the point x
c      alamda  levenberg-marquardt parameter

       parameter (mmax=20,lwork=1024)
       integer lista(ma),ipiv(mmax),inert(3),info
       real x(ndata), y(ndata), sig(ndata), a(ma), covar(nca,nca),
     1      alpha(nca,nca), atry(mmax), beta(mmax), da(mmax), alamda,
     1     chisq,ochisq, work(lwork)
       save ochisq,atry,beta,da

       if (alamda.lt.0) then
c      check that lista has been properly set up in calling routine
          kk=mfit+1
          do 12, j=1,ma
            ihit=0
            do 11, k=1,mfit
              if (lista(k).eq.j) ihit=ihit+1
11          continue
            if (ihit.eq.0) then
               lista(kk)=j
               kk=kk+1
            elseif (ihit.gt.1) then
               write(0,*) 'Improper permutation in lista array'
               stop
            endif
12        continue
          if (kk.ne.(ma+1)) then
               write(0,*) 'Improper permutation in lista array'
               stop
          endif

c      initialize routine
          alamda=.001
          call mrqcof(x,y,sig,ndata,a,ma,lista,mfit,alpha,beta,nca,
     1                chisq,funcs)
          ochisq=chisq
          do 13, j=1,ma
            atry(j)=a(j)
13        continue
       endif

c      now perform 1 iteration of optimization
       do 15, j=1,mfit
          do 14 k=1,mfit
             covar(j,k)=alpha(j,k)
14        continue
          covar(j,j)=alpha(j,j)*(1.0+alamda)
          da(j)=beta(j)
15     continue

c     now use lapack routines for matrix inversion and
c     solving linear equations
c      write(0,*) 'Covariance Array'
c      do 30 i=1,mfit
c30        write(0,*) (real(covar(i,j)),j=1,mfit)

       info=0
       call ssysv('U',mfit,1,covar,nca,ipiv,da,mmax,work,
     1   lwork,info)
c       do 33 i=1,mfit
c33        write(0,*) (real(covar(i,j)),j=1,mfit)
       if (info.ne.0) goto 99

       info=0
       call ssytri('U',mfit,covar,nca,ipiv,work,info)
       if (info.ne.0) goto 99

       do 31 i=1,mfit
	       do 32 j=i+1,mfit
	        covar(j,i)=covar(i,j)
32         continue
31      continue

       if (alamda.eq.0) then
         call covsrt(covar,nca,ma,lista,mfit)
         return
       endif

       do 16, j=1,mfit
         atry(lista(j))=a(lista(j))+da(j)
16     continue
       call mrqcof(x,y,sig,ndata,atry,ma,lista,mfit,covar,da,nca,chisq,
     1             funcs)

       if (chisq.lt.ochisq) then
         alamda=0.1*alamda
         ochisq=chisq
         do 18 j=1,mfit
            do 17 k=1,mfit
               alpha(j,k)=covar(j,k)
17          continue
            beta(j)=da(j)
            a(lista(j))=atry(lista(j))
18       continue
       else
         alamda=10.*alamda
         chisq=ochisq
       endif
       return

c      error return for singular matrix
99     alambda=10.*alamda
       chisq=ochisq
       write(0,*) 'ssysv: ',info,'th pivot block is singular'
       return

       end

