       subroutine mrqcof(x,y,sig,ndata,a,ma,lista,mfit,alpha,beta,nalp,
     1                   chisq, funcs)

c      Routine to evaluate the linearized fitting matrix

       external funcs

       parameter (mmax=20)
       integer lista(mfit)
       real x(ndata), y(ndata), sig(ndata), alpha(nalp,nalp),
     1      beta(ma), dyda(mmax), a(ma), chisq, ymod, sig2i, wt, dy

       do 12 j=1,mfit
         do 11 k=1,j
            alpha(j,k)=0
11       continue
         beta(j)=0
12     continue
       chisq=0.0
       do 15, i=1,ndata
         call funcs(x(i),a,ymod,dyda,ma)
         sig2i=1/sig(i)**2
         dy=y(i)-ymod
         do 14 j=1,mfit
           wt=dyda(lista(j))*sig2i
           do 13 k=1,j
             alpha(j,k)=alpha(j,k)+wt*dyda(lista(k))
13         continue
           beta(j)=beta(j)+dy*wt
14       continue
         chisq=chisq+dy*dy*sig2i
15     continue

       do 17, j=2,mfit
         do 16 k=1,j-1
             alpha(k,j)=alpha(j,k)
16       continue
17     continue
       return
       end

