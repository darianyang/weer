c     zbrent.f				ver. 1.0  1Aug92  AGP
c         optimize tm to R2/R1 ratio

      subroutine ZBRENT(tm,r2r1,X1,X2,TOL)

      PARAMETER (ITMAX=100,EPS=3.E-8)

      A=X1
      B=X2
      FA=FTM(A)-r2r1
      FB=FTM(B)-r2r1
      IF(FB*FA.GT.0.) PAUSE 'Root must be bracketed for ZBRENT.'
      FC=FB
      DO 11 ITER=1,ITMAX
        IF(FB*FC.GT.0.) THEN
          C=A
          FC=FA
          D=B-A
          E=D
        ENDIF
        IF(ABS(FC).LT.ABS(FB)) THEN
          A=B
          B=C
          C=A
          FA=FB
          FB=FC
          FC=FA
        ENDIF
        TOL1=2.*EPS*ABS(B)+0.5*TOL
        XM=.5*(C-B)
        IF(ABS(XM).LE.TOL1 .OR. FB.EQ.0.)THEN
          tm=B
          RETURN
        ENDIF
        IF(ABS(E).GE.TOL1 .AND. ABS(FA).GT.ABS(FB)) THEN
          S=FB/FA
          IF(A.EQ.C) THEN
            P=2.*XM*S
            Q=1.-S
          ELSE
            Q=FA/FC
            R=FB/FC
            P=S*(2.*XM*Q*(Q-R)-(B-A)*(R-1.))
            Q=(Q-1.)*(R-1.)*(S-1.)
          ENDIF
          IF(P.GT.0.) Q=-Q
          P=ABS(P)
          IF(2.*P .LT. MIN(3.*XM*Q-ABS(TOL1*Q),ABS(E*Q))) THEN
            E=D
            D=P/Q
          ELSE
            D=XM
            E=D
          ENDIF
        ELSE
          D=XM
          E=D
        ENDIF
        A=B
        FA=FB
        IF(ABS(D) .GT. TOL1) THEN
          B=B+D
        ELSE
          B=B+SIGN(TOL1,XM)
        ENDIF
        FB=FTM(B)-r2r1
11    CONTINUE
      PAUSE 'ZBRENT exceeding maximum iterations.'
      tm=B
      RETURN
      END


      SUBROUTINE ZBRAC(r2r1,X1,X2,SUCCES)
      PARAMETER (FACTOR=1.6,NTRY=50)
      LOGICAL SUCCES
      IF(X1.EQ.X2)PAUSE 'You have to guess an initial range'
      F1=FTM(X1)-r2r1
      F2=FTM(X2)-r2r1
      SUCCES=.TRUE.
      DO 11 J=1,NTRY
        IF(F1*F2.LT.0.)RETURN
        IF(ABS(F1).LT.ABS(F2))THEN
          X1=X1+FACTOR*(X1-X2)
	  if (x1.lt.0.0) x1=0.0
          F1=FTM(X1)-r2r1
        ELSE
          X2=X2+FACTOR*(X2-X1)
	  if (x2.lt.0.0) x2=0.0
          F2=FTM(X2)-r2r1
        ENDIF
11    CONTINUE
      SUCCES=.FALSE.
      RETURN
      END
