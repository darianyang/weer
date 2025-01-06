c     getseed.f
c     NOTE: not all systems may support the 'time' subroutine

      subroutine getseed(idum)
c     This is an SGI/PGF specific call
#if defined(USE_SGITIME)

      integer ihour,imin,isec,idum

      character*8 tstr

      call time(tstr)
      ihour =10*(ichar(tstr(1:1))-48) + ichar(tstr(2:2)) - 48
      imin = 10*(ichar(tstr(4:4))-48) + ichar(tstr(5:5)) - 48
      isec = 10*(ichar(tstr(7:7))-48) + ichar(tstr(8:8)) - 48
      idum = 3600*ihour + 60*imin + isec
      idum=-abs(idum)
#endif

c     This is a fairly standard call
#if defined(USE_TIME)
      integer idum

      idum=-abs(time())
#endif

c     This is a G77 call
#if defined(USE_TIME8)
      integer idum

      idum=-abs(time8())
#endif

c     This is the Intel IFORT call
#if defined(USE_CPU_TIME)
      integer idum
      character*8 tsecs

      call time(tsecs)
      print*,tsecs
      idum=-abs(idum)
#endif
      return
      end

