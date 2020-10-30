MODULE lorenzN

USE common_tools
!USE openmp

CONTAINS


SUBROUTINE lorenzN_core(xin,xout,nx,force,dt)
! Lorenz N model (1 scale) 
  IMPLICIT NONE
  INTEGER,INTENT(IN) :: nx
  REAL(r_sngl),INTENT(IN) :: xin(nx),force(nx),dt
  REAL(r_sngl),INTENT(OUT) :: xout(nx)
  INTEGER :: i


  xout(1) = xin(nx) * ( xin(2) - xin(nx-1) ) - xin(1) + force(1) 
  xout(2) = xin(1) * ( xin(3) - xin(nx) ) - xin(2) + force(2) 
  DO i=3,nx-1
    xout(i) = xin(i-1) * ( xin(i+1) - xin(i-2) ) - xin(i) + force(i)
  END DO
  xout(nx) = xin(nx-1) * ( xin(1) - xin(nx-2) ) - xin(nx) + force(nx)

  xout(:) = dt * xout(:)

  RETURN
END SUBROUTINE lorenzN_core

SUBROUTINE lorenzN_ss_core(xssin,xin,xssout,nx,nxss,dtss,css,bss,hint)
  ! Lorenz small-scale equations
  ! Fixed parameters: css,bss,hint
  ! Cyclic conditions are imposed at the extremes 
  ! Notation taken from Wilks QJ 2005.
  IMPLICIT NONE
  INTEGER,INTENT(IN) :: nx,nxss
  REAL(r_sngl),INTENT(IN)  :: xssin(1:nxss),xin(1:nx),dtss
  REAL(r_sngl),INTENT(OUT) :: xssout(1:nxss)
  INTEGER     ,INTENT(IN)  :: css , bss , hint !Small scale parameters and coupling parameters
  INTEGER :: i,ix,j,nj
  REAL(r_sngl) :: a,fss

  nj=nxss/nx

  a=css*bss 
  fss=hint*css/bss

! for ix = 1 
  xssout(1) = a * xssin(2) * ( xssin(nxss) - xssin(3) ) - css * xssin(1) + fss*xin(1)
  DO i=2,nj
    xssout(i) = a * xssin(i+1) * ( xssin(i-1) - xssin(i+2) ) - css * xssin(i) + fss*xin(1)
  ENDDO

! for ix = 2:nx-1
  i=nj
  DO ix=2,nx-1
     DO j=1,nj
        i=i+1
        xssout(i) = a * xssin(i+1) * ( xssin(i-1) - xssin(i+2) ) - css* xssin(i) + fss*xin(ix)
     ENDDO
  ENDDO

! for ix = nx
  DO i=nj*(nx-1)+1,nxss-2 !nxss=nj*nx
    xssout(i) = a *xssin(i+1) * ( xssin(i-1) - xssin(i+2) ) - css* xssin(i) + fss*xin(nx)
  END DO
  xssout(nxss-1) = a * xssin(nxss) * ( xssin(nxss-2) - xssin(1) ) - css*xssin(nxss-1) + fss*xin(nx)
  xssout(nxss) = a * xssin(1) * ( xssin(nxss-1) - xssin(2) ) - css*xssin(nxss) + fss*xin(nx)

  xssout(:) = dtss * xssout(:)

  RETURN
END SUBROUTINE lorenzN_ss_core


SUBROUTINE tinteg_rk4(nens,nt,ntout,x0,xss0,rf0,phi,sigma,c0,crf0,cphi,csigma,nx,nxss,ncoef,  &
     &                param,dt,dtss,xout,xssout,dfout,rfout,ssfout,crfout,cout)

! RK4 time integration of Lorenz N model with deterministic-stochastic parametrization and two scales.
! stochastic forcing changes at each time step.

  IMPLICIT NONE
  !INPUT VARIABLES
  INTEGER , INTENT(IN)      :: nens                      !Number of ensemble members
  INTEGER , INTENT(IN)      :: nx , nxss , ncoef         !Number of large-scale grid points, small scale grid points and number of coefficients
  INTEGER , INTENT(IN)      ::  nt , ntout               !number of large scale time steps , number of model outputs
  REAL(r_sngl) , INTENT(IN) :: dt , dtss                 !Large scale time increment and small scale time increment
  REAL(r_sngl) , INTENT(IN) :: x0(nx,nens)               !Initial condition for the large scale variables
  REAL(r_sngl) , INTENT(IN) :: xss0(nxss,nens)           !Initial condition for the small scale variables
  REAL(r_sngl) , INTENT(IN) :: rf0(nx,nens)              !Initial state of the random forcing.
  REAL(r_sngl) , INTENT(IN) :: c0(nx,nens,ncoef)         !Initial state of parametrization coefficients at t0.
  REAL(r_sngl) , INTENT(IN) :: crf0(nens,ncoef)          !Initial state of the random forcing for the parameter.
  REAL(r_sngl) , INTENT(IN) :: phi  , sigma              !Parameters for AR1 random forcing.
  REAL(r_sngl) , INTENT(IN) :: cphi , csigma(ncoef)      !Parameters for AR1 parameter evolution in time.
  INTEGER      , INTENT(IN) :: param(3)                  !Small scale and coupling parameters css,bss,hint
  !OUTPUT VARIABLES
  REAL(r_sngl),INTENT(OUT) ::  xout(nx,nens,ntout)         !Large scale state ouput
  REAL(r_sngl),INTENT(OUT) ::  xssout(nxss,nens,ntout)     !Small scale state ouput
  REAL(r_sngl),INTENT(OUT) ::  dfout(nx,nens,ntout)        !Deterministic forcing output.
  REAL(r_sngl),INTENT(OUT) ::  rfout(nx,nens,ntout)        !Random forcing output.
  REAL(r_sngl),INTENT(OUT) ::  ssfout(nx,nens,ntout)       !Small scale forcing output.
  REAL(r_sngl),INTENT(OUT) ::  crfout(nens,ncoef,ntout)    !Random forcing output for the deterministic parameters.
  REAL(r_sngl),INTENT(OUT) ::  cout(nx,nens,ncoef,ntout)   !Deterministic forcing parameters output.

  !OTHER VARIABLES
  REAL(r_sngl)              :: x(nx,nens)                !Large-scale model state
  REAL(r_sngl)              :: xss(nxss,nens)            !Small-scale model state
  REAL(r_sngl)              :: rf(nx,nens)               !Random forcing
  REAL(r_sngl)              :: df(nx,nens)               !Deterministic forcing
  REAL(r_sngl)              :: ssf(nx,nens)              !Small scale forcing
  REAL(r_sngl)              :: coef(nx,nens,ncoef)       !Deterministic forcing coefficients
  REAL(r_sngl)              :: crf(nens,ncoef)           !Random forcing for the parameters
  
  REAL(r_size) :: random_forcing_vect(nx) , random_forcing_scalar(1)
  REAL(r_sngl) :: xtmp(nx) , xf1(nx) , xf2(nx) , xf3(nx) , xf4(nx) , force(nx,nens) 
  REAL(r_sngl) :: xsstmp(nxss) , xssf1(nxss) , xssf2(nxss) , xssf3(nxss) , xssf4(nxss)  

  INTEGER                  ::  outputfreq                  !Output frequency (in model time steps) 

  REAL(r_sngl)             :: fct 
  INTEGER :: k,kss,ktss,i,kout,ie,css,bss,hint,nj

!============================================================================
!INITIALIZE SOME VARIABLES AND PARAMETERS
!============================================================================

  css = int(param(1))
  bss = int(param(2))
  hint = int(param(3))

  fct=real(hint,r_sngl)*real(css,r_sngl)/real(bss,r_sngl)      !Coupling parameter

  nj=nxss/nx            !Number of small scale grid points per large scale grid point.
  ktss=int(dt/dtss)     !Number of small scale time steps per large scale time step.

  IF( ntout > 1 )THEN
    outputfreq=int( nt / (ntout-1) )
  ELSE
    write(*,*)'[Error]: ntout should be greather or equal to 2 '
    stop
  ENDIF

! Initialize variables.

  x=x0
  rf=rf0
  xss=xss0

  crf=crf0

!============================================================================
!LOOP OVER THE ENSEMBLE MEMBERS.
!============================================================================

!$OMP PARALLEL DO PRIVATE(random_forcing_scalar,random_forcing_vect,ie,k,i,kout,xtmp,xf1,xf2,xf3,xf4,xsstmp,xssf1,xssf2,xssf3,xssf4)

DO ie = 1,nens

  kout=1
  !============================================================================
  !Write the initial conditions to the output.
  !============================================================================
  xout(:,ie,kout)=x(:,ie)
  xssout(:,ie,kout)=xss(:,ie)
  dfout(:,ie,kout)=0
  DO i=1,ncoef
     cout(:,ie,i,kout)=c0(:,ie,i) + crf(ie,i)
     dfout(:,ie,kout)=dfout(:,ie,kout) + cout(:,ie,i,kout)*x(:,ie)**(i-1)
  END DO
  rfout(:,ie,kout)=rf(:,ie)
  ssfout=0.0d0
  IF( fct /= 0.0 )then
    DO i=1,nx
       ssfout(i,ie,kout)=-fct*sum(xss((i-1)*nj+1:i*nj,ie))
    ENDDO
  END IF
  crfout(ie,:,kout)=crf(ie,:)

  !============================================================================
  !LOOP OVER TIME STEPS.
  !============================================================================
  DO k=1,nt

    !============================================================================
    !Compute the forcing: deterministic + random + small scale
    !============================================================================

    !Compute coefficient for deterministic forcing
    CALL com_randn( 1 , random_forcing_scalar )
    crf(ie,:)=cphi*crf(ie,:)+csigma(:)*random_forcing_scalar(1)  !AR1 process for the parameter evolution.
    DO i=1,ncoef 
       coef(:,ie,i)=c0(:,ie,i) + crf(ie,i) 
    ENDDO

    !Compute deterministic forcing
    df(:,ie)=0
    DO i=1,ncoef
      df(:,ie)=df(:,ie)+coef(:,ie,i)*x(:,ie)**(i-1)
    ENDDO

    !Compute random forcing.
    CALL com_randn( nx ,random_forcing_vect )
    rf(:,ie)=phi*rf(:,ie)+(1-phi**2.0)**(0.5)*sigma*random_forcing_vect(:) 

    !Compute small scale forcing
    ssf=0.0d0
    IF( fct /= 0.0 )then
       DO i=1,nx
          ssf(i,ie)=-fct*sum(xss((i-1)*nj+1:i*nj,ie))
       ENDDO
    END IF

    !Compute total forcing
    force(:,ie)=df(:,ie)+rf(:,ie)+ssf(:,ie)

    !============================================================================
    !Update large scale variables 
    !============================================================================

    xtmp = x(:,ie)
    CALL lorenzN_core(xtmp,xf1,nx,force(:,ie),dt)
    xtmp = x(:,ie) + 0.5 * xf1
    CALL lorenzN_core(xtmp,xf2,nx,force(:,ie),dt)
    xtmp = x(:,ie) + 0.5 * xf2
    CALL lorenzN_core(xtmp,xf3,nx,force(:,ie),dt)
    xtmp = x(:,ie) + xf3
    CALL lorenzN_core(xtmp,xf4,nx,force(:,ie),dt)

    xtmp = x(:,ie) 

    x(:,ie) = x(:,ie) + ( xf1 + 2.0 * xf2 + 2.0 * xf3 + xf4 ) / 6.0

    !============================================================================
    !Update small scale variables
    !============================================================================

    IF( fct /= 0.0 )then 
      !Small scale variables are only updated if the coupling strength is not 0.
      !If the coupling strength is 0, then small scale variables has no effect upon
      !large scale variables.
      DO kss=1,ktss
        xsstmp = xss(:,ie)
        CALL lorenzN_ss_core(xsstmp,xtmp,xssf1,nx,nxss,dtss,css,bss,hint)
        xsstmp = xss(:,ie) + 0.5 * xssf1
        CALL lorenzN_ss_core(xsstmp,xtmp,xssf2,nx,nxss,dtss,css,bss,hint)
        xsstmp = xss(:,ie) + 0.5 * xssf2
        CALL lorenzN_ss_core(xsstmp,xtmp,xssf3,nx,nxss,dtss,css,bss,hint)
        xsstmp = xss(:,ie) + xssf3
        CALL lorenzN_ss_core(xsstmp,xtmp,xssf4,nx,nxss,dtss,css,bss,hint)

        xss(:,ie) = xss(:,ie) + ( xssf1 + 2.0 * xssf2 + 2.0 * xssf3 + xssf4 ) / 6.0
      END DO
    END IF
    

    !============================================================================
    !Store the output
    !============================================================================

    IF ( MOD( k , outputfreq ) == 0  ) THEN

       kout=kout+1 

       xout(:,ie,kout)=x(:,ie)
       xssout(:,ie,kout)=xss(:,ie)
       dfout(:,ie,kout)=df(:,ie)
       rfout(:,ie,kout)=rf(:,ie)
       crfout(ie,:,kout)=crf(ie,:)
       cout(:,ie,:,kout)=coef(:,ie,:)
       !Compute small scale forcing
       ssfout(:,ie,kout)=0.0
       IF( fct /= 0.0 )then
          DO i=1,nx
             ssfout(i,ie,kout)=-fct*sum(xss((i-1)*nj+1:i*nj,ie))
          ENDDO
       END IF

    ENDIF


  END DO  !End do of time steps

END DO  !End do of ensemble members

!$OMP END PARALLEL DO


RETURN

END SUBROUTINE tinteg_rk4


END MODULE lorenzN
