MODULE common_da_tools
!=======================================================================
!
! [PURPOSE:] Data assimilation tools for 1D models
!
!=======================================================================
!$USE OMP_LIB
  USE common_tools
  USE common_letkf
  USE common_pf
  USE common_gm
  USE rand_matrix
  IMPLICIT NONE
  PUBLIC
CONTAINS

!=======================================================================
!  LETKF DA for the 1D model
!=======================================================================
SUBROUTINE da_letkf(nx,nt,no,nens,nvar,xloc,tloc,xfens,xaens,obs,obsloc,ofens,Rdiag, &   
         &          loc_scale,inf_coefs,update_smooth_coef,temp_factor)

IMPLICIT NONE
INTEGER,INTENT(IN)         :: nx , nt , nvar             !State dimensions, space, time and variables
INTEGER,INTENT(IN)         :: no                         !Number of observations 
INTEGER,INTENT(IN)         :: nens                       !Number of ensemble members
REAL(r_size),INTENT(IN)    :: xloc(nx)                   !Location of state grid points (space)
REAL(r_size),INTENT(IN)    :: tloc(nt)                   !Location of state grid points (time)
REAL(r_size),INTENT(IN)    :: obsloc(no,2)               !Location of obs (space , time)
REAL(r_size),INTENT(IN)    :: xfens(nx,nens,nvar,nt)     !Forecast state ensemble  
REAL(r_size),INTENT(OUT)   :: xaens(nx,nens,nvar,nt)     !Analysis state ensemble 
REAL(r_size),INTENT(IN)    :: ofens(no,nens)             !Ensemble in observation space
REAL(r_size),INTENT(IN)    :: obs(no)                    !Observations 
REAL(r_size),INTENT(IN)    :: Rdiag(no)                  !Diagonal of observation error covariance matrix.
REAL(r_size),INTENT(INOUT) :: inf_coefs(5)               !Mult inf, RTPS , RTPP , EPES, Additive inflation (State variables)
REAL(r_size),INTENT(IN)    :: update_smooth_coef         !Update smooth parameter.
REAL(r_size),INTENT(IN)    :: temp_factor(nx,nt)         !Tempering factor ( R -> R*temp_factor)
REAL(r_size)               :: xfpert(nx,nens,nvar,nt)       !State and parameter forecast perturbations
REAL(r_size)               :: xapert(nx,nens,nvar,nt)       !State and parameter analysis perturbations
REAL(r_size)               :: xfmean(nx,nvar,nt)            !State and parameter ensemble mean (forecast)
REAL(r_size)               :: ofmean(no) , ofpert(no,nens)                         !Ensemble mean in obs space, ensemble perturbations in obs space (forecast)
REAL(r_size)               :: d(no)                                                !Observation departure



REAL(r_size)               :: ofpert_loc(no,nens) , Rdiag_loc(no)                  !Localized ensemble in obs space and diag of R
REAL(r_size)               :: Rwf_loc(no)                                          !Localization weigths.
REAL(r_size)               :: d_loc(no)                                            !Localized observation departure.
INTEGER                    :: no_loc                                               !Number of observations in the local domain.
REAL(r_size),INTENT(IN)    :: loc_scale(2)                                         !Localization scales (space,time)

REAL(r_size)               :: wa(nens,nens)                                        !Analysis weights
REAL(r_size)               :: wainf(nens,nens)                                     !Analysis weights after inflation.
REAL(r_size)               :: wamean(nens)                                         !Mean analysis weights
REAL(r_size)               :: pa(nens,nens)                                        !Analysis cov matrix in ensemble space)

REAL(r_size),PARAMETER     :: min_infl=1.0d0                                       !Minumn allowed multiplicative inflation.
REAL(r_size)               :: mult_inf
REAL(r_size)               :: grid_loc(2)
REAL(r_size)               :: work1d(nx)

INTEGER                    :: ix,ie,ke,it,iv,io
REAL(r_size)               :: dx

!Initialization
xfmean=0.0d0
xfpert=0.0d0
xaens =0.0d0
ofmean=0.0d0
ofpert=0.0d0
d     =0.0d0

wa=0.0d0
wamean=0.0d0
wainf =0.0d0
pa=0.0d0

dx=xloc(2)-xloc(1)    !Assuming regular grid

!Compute forecast ensemble mean and perturbations.


DO it = 1,nt

 DO ix = 1,nx

  DO iv = 1,nvar

   CALL com_mean( nens,xfens(ix,:,iv,it),xfmean(ix,iv,it) )
 
   xfpert(ix,:,iv,it) = xfens(ix,:,iv,it) - xfmean(ix,iv,it) 

  END DO

 END DO

END DO



!Compute mean departure and HXf


DO io = 1,no

    CALL com_mean( nens , ofens(io,:) , ofmean(io) )

    ofpert(io,:)=ofens(io,:) - ofmean(io) 

    d(io) = obs(io) - ofmean(io)   

ENDDO




!Main assimilation loop.

DO it = 1,nt


!$OMP PARALLEL DO SCHEDULE(DYNAMIC) PRIVATE(grid_loc,no_loc,ofpert_loc,d_loc   &
!$OMP &          ,mult_inf,Rdiag_loc,Rwf_loc,wa,wamean,pa,wainf,ie,iv,ke)
  DO ix = 1,nx

   !Localize observations
   grid_loc(1)=xloc(ix)  !Set the grid point location in space
   grid_loc(2)=tloc(it)  !Set the grid point location in time

   CALL r_localization(no,no_loc,nens,ofpert,d,Rdiag,ofpert_loc,     &
                     d_loc,Rdiag_loc,Rwf_loc,grid_loc,xloc(1),xloc(nx),dx,obsloc,loc_scale)

   !Aplly local tempering factor to the error covariance matrix.
   Rdiag_loc(1:no_loc) = Rdiag_loc(1:no_loc) * temp_factor(ix,it)

   IF( no_loc > 0 )THEN   
    !We have observations for this grid point. Let's compute the analysis.

    !Set multiplicative inflation
    mult_inf=inf_coefs(1) ** ( 1.0d0 / temp_factor(ix,it) ) 
  
    !Compute analysis weights
 
    CALL letkf_core( nens,no_loc,ofpert_loc(1:no_loc,:),Rdiag_loc(1:no_loc),   &
                    Rwf_loc(1:no_loc),d_loc(1:no_loc),mult_inf,wa,wamean,pa,min_infl )

    !Update state variables (apply RTPP and RTPS )
    IF( inf_coefs(2) /= 0.0d0) THEN                                                         !GYL - RTPP method (Zhang et al. 2005)
      CALL weight_RTPP(nens,inf_coefs(2),wa,wainf)                                          !GYL
    ELSE IF( inf_coefs(4) /= 0.0d0) THEN                                                    !EPES
      CALL weight_EPES(nens,inf_coefs(4),wa,wainf)
    !ELSE IF( inf_coefs(5) /= 0.0d0) THEN                                                    !TODO: Code additive inflation
      !write(*,*)"[Warning]: Additive inflation not implemented for LETKF yet"
    ELSE
      wainf = wa                                                                            !GYL
    END IF  

    !Apply the weights and update the state variables. 
   
    DO iv=1,nvar
      !RTPS inflation is variable dependent, so we have to implement it here. 
      IF( inf_coefs(3) /= 0.0d0) THEN
        CALL weight_RTPS(nens,inf_coefs(3),wa,pa,xfens(ix,:,iv,it),wainf)
      ENDIF

      DO ie=1,nens
       xaens(ix,ie,iv,it) = xfmean(ix,iv,it)
       DO ke = 1,nens
          xaens(ix,ie,iv,it) = xaens(ix,ie,iv,it) &                                         !GYL - sum trans and transm here
              & + xfpert(ix,ke,iv,it) * (wainf(ke,ie) + wamean(ke))                         !GYL
       END DO
      END DO
    END DO

   ELSE  

    !We don't have observations for this grid point. We can do nothing :( 
    xaens(ix,:,:,it)=xfens(ix,:,:,it)
 
   ENDIF

  END DO
!$OMP END PARALLEL DO

END DO  




IF( update_smooth_coef > 0.0d0 )THEN


! allocate( work1d( nx ) )

 DO it = 1,nt

  DO iv = 1,nvar

   DO ie = 1,nens

    !Smooth the update of each ensemble member.
    work1d = xaens(:,ie,iv,it) - xfens(:,ie,iv,it) !DA update
    CALL com_filter_lanczos( nx , update_smooth_coef , work1d )
    xaens(:,ie,iv,it) = xfens(:,ie,iv,it) + work1d 

   END DO

  END DO

 END DO

! deallocate( work1d )

ENDIF

END SUBROUTINE da_letkf

!=======================================================================
!  ETKF DA for 1D models
!=======================================================================

SUBROUTINE da_etkf(nx,nt,no,nens,nvar,xfens,xaens,obs,ofens,Rdiag,inf_coefs)

IMPLICIT NONE
INTEGER,INTENT(IN)         :: no , nt , nx , nvar        !Number of observations , number of times , number of state variables ,number of variables
INTEGER,INTENT(IN)         :: nens                       !Number of ensemble members
REAL(r_size),INTENT(IN)    :: xfens(nx,nens,nvar,nt)     !Forecast state ensemble  
REAL(r_size),INTENT(OUT)   :: xaens(nx,nens,nvar,nt)     !Analysis state ensemble 
REAL(r_size),INTENT(IN)    :: ofens(no,nens)             !Ensemble in observation space
REAL(r_size),INTENT(IN)    :: obs(no)                    !Observations 
REAL(r_size),INTENT(IN)    :: Rdiag(no)                  !Diagonal of observation error covariance matrix.
REAL(r_size),INTENT(INOUT) :: inf_coefs(6)               !Mult inf, RTPS , RTPP , EPES, Additive inflation (State variables)
REAL(r_size)               :: xfpert(nx,nens,nvar,nt)    !State and parameter forecast perturbations
REAL(r_size)               :: xfmean(nx,nvar,nt)         !State and parameter ensemble mean (forecast)
REAL(r_size)               :: ofmean(no) , ofpert(no,nens)                         !Ensemble mean in obs space, ensemble perturbations in obs space (forecast)
REAL(r_size)               :: d(no)                                                !Observation departure

REAL(r_size)               :: wa(nens,nens)                                        !Analysis weights
REAL(r_size)               :: wainf(nens,nens)                                     !Analysis weights after inflation.
REAL(r_size)               :: wamean(nens)                                         !Mean analysis weights
REAL(r_size)               :: pa(nens,nens)                                        !Analysis cov matrix in ensemble space)

REAL(r_size),PARAMETER     :: min_infl=1.0d0                                       !Minumn allowed multiplicative inflation.
REAL(r_size)               :: mult_inf
REAL(r_size)               :: work

REAL(r_size)               :: Rwf(no)                                              !Dummy variable.

INTEGER                    :: ie,ke,iv,io,ix,it

!Initialization
xfmean=0.0
xfpert=0.0

!Compute forecast ensemble mean and perturbations.

DO it = 1,nt

 DO ix = 1,nx

  DO iv = 1,nvar

   CALL com_mean( nens,xfens(ix,:,iv,it),xfmean(ix,iv,it) )
 
   xfpert(ix,:,iv,it) = xfens(ix,:,iv,it) - xfmean(ix,iv,it) 

  END DO

 END DO

END DO


!Compute mean departure and HXf

DO io = 1,no

    CALL com_mean( nens , ofens(io,:) , ofmean(io) )

    ofpert(io,:)=ofens(io,:) - ofmean(io) 

    d(io) = obs(io) - ofmean(io)   

ENDDO

!Set multiplicative inflation

mult_inf=inf_coefs(1)

!Main assimilation loop.

IF( no > 0 )THEN   
  !We have observations for this grid point. Let's compute the analysis.
   
  !Compute analysis weights
  Rwf=1.0d0 
  CALL letkf_core( nens,no,ofpert,Rdiag,Rwf,d,   &
                   mult_inf,wa,wamean,pa,min_infl )


  !Update state variables (apply RTPP and RTPS )
  IF( inf_coefs(2) /= 0.0d0) THEN                                                            !GYL - RTPP method (Zhang et al. 2005)
    CALL weight_RTPP(nens,inf_coefs(2),wa,wainf)                                             !GYL
  ELSE IF( inf_coefs(4) /= 0.0d0) THEN                                                       !EPES
    CALL weight_EPES(nens,inf_coefs(4),wa,wainf)
  ELSE IF( inf_coefs(5) /= 0.0d0) THEN                                                       !TODO: Code additive inflation
    write(*,*)"[Warning]: Additive inflation not implemented for ETKF yet"
  ELSE
    wainf = wa                                                                               !GYL
  END IF                                                                                     !GYL
   
  !Apply the weights and update the state variables.
  DO it=1,nt
   DO ix=1,nx 
    DO iv=1,nvar
     !RTPS inflation is variable dependent, so we have to implement it here. 
     IF( inf_coefs(3) /= 0.0d0) THEN
       CALL weight_RTPS(nens,inf_coefs(3),wa,pa,xfens(ix,:,iv,it),wainf)
     ENDIF

     DO ie=1,nens
       xaens(ix,ie,iv,it) = xfmean(ix,iv,it)
       DO ke = 1,nens
          xaens(ix,ie,iv,it) = xaens(ix,ie,iv,it) &                                          !GYL - sum trans and transm here
                        & + xfpert(ix,ke,iv,it) * ( wainf(ke,ie) + wamean(ke) )               !GYL
       END DO
     END DO

    END DO
   END DO
  END DO

ELSE  

  !We don't have observations for this grid point. We can do nothing :( 
  xaens(:,:,:,:)=xfens(:,:,:,:)

ENDIF



END SUBROUTINE da_etkf


!=======================================================================
!  GaussianMixture with deterministic resampling DA for the 1D model
!  Liu et al. 2016 MWR
!=======================================================================
SUBROUTINE da_gmdr(nx,nt,no,nens,nvar,xloc,tloc,xfens,xaens,w_pf,obs,obsloc,ofens,Rdiag, &
                   loc_scale,inf_coefs,beta_coef,gamma_coef,resampling_type,temp_factor)

IMPLICIT NONE
INTEGER,INTENT(IN)         :: nx , nt , nvar             !State dimensions, space, time and variables
INTEGER,INTENT(IN)         :: no                         !Number of observations 
INTEGER,INTENT(IN)         :: nens                       !Number of ensemble members
REAL(r_size),INTENT(IN)    :: xloc(nx)                   !Location of state grid points (space)
REAL(r_size),INTENT(IN)    :: tloc(nt)                   !Location of state grid points (time)
REAL(r_size),INTENT(IN)    :: obsloc(no,2)               !Location of obs (space , time)
REAL(r_size),INTENT(IN)    :: xfens(nx,nens,nvar,nt)     !Forecast state ensemble  
REAL(r_size),INTENT(OUT)   :: xaens(nx,nens,nvar,nt)     !Analysis state ensemble 
REAL(r_size),INTENT(IN)    :: ofens(no,nens)             !Ensemble in observation space
REAL(r_size),INTENT(IN)    :: obs(no)                    !Observations 
REAL(r_size),INTENT(IN)    :: Rdiag(no)                  !Diagonal of observation error covariance matrix.
REAL(r_size),INTENT(INOUT) :: inf_coefs(5)               !Mult inf, RTPS , RTPP , EPES, Additive inflation (State variables)
REAL(r_size),INTENT(IN)    :: beta_coef                  !Gaussian Kernel scalling parameter
REAL(r_size),INTENT(IN)    :: gamma_coef                 !Weigths nudging parameter
REAL(r_size),INTENT(IN)    :: temp_factor(nx,nt)         !Tempering factor ( R -> R*temp_factor)
REAL(r_size)               :: xfpert(nx,nens,nvar,nt)       !State and parameter forecast perturbations
REAL(r_size)               :: xfmean(nx,nvar,nt)            !State and parameter ensemble mean (forecast)
REAL(r_size)               :: xamean , xawmean , xapert(nens)             !For deterministic resampling.
REAL(r_size)               :: ofmean(no) , ofpert(no,nens)                         !Ensemble mean in obs space, ensemble perturbations in obs space (forecast)
REAL(r_size)               :: d(no)                                                !Observation departure
REAL(r_size),INTENT(OUT)   :: w_pf(nx,nens,nt)              !Posterior weigths
REAL(r_size)               :: ofpert_loc(no,nens) , Rdiag_loc(no)                  !Localized ensemble in obs space and diag of R
REAL(r_size)               :: Rwf_loc(no)                                          !Localization weigths.
REAL(r_size)               :: d_loc(no)                                            !Localized mean departure
INTEGER                    :: no_loc                                               !Number of observations in the local domain.
REAL(r_size),INTENT(IN)    :: loc_scale(2)                                         !Localization scales (space,time)
REAL(r_size)               :: w(nens,nens)                                        !Analysis weights
REAL(r_size),PARAMETER     :: min_infl=1.0d0                                       !Minumn allowed multiplicative inflation.
REAL(r_size)               :: mult_inf
REAL(r_size)               :: grid_loc(2)
REAL(r_size)               :: work1d(nx)
REAL(r_size)               :: m(nens,nens) , wt(nens) , delta(nens,nens) , wf(nens,nens) , rr_matrix(nens,nens)
REAL(r_size)               :: tmp_ens(nens,nvar)
INTEGER,INTENT(IN)         :: resampling_type         !1-Liu, 2-Reich , NETPF without rotation ,NETPF with random rotation

!
INTEGER                    :: ix,ie,ke,it,iv,io
REAL(r_size)               :: dx

!Initialization
xfmean=0.0d0
xfpert=0.0d0
xaens =0.0d0
ofmean=0.0d0
ofpert=0.0d0
d     =0.0d0

w=1.0d0/REAL(nens,r_size)
w_pf=1.0d0/REAL(nens,r_size)

dx=xloc(2)-xloc(1)    !Assuming regular grid

!Compute forecast ensemble mean and perturbations.
DO it = 1,nt
 DO ix = 1,nx
  DO iv = 1,nvar
   CALL com_mean( nens,xfens(ix,:,iv,it),xfmean(ix,iv,it) )
   xfpert(ix,:,iv,it) = xfens(ix,:,iv,it) - xfmean(ix,iv,it) 
  END DO
 END DO
END DO

!!Compute mean departure and HXf
DO io = 1,no
    CALL com_mean( nens , ofens(io,:) , ofmean(io) )
    ofpert(io,:)=ofens(io,:) - ofmean(io)
    d(io) = obs(io) - ofmean(io)
ENDDO

wt = 1.0d0 / REAL( nens , r_size )  !Compute the target weigths (equal weigths in this case)

!!Main assimilation loop.

IF ( resampling_type == 4 )THEN
  !Compute random rotation matrix
  !Random rotation matrix generator from PDAF
  CALL PDAF_generate_rndmat( nens, rr_matrix , 2)
ENDIF


DO it = 1,nt


!$OMP PARALLEL DO SCHEDULE(DYNAMIC) PRIVATE(grid_loc,no_loc,ofpert_loc,d_loc   &
!$OMP & ,mult_inf,Rdiag_loc,w,Rwf_loc,ie,iv,ke,wf,delta,tmp_ens,m,xamean,xapert,xawmean)
  DO ix = 1,nx

!   !Localize observations
   grid_loc(1)=xloc(ix)  !Set the grid point location in space
   grid_loc(2)=tloc(it)  !Set the grid point location in time

   CALL r_localization(no,no_loc,nens,ofpert,d,Rdiag,ofpert_loc,     &
                     d_loc,Rdiag_loc,Rwf_loc,grid_loc,xloc(1),xloc(nx),dx,obsloc,loc_scale)

   !Aplly local tempering factor to the error covariance matrix.
   Rdiag_loc(1:no_loc) = Rdiag_loc(1:no_loc) * temp_factor(ix,it)

   IF( no_loc > 0 )THEN   
    !We have observations for this grid point. Let's compute the analysis.

    !w_pf=1.0d0/REAL(nens,r_size)
    !Compute weigths taking into account Gaussian Kernels.
    CALL pf_weigth_core( nens , no_loc , ofpert_loc(1:no_loc,:),d_loc(1:no_loc), Rdiag_loc(1:no_loc) , &
                         beta_coef , gamma_coef , w_pf(ix,:,it) )  
    !Set multiplicative inflation
    !Currently this is set to 1. The reason is because EnKF update is alredy controled by beta, so there is no reason
    !to add another parameter controling the update. Beta plays the role of a deflation of the ensemble actually. 
    mult_inf=1.0d0 
    !Compute analysis weights

    !w = 1.0d0/REAL(nens,r_size) 
    CALL letkf_gm_core(nens,no_loc,ofpert_loc(1:no_loc,:),Rdiag_loc(1:no_loc),d_loc(1:no_loc),mult_inf,w,min_infl,beta_coef)
   
    !Shift the particle according to the LETKF update. 

    DO iv=1,nvar
      DO ie=1,nens
       xaens(ix,ie,iv,it) = xfens(ix,ie,iv,it)   
       DO ke = 1,nens
          xaens(ix,ie,iv,it) = xaens(ix,ie,iv,it) &  
              & + xfpert(ix,ke,iv,it) * w(ke,ie)       
       END DO
      END DO
    END DO

   ELSE  

    !We don't have observations for this grid point. We can do nothing :( 
    xaens(ix,:,:,it)=xfens(ix,:,:,it)
 
   ENDIF


   !--------------------------------------------------------------------
   ! Deterministic Resampling Step
   !--------------------------------------------------------------------

   IF ( resampling_type == 1 ) THEN
      !-----------------------------------------------------------------
      ! Liu et al 2016 Deterministic resampling
      !-----------------------------------------------------------------
      DO iv=1,nvar
        xamean = SUM( xaens(ix,:,iv,it) ) / REAL(nens,r_size)
        xawmean= SUM( xaens(ix,:,iv,it) * w_pf(ix,:,it) )/SUM( w_pf(ix,:,it) )
        xaens(ix,:,iv,it) = SQRT(inf_coefs(1)+beta_coef) * ( xaens(ix,:,iv,it) - xamean ) + xawmean
      END DO
   ELSEIF( resampling_type == 2 )THEN
      !-----------------------------------------------------------------
      ! Acevedo et al 2016 Deterministic resampling
      !-----------------------------------------------------------------
      !Compute analysis weights
      CALL get_distance_matrix( nens , 1 , nvar , 1 , xfens(ix,:,:,it) , m )
      !Solve the regularized optimal transport problem.
      CALL sinkhorn_ot( nens , w_pf(ix,:,it) , wt , m , wf , lambda_reg , stop_threshold_sinkhorn , max_iter_sinkhorn )
      !Call Riccati solver
      delta = 0.0d0
      mult_inf = inf_coefs(1) ** ( 1.0d0 / temp_factor(ix,it) ) 
      CALL riccati_solver( nens ,wf,w_pf(ix,:,it),dt_riccati,stop_threshold_riccati,max_iter_riccati,delta,mult_inf)
      wf = wf + delta
      !Compute the updated ensemble mean, std and mode.
      tmp_ens = 0.0d0
      DO ie=1,nens
         DO ke=1,nens
           tmp_ens(ie,:) =  tmp_ens(ie,:) + xaens(ix,ke,:,it) * wf(ke,ie)
         ENDDO
      ENDDO
      xaens(ix,:,:,it) = tmp_ens
   ELSEIF ( resampling_type == 3 .or. resampling_type == 4 )THEN
      !-----------------------------------------------------------------
      ! Todter and Ahrens 2015 MWR
      !-----------------------------------------------------------------
      CALL netpf_w( nens , w_pf(ix,:,it) , wf )
      wf = SQRT( inf_coefs(1) ) * wf

      IF( resampling_type == 4 )THEN
         wf = MATMUL( wf , rr_matrix ) 
      ENDIF

      !Recompute the ensemble mean and perturbations
      DO iv = 1,nvar
          CALL com_mean( nens,xaens(ix,:,iv,it),xfmean(ix,iv,it) )
          xfpert(ix,:,iv,it) = xaens(ix,:,iv,it) - xfmean(ix,iv,it)
      END DO

      !Compute the new analysis mean.
      DO iv=1,nvar
        xamean  = xfmean(ix,iv,it)
        DO ie = 1 ,nens
           xamean = xamean + w_pf(ix,ie,it) * xfpert(ix,ie,iv,it)
        ENDDO
        xapert = 0.0d0
        DO ie=1,nens
          DO ke=1,nens
           xapert(ie) =  xapert(ie) + xfpert(ix,ke,iv,it) * wf(ke,ie)
          ENDDO
        ENDDO
        xaens(ix,:,iv,it) = xamean + xapert
      END DO


   ENDIF

  END DO
!$OMP END PARALLEL DO



END DO  

END SUBROUTINE da_gmdr

!=======================================================================
!  GaussianMixture with deterministic resampling DA for the 1D model
!  Local linealization of the observation operator. 
!  Liu et al. 2016 MWR
!=======================================================================
!In this implementation we centered all the perturbations around each ensemble member
!in order to compute a local linearization of the observation operator.
!The observation operator is applied outside this subroutine, this is why here the input for 
!the observation operator has one more dimension.

SUBROUTINE da_gmdr_localh(nx,nt,no,nens,nvar,xloc,tloc,xfens,xaens,w_pf,obs,obsloc,ofens,Rdiag, &
                   loc_scale,inf_coefs,beta_coef,gamma_coef,resampling_type,temp_factor)

IMPLICIT NONE
INTEGER,INTENT(IN)         :: nx , nt , nvar             !State dimensions, space, time and variables
INTEGER,INTENT(IN)         :: no                         !Number of observations 
INTEGER,INTENT(IN)         :: nens                       !Number of ensemble members
REAL(r_size),INTENT(IN)    :: xloc(nx)                   !Location of state grid points (space)
REAL(r_size),INTENT(IN)    :: tloc(nt)                   !Location of state grid points (time)
REAL(r_size),INTENT(IN)    :: obsloc(no,2)               !Location of obs (space , time)
REAL(r_size),INTENT(IN)    :: xfens(nx,nens,nvar,nt)     !Forecast state ensemble  
REAL(r_size),INTENT(OUT)   :: xaens(nx,nens,nvar,nt)     !Analysis state ensemble 
REAL(r_size),INTENT(IN)    :: ofens(no,nens,nens)        !Ensemble in observation space ofens(:,:,iens) are the ensemble in observation space centered around ensemble member iens.
REAL(r_size),INTENT(IN)    :: obs(no)                    !Observations 
REAL(r_size),INTENT(IN)    :: Rdiag(no)                  !Diagonal of observation error covariance matrix.
REAL(r_size),INTENT(INOUT) :: inf_coefs(5)               !Mult inf, RTPS , RTPP , EPES, Additive inflation (State variables)
REAL(r_size),INTENT(IN)    :: beta_coef                  !Gaussian Kernel scalling parameter
REAL(r_size),INTENT(IN)    :: gamma_coef                 !Weigths nudging parameter
REAL(r_size),INTENT(IN)    :: temp_factor(nx,nt)         !Tempering factor ( R -> R*temp_factor)
REAL(r_size)               :: xfpert(nx,nens,nvar,nt)       !State and parameter forecast perturbations
REAL(r_size)               :: xfmean(nx,nvar,nt)            !State and parameter ensemble mean (forecast)
REAL(r_size)               :: xamean , xawmean , xapert(nens)             !For deterministic resampling.
REAL(r_size)               :: ofmean(no,nens) , ofpert(no,nens,nens)                         !Ensemble mean in obs space, ensemble perturbations in obs space (forecast)
REAL(r_size)               :: d(no,nens)                                  !Observation departure
REAL(r_size),INTENT(OUT)   :: w_pf(nx,nens,nt)              !Posterior weigths
REAL(r_size)               :: ofpert_loc(no,nens,nens) , Rdiag_loc(no)                  !Localized ensemble in obs space and diag of R
REAL(r_size)               :: Rwf_loc(no)                                          !Localization weigths.
REAL(r_size)               :: d_loc(no,nens)                                            !Localized mean departure
INTEGER                    :: no_loc                                               !Number of observations in the local domain.
REAL(r_size),INTENT(IN)    :: loc_scale(2)                                         !Localization scales (space,time)
REAL(r_size)               :: w(nens,nens)                                        !Analysis weights
REAL(r_size),PARAMETER     :: min_infl=1.0d0                                       !Minumn allowed multiplicative inflation.
REAL(r_size)               :: mult_inf
REAL(r_size)               :: grid_loc(2)
REAL(r_size)               :: work1d(nx)
REAL(r_size)               :: m(nens,nens) , wt(nens) , delta(nens,nens) , wf(nens,nens) , rr_matrix(nens,nens)
REAL(r_size)               :: tmp_ens(nens,nvar)
INTEGER,INTENT(IN)         :: resampling_type         !1-Liu, 2-Reich , NETPF without rotation ,NETPF with random rotation

!
INTEGER                    :: ix,ie,ke,it,iv,io
REAL(r_size)               :: dx

!Initialization
xfmean=0.0d0
xfpert=0.0d0
xaens =0.0d0
ofmean=0.0d0
ofpert=0.0d0
d     =0.0d0
no_loc=0

w=1.0d0/REAL(nens,r_size)
w_pf=1.0d0/REAL(nens,r_size)
wt = 1.0d0/REAL( nens , r_size )  !Compute the target weigths (equal weigths in this case)


dx=xloc(2)-xloc(1)    !Assuming regular grid

!Compute forecast ensemble mean and perturbations.
DO it = 1,nt
 DO ix = 1,nx
  DO iv = 1,nvar
   CALL com_mean( nens,xfens(ix,:,iv,it),xfmean(ix,iv,it) )
   xfpert(ix,:,iv,it) = xfens(ix,:,iv,it) - xfmean(ix,iv,it) 
  END DO
 END DO
END DO

!!Compute mean departure and HXf
DO io = 1,no
   DO ie = 1,nens
    CALL com_mean( nens , ofens(io,:,ie) , ofmean(io,ie) )
    ofpert(io,:,ie)=ofens(io,:,ie) - ofmean(io,ie)
    d(io,ie) = obs(io) - ofmean(io,ie)
   ENDDO
ENDDO

!!Main assimilation loop.

IF ( resampling_type == 4 )THEN
  !Compute random rotation matrix
  !Random rotation matrix generator from PDAF
  CALL PDAF_generate_rndmat( nens, rr_matrix , 2)
ENDIF


DO it = 1,nt

!$OMP PARALLEL DO SCHEDULE(DYNAMIC) PRIVATE(grid_loc,no_loc,ofpert_loc,d_loc   &
!$OMP & ,mult_inf,Rdiag_loc,w,Rwf_loc,ie,iv,ke,wf,delta,tmp_ens,m,xamean,xapert)
  DO ix = 1,nx

   !Localize observations
   grid_loc(1)=xloc(ix)  !Set the grid point location in space
   grid_loc(2)=tloc(it)  !Set the grid point location in time

   !Localize all the ensembles 
   CALL r_localization_localh(no,no_loc,nens,ofpert,d,Rdiag,ofpert_loc,     &
                     d_loc,Rdiag_loc,Rwf_loc,grid_loc,xloc(1),xloc(nx),dx,obsloc,loc_scale)

   !Aplly local tempering factor to the error covariance matrix.
   Rdiag_loc(1:no_loc) = Rdiag_loc(1:no_loc) * temp_factor(ix,it)

   IF( no_loc > 0 )THEN   
    !We have observations for this grid point. Let's compute the analysis.

    !w_pf=1.0d0/REAL(nens,r_size)
    !Compute weigths taking into account Gaussian Kernels.
    CALL pf_weigth_localh_core( nens , no_loc , ofpert_loc(1:no_loc,:,:),d_loc(1:no_loc,:), Rdiag_loc(1:no_loc) , &
                         beta_coef , gamma_coef , w_pf(ix,:,it) )  
   
    !Set multiplicative inflation
    !Currently this is set to 1. The reason is because EnKF update is alredy controled by beta, so there is no reason
    !to add another parameter controling the update. Beta plays the role of a deflation of the ensemble actually. 
    mult_inf=1.0d0 
    !Compute analysis weights

    !w = 1.0d0/REAL(nens,r_size) 
    CALL letkf_gm_localh_core(nens,no_loc,ofpert_loc(1:no_loc,:,:),Rdiag_loc(1:no_loc),  &
        &                     d_loc(1:no_loc,:),mult_inf,w,min_infl,beta_coef)
   
    !Shift the particle according to the LETKF update. 

    DO iv=1,nvar
      DO ie=1,nens
       xaens(ix,ie,iv,it) = xfens(ix,ie,iv,it)   
       DO ke = 1,nens
          xaens(ix,ie,iv,it) = xaens(ix,ie,iv,it) &  
              & + xfpert(ix,ke,iv,it) * w(ke,ie)       
       END DO
      END DO
    END DO

   ELSE  

    !We don't have observations for this grid point. We can do nothing :( 
    xaens(ix,:,:,it)=xfens(ix,:,:,it)
 
   ENDIF

   !--------------------------------------------------------------------
   ! Deterministic Resampling Step
   !--------------------------------------------------------------------

   IF( no_loc > 0 )THEN
     IF ( resampling_type == 1 ) THEN
        !-----------------------------------------------------------------
        ! Liu et al 2016 Deterministic resampling
        !-----------------------------------------------------------------
        DO iv=1,nvar
          xamean  = 0.0d0
          xawmean = 0.0d0 
          DO ie = 1 ,nens
             xamean  = xamean  + xaens(ix,ie,iv,it) 
             xawmean = xawmean + w_pf(ix,ie,it) * xaens(ix,ie,iv,it) 
          ENDDO
          xamean  = xamean  / REAL( nens , r_size )
         !Expand perturbations by the factor sqrt(1+beta) and recenter around the PF mean.
          xaens(ix,:,iv,it) = SQRT(1.0d0+0.6*beta_coef) * ( xaens(ix,:,iv,it) - xamean ) + xawmean
        END DO
     ELSEIF( resampling_type == 2 )THEN
        !-----------------------------------------------------------------
        ! Acevedo et al 2016 Deterministic resampling
        !-----------------------------------------------------------------
        !Compute analysis weights
        CALL get_distance_matrix( nens , 1 , nvar , 1 , xfens(ix,:,:,it) , m )
        !Solve the regularized optimal transport problem.
        CALL sinkhorn_ot( nens , w_pf(ix,:,it) , wt , m , wf , lambda_reg , stop_threshold_sinkhorn , max_iter_sinkhorn )
        !Call Riccati solver
        delta = 0.0d0
        mult_inf = inf_coefs(1) ** ( 1.0d0 / temp_factor(ix,it) ) 
        CALL riccati_solver( nens ,wf,w_pf(ix,:,it),dt_riccati,stop_threshold_riccati,max_iter_riccati,delta,mult_inf)
        wf = wf + delta
        !Compute the updated ensemble mean, std and mode.
        tmp_ens = 0.0d0
        DO ie=1,nens
           DO ke=1,nens
             tmp_ens(ie,:) =  tmp_ens(ie,:) + xaens(ix,ke,:,it) * wf(ke,ie)
           ENDDO
        ENDDO
        xaens(ix,:,:,it) = tmp_ens
     ELSEIF ( resampling_type == 3 .or. resampling_type == 4 )THEN
        !-----------------------------------------------------------------
        ! Todter and Ahrens 2015 MWR
        !-----------------------------------------------------------------
        CALL netpf_w( nens , w_pf(ix,:,it) , wf )
        wf = SQRT( inf_coefs(1) ) * wf

        IF( resampling_type == 4 )THEN
           wf = MATMUL( wf , rr_matrix ) 
        ENDIF

        !Recompute the ensemble mean and perturbations
        DO iv = 1,nvar
           CALL com_mean( nens,xaens(ix,:,iv,it),xfmean(ix,iv,it) )
           xfpert(ix,:,iv,it) = xaens(ix,:,iv,it) - xfmean(ix,iv,it)
        END DO

        !Compute the new analysis mean.
        DO iv=1,nvar
          xamean  = xfmean(ix,iv,it)
          DO ie = 1 ,nens
             xamean = xamean + w_pf(ix,ie,it) * xfpert(ix,ie,iv,it)
          ENDDO
          xapert = 0.0d0
          DO ie=1,nens
            DO ke=1,nens
              xapert(ie) =  xapert(ie) + xfpert(ix,ke,iv,it) * wf(ke,ie)
            ENDDO
          ENDDO
          xaens(ix,:,iv,it) = xamean + xapert
        END DO
     ENDIF
   ENDIF

  END DO
!$OMP END PARALLEL DO


END DO  

END SUBROUTINE da_gmdr_localh




!=======================================================================
!  L-ETPF DA for the 1D model
!=======================================================================
SUBROUTINE da_letpf(nx,nt,no,nens,nvar,xloc,tloc,xfens,xaens,obs,obsloc,ofens,Rdiag,loc_scale,multinf,rejuv_param,wa,temp_factor)

IMPLICIT NONE
INTEGER,INTENT(IN)         :: nx , nt , nvar             !State dimensions, space, time and variables
INTEGER,INTENT(IN)         :: no                         !Number of observations 
INTEGER,INTENT(IN)         :: nens                       !Number of ensemble members
REAL(r_size),INTENT(IN)    :: xloc(nx)                   !Location of state grid points (space)
REAL(r_size),INTENT(IN)    :: tloc(nt)                   !Location of state grid points (time)
REAL(r_size),INTENT(IN)    :: obsloc(no,2)               !Location of obs (space , time)
REAL(r_size),INTENT(IN)    :: xfens(nx,nens,nvar,nt)     !Forecast state ensemble 
REAL(r_size),INTENT(IN)    :: multinf                    !Multiplicative inflation 
REAL(r_size),INTENT(OUT)   :: xaens(nx,nens,nvar,nt)     !Analysis state ensemble
REAL(r_size),INTENT(OUT)   :: wa(nx,nens)                !Posterior weigths

REAL(r_size)               :: multinf_loc                !Local multinf

REAL(r_size)               :: xfpert(nx,nens,nvar,nt)    !State and parameter forecast perturbations
REAL(r_size)               :: xfmean(nx,nvar,nt)         !State and parameter ensemble mean (forecast)
REAL(r_size)               :: random_rejuv_matrix(nens,nens) !Random coefficients for particle rejuvenation.
REAL(r_size)               :: m(nens,nens)                   !Distance matrix

REAL(r_size),INTENT(IN)    :: ofens(no,nens)
REAL(r_size)               :: dens(no,nens)              !Ensemble mean in observation space and innovation
REAL(r_size),INTENT(IN)    :: obs(no)                    !Observations 
REAL(r_size),INTENT(IN)    :: Rdiag(no)                  !Diagonal of observation error covariance matrix.
REAL(r_size)               :: d(no)                                                !Observation departure


REAL(r_size)               :: dens_loc(no,nens) , Rdiag_loc(no)                   !Localized ensemble in obs space and diag of R
REAL(r_size)               :: Rwf_loc(no)                                          !Localization weigths.
REAL(r_size)               :: d_loc(no)                                            !Localized observation departure.
INTEGER                    :: no_loc                                               !Number of observations in the local domain.
REAL(r_size),INTENT(IN)    :: loc_scale(2)                                         !Localization scales (space,time)
REAL(r_size),INTENT(IN)    :: rejuv_param
REAL(r_size),INTENT(IN)    :: temp_factor(nx,nt)         !Tempering factor ( R -> R*temp_factor)

REAL(r_size)               :: wamean(nens)                                         !Mean analysis weights
REAL(r_size)               :: grid_loc(2)
REAL(r_size)               :: work1d(nx)

REAL(r_size)               :: W(nens,nens) !LETKF transformation matrix
REAL(r_size)               :: infpert(nx,nens,nvar,nt) , tmp_mean

INTEGER                    :: ix,ie,je,ke,it,iv,io
REAL(r_size)               :: dx , tmp

!Initialization
d=0.0d0
wa=0.0d0

dx=xloc(2)-xloc(1)    !Assuming regular grid


!Initialization
xfmean=0.0d0
xfpert=0.0d0
W=0.0d0


!Compute mean departure and HXf

DO io = 1,no
    dens(io,:) = obs(io) - ofens(io,:) 
    CALL com_mean( nens , dens(io,:) , d(io) )
ENDDO

!Main assimilation loop.

DO it = 1,nt

!$OMP PARALLEL DO SCHEDULE(DYNAMIC) PRIVATE(grid_loc,no_loc,dens_loc,d_loc   &
!$OMP &          ,Rdiag_loc,Rwf_loc,W,ie,je,m,multinf_loc)

  DO ix = 1,nx

   !Localize observations
   grid_loc(1)=xloc(ix)  !Set the grid point location in space
   grid_loc(2)=tloc(it)  !Set the grid point location in time

   CALL r_localization(no,no_loc,nens,dens,d,Rdiag,dens_loc,     &
                     d_loc,Rdiag_loc,Rwf_loc,grid_loc,xloc(1),xloc(nx),dx,obsloc,loc_scale)

   !Aplly local tempering factor to the error covariance matrix.
   Rdiag_loc(1:no_loc) = Rdiag_loc(1:no_loc) * temp_factor(ix,it)
   multinf_loc = multinf ** ( 1.0d0 / temp_factor(ix,it) )

   IF( no_loc > 0 )THEN   
    !We have observations for this grid point. Let's compute the analysis.
    !Compute analysis weights
    CALL get_distance_matrix( nens , 1 , nvar , 1 , xfens(ix,:,:,it) , m )
 
    CALL letpf_core( nens,1,no_loc,dens_loc(1:no_loc,:),m,Rdiag_loc(1:no_loc)   &
                    ,wa(ix,:),W, multinf_loc )

    !Compute the updated ensemble mean, std and mode.
    DO ie=1,nens
       xaens(ix,ie,:,it) = 0.0d0
       DO je=1,nens
         xaens(ix,ie,:,it) =  xaens(ix,ie,:,it) + xfens(ix,je,:,it) * W(je,ie)
       ENDDO
    ENDDO
   
   ELSE  

    !We don't have observations for this grid point. We can do nothing :(
    xaens(ix,:,:,it) = xfens(ix,:,:,it) 

   ENDIF

  END DO
  !$OMP END PARALLEL DO

END DO  

IF ( rejuv_param > 0.0d0 ) THEN

   !Compute forecast ensemble mean and perturbations.
   DO it = 1,nt
    DO ix = 1,nx
     DO iv = 1,nvar
      CALL com_mean( nens,xfens(ix,:,iv,it),xfmean(ix,iv,it) )
      xfpert(ix,:,iv,it) = xfens(ix,:,iv,it) - xfmean(ix,iv,it)
     END DO
    END DO
   END DO

   infpert = 0.0d0
   !Perform particle rejuvenation on the global ensemble.a
   DO ie = 1,nens
      CALL com_randn( nens , random_rejuv_matrix(ie,:) , 10 )
   ENDDO
   random_rejuv_matrix = rejuv_param * random_rejuv_matrix / ( REAL(nens,r_size) ** 0.5 )

   DO ie = 1,nens
      DO je = 1,nens
         infpert(:,ie,:,:) = infpert(:,ie,:,:) +  &
                               xfpert(:,je,:,:)*random_rejuv_matrix(ie,je)
      ENDDO
   ENDDO       
   !Recenter the perturbations around the ETPF mean.
   DO ix = 1,nx
     DO iv = 1,nvar
       DO it = 1,nt
         tmp_mean = SUM( infpert(ix,:,iv,it) )/( REAL(nens,r_size) )
         infpert(ix,:,iv,it) = infpert(ix,:,iv,it) - tmp_mean
       ENDDO
     ENDDO
   ENDDO

   !Add the rejuvenation perturbations to the ensemble.
   xaens = xaens + infpert
ENDIF

END SUBROUTINE da_letpf


!=======================================================================
!  ADDAPTIVE TEMPERING STEP COMPUTATION
!=======================================================================
!This routine computes an optimal estimation of the tempering steps.
!The output consists of coeficients of a linear function defining dt in pseudo time
!for each iteration.
!dt = a + b ( i ) where i is the iteration number from 1 to NIter
!a and b are computed independently fore each model grid point because they depend on the 
!local ration between the observation error and the ensemble spread.

SUBROUTINE da_pseudo_time_step(nx,nt,no,nens,nvar,xloc,tloc,obsloc,ofens,Rdiag,loc_scale,NIter,a_coef,b_coef)

IMPLICIT NONE
INTEGER,INTENT(IN)         :: nx , nt , nvar             !State dimensions, space, time and variables
INTEGER,INTENT(IN)         :: NIter                      !Number of iterations
INTEGER,INTENT(IN)         :: no                         !Number of observations 
INTEGER,INTENT(IN)         :: nens                       !Number of ensemble members
REAL(r_size),INTENT(IN)    :: loc_scale(2)               !Localization scales in space and time.
REAL(r_size),INTENT(IN)    :: xloc(nx)                   !Location of state grid points (space)
REAL(r_size),INTENT(IN)    :: tloc(nt)                   !Location of state grid points (time)
REAL(r_size),INTENT(IN)    :: obsloc(no,2)               !Location of obs (space , time)
REAL(r_size),INTENT(IN)    :: ofens(no,nens)             !Ensemble in observation space
REAL(r_size),INTENT(IN)    :: Rdiag(no)                  !Diagonal of observation error covariance matrix.
REAL(r_size),INTENT(OUT)   :: a_coef(nx,nt) , b_coef(nx,nt)    !Coeficients to compute the size of the tempering iteration.
REAL(r_size)               :: ofmean(no) , ofpert(no,nens)                         !Ensemble mean in obs space, ensemble perturbations in obs space (forecast)
REAL(r_size)               :: HPHtdiag(no) , HPHtdiag_loc(no)                      !Ensemble spread in observation space.
REAL(r_size)               :: ofpert_loc(no,nens) , Rdiag_loc(no)                  !Localized ensemble in obs space and diag of R
REAL(r_size)               :: Rwf_loc(no)                                          !Localization weigths.
REAL(r_size)               :: d_loc(no)                                            !Localized observation departure.
INTEGER                    :: no_loc                                               !Number of observations in the local domain.
REAL(r_size)               :: grid_loc(2)
REAL(r_size)               :: target_inc , tmp_coef , dt_1 , tmp

INTEGER                    :: ix,ie,ke,it,iv,io
REAL(r_size)               :: dx

!Initialization
ofmean=0.0d0
ofpert=0.0d0
ofpert_loc=0.0d0
target_inc = 1.0d0 / REAL( NIter , r_size )

dx=xloc(2)-xloc(1)    !Assuming regular grid

!Compute mean departure and HXf

DO io = 1,no
    CALL com_mean( nens , ofens(io,:) , ofmean(io) )
    ofpert(io,:)=ofens(io,:) - ofmean(io)
    CALL com_var( nens , ofpert(io,:) , HPHtdiag(io) )
ENDDO

!These values will be asigned for the no local observation case.
a_coef = 1.0d0/NIter
b_coef = 0.0d0

tmp = 0.5d0*REAL(NIter,r_size)*(REAL(NIter,r_size)+1.0d0) -REAL(NIter,r_size)

IF ( NIter > 1 ) THEN !Only perform this computation if NIter is greather than one.

   DO it = 1,nt


  !$OMP PARALLEL DO SCHEDULE(DYNAMIC) PRIVATE(grid_loc,no_loc,ofpert_loc,d_loc   &
  !$OMP &          ,Rdiag_loc,Rwf_loc,HPHtdiag_loc,tmp_coef,dt_1)
     DO ix = 1,nx

      !Localize observations
      grid_loc(1)=xloc(ix)  !Set the grid point location in space
      grid_loc(2)=tloc(it)  !Set the grid point location in time

      CALL r_localization(no,no_loc,nens,ofpert,HPHtdiag,Rdiag,ofpert_loc,     &
                HPHtdiag_loc,Rdiag_loc,Rwf_loc,grid_loc,xloc(1),xloc(nx),dx,obsloc,loc_scale)

      IF( no_loc > 0 )THEN 
   !
       tmp_coef=( ( 1.0d0 - target_inc ) / target_inc ) 
       tmp_coef=tmp_coef*SUM(HPHtdiag_loc(1:no_loc))/SUM(Rdiag_loc(1:no_loc)*Rwf_loc(1:no_loc))
       tmp_coef= tmp_coef + 1.0d0 / ( target_inc ) 
       
       dt_1 = 1.0d0 / tmp_coef
       !write(*,*)dt_1 
       b_coef(ix,it) = (1.0d0 - REAL(NIter,r_size)*dt_1)/tmp
       a_coef(ix,it) = dt_1 - b_coef(ix,it)
      ENDIF

     END DO  
   !$OMP END PARALLEL DO
   END DO
ENDIF  

END SUBROUTINE da_pseudo_time_step



!=======================================================================
!  R-LOCALIZATION for 1D models
!=======================================================================

SUBROUTINE  r_localization(no,no_loc,nens,ofpert,d,Rdiag,ofpert_loc,d_loc,Rdiag_loc    &
            &              ,Rwf_loc,grid_loc,x_min,x_max,dx,obsloc,loc_scale)


IMPLICIT NONE
INTEGER,INTENT(IN)         :: no , nens                  !Number of observations , number of ensemble members
REAL(r_size),INTENT(IN)    :: obsloc(no,2)               !Location of obs (space,time)
REAL(r_size),INTENT(IN)    :: grid_loc(2)                !Current grid point location (space,time)
REAL(r_size),INTENT(IN)    :: x_min , x_max              !Space grid limits
REAL(r_size),INTENT(IN)    :: dx                         !Grid resolution
REAL(r_size),INTENT(IN)    :: ofpert(no,nens)            !Ensemble in observation space
REAL(r_size),INTENT(IN)    :: Rdiag(no)                  !Diagonal of observation error covariance matrix.
REAL(r_size),INTENT(IN)    :: d(no)                      !Observation departure
REAL(r_size),INTENT(IN)    :: loc_scale(2)               !Localization scale (space,time) , note that negative values
                                                         !will remove localization for that dimension.

INTEGER,INTENT(OUT)        :: no_loc                     !Number of observations in the local domain


REAL(r_size),INTENT(OUT)   :: ofpert_loc(no,nens) , Rdiag_loc(no)   !Localized ensemble in obs space and diag of R
REAL(r_size),INTENT(OUT)   :: Rwf_loc(no)                           !Localization weigthing function.
REAL(r_size),INTENT(OUT)   :: d_loc(no)                             !Localized observation departure.


INTEGER                    :: io
INTEGER                    :: if_loc(2)                        !1- means localize, 0- means no localization
REAL(r_size)               :: distance(2)                      !Distance between the current grid point and the observation (space,time)
REAL(r_size)               :: distance_tr(2)                   !Ignore observations farther than this threshold (space,time)
REAL(r_size)               :: floc_scale(2)                    !Final loc scale that will be used

ofpert_loc=0.0d0
d_loc     =0.0d0
Rdiag_loc =0.0d0
Rwf_loc   =1.0d0
no_loc    =0
floc_scale = 1.0d0

!If space localization will be applied.
if( loc_scale(1) <= 0 )then
    !No localization in space.
    if_loc(1) = 0
    floc_scale(1)=1.0d0
else 
    if_loc(1) = 1
    floc_scale(1)=loc_scale(1)
endif
!If temporal localization will be applied.
if( loc_scale(2) <= 0 )then
    !No localization in time.
    if_loc(2) = 0
    floc_scale(2)=1.0d0
else 
    if_loc(2) = 1
    floc_scale(2)=loc_scale(2)
endif
!NOTE: To remove both space and time localizations is better to use the da_etkf routine.

!Compute distance threshold (space , time)
distance_tr = floc_scale * SQRT(10.0d0/3.0d0) * 2.0d0


DO io = 1,no

   !Compute distance in space
   CALL distance_x( grid_loc(1) , obsloc(io,1) , x_min , x_max , dx , distance(1) )

   !write(*,*) grid_loc(1) , obsloc(io,1) , distance(1) 

   !Compute distance in time 
   distance(2)=abs( grid_loc(2) - obsloc(io,2) )

   !distance_tr check
   IF(  distance(1)*if_loc(1) < distance_tr(1) .and. distance(2)*if_loc(2) < distance_tr(2) )THEN

     !This observation will be assimilated!
     no_loc = no_loc + 1

     ofpert_loc(no_loc,:) =ofpert(io,:)
     d_loc(no_loc)        =d(io)
     !Apply localization to the R matrix
     Rwf_loc(no_loc)      =  exp( -0.5d0 * (( REAL( if_loc(1) , r_size ) * distance(1)/floc_scale(1))**2 + &
                                           ( REAL( if_loc(2) , r_size ) * distance(2)/floc_scale(2))**2))
     Rdiag_loc(no_loc)    =  Rdiag(io) / Rwf_loc( no_loc )


   ENDIF

ENDDO

END SUBROUTINE r_localization

!=======================================================================
!  R-LOCALIZATION for 1D models
!  This version is to be used with the Gaussian Mixture with local H approach
!  There is only a minor difference with respect to the previous version and is
!  the dimension of ofpert, ofpert_loc, d and d_loc
!=======================================================================

SUBROUTINE  r_localization_localh(no,no_loc,nens,ofpert,d,Rdiag,ofpert_loc,d_loc,Rdiag_loc    &
            &              ,Rwf_loc,grid_loc,x_min,x_max,dx,obsloc,loc_scale)


IMPLICIT NONE
INTEGER,INTENT(IN)         :: no , nens                  !Number of observations , number of ensemble members
REAL(r_size),INTENT(IN)    :: obsloc(no,2)               !Location of obs (space,time)
REAL(r_size),INTENT(IN)    :: grid_loc(2)                !Current grid point location (space,time)
REAL(r_size),INTENT(IN)    :: x_min , x_max              !Space grid limits
REAL(r_size),INTENT(IN)    :: dx                         !Grid resolution
REAL(r_size),INTENT(IN)    :: ofpert(no,nens,nens)       !Ensemble in observation space (nens ensembles centered around each ensemble member)
REAL(r_size),INTENT(IN)    :: Rdiag(no)                  !Diagonal of observation error covariance matrix.
REAL(r_size),INTENT(IN)    :: d(no,nens)                      !Observation departure
REAL(r_size),INTENT(IN)    :: loc_scale(2)               !Localization scale (space,time) , note that negative values
                                                         !will remove localization for that dimension.
INTEGER,INTENT(OUT)        :: no_loc                     !Number of observations in the local domain
REAL(r_size),INTENT(OUT)   :: ofpert_loc(no,nens,nens) , Rdiag_loc(no)   !Localized ensemble in obs space and diag of R
REAL(r_size),INTENT(OUT)   :: Rwf_loc(no)                           !Localization weigthing function.
REAL(r_size),INTENT(OUT)   :: d_loc(no,nens)                             !Localized observation departure.
INTEGER                    :: io
INTEGER                    :: if_loc(2)                        !1- means localize, 0- means no localization
REAL(r_size)               :: distance(2)                      !Distance between the current grid point and the observation (space,time)
REAL(r_size)               :: distance_tr(2)                   !Ignore observations farther than this threshold (space,time)
REAL(r_size)               :: floc_scale(2)                    !Final loc scale that will be used

ofpert_loc=0.0d0
d_loc     =0.0d0
Rdiag_loc =0.0d0
Rwf_loc   =1.0d0
no_loc    =0
floc_scale = 1.0d0

!If space localization will be applied.
if( loc_scale(1) <= 0 )then
    !No localization in space.
    if_loc(1) = 0
    floc_scale(1)=1.0d0
else 
    if_loc(1) = 1
    floc_scale(1)=loc_scale(1)
endif
!If temporal localization will be applied.
if( loc_scale(2) <= 0 )then
    !No localization in time.
    if_loc(2) = 0
    floc_scale(2)=1.0d0
else 
    if_loc(2) = 1
    floc_scale(2)=loc_scale(2)
endif
!NOTE: To remove both space and time localizations is better to use the da_etkf routine.
!Compute distance threshold (space , time)
distance_tr = floc_scale * SQRT(10.0d0/3.0d0) * 2.0d0


DO io = 1,no
   !Compute distance in space
   CALL distance_x( grid_loc(1) , obsloc(io,1) , x_min , x_max , dx , distance(1) )
   !write(*,*) grid_loc(1) , obsloc(io,1) , distance(1) 
   !Compute distance in time 
   distance(2)=abs( grid_loc(2) - obsloc(io,2) )
   !distance_tr check
   IF(  distance(1)*if_loc(1) < distance_tr(1) .and. distance(2)*if_loc(2) < distance_tr(2) )THEN
     !This observation will be assimilated!
     no_loc = no_loc + 1
     ofpert_loc(no_loc,:,:) =ofpert(io,:,:)
     d_loc(no_loc,:)        =d(io,:)
     !Apply localization to the R matrix
     Rwf_loc(no_loc)      =  exp( -0.5d0 * (( REAL( if_loc(1) , r_size ) * distance(1)/floc_scale(1))**2 + &
                                           ( REAL( if_loc(2) , r_size ) * distance(2)/floc_scale(2))**2))
     Rdiag_loc(no_loc)    =  Rdiag(io) / Rwf_loc( no_loc )
   ENDIF
ENDDO

END SUBROUTINE r_localization_localh




!Compute distance between two grid points assuming cyclic boundary conditions.
SUBROUTINE distance_x( x1 , x2 , x_min , x_max , dx , distance)
IMPLICIT NONE
REAL(r_size), INTENT(IN)  :: x1,x2        !Possition
REAL(r_size), INTENT(IN)  :: dx           !Grid resolution
REAL(r_size), INTENT(IN)  :: x_min,x_max  !Grid limits
REAL(r_size), INTENT(OUT) :: distance     !Distance between x1 and x2


  distance=abs(x2 - x1) 

  if( distance > ( x_max - x1 ) + ( x2 - x_min ) + dx )then

    distance = ( x_max - x1 ) + ( x2 - x_min ) + dx

  endif

  if( distance > ( x_max - x2 ) + ( x1 - x_min ) + dx )then

    distance = ( x_max - x2 ) + ( x1 - x_min ) + dx 

  endif

END SUBROUTINE distance_x

!=======================================================================
!  Get ensemble mean and perturbations
!=======================================================================

SUBROUTINE da_mean_and_pert(nx,nt,nvar,nens,xfens,xfmean,xfpert)

IMPLICIT NONE
INTEGER,INTENT(IN)         :: nt , nx , nvar             !Number of observations , number of times , number of state variables ,number of variables
INTEGER,INTENT(IN)         :: nens                       !Number of ensemble members
REAL(r_size),INTENT(IN)    :: xfens(nx,nens,nvar,nt)     !Forecast state ensemble  
REAL(r_size),INTENT(OUT)   :: xfpert(nx,nens,nvar,nt)    !State and parameter forecast perturbations
REAL(r_size),INTENT(OUT)   :: xfmean(nx,nvar,nt)         !State and parameter ensemble mean (forecast)
INTEGER                    :: iv,ix,it

!Initialization
xfmean=0.0
xfpert=0.0
!Compute forecast ensemble mean and perturbations.
DO it = 1,nt
 DO ix = 1,nx
  DO iv = 1,nvar
   CALL com_mean( nens,xfens(ix,:,iv,it),xfmean(ix,iv,it) )
   xfpert(ix,:,iv,it) = xfens(ix,:,iv,it) - xfmean(ix,iv,it) 
  END DO
 END DO
END DO

END SUBROUTINE da_mean_and_pert


!=======================================================================
!  Get ensemble mean and perturbations
!=======================================================================

SUBROUTINE da_particle_rejuvenation(nx,nt,nvar,nens,xaens,xfens,xaens_rejuv,  &
      &                             xpert_rejuv,rejuv_param,temp_factor)

IMPLICIT NONE
INTEGER,INTENT(IN)         :: nt , nx , nvar             !Number of observations , number of times , number of state variables ,number of variables
INTEGER,INTENT(IN)         :: nens                       !Number of ensemble members
REAL(r_size),INTENT(IN)    :: xaens(nx,nens,nvar,nt)     !Analysis state ensemble
REAL(r_size),INTENT(IN)    :: xfens(nx,nens,nvar,nt)     !Forecast state ensemble
REAL(r_size),INTENT(IN)    :: temp_factor(nx,nt)         !Tempering factor
REAL(r_size),INTENT(IN)    :: rejuv_param 
REAL(r_size),INTENT(OUT)   :: xpert_rejuv(nx,nens,nvar,nt)    !Ensemble after particle rejuvenation.
REAL(r_size),INTENT(OUT)   :: xaens_rejuv(nx,nens,nvar,nt)    !Analysis ensemble with rejuvenation
REAL(r_size)               :: xfpert(nx,nens,nvar,nt),xfmean(nx,nvar,nt)
INTEGER                    :: iv,ix,it,ie,je
REAL(r_size)               :: random_rejuv_matrix(nens,nens) , tmp_mean

xpert_rejuv = 0.0d0
IF ( rejuv_param > 0.0d0 ) THEN

   CALL da_mean_and_pert(nx,nt,nvar,nens,xfens,xfmean,xfpert)

   !Perform particle rejuvenation on the global ensemble.a
   DO ie = 1,nens
      CALL com_randn( nens , random_rejuv_matrix(ie,:) , 10 )
   ENDDO
   random_rejuv_matrix = rejuv_param * random_rejuv_matrix / ( REAL(nens,r_size) ** 0.5 )

   DO ix = 1,nx
    DO it = 1,nt
     DO ie = 1,nens
      DO je = 1,nens
         xpert_rejuv(ix,ie,:,it) = xpert_rejuv(ix,ie,:,it) +  &
                      xfpert(ix,je,:,it)*random_rejuv_matrix(je,ie) * temp_factor(ix,it)
      ENDDO
     ENDDO
    ENDDO
   ENDDO
   !Recenter the perturbations around the original ensemble mean.
   DO ix = 1,nx
     DO iv = 1,nvar
       DO it = 1,nt
         tmp_mean = SUM( xpert_rejuv(ix,:,iv,it) )/( REAL(nens,r_size) )
         xpert_rejuv(ix,:,iv,it) = xpert_rejuv(ix,:,iv,it) - tmp_mean 
       ENDDO
     ENDDO
   ENDDO

ENDIF

xaens_rejuv = xaens + xpert_rejuv

END SUBROUTINE da_particle_rejuvenation


END MODULE common_da_tools






