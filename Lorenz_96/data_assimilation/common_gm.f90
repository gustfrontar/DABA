MODULE common_gm
!=======================================================================
!
! [PURPOSE:] Local Ensemble Transform Kalman Filtering (LETKF)
!            Model Independent Core Module
!
! [REFERENCES:]
!  [1] Ott et al., 2004: A local ensemble Kalman filter for atmospheric
!    data assimilation. Tellus, 56A, 415-428.
!  [2] Hunt et al., 2007: Efficient Data Assimilation for Spatiotemporal
!    Chaos: A Local Ensemble Transform Kalman Filter. Physica D, 230,
!    112-126.
!
! [HISTORY:]
!  01/21/2009 Takemasa Miyoshi  Created at U. of Maryland, College Park
!
!=======================================================================
!$USE OMP_LIB
  USE common_tools
  USE common_mtx
  IMPLICIT NONE
  PUBLIC
CONTAINS
!=======================================================================
!  Main Subroutine of GM-LETKF Core
!   INPUT
!     ne               : ensemble size                                           !GYL
!     nobsl            : total number of observation assimilated at the point
!     hdxb(nobsl,ne)   : obs operator times fcst ens perturbations
!     rdiag(nobsl)     : observation error variance
!     rloc(nobsl)      : localization weightning function
!     dep(nobsl)       : observation departure (yo-Hxb)
!     parm_infl        : covariance inflation parameter
!     minfl            : (optional) minimum covariance inflation parameter       !GYL
!     beta_coef        : parameter to control the width of the Gaussian kernel 
!   OUTPUT
!     parm_infl        : updated covariance inflation parameter
!     trans(ne,ne)     : transformation matrix (each column of this matrix
!                        contains the weights to shift the particles towards the observations
!                        following a traditional LETKF update with ensemble covariance matrix scaled
!                        by a factor beta_coef.
!     trans_pert(ne,ne): This is the transformation matrix for the perturbations. 
!                        Depending on the implementation of the GM method this can be useful or not.
! 

!=======================================================================
SUBROUTINE letkf_gm_core(ne,nobsl,hdxb,rdiag,dep,parm_infl,trans,trans_pert,minfl,beta_coef)
  IMPLICIT NONE
  INTEGER,INTENT(IN) :: ne                      !GYL
  INTEGER,INTENT(IN) :: nobsl
  REAL(r_size),INTENT(IN) :: hdxb(1:nobsl,1:ne)
  REAL(r_size),INTENT(IN) :: rdiag(1:nobsl)
  REAL(r_size),INTENT(IN) :: dep(1:nobsl)
  REAL(r_size),INTENT(INOUT) :: parm_infl
  REAL(r_size),INTENT(OUT) :: trans(ne,ne)
  REAL(r_size),INTENT(OUT) :: trans_pert(ne,ne) 
  REAL(r_size),INTENT(IN)  :: minfl     !GYL
  REAL(r_size),INTENT(IN)  :: beta_coef
  REAL(r_size) :: hdxb_rinv(nobsl,ne)
  REAL(r_size) :: hdxb_tmp(nobsl,ne)
  REAL(r_size) :: eivec(ne,ne)
  REAL(r_size) :: eival(ne)
  REAL(r_size) :: pa(ne,ne)
  REAL(r_size) :: work1(ne,ne)
  REAL(r_size) :: work2(ne,nobsl)
  REAL(r_size) :: work3(ne)
  REAL(r_size) :: work4(nobsl,1)
  REAL(r_size) :: mem_departure(nobsl,ne)
  REAL(r_size) :: rho
  INTEGER :: i,j,k
  trans = 1.0d0
!-----------------------------------------------------------------------
!  hdxb Rinv
!-----------------------------------------------------------------------
    DO j=1,ne                                     !GYL
      DO i=1,nobsl                                !GYL
        hdxb_rinv(i,j) = hdxb(i,j) / rdiag(i)     !GYL
      END DO                                      !GYL
    END DO                                        !GYL
!-----------------------------------------------------------------------
!  hdxb^T Rinv hdxb
!-----------------------------------------------------------------------
  !CALL dgemm('t','n',ne,ne,nobsl,1.0d0,hdxb_rinv,nobsl,hdxb_tmp,&
  !  & nobsl,0.0d0,work1,ne)
  work1 = MATMUL( TRANSPOSE( hdxb_rinv ) , hdxb )
!-----------------------------------------------------------------------
!  hdxb^T Rinv hdxb + (m-1) I / rho (covariance inflation)
!-----------------------------------------------------------------------
    IF (minfl > 0.0d0 .AND. parm_infl < minfl) THEN   !GYL
      parm_infl = minfl                               !GYL
    END IF                                            !GYL
  rho = 1.0d0 / ( parm_infl * beta_coef ) 
  DO i=1,ne
    work1(i,i) = work1(i,i) + REAL(ne-1,r_size) * rho
  END DO
!-----------------------------------------------------------------------
!  Pa = [ hdxb^T Rinv hdxb + (m-1) I ]inv
!-----------------------------------------------------------------------
  CALL mtx_inv_eivec( ne , work1 , pa )
!-----------------------------------------------------------------------
!  Pa hdxb_rinv^T
!-----------------------------------------------------------------------
  work2 = MATMUL( pa , TRANSPOSE( hdxb_rinv ) )
!-----------------------------------------------------------------------
!  Pa hdxb_rinv^T (dep - hdxb(:,iens))
!  This step is performed for each ensemble member
!-----------------------------------------------------------------------
  DO i=1,nobsl
   DO j=1,ne
     mem_departure(i,j) = dep(i) - hdxb(i,j) 
   END DO
  END DO 
  trans = MATMUL( work2 , mem_departure )
!-----------------------------------------------------------------------
!  Wa = sqrt( (ne-1)pa)  !dep contains departures centered at each ensemble member
!  This step is performed for each ensemble member
!-----------------------------------------------------------------------
  CALL mtx_sqrt( ne , REAL(ne-1,r_size) * pa , trans_pert )

  RETURN
END SUBROUTINE letkf_gm_core

!=======================================================================
!  Main Subroutine of GM-LETKF with local h linearization Core 
!   INPUT
!     ne               : ensemble size                                           !GYL
!     nobsl            : total number of observation assimilated at the point
!     hdxb(nobsl,ne)   : obs operator times fcst ens perturbations
!     rdiag(nobsl)     : observation error variance
!     rloc(nobsl)      : localization weightning function
!     dep(nobsl)       : observation departure (yo-Hxb)
!     parm_infl        : covariance inflation parameter
!     minfl            : (optional) minimum covariance inflation parameter       !GYL
!     beta_coef        : parameter to control the width of the Gaussian kernel 
!   OUTPUT
!     parm_infl        : updated covariance inflation parameter
!     trans(ne,ne)     : transformation matrix (each column of this matrix
!                        contains the weights to shift the particles towards the observations
!                        following a traditional LETKF update with ensemble covariance matrix scaled
!                        by a factor beta_coef.

!=======================================================================
SUBROUTINE letkf_gm_localh_core(ne,nobsl,hdxb,rdiag,dep,parm_infl,trans,minfl,beta_coef)
  IMPLICIT NONE
  INTEGER,INTENT(IN) :: ne                      !GYL
  INTEGER,INTENT(IN) :: nobsl
  REAL(r_size),INTENT(IN) :: hdxb(1:nobsl,1:ne,1:ne) !We have one Y for each ensemble member
  REAL(r_size),INTENT(IN) :: rdiag(1:nobsl)
  REAL(r_size),INTENT(IN) :: dep(1:nobsl,1:ne)       !We have one dep for each ensemble member
  REAL(r_size),INTENT(INOUT) :: parm_infl
  REAL(r_size),INTENT(OUT) :: trans(ne,ne)           !We obtain one weight vector for each ensemble member
  REAL(r_size),INTENT(IN)  :: minfl     !GYL
  REAL(r_size),INTENT(IN)  :: beta_coef

  REAL(r_size) :: hdxb_rinv(nobsl,ne)
  REAL(r_size) :: hdxb_tmp(nobsl,ne)
  REAL(r_size) :: hdxb_local(nobsl,ne)
  REAL(r_size) :: eivec(ne,ne)
  REAL(r_size) :: eival(ne)
  REAL(r_size) :: pa(ne,ne)
  REAL(r_size) :: work1(ne,ne)
  REAL(r_size) :: work2(ne,nobsl)
  REAL(r_size) :: work3(ne)
  REAL(r_size) :: work4(nobsl,1)
  REAL(r_size) :: mem_departure(nobsl,ne)
  REAL(r_size) :: rho
  INTEGER :: i,j,k,ie

  trans = 1.0d0

DO ie = 1 , ne  !Loop over sub ensembles.
   hdxb_local = hdxb(:,:,ie) 
!-----------------------------------------------------------------------
!  hdxb Rinv
!-----------------------------------------------------------------------
   hdxb_rinv=0.0d0
   DO j=1,ne                                        !GYL
     DO i=1,nobsl                                   !GYL
       hdxb_rinv(i,j) = hdxb_local(i,j) / rdiag(i)  !GYL
     END DO                                         !GYL
   END DO                                           !GYL
!-----------------------------------------------------------------------
!  hdxb^T Rinv hdxb
!-----------------------------------------------------------------------
   work1 = MATMUL( TRANSPOSE( hdxb_rinv ) , hdxb_local )
!-----------------------------------------------------------------------
!  hdxb^T Rinv hdxb + (m-1) I / rho (covariance inflation)
!-----------------------------------------------------------------------
   IF (minfl > 0.0d0 .AND. parm_infl < minfl) THEN   !GYL
      parm_infl = minfl                               !GYL
   END IF                                            !GYL
   rho = 1.0d0 / ( parm_infl * beta_coef ) 
   DO i=1,ne
    work1(i,i) = work1(i,i) + REAL(ne-1,r_size) * rho
   END DO
!-----------------------------------------------------------------------
!  Pa = [ hdxb^T Rinv hdxb + (m-1) I ]inv
!-----------------------------------------------------------------------
   CALL mtx_inv_eivec( ne , work1 , pa )
!-----------------------------------------------------------------------
!  Pa hdxb_rinv^T
!-----------------------------------------------------------------------
   work2 = MATMUL( pa , TRANSPOSE( hdxb_rinv ) )
!-----------------------------------------------------------------------
!  Pa hdxb_rinv^T (dep)  !dep contains departures centered at each ensemble member
!  This step is performed for each ensemble member
!-----------------------------------------------------------------------
   trans(:,ie) = MATMUL( work2 , dep(:,ie) )
!-----------------------------------------------------------------------
!  Wa = sqrt( (ne-1)pa)  !dep contains departures centered at each ensemble member
!  This step is performed for each ensemble member
!-----------------------------------------------------------------------

!CALL mtx_sqrt( ne , REAL(ne-1,r_size) * pa , Wa )

END DO




  RETURN
END SUBROUTINE letkf_gm_localh_core

!=======================================================================
!  Main Subroutine for weight computation in the Gaussian Mixture PF
!   INPUT
!     ne               : ensemble size                                         
!     nobsl            : total number of observation assimilated at the point
!     dens(nobsl,ne)   : distance between each ensemble member and the observation
!     rdiag(nobsl)     : observation error variance
!     beta_coef        : Gaussian Kernel width parameter
!     gamma_coef       : weight nudging parameter
!     y                : ensemble in local observation space
!     d                : mean departure
!   OUTPUT
!     wa(ne)           : PF weights
!=======================================================================

SUBROUTINE pf_weight_core(ne,nobsl,hdxb,dep,rdiag,beta_coef,gamma_coef,wa)
  IMPLICIT NONE
  INTEGER,INTENT(IN) :: ne                     
  INTEGER,INTENT(IN) :: nobsl
  REAL(r_size),INTENT(IN) :: hdxb(nobsl,ne)
  REAL(r_size),INTENT(IN) :: dep(nobsl)
  REAL(r_size),INTENT(IN) :: rdiag(nobsl)
  REAL(r_size),INTENT(IN) :: beta_coef , gamma_coef
  REAL(r_size),INTENT(OUT) :: wa(ne)    ! 
  REAL(r_size)  :: wu                   !Uniform weight (1/ne)
  REAL(r_size)  :: log_w_sum
  REAL(r_size)  :: work1(nobsl,nobsl)
  REAL(r_size)  :: mem_departure(nobsl,1)
  REAL(r_size) :: eivec(nobsl,nobsl)
  REAL(r_size) :: eival(nobsl)
  REAL(r_size) :: work2(1,nobsl),work3(1,1)
  INTEGER :: i,j,k


  wa=0.0d0
  
  IF( beta_coef > 0.0d0 )THEN 
    !Compute ( HPHt + R )^-1
    
    work1 = MATMUL( hdxb , TRANSPOSE( hdxb )  )

    DO i = 1,nobsl
      work1(i,i) = work1(i,i) + rdiag(i)
    ENDDO

    CALL mtx_inv_eivec( nobsl , work1 , work1 )

  ELSE 
    !-----------------------------------------------------------------------
    !Compute weights without Gaussian Kernel
    !-----------------------------------------------------------------------

    work1=0.0d0
    DO i=1,nobsl
       work1(i,i)=1.0d0/rdiag(i)
    ENDDO
  ENDIF


  !Compute the logaritm of the weights
  DO i=1,ne
   mem_departure(:,1) = dep - hdxb(:,i)
   work3 = MATMUL( MATMUL( TRANSPOSE( mem_departure ) , work1 ) , mem_departure )
   wa(i) = -0.5d0 * work3(1,1)
  ENDDO

  !-----------------------------------------------------------------------
  !Normalize log of the weights (to avoid underflow issues)
  !-----------------------------------------------------------------------

  CALL log_sum_vec( ne , wa , log_w_sum )
  DO i=1,ne
     wa(i) = EXP( wa(i) - log_w_sum )
  ENDDO
  
  !-----------------------------------------------------------------------
  !Weigth nudging
  !-----------------------------------------------------------------------
  wu=1.0d0/REAL(ne,r_size)
  DO i = 1,ne
    wa(i) = gamma_coef * wa(i) + (1.0d0 - gamma_coef )*wu
  ENDDO
  

  RETURN
END SUBROUTINE pf_weight_core

!=======================================================================
!  Main Subroutine for weight computation in the Gaussian Mixture PF
!   INPUT
!     ne               : ensemble size                                         
!     nobsl            : total number of observation assimilated at the point
!     dens(nobsl,ne)   : distance between each ensemble member and the observation
!     rdiag(nobsl)     : observation error variance
!     beta_coef        : Gaussian Kernel width parameter
!     gamma_coef       : weight nudging parameter
!     y                : ensemble in local observation space
!     d                : mean departure
!   OUTPUT
!     wa(ne)           : PF weights
!=======================================================================

SUBROUTINE pf_weight_localh_core(ne,nobsl,hdxb,dep,rdiag,beta_coef,gamma_coef,wa)
  IMPLICIT NONE
  INTEGER,INTENT(IN) :: ne                     
  INTEGER,INTENT(IN) :: nobsl
  REAL(r_size),INTENT(IN) :: hdxb(nobsl,ne,ne)
  REAL(r_size),INTENT(IN) :: dep(nobsl,ne)
  REAL(r_size),INTENT(IN) :: rdiag(nobsl)
  REAL(r_size),INTENT(IN) :: beta_coef , gamma_coef
  REAL(r_size),INTENT(OUT) :: wa(ne)    ! 
  REAL(r_size)             :: hdxb_local(nobsl,ne)
  REAL(r_size)  :: wu                   !Uniform weight (1/ne)
  REAL(r_size)  :: log_w_sum
  REAL(r_size)  :: work1(nobsl,nobsl)
  REAL(r_size)  :: mem_departure(nobsl,1)
  REAL(r_size) :: eivec(nobsl,nobsl)
  REAL(r_size) :: eival(nobsl)
  REAL(r_size) :: work2(1,nobsl),work3(1,1)
  INTEGER :: i,j,k,ie

  wa=1.0d0
  
  DO ie = 1 , ne
    IF( beta_coef > 0.0d0 )THEN 
      !Compute ( HPHt + R )^-1
      hdxb_local = hdxb(:,:,ie)
      work1 = MATMUL( hdxb_local , TRANSPOSE( hdxb_local )  )
      DO i = 1,nobsl
        work1(i,i) = work1(i,i) + rdiag(i)
      ENDDO
      !-----------------------------------------------------------------------
      !  eigenvalues and eigenvectors of [ HPHt + R ]  
      !-----------------------------------------------------------------------
      i=ne
      CALL mtx_eigen(1,nobsl,work1,eival,eivec,i)
      !-----------------------------------------------------------------------
      !  [ HPHt ]^-1
      !-----------------------------------------------------------------------
      DO j=1,nobsl
        DO i=1,nobsl
          work1(i,j) = eivec(i,j) / eival(j)
        END DO
      END DO
      work1 = MATMUL( work1 , TRANSPOSE( eivec ) )
    ELSE 
      !-----------------------------------------------------------------------
      !Compute weights without Gaussian Kernel
      !-----------------------------------------------------------------------
      work1=0.0d0
      DO i=1,nobsl
        work1(i,i)=1.0d0/rdiag(i)
      ENDDO
    ENDIF
    !Compute the logaritm of the weights
    wa(ie) = 0.0d0
    mem_departure(:,1) = dep(:,ie)
    work3 = MATMUL( MATMUL( TRANSPOSE( mem_departure ) , work1 ) , mem_departure )
    wa(ie) = -0.5d0 * work3(1,1)
  END DO

  !-----------------------------------------------------------------------
  !Normalize log of the weights (to avoid underflow issues)
  !-----------------------------------------------------------------------

  CALL log_sum_vec( ne , wa , log_w_sum )
  DO ie=1,ne
     wa(ie) = EXP( wa(ie) - log_w_sum )
  ENDDO
  
  !-----------------------------------------------------------------------
  !Weigth nudging
  !-----------------------------------------------------------------------
  wu=1.0d0/REAL(ne,r_size)
  DO ie = 1,ne
    wa(ie) = gamma_coef * wa(ie) + (1.0d0 - gamma_coef )*wu
  ENDDO
  
  RETURN

END SUBROUTINE pf_weight_localh_core


SUBROUTINE netpf_w(ne,wa_in,w)
  IMPLICIT NONE
  INTEGER,INTENT(IN) :: ne               
  REAL(r_size),INTENT(IN)  :: wa_in(ne) 
  REAL(r_size),INTENT(OUT) :: w(ne,ne)
  REAL(r_size) :: wa(ne,1)
  REAL(r_size) :: work1(ne,ne)
  INTEGER :: i,j,k

  wa(:,1)=wa_in

!-----------------------------------------------------------------------
!   W - w * w^T 
!-----------------------------------------------------------------------

  work1 = -1.0d0 * MATMUL( wa , TRANSPOSE(wa) )
  DO i=1,ne
    work1(i,i) = wa_in(i) + work1(i,i) 
  END DO

!-----------------------------------------------------------------------
!  w = sqrt(m) * [ W - w * w^T ]^1/2
!-----------------------------------------------------------------------
  CALL mtx_sqrt(ne,work1,w)
  w = sqrt(REAL(ne,r_size)) * w


END SUBROUTINE netpf_w


END MODULE common_gm
