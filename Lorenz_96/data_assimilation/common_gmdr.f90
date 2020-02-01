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
!  Main Subroutine of LETKF Core
!   INPUT
!     ne               : ensemble size                                           !GYL
!     nobsl            : total number of observation assimilated at the point
!     hdxb(nobsl,ne)   : obs operator times fcst ens perturbations
!     rdiag(nobsl)     : observation error variance
!     rloc(nobsl)      : localization weigthning function
!     dep(nobsl)       : observation departure (yo-Hxb)
!     parm_infl        : covariance inflation parameter
!     minfl            : (optional) minimum covariance inflation parameter       !GYL
!     beta_coef        : parameter to control the width of the Gaussian kernel 
!   OUTPUT
!     parm_infl        : updated covariance inflation parameter
!     trans(ne,ne)     : transformation matrix (each column of this matrix
!                        contains the weigths to shift the particles towards the observations
!                        following a traditional LETKF update with ensemble covariance matrix scaled
!                        by a factor beta_coef.

!=======================================================================
SUBROUTINE letkf_gm_core(ne,nobsl,hdxb,rdiag,rloc,dep,parm_infl,trans,minfl)
  IMPLICIT NONE
  INTEGER,INTENT(IN) :: ne                      !GYL
  INTEGER,INTENT(IN) :: nobsl
  REAL(r_size),INTENT(IN) :: hdxb(1:nobsl,1:ne)
  REAL(r_size),INTENT(IN) :: rdiag(1:nobsl)
  REAL(r_size),INTENT(IN) :: rloc(1:nobsl)
  REAL(r_size),INTENT(IN) :: dep(1:nobsl)
  REAL(r_size),INTENT(INOUT) :: parm_infl
  REAL(r_size),INTENT(OUT) :: trans(ne,ne)
  REAL(r_size),INTENT(IN)  :: minfl     !GYL
  REAL(r_size),INTENT(IN)  :: beta_coef

  REAL(r_size) :: hdxb_rinv(nobsl,ne)
  REAL(r_size) :: eivec(ne,ne)
  REAL(r_size) :: eival(ne)
  REAL(r_size) :: pa(ne,ne)
  REAL(r_size) :: work1(ne,ne)
  REAL(r_size) :: work2(ne,nobsl)
  REAL(r_size) :: work3(ne)
  REAL(r_size) :: rho
  INTEGER :: i,j,k


  IF(nobsl == 0) THEN
    transm = 1.0d0            !GYL
    RETURN
  ELSE
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
  CALL dgemm('t','n',ne,ne,nobsl,1.0d0,hdxb_rinv,nobsl,hdxb(1:nobsl,:),&
    & nobsl,0.0d0,work1,ne)
!-----------------------------------------------------------------------
!  hdxb^T Rinv hdxb + (m-1) I / rho (covariance inflation)
!-----------------------------------------------------------------------
!  IF (PRESENT(minfl)) THEN                           !GYL
    IF (minfl > 0.0d0 .AND. parm_infl < minfl) THEN   !GYL
      parm_infl = minfl                               !GYL
    END IF                                            !GYL
!  END IF                                             !GYL
  rho = 1.0d0 / ( parm_infl * beta_coef )
  DO i=1,ne
    work1(i,i) = work1(i,i) + REAL(ne-1,r_size) * rho
  END DO
!-----------------------------------------------------------------------
!  eigenvalues and eigenvectors of [ hdxb^T Rinv hdxb + (m-1) I ]
!-----------------------------------------------------------------------
  i=ne
  CALL mtx_eigen(1,ne,work1,eival,eivec,i)

!-----------------------------------------------------------------------
!  Pa = [ hdxb^T Rinv hdxb + (m-1) I ]inv
!-----------------------------------------------------------------------
  DO j=1,ne
    DO i=1,ne
      work1(i,j) = eivec(i,j) / eival(j)
    END DO
  END DO
  CALL dgemm('n','t',ne,ne,ne,1.0d0,work1,ne,eivec,&
    & ne,0.0d0,pa,ne)
!-----------------------------------------------------------------------
!  Pa hdxb_rinv^T
!-----------------------------------------------------------------------
  CALL dgemm('n','t',ne,nobsl,ne,1.0d0,pa,ne,hdxb_rinv,&
    & nobsl,0.0d0,work2,ne)
!-----------------------------------------------------------------------
!  Pa hdxb_rinv^T dep - hdxb(:,iens)
!  This step is performed for each ensemble member
!-----------------------------------------------------------------------
  DO iens = 1,ne
    trans(:,iens) = MATMUL( work2 , dep - hdxb(:,iens) )
  END DO    

!The transformation matrix contains the weigths to shift each particle according
!to the LETKF update. 


  RETURN
  END IF
END SUBROUTINE letkf_gm_core

!=======================================================================
!  Main Subroutine for weigth computation in the Gaussian Mixture PF
!   INPUT
!     ne               : ensemble size                                         
!     nobsl            : total number of observation assimilated at the point
!     dens(nobsl,ne)   : distance between each ensemble member and the observation
!     rdiag(nobsl)     : observation error variance
!     beta_coef        : Gaussian Kernel width parameter
!     gamma_coef       : weigth nudging parameter
!     y                : ensemble in local observation space
!     d                : mean departure
!   OUTPUT
!     wa(ne)           : PF weigths
!=======================================================================

SUBROUTINE pf_weigth_core(ne,nobsl,y,d,rdiag,beta_coef,gamma_coef,wa)
  IMPLICIT NONE
  INTEGER,INTENT(IN) :: ne , ndim                      
  INTEGER,INTENT(IN) :: nobsl
  REAL(r_size),INTENT(IN) :: dens(1:nobsl,1:ne)
  REAL(r_size),INTENT(IN) :: m(1:ne,1:ne)   !Distance matrix
  REAL(r_size),INTENT(IN) :: rdiag(1:nobsl)
  REAL(r_size),INTENT(IN) :: beta_coef , gamma_coef
  REAL(r_size),INTENT(OUT) :: wa(ne)    ! 
  REAL(r_size)  :: wu                   !Uniform weigth (1/ne)
  REAL(r_size)  :: delta(ne,ne)         !Correction to get a second order exact filter.
  REAL(r_size)  :: log_w_sum
  REAL(r_size)  :: work1(1:nobsl,1:nobsl)
  REAL(r_size)  :: mem_departure(1:nobsl,1)
  INTEGER :: i,j,k
  
  IF( beta_coef > 0.0d0 )THEN 
    !Compute ( HPHt + R )^-1
    work1 = beta_coef * MATMUL( y , TRANSPOSE( y ) )
    DO i = 1,ne
      work1(i,i) = work1(i,i) + rdiag(i)
    ENDDO
  
    !-----------------------------------------------------------------------
    !  eigenvalues and eigenvectors of [ HPHt + R ]  
    !-----------------------------------------------------------------------
    i=ne
    CALL mtx_eigen(1,ne,hPht,eival,eivec,i)

    !-----------------------------------------------------------------------
    !  [ HPHt ]^-1
    !-----------------------------------------------------------------------
    DO j=1,ne
      DO i=1,ne
        work1(i,j) = eivec(i,j) / eival(j)
      END DO
    END DO
    CALL dgemm('n','t',ne,ne,ne,1.0d0,work1,ne,eivec,&
              & ne,0.0d0,work1,ne)

    !-----------------------------------------------------------------------
    !Compute the weights.
    !-----------------------------------------------------------------------
  ELSE 
    !Compute weigths without Gaussian Kernel
    work1=0.0d0
    DO i=1,ne
       work1(i,i)=1.0d0/rdiag(i)
    ENDDO
  ENDIF

  wa=0.0d0
  DO i=1,ne
     mem_departure(:,1) = d - y(:,i)
     wa(i)=wa(i) - 0.5* MATMUL( TRANSPOSE( mem_departure , MATMUL( work1 , mem_departure )  ) 
  ENDDO

  !-----------------------------------------------------------------------
  !Normalize log of the weigths (to avoid underflow issues)
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
    w(i) = gamma_parameter * w(i) + (1.0d0 - gamma_parameter )*wu
  ENDDO
  

  RETURN
END SUBROUTINE letpf_core


END MODULE common_gm
