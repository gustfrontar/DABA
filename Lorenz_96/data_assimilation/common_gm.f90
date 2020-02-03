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
SUBROUTINE letkf_gm_core(ne,nobsl,hdxb,rdiag,dep,parm_infl,trans,minfl,beta_coef)
  IMPLICIT NONE
  INTEGER,INTENT(IN) :: ne                      !GYL
  INTEGER,INTENT(IN) :: nobsl
  REAL(r_size),INTENT(IN) :: hdxb(1:nobsl,1:ne)
  REAL(r_size),INTENT(IN) :: rdiag(1:nobsl)
  REAL(r_size),INTENT(IN) :: dep(1:nobsl)
  REAL(r_size),INTENT(INOUT) :: parm_infl
  REAL(r_size),INTENT(OUT) :: trans(ne,ne)
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
  pa = MATMUL( work1 , TRANSPOSE( eivec ) )
  !CALL dgemm('n','t',ne,ne,ne,1.0d0,work1,ne,eivec,&
  !  & ne,0.0d0,pa,ne)
!-----------------------------------------------------------------------
!  Pa hdxb_rinv^T
!-----------------------------------------------------------------------
  work2 = MATMUL( pa , TRANSPOSE( hdxb_rinv ) )
  !CALL dgemm('n','t',ne,nobsl,ne,1.0d0,pa,ne,hdxb_rinv,&
  !  & nobsl,0.0d0,work2,ne)
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

  !DO k=1,ne
  !  DO i=1,ne
  !    trans(i,k) = 0.0d0
  !    DO j=1,nobsl
  !      trans(i,k) = trans(i,k) + work2(i,j) * ( dep(j) - hdxb(j,k) )
  !    END DO
  !  END DO
  !ENDDO


!The transformation matrix contains the weigths to shift each particle according
!to the LETKF update. 


  RETURN
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

SUBROUTINE pf_weigth_core(ne,nobsl,hdxb,dep,rdiag,beta_coef,gamma_coef,wa)
  IMPLICIT NONE
  INTEGER,INTENT(IN) :: ne                     
  INTEGER,INTENT(IN) :: nobsl
  REAL(r_size),INTENT(IN) :: hdxb(nobsl,ne)
  REAL(r_size),INTENT(IN) :: dep(nobsl)
  REAL(r_size),INTENT(IN) :: rdiag(nobsl)
  REAL(r_size),INTENT(IN) :: beta_coef , gamma_coef
  REAL(r_size),INTENT(OUT) :: wa(ne)    ! 
  REAL(r_size)  :: wu                   !Uniform weigth (1/ne)
  REAL(r_size)  :: log_w_sum
  REAL(r_size)  :: work1(nobsl,nobsl)
  REAL(r_size)  :: mem_departure(nobsl,1)
  REAL(r_size) :: eivec(nobsl,nobsl)
  REAL(r_size) :: eival(nobsl)
  REAL(r_size) :: work2(1,nobsl),work3(1,1)
  INTEGER :: i,j,k


  wa=1.0d0
  
  IF( beta_coef > 0.0d0 )THEN 
    !Compute ( HPHt + R )^-1
    
    !CALL dgemm('t','n',ne,ne,nobsl,1.0d0,hdxb,nobsl,hdxb,&
    !          & nobsl,0.0d0,work1,ne)
    work1 = MATMUL( hdxb , TRANSPOSE( hdxb ) )

    DO i = 1,ne
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
!    CALL dgemm('n','t',ne,ne,ne,1.0d0,work1,ne,eivec,&
!              & ne,0.0d0,work1,ne)

  ELSE 
    !-----------------------------------------------------------------------
    !Compute weigths without Gaussian Kernel
    !-----------------------------------------------------------------------

    work1=0.0d0
    DO i=1,ne
       work1(i,i)=1.0d0/rdiag(i)
    ENDDO
  ENDIF

  

  !Compute the logaritm of the weigths
  DO i=1,ne

   wa(i) = 0.0d0
   mem_departure(:,1) = dep - hdxb(:,i)

   work3 = MATMUL( MATMUL( TRANSPOSE( mem_departure ) , work1 ) , mem_departure )

!   CALL dgemm('n','n',nobsl,1,nobsl,1.0d0,work1,nobsl,mem_departure,&
!           & nobsl,0.0d0,work3,nobsl)

     wa(i) = -0.5d0 * work3(1,1)

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
    wa(i) = gamma_coef * wa(i) + (1.0d0 - gamma_coef )*wu
  ENDDO
  

  RETURN
END SUBROUTINE pf_weigth_core


END MODULE common_gm
