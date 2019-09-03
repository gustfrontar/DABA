MODULE common_letkf
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
!   OUTPUT
!     parm_infl        : updated covariance inflation parameter
!     trans(ne,ne)     : transformation matrix
!     transm(ne)       : (optional) transformation matrix mean                   !GYL
!     pao(ne,ne)       : (optional) analysis covariance matrix in ensemble space !GYL
!=======================================================================
SUBROUTINE letkf_core(ne,nobsl,hdxb,rdiag,rloc,dep,parm_infl,trans,transm,pao,minfl)
  IMPLICIT NONE
  INTEGER,INTENT(IN) :: ne                      !GYL
  INTEGER,INTENT(IN) :: nobsl
  REAL(r_size),INTENT(IN) :: hdxb(1:nobsl,1:ne)
  REAL(r_size),INTENT(IN) :: rdiag(1:nobsl)
  REAL(r_size),INTENT(IN) :: rloc(1:nobsl)
  REAL(r_size),INTENT(IN) :: dep(1:nobsl)
  REAL(r_size),INTENT(INOUT) :: parm_infl
  REAL(r_size),INTENT(OUT) :: trans(ne,ne)
  REAL(r_size),INTENT(OUT) :: transm(ne)
  REAL(r_size),INTENT(OUT) :: pao(ne,ne)
  REAL(r_size),INTENT(IN)  :: minfl     !GYL

  REAL(r_size) :: hdxb_rinv(nobsl,ne)
  REAL(r_size) :: eivec(ne,ne)
  REAL(r_size) :: eival(ne)
  REAL(r_size) :: pa(ne,ne)
  REAL(r_size) :: work1(ne,ne)
  REAL(r_size) :: work2(ne,nobsl)
  REAL(r_size) :: work3(ne)
  REAL(r_size) :: rho
  REAL(r_size) :: parm(4),sigma_o,gain
  REAL(r_size),PARAMETER :: sigma_b = 0.04d0 !error stdev of parm_infl
  INTEGER :: i,j,k


  IF(nobsl == 0) THEN
    trans = 0.0d0
    DO i=1,ne
      trans(i,i) = SQRT(parm_infl)
    END DO
!    IF (PRESENT(transm)) THEN   !GYL
      transm = 0.0d0            !GYL
!    END IF                      !GYL
!    IF (PRESENT(pao)) THEN                        !GYL
      pao = 0.0d0                                 !GYL
      DO i=1,ne                                   !GYL
        pao(i,i) = parm_infl / REAL(ne-1,r_size)  !GYL
      END DO                                      !GYL
!    END IF                                        !GYL
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
!  DO j=1,ne
!    DO i=1,ne
!      work1(i,j) = hdxb_rinv(1,i) * hdxb(1,j)
!      DO k=2,nobsl
!        work1(i,j) = work1(i,j) + hdxb_rinv(k,i) * hdxb(k,j)
!      END DO
!    END DO
!  END DO
!-----------------------------------------------------------------------
!  hdxb^T Rinv hdxb + (m-1) I / rho (covariance inflation)
!-----------------------------------------------------------------------
!  IF (PRESENT(minfl)) THEN                           !GYL
    IF (minfl > 0.0d0 .AND. parm_infl < minfl) THEN   !GYL
      parm_infl = minfl                               !GYL
    END IF                                            !GYL
!  END IF                                             !GYL
  rho = 1.0d0 / parm_infl
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
!  DO j=1,ne
!    DO i=1,ne
!      pa(i,j) = work1(i,1) * eivec(j,1)
!      DO k=2,ne
!        pa(i,j) = pa(i,j) + work1(i,k) * eivec(j,k)
!      END DO
!    END DO
!  END DO
!-----------------------------------------------------------------------
!  Pa hdxb_rinv^T
!-----------------------------------------------------------------------
  CALL dgemm('n','t',ne,nobsl,ne,1.0d0,pa,ne,hdxb_rinv,&
    & nobsl,0.0d0,work2,ne)
!  DO j=1,nobsl
!    DO i=1,ne
!      work2(i,j) = pa(i,1) * hdxb_rinv(j,1)
!      DO k=2,ne
!        work2(i,j) = work2(i,j) + pa(i,k) * hdxb_rinv(j,k)
!      END DO
!    END DO
!  END DO
!-----------------------------------------------------------------------
!  Pa hdxb_rinv^T dep
!-----------------------------------------------------------------------
  DO i=1,ne
    work3(i) = work2(i,1) * dep(1)
    DO j=2,nobsl
      work3(i) = work3(i) + work2(i,j) * dep(j)
    END DO
  END DO
!-----------------------------------------------------------------------
!  T = sqrt[(m-1)Pa]
!-----------------------------------------------------------------------
  DO j=1,ne
    rho = SQRT( REAL(ne-1,r_size) / eival(j) )
    DO i=1,ne
      work1(i,j) = eivec(i,j) * rho
    END DO
  END DO
  CALL dgemm('n','t',ne,ne,ne,1.0d0,work1,ne,eivec,&
    & ne,0.0d0,trans,ne)
!  DO j=1,ne
!    DO i=1,ne
!      trans(i,j) = work1(i,1) * eivec(j,1)
!      DO k=2,ne
!        trans(i,j) = trans(i,j) + work1(i,k) * eivec(j,k)
!      END DO
!    END DO
!  END DO
!-----------------------------------------------------------------------
!  T + Pa hdxb_rinv^T dep
!-----------------------------------------------------------------------
!  IF (PRESENT(transm)) THEN                !GYL - if transm is present,
    transm = work3                         !GYL - return both trans and transm without adding them
!  ELSE                                     !GYL
!    DO j=1,ne
!      DO i=1,ne
!        trans(i,j) = trans(i,j) + work3(i)
!      END DO
!    END DO
!  END IF                                   !GYL
!  IF (PRESENT(pao))THEN
     pao = pa               !GYL
!  ENDIF
!-----------------------------------------------------------------------
!  Inflation estimation
!-----------------------------------------------------------------------
  parm = 0.0d0
    DO i=1,nobsl                                  !GYL
      parm(1) = parm(1) + dep(i)*dep(i)/rdiag(i)  !GYL
    END DO                                        !GYL
  DO j=1,ne
    DO i=1,nobsl
      parm(2) = parm(2) + hdxb_rinv(i,j) * hdxb(i,j)
    END DO
  END DO
  parm(2) = parm(2) / REAL(ne-1,r_size)
  parm(3) = SUM(rloc(1:nobsl))
  parm(4) = (parm(1)-parm(3))/parm(2) - parm_infl
!  sigma_o = 1.0d0/REAL(nobsl,r_size)/MAXVAL(rloc(1:nobsl))
  sigma_o = 2.0d0/parm(3)*((parm_infl*parm(2)+parm(3))/parm(2))**2
  gain = sigma_b**2 / (sigma_o + sigma_b**2)
  parm_infl = parm_infl + gain * parm(4)

  RETURN
  END IF
END SUBROUTINE letkf_core

!-----------------------------------------------------------------------
! Relaxation via LETKF weight - RTPS method
!-----------------------------------------------------------------------
subroutine weight_RTPS( ne, relax_alpha_spread , w, pa, xb, wrlx )
  implicit none
  integer     , intent(in) :: ne
  real(r_size), intent(in) :: w(ne,ne)
  real(r_size), intent(in) :: pa(ne,ne)
  real(r_size), intent(in) :: xb(ne)
  real(r_size), intent(in) :: relax_alpha_spread
  real(r_size), intent(out) :: wrlx(ne,ne)
  real(r_size)              :: infl
  real(r_size) :: var_g, var_a
  integer :: m, k

  var_g = 0.0d0
  var_a = 0.0d0
  do m = 1, ne
    var_g = var_g + xb(m) * xb(m)
    do k = 1, ne
      var_a = var_a + xb(k) * pa(k,m) * xb(m)
    end do
  end do
  if (var_g > 0.0d0 .and. var_a > 0.0d0) then
    infl = relax_alpha_spread * sqrt(var_g / (var_a * real(ne-1,r_size))) - relax_alpha_spread + 1.0d0   ! Whitaker and Hamill 2012
    wrlx = w * infl
  else
    wrlx = w
    infl = 1.0d0
  end if

  return
end subroutine weight_RTPS

!-----------------------------------------------------------------------
! Relaxation via LETKF weight - RTPP method
!-----------------------------------------------------------------------
subroutine weight_RTPP(ne , relax_alpha ,w, wrlx)

  implicit none
  integer     , intent(in)  :: ne
  real(r_size), intent(in)  :: w(ne,ne)
  real(r_size), intent(out) :: wrlx(ne,ne)
  real(r_size), intent(in)  :: relax_alpha
  integer :: m

  wrlx = (1.0d0 - relax_alpha ) * w
  do m = 1, ne
    wrlx(m,m) = wrlx(m,m) + relax_alpha
  end do

  return

end subroutine weight_RTPP

!-----------------------------------------------------------------------
! Relaxation via LETKF weight - EPES method
!-----------------------------------------------------------------------
subroutine weight_EPES(ne, relax_alpha_spreadw , w, wrlx)
implicit none
integer , intent(in)       :: ne
real(r_size) , intent(in)  :: w(ne,ne)
real(r_size) , intent(out) :: wrlx(ne,ne) 
real(r_size) , intent(in)  :: relax_alpha_spreadw
real(r_size)               :: w_var(ne),w_mean(ne) , infl
integer                    :: ie 

!EPES is like relaxation to prior spread in the ensemble space.
!Wb is the identity matrix. In this approach we will inflate the 
!variance of the columns of Wa so that they are equal to the variance
!of the columns of Wb.

IF( relax_alpha_spreadw == 0.0d0 )THEN
  wrlx=w
  return
ENDIF

!First remove the mean of the columns of W.
w_mean=0.0d0
DO ie=1,ne
  CALL com_mean(ne,w(:,ie),w_mean(ie))
  wrlx(:,ie)=w(:,ie)-w_mean(ie) 
END DO

!Compute W column variance
w_var=0.0d0
DO ie=1,ne
   CALL com_covar(ne,wrlx(:,ie),wrlx(:,ie),w_var(ie))
ENDDO

!Compute the inflation factor and update W matrix.
infl=relax_alpha_spreadw * sqrt( 1 / sum(w_var) )

wrlx=infl * wrlx

end subroutine weight_EPES

!-----------------------------------------------------------------------
! Relaxation via LETKF weight - modified EPES method
! Modification of the original method as suggested by Bryan Hunt
!-----------------------------------------------------------------------
subroutine weight_modEPES(ne, relax_alpha_spreadw , w, wrlx)
implicit none
integer , intent(in)       :: ne
real(r_size) , intent(in)  :: w(ne,ne)
real(r_size) , intent(out) :: wrlx(ne,ne)
real(r_size) , intent(in)  :: relax_alpha_spreadw
real(r_size)               :: w_trace,w_mean(ne) , infl
integer                    :: ie

IF( relax_alpha_spreadw == 0.0d0 )THEN
  wrlx=w
  return
ENDIF

!First remove the mean of the columns of W.
w_mean=0.0d0
DO ie=1,ne
  CALL com_mean(ne,w(:,ie),w_mean(ie))
  wrlx(:,ie)=w(:,ie)-w_mean(ie)
END DO

!Compute the trace of W
w_trace=0.0d0
DO ie=1,ne
   w_trace=w_trace+wrlx(ie,ie)
ENDDO

!Compute the inflation coefficient and update W
infl=REAL(ne,r_size)/w_trace

wrlx= infl  * wrlx

END SUBROUTINE weight_modEPES

END MODULE common_letkf
