MODULE common_pf
!=======================================================================
!
! [PURPOSE:] Local Particle Filter
!
!=======================================================================
!$USE OMP_LIB
  USE common_tools

  IMPLICIT NONE

  REAL(r_size),PARAMETER  :: stop_threshold_sinkhorn = 1.0d-8  !Stoping threshold for Sinkhorn iteration
  INTEGER     ,PARAMETER  :: max_iter_sinkhorn = 10000         !Max number of iterations for Sinkhorn iteration
  REAL(r_size),PARAMETER  :: lambda = 40.0            !Inverse of regularization parameter in Sinkhorn iteration

  REAL(r_size),PARAMETER  :: stop_threshold_riccati = 1.0d-3
  REAL(r_size),PARAMETER  :: dt_riccati = 0.1
  INTEGER     ,PARAMETER  :: max_iter_riccati=1000

  PUBLIC

CONTAINS

!=======================================================================
!  Main Subroutine of LETPF Core
!   INPUT
!     ne               : ensemble size                                         
!     nobsl            : total number of observation assimilated at the point
!     dens(nobsl,ne)   : distance between each ensemble member and the observation
!     rdiag(nobsl)     : observation error variance
!     rloc(nobsl)      : localization weigthning function
!   OUTPUT
!     wa(ne)           : PF weigths
!=======================================================================
SUBROUTINE letpf_core(ne,ndim,nobsl,dens,m,rdiag,rloc,wa,W)
  IMPLICIT NONE
  INTEGER,INTENT(IN) :: ne , ndim                      
  INTEGER,INTENT(IN) :: nobsl
  REAL(r_size),INTENT(IN) :: dens(1:nobsl,1:ne)
  REAL(r_size),INTENT(IN) :: m(1:ne,1:ne)   !Distance matrix
  REAL(r_size),INTENT(IN) :: rdiag(1:nobsl)
  REAL(r_size),INTENT(IN) :: rloc(1:nobsl)
  REAL(r_size),INTENT(OUT) :: wa(ne)    !
  REAL(r_size),INTENT(OUT) :: W(ne,ne)  !Transformation matrix
  REAL(r_size)  :: wt(ne)               !Target weigths 
  REAL(r_size)  :: delta(ne,ne)         !Correction to get a second order exact filter.
  REAL(r_size)  :: log_w_sum        

  INTEGER :: i,j,k
  
  !Compute the weights.
  wa=0.0d0
  DO i=1,ne
    DO j=1,nobsl
       wa(i)=wa(i) - ( dens(j,i)**2 ) / ( rdiag(j) * rloc(j) )
    ENDDO
  ENDDO
  !Normalize log of the weigths (to avoid underflow issues)
  CALL log_sum_vec( ne , wa , log_w_sum )
  DO i=1,ne
     wa(i) = EXP( wa(i) - log_w_sum )
  ENDDO


  !Normalize the weigths (just to remove any precission issue)
  wa = wa / sum(wa)

  wt = 1.0 / REAL( ne , r_size )  !Compute the target weigths (equal weigths in this case)

  !Solve the regularized optimal transport problem.
  CALL sinkhorn_ot_robust( ne , wa , wt , m , W , lambda , stop_threshold_sinkhorn , max_iter_sinkhorn )

  !Call Riccati solver
  CALL riccati_solver( ne , W , wa , dt_riccati , stop_threshold_riccati , max_iter_riccati , delta )

  W = W + delta
  
  RETURN
END SUBROUTINE letpf_core

SUBROUTINE riccati_solver( ne , D , win , dt , stop_threshold , max_iter , delta )
IMPLICIT NONE
REAL(r_size) , INTENT(IN)    :: D(ne,ne) 
REAL(r_size) , INTENT(IN)    :: win(ne)
REAL(r_size) , INTENT(IN)    :: dt , stop_threshold
INTEGER      , INTENT(IN)    :: max_iter , ne 
REAL(r_size) , INTENT(OUT)   :: delta(ne,ne)
REAL(r_size)                 :: ones(ne,1) , wa(ne,1) , B(ne,ne) , A(ne,ne) , W(ne,ne) , delta_old(ne,ne)
INTEGER                      :: i , it_num

ones = 1.0d0
wa(:,1) = win
delta = 0.0d0
W=0.0d0

DO i=1,ne
   W(i,i)=win(i)
ENDDO

B = D  - MATMUL( W , TRANSPOSE( ones ) )
A = REAL(ne,r_size) * ( W - MATMUL( wa , TRANSPOSE(wa) ) ) - MATMUL( B , TRANSPOSE(B) )

it_num = 0    
DO  !Do until exit

  delta_old = delta 
  delta = delta + dt * ( -MATMUL( B , delta) -MATMUL( delta , TRANSPOSE(B) ) + A - MATMUL(delta,delta) )
  it_num = it_num + 1
  IF( MAXVAL( delta - delta_old ) < stop_threshold )THEN
      exit 
  ENDIF
  IF( it_num > max_iter )THEN
      exit
      WRITE(*,*)'Warning: Iteration limit reached in Riccati solver'
  ENDIF

ENDDO

END SUBROUTINE riccati_solver


SUBROUTINE get_distance_matrix( ne , nx , nvar , nt , xens , m )
IMPLICIT NONE
INTEGER,INTENT(IN) :: ne   !Ensemble size
INTEGER,INTENT(IN) :: nx , nvar , nt  !Ensemble dimensions
REAL(r_size),INTENT(IN) :: xens(nx,ne,nvar,nt)
REAL(r_size),INTENT(OUT):: m(ne,ne)
INTEGER :: i , j , k , ix , iv , it

  !For Lorenz 96 all variables are equal 
  !For other models we would need to normalize the distance
  !computed from different variables.

  DO i=1,ne
    DO j=i,ne
      m(i,j)=0.0d0
      DO ix=1,nx
        DO iv=1,nvar
          DO it=1,nt
            m(i,j) = m(i,j) + ( xens(ix,i,iv,it) - xens(ix,j,iv,it) ) ** 2
          ENDDO
        ENDDO
      ENDDO
      IF( i .ne. j )THEN
        m(j,i) = m(i,j)
      ENDIF
    ENDDO
  ENDDO

END SUBROUTINE get_distance_matrix

SUBROUTINE sinkhorn_ot( ne , wi , wt , m , W , lambda , stop_threshold , max_iter )
IMPLICIT NONE
INTEGER     ,INTENT(IN) :: ne
REAL(r_size),INTENT(IN) :: wi(ne) , wt(ne) !Initial and target weigths.
REAL(r_size),INTENT(IN) :: m(ne,ne) !Cost matrix for the optimal transport problem.
REAL(r_size),INTENT(IN) :: lambda , stop_threshold
INTEGER     ,INTENT(IN) :: max_iter
REAL(r_size),INTENT(OUT):: w(ne,ne) !Transformation matrix.
REAL(r_size)            :: u(ne) , v(ne) , K(ne,ne) , lnK(ne,ne) , lnKmax , wdiff(ne) , west(ne) , metric
REAL(r_size),PARAMETER  :: lnKmaxTr = 200.0d0
INTEGER                 :: it_num , i  , j
REAL(r_size)            :: tmp_val

!Solves the Sinkhorn optimal transport problem following Acevedo et al. 2017 SIAM
u=1.0d0
v=1.0d0    
K=1.0d0
    
lnK =  -lambda * ( m )
!Normalize lnK to avoid the divergence of the iteration.
lnKmax = maxval( lnK )
IF ( lnKmax > lnKmaxTr ) THEN
   lnK = lnK * lnKmaxTr /  lnKmax
ENDIF
K = EXP( lnK )
it_num = 0
DO !This loop last until termination conditions mets
  it_num = it_num + 1 
  DO i=1,ne
   tmp_val = 0.0d0
   DO j=1,ne
     tmp_val = tmp_val + K(i,j) * v(j)   
   ENDDO
   u(i) = REAL(ne,r_size) * wi(i) / tmp_val
  ENDDO
  DO i=1,ne
   tmp_val = 0.0d0
   DO j=1,ne
     tmp_val = tmp_val + K(i,j) * u(j) 
   ENDDO
   v(i) = 1.0d0 / tmp_val
  ENDDO
  !Check stoping criteria once every 10 time steps
  IF( mod( it_num , 10 ) .eq. 0 )THEN
    W = K
    DO i = 1,ne 
      W(:,i)=W(:,i)*v(i)
      W(i,:)=W(i,:)*u(i)
    ENDDO
    DO i = 1,ne
      west(i)  = SUM( W(i,:) )/REAL(ne,r_size)
      wdiff(i) = ( SUM( W(i,:) )/REAL(ne,r_size) - wi(i) )**2
    ENDDO
      metric = SUM( wdiff )
      IF( metric < stop_threshold ) THEN
        EXIT
      ENDIF
      IF( it_num > max_iter )THEN
        WRITE(*,*)'Warning: Iteration limit reached in Sinkhorn minimization'
        EXIT
      ENDIF
  ENDIF

ENDDO
!Once the iteration is complete correct W.
!La siguiente linea no esta igual al paper pero me parece que hay un error en el trabajo
!en la ecuacion 5.7. Poniendo los terminos como esta a continuacion se verifican las 
!restricciones que indica el paper, pero si se usa lo que dice la ecuacion 5.7 esas restricciones
!no se cumplen y el filtro diverge.
DO i=1,ne
   W(:,i) = W(:,i) -  west(:) + wi(:) 
ENDDO    

END SUBROUTINE sinkhorn_ot

SUBROUTINE sinkhorn_ot_robust( ne , wi , wt , m , W , lambda , stop_threshold , max_iter )
IMPLICIT NONE
INTEGER     ,INTENT(IN) :: ne
REAL(r_size),INTENT(IN) :: wi(ne) , wt(ne) !Initial and target weigths.
REAL(r_size),INTENT(IN) :: m(ne,ne) !Cost matrix for the optimal transport problem.
REAL(r_size),INTENT(IN) :: lambda , stop_threshold
INTEGER     ,INTENT(IN) :: max_iter
REAL(r_size),INTENT(OUT):: w(ne,ne) !Transformation matrix.
REAL(r_size)            :: lnu(ne) , lnv(ne) , lnK(ne,ne) , wdiff(ne) , west(ne) , metric
INTEGER                 :: it_num , i  , j
REAL(r_size)            :: tmp_val

!Solves the Sinkhorn optimal transport problem following Acevedo et al. 2017 SIAM
lnu=0.0d0
lnv=0.0d0
lnK =  -lambda * ( m )

it_num = 0
DO !This loop last until termination conditions mets
  it_num = it_num + 1
  DO i=1,ne
   CALL log_sum_vec( ne , lnk(i,:) + lnv , tmp_val )
   lnu(i) = LOG( REAL(ne,r_size) * wi(i) ) - tmp_val
  ENDDO
  DO i=1,ne
   CALL log_sum_vec( ne , lnk(i,:) + lnu , tmp_val )
   lnv(i) = - tmp_val
  ENDDO
  !Check stoping criteria once every 10 time steps
  IF( mod( it_num , 10 ) .eq. 0 )THEN
    W = 0.0d0
    DO i = 1,ne
      DO j = 1,ne
        W(j,i)=lnu(i) + lnk(i,j) + lnv(j)
      ENDDO
    ENDDO
    W = EXP(W) 
    DO i = 1,ne
      west(i)  = SUM( W(i,:) )/REAL(ne,r_size)
      wdiff(i) = ( SUM( W(i,:) )/REAL(ne,r_size) - wi(i) )**2
    ENDDO
      metric = SUM( wdiff )
      IF( metric < stop_threshold ) THEN
        EXIT
      ENDIF
      IF( it_num > max_iter )THEN
        WRITE(*,*)'Warning: Iteration limit reached in Sinkhorn minimization'
        EXIT
      ENDIF
  ENDIF

ENDDO
!Once the iteration is complete correct W.
!La siguiente linea no esta igual al paper pero me parece que hay un error en el trabajo
!en la ecuacion 5.7. Poniendo los terminos como esta a continuacion se verifican las
!restricciones que indica el paper, pero si se usa lo que dice la ecuacion 5.7 esas restricciones
!no se cumplen y el filtro diverge.
DO i=1,ne
   W(:,i) = W(:,i) -  west(:) + wi(:)
ENDDO

END SUBROUTINE sinkhorn_ot_robust

SUBROUTINE log_sum_vec( ne , logvec , log_sum )
IMPLICIT NONE
INTEGER, INTENT(IN)       :: ne
REAL(r_size) , INTENT(IN) :: logvec(ne)
REAL(r_size) , INTENT(OUT):: log_sum
REAL(r_size)              :: max_log_vec

max_log_vec = MAXVAL( logvec )

log_sum = max_log_vec + LOG( SUM( EXP( logvec - max_log_vec ) ) )

END SUBROUTINE log_sum_vec



END MODULE common_pf
