MODULE common_pf
!=======================================================================
!
! [PURPOSE:] Local Particle Filter
!
!=======================================================================
!$USE OMP_LIB
  USE common_tools

  IMPLICIT NONE

  REAL(r_size),PARAMETER  :: stop_threshold = 1.0e-8  !Stoping threshold for Sinkhorn iteration
  INTEGER     ,PARAMETER  :: max_iter = 10000         !Max number of iterations for Sinkhorn iteration
  REAL(r_size),PARAMETER  :: lambda = 40.0            !Inverse of regularization parameter in Sinkhorn iteration


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
SUBROUTINE letpf_core(ne,ndim,nobsl,dens,xens,rdiag,rloc,wa,W)
  IMPLICIT NONE
  INTEGER,INTENT(IN) :: ne , ndim                      
  INTEGER,INTENT(IN) :: nobsl
  REAL(r_size),INTENT(IN) :: dens(1:nobsl,1:ne)
  REAL(r_size),INTENT(IN) :: xens(1:ndim,1:ne)   !Local ensemble.
  REAL(r_size),INTENT(IN) :: rdiag(1:nobsl)
  REAL(r_size),INTENT(IN) :: rloc(1:nobsl)
  REAL(r_size),INTENT(OUT) :: wa(ne) 
  REAL(r_size),INTENT(OUT) :: W(ne,ne)  !Transformation matrix

  REAL(r_size)  :: m(ne,ne)  , wt(ne) 

  REAL(r_size)  :: tw(ne)  !Target weigths

  INTEGER :: i,j,k
  
  !Compute the weights.
  wa=0.0d0
  DO i=1,ne
    DO j=1,nobsl
       wa(i)=wa(i)+ ( dens(j,i)**2 ) / ( rdiag(j) * rloc(j) )
    ENDDO
  ENDDO

  wa = exp( -1.0d0 * wa )

  !Normalize the weigths.
  wa = wa / sum(wa)

  wt = 1.0 / REAL( ne , r_size )  !Compute the target weigths (equal weigths in this case)

  !Get the distance matrix (the cost matrix for the optimal transport problem)
  CALL get_distance_matrix( ne , ndim , xens , m ) 
  !Solve the regularized optimal transport problem.
  CALL sinkhorn_ot( ne , wa , wt , m , W , lambda , stop_threshold , max_iter )
   
  
  RETURN
END SUBROUTINE letpf_core


SUBROUTINE get_distance_matrix( ne , ndim , xens , m )
IMPLICIT NONE
INTEGER,INTENT(IN) :: ne   !Ensemble size
INTEGER,INTENT(IN) :: ndim !Number of state variables
REAL(r_size),INTENT(IN) :: xens(ndim,ne)
REAL(r_size),INTENT(OUT):: m(ne,ne)
INTEGER :: i , j , k

  DO i=1,ne
    DO j=i,ne
      m(i,j)=0.0d0
      DO k=1,ndim
        m(i,j)=m(i,j) + ( xens(k,i) - xens(k,j) ) ** 2
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


END MODULE common_pf
