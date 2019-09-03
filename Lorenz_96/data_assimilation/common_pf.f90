MODULE common_pf
!=======================================================================
!
! [PURPOSE:] Local Particle Filter
!
!=======================================================================
!$USE OMP_LIB
  USE common_tools

  IMPLICIT NONE

  PUBLIC

CONTAINS

!=======================================================================
!  Main Subroutine of LETKF Core
!   INPUT
!     ne               : ensemble size                                         
!     nobsl            : total number of observation assimilated at the point
!     dens(nobsl,ne)   : distance between each ensemble member and the observation
!     rdiag(nobsl)     : observation error variance
!     rloc(nobsl)      : localization weigthning function
!   OUTPUT
!     wa(ne)           : PF weigths
!=======================================================================
SUBROUTINE lpf_core(ne,nobsl,dens,rdiag,rloc,wa)
  IMPLICIT NONE
  INTEGER,INTENT(IN) :: ne                      !GYL
  INTEGER,INTENT(IN) :: nobsl
  REAL(r_size),INTENT(IN) :: dens(1:nobsl,1:ne)
  REAL(r_size),INTENT(IN) :: rdiag(1:nobsl)
  REAL(r_size),INTENT(IN) :: rloc(1:nobsl)
  REAL(r_size),INTENT(OUT) :: wa(ne)

  INTEGER :: i,j,k

  
  !Compute the weights.

  !TODO chequear la rutina de claculo de los pesos. 
  wa=0.0d0
  DO i=1,ne
    DO j=1,nobsl
       wa(i)=wa(i)+ ( dens(j,i)**2 ) / ( rdiag(j) * rloc(j) )
    ENDDO
  ENDDO

  !WRITE(*,*) "WA",wa

  wa = exp( -1.0d0 * wa )
  !wa = 1.0/(1.0 + wa)

  !WRITE(*,*) "WA",wa


  !Normalize the weigths.
  wa = wa / sum(wa)
   
  
  RETURN
END SUBROUTINE lpf_core

END MODULE common_pf
