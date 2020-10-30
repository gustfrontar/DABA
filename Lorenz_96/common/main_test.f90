PROGRAM test

USE common_tools

IMPLICIT NONE
INTEGER,PARAMETER :: ne = 10 , nt = 3 , nx = 3
REAL :: delta(ne,ne)

CALL test_sub( nx , ne , nt )


END PROGRAM test



SUBROUTINE test_sub( nx , ne , nt)
  USE common_tools
  IMPLICIT NONE
  INTEGER , INTENT(IN) ::  nx , ne , nt
  REAL(r_size) :: random_data(nx,ne,nt)
  INTEGER :: ix , ie , it
  

!============================================================================
!INITIALIZE SOME VARIABLES AND PARAMETERS
!============================================================================

!$OMP PARALLEL DO PRIVATE(ix,ie,it)

DO ie = 1,ne
   DO it = 1,nt

    CALL com_randn( nx , random_data(:,ie,it) )

  END DO  !End do of time steps
END DO  !End do of ensemble members

!$OMP END PARALLEL DO

DO it = 1 , nt
  WRITE(*,*)'my time is ',it
  DO ie = 1,ne
     WRITE(*,*)'member ',ie,random_data(:,ie,it)
  END DO
END DO

RETURN

END SUBROUTINE test_sub

