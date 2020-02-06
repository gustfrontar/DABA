PROGRAM test

USE rand_matrix 
USE common_tools

IMPLICIT NONE
INTEGER,PARAMETER :: ne = 10
REAL(r_size) :: delta(ne,ne)



CALL PDAF_generate_rndmat(ne, delta, 2)

WRITE(*,*)delta




END PROGRAM test
