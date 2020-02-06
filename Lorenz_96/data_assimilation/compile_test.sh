gfortran -c SFMT.f90
gfortran -c common_tools.f90
gfortran -c PDAF_generate_rndmat.F90
gfortran -c main_test.f90
gfortran -o main_test.exe *.o

