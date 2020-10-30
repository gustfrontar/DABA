#!/bin/bash

#COMPILER=f2py3
#COMPILER='f2py -c ' #--fcompiler=intelem'
COMPILER='gfortran'
FFLAGS='-O3 '
F90FLAGS='-fopenmp -lgomp'

#$COMPILER -O3 -fopenmp -lgomp -c SFMT.f90
$COMPILER -O3 -fopenmp -lgomp -c common_tools.f90 
$COMPILER -O3 -fopenmp -lgomp -c main_test.f90
$COMPILER -O3 -fopenmp -lgomp -o test.exe  *.o  #> compile.out 2>&1



