
COMPILER='f2py -c ' #--fcompiler=intelem'
FFLAGS='-O3 '
F90FLAGS='-fopenmp -lgomp'

$COMPILER --f90flags=$F90FLAGS --opt=$FFLAGS lorenz_63.f90 -m lorenz_63
