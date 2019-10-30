#!/bin/bash

#COMPILER=f2py3
COMPILER=f2py
FFLAGS='-O3'
#FFLAGS='-O1 -fcheck=all'  #For debug

#This script compiles the fortran modules required to run the python experiments.

echo "Compiling Observation operator"
cd data_assimilation
ln -sf ../common/netlib.f90        .
ln -sf ../common/SFMT.f90          .
ln -sf ../common/common_tools.f90  .
ln -sf ../common/common_mtx.f90    .

$COMPILER -c --opt=$FFLAGS netlib.f90 SFMT.f90 common_tools.f90 common_obs_lorenzN.f90 -m obsope > compile.out 2>&1

rm netlib.f90 SFMT.f90 common_tools.f90 common_mtx.f90


cd ../

echo "Compiling DA routines"
cd data_assimilation
ln -sf ../common/netlib.f90        .
ln -sf ../common/SFMT.f90          .
ln -sf ../common/common_tools.f90  .
ln -sf ../common/common_mtx.f90    .

$COMPILER -c --opt=$FFLAGS netlib.f90 SFMT.f90 common_tools.f90 common_mtx.f90 common_letkf.f90 common_pf.f90 common_da_tools_1d.f90 -m da #> compile.out 2>&1

rm netlib.f90 SFMT.f90 common_tools.f90 common_mtx.f90


cd ../

echo "Compiling model routines"
cd model
ln -sf ../common/SFMT.f90          .
ln -sf ../common/common_tools.f90  .
#Two scale model - stochastic parametrization model.
$COMPILER -c --opt=$FFLAGS SFMT.f90 common_tools.f90 lorenzN.f90 -m model > compile.out 2>&1 


rm SFMT.f90 common_tools.f90 


cd ../

echo "Normal end"

#ISSUES>

#If you have installed Anaconda from scratch and you experience issues with the compilation of the fortran code, try
#conda update anaconda 
#Before running the compilation script again.

