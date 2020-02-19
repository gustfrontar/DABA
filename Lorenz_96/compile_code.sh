#!/bin/bash

#COMPILER=f2py3
COMPILER='f2py -c ' #--fcompiler=intelem'
FFLAGS='-O3 '
F90FLAGS='-fopenmp -lgomp'
#FFLAGS='-O1 -fcheck=all'  #For debug

#This script compiles the fortran modules required to run the python experiments.
COMPILE_OBS=0
COMPILE_DA=0
COMPILE_MOD=0

if [ $1 == 'all' ] ;then
COMPILE_OBS=1
COMPILE_DA=1
COMPILE_MOD=1
fi

if [ $1 == 'operator' ];then
COMPILE_OBS=1
fi
if [ $1 == 'da' ];then
COMPILE_DA=1
fi
if [ $1 == 'model' ];then
COMPILE_MOD=1
fi



if [ $COMPILE_OBS -eq 1 ] ; then

echo "Compiling Observation operator"
cd data_assimilation
ln -sf ../common/netlib.f90        .
ln -sf ../common/SFMT.f90          .
ln -sf ../common/common_tools.f90  .
ln -sf ../common/common_mtx.f90    .

$COMPILER --f90flags=$F90FLAGS --opt=$FFLAGS netlib.f90 SFMT.f90 common_tools.f90 common_obs_lorenzN.f90 -m obsope #> compile.out 2>&1

rm netlib.f90 SFMT.f90 common_tools.f90 common_mtx.f90

cd ../

fi

if [ $COMPILE_DA -eq 1 ] ; then

echo "Compiling DA routines"
cd data_assimilation
ln -sf ../common/netlib.f90        .
ln -sf ../common/SFMT.f90          .
ln -sf ../common/common_tools.f90  .
ln -sf ../common/common_mtx.f90    .


$COMPILER --f90flags=$F90FLAGS --opt=$FFLAGS SFMT.f90 netlib.f90 common_tools.f90 common_mtx.f90 -m mtx_oper #> compile.out 2>&1

$COMPILER --f90flags=$F90FLAGS --opt=$FFLAGS netlib.f90 SFMT.f90 common_tools.f90  PDAF_generate_rndmat.F90 common_mtx.f90 common_letkf.f90 common_pf.f90 common_gm.f90 common_da_tools_1d.f90 -m da #> compile.out 2>&1
rm netlib.f90 SFMT.f90 common_tools.f90 common_mtx.f90

cd ../

fi

if [ $COMPILE_MOD -eq 1 ] ; then

echo "Compiling model routines"
cd model
ln -sf ../common/SFMT.f90          .
ln -sf ../common/common_tools.f90  .
#Two scale model - stochastic parametrization model.
$COMPILER --f90flags=$F90FLAGS --opt=$FFLAGS SFMT.f90 common_tools.f90 lorenzN.f90 -m model #> compile.out 2>&1 


rm SFMT.f90 common_tools.f90 


cd ../

fi

echo "Normal end"

#ISSUES>

#If you have installed Anaconda from scratch and you experience issues with the compilation of the fortran code, try
#conda update anaconda 
#Before running the compilation script again.

