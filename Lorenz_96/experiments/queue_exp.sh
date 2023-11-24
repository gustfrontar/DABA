#!/bin/bash
MY_EXP=$1
BASEPATH=$(pwd)
NCORES=20

#Tempering and LETKF with linear operator.
expnum=1
#MY_EXP=sensitivit_experiment_adtempering_multinf_LETKF_R1_D1_Freq4_Hlinear.py
echo "#!/bin/bash                                                     "  > ./tmp_script${expnum}.bash
echo "source /opt/load-libs.sh 3                                      " >> ./tmp_script${expnum}.bash
echo "cd $BASEPATH                                                    " >> ./tmp_script${expnum}.bash
echo "export PATH="/opt/intel/oneapi/intelpython/latest/bin:$PATH"    " >> ./tmp_script${expnum}.bash
echo "export OMP_NUM_THREADS=$NCORES                                  " >> ./tmp_script${expnum}.bash
echo "python -u ./${MY_EXP} compute  > ./logs/${MY_EXP}.log           " >> ./tmp_script${expnum}.bash
echo qsub -l nodes=1:ppn=$NCORES ./tmp_script${expnum}.bash
qsub -l nodes=1:ppn=$NCORES ./tmp_script${expnum}.bash 


#Tempering and LETKF with radar
#MY_EXP=sensitivit_experiment_adtempering_multinf_LETKF_R1_D1_Freq4_Hradar.py
#python -u ./${MY_EXP} compute  > ./logs/${MY_EXP}.log
























