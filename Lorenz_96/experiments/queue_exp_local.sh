#!/bin/bash
MY_EXP=$1
BASEPATH=$(pwd)
NCORES=1

#Tempering and LETKF with linear operator.
expnum=1
#MY_EXP=sensitivit_experiment_adtempering_multinf_LETKF_R1_D1_Freq4_Hlinear.py
#echo "#!/bin/bash                                                     "  > ./tmp_script${expnum}.bash
#echo "source /opt/load-libs.sh 3                                      " >> ./tmp_script${expnum}.bash
#echo "cd $BASEPATH                                                    " >> ./tmp_script${expnum}.bash
#echo "export PATH="/opt/intel/oneapi/intelpython/latest/bin:$PATH"    " >> ./tmp_script${expnum}.bash
export OMP_NUM_THREADS=$NCORES                  
python -u ./${MY_EXP} compute  > ./logs/${MY_EXP}.log  &    
#echo qsub -l nodes=1:ppn=$NCORES ./tmp_script${expnum}.bash
#bash ./tmp_script${expnum}.bash  &
























