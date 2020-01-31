#!/bin/bash
#PBS -l nodes=1:ppn=40

module load anaconda3/2019.10
source activate daba

cd /home/jruiz/DABA/Lorenz_96/experiments/

#TEMPERING
#Different levels of non-linearity and different sources of non-linearity
#MY_EXP=sensitivit_experiment_tempering_addinf_LETKF_R4_D05_Freq8_Hlogaritmic
#python -u ./${MY_EXP}.py compute > ./logs/${MY_EXP}.log    
#MY_EXP=sensitivit_experiment_tempering_addinf_LETKF_R8_D05_Freq8_Hlinear
#python -u ./${MY_EXP}.py compute  > ./logs/${MY_EXP}.log
#MY_EXP=sensitivit_experiment_tempering_addinf_LETKF_R02_D1_Freq4_Hlinear
#python -u ./${MY_EXP}.py compute  > ./logs/${MY_EXP}.log

#Different levels of non-linearity and different sources of non-linearity pero con un ensamble mas pequenio.
#MY_EXP=sensitivit_experiment_tempering_addinf_LETKF_R4_D05_Freq8_Hlogaritmic_ENSSIZE10
#python -u ./${MY_EXP}.py compute  > ./logs/${MY_EXP}.log
#MY_EXP=sensitivit_experiment_tempering_addinf_LETKF_R8_D05_Freq8_Hlinear_ENSSIZE10
#python -u ./${MY_EXP}.py compute  > ./logs/${MY_EXP}.log
#MY_EXP=sensitivit_experiment_tempering_addinf_LETKF_R02_D1_Freq4_Hlinear_ENSSIZE10
#python -u ./${MY_EXP}.py compute  > ./logs/${MY_EXP}.log


#Different levels of non-linearity and different sources of non-linearity pero con un ensamble mas pequenio.
#MY_EXP=sensitivit_experiment_tempering_addinf_LETKF_R4_D05_Freq8_Hlogaritmic_ImpMod
#python -u ./${MY_EXP}.py compute  > ./logs/${MY_EXP}.log
#MY_EXP=sensitivit_experiment_tempering_addinf_LETKF_R8_D05_Freq8_Hlinear_ImpMod
#python -u ./${MY_EXP}.py compute  > ./logs/${MY_EXP}.log
#MY_EXP=sensitivit_experiment_tempering_addinf_LETKF_R02_D1_Freq4_Hlinear_ImpMod
#python -u ./${MY_EXP}.py compute  > ./logs/${MY_EXP}.log


#Different levels of non-linearity and different sources of non-linearity
#MY_EXP=sensitivit_experiment_rip_addinf_LETKF_R4_D05_Freq8_Hlogaritmic
#python -u ./${MY_EXP}.py compute > ./logs/${MY_EXP}.log
#MY_EXP=sensitivit_experiment_rip_addinf_LETKF_R8_D05_Freq8_Hlinear
#python -u ./${MY_EXP}.py compute  > ./logs/${MY_EXP}.log
#MY_EXP=sensitivit_experiment_rip_addinf_LETKF_R02_D1_Freq4_Hlinear
#python -u ./${MY_EXP}.py compute  > ./logs/${MY_EXP}.log

#Different levels of non-linearity and different sources of non-linearity pero con un ensamble mas pequenio.
#MY_EXP=sensitivit_experiment_rip_addinf_LETKF_R4_D05_Freq8_Hlogaritmic_ENSSIZE10
#python -u ./${MY_EXP}.py compute  > ./logs/${MY_EXP}.log
#MY_EXP=sensitivit_experiment_rip_addinf_LETKF_R8_D05_Freq8_Hlinear_ENSSIZE10
#python -u ./${MY_EXP}.py compute  > ./logs/${MY_EXP}.log
#MY_EXP=sensitivit_experiment_rip_addinf_LETKF_R02_D1_Freq4_Hlinear_ENSSIZE10
#python -u ./${MY_EXP}.py compute  > ./logs/${MY_EXP}.log


#Different levels of non-linearity and different sources of non-linearity pero con un ensamble mas pequenio.
#MY_EXP=sensitivit_experiment_rip_addinf_LETKF_R4_D05_Freq8_Hlogaritmic_ImpMod
#python -u ./${MY_EXP}.py compute  > ./logs/${MY_EXP}.log
#MY_EXP=sensitivit_experiment_rip_addinf_LETKF_R8_D05_Freq8_Hlinear_ImpMod
#python -u ./${MY_EXP}.py compute  > ./logs/${MY_EXP}.log
#MY_EXP=sensitivit_experiment_rip_addinf_LETKF_R02_D1_Freq4_Hlinear_ImpMod
#python -u ./${MY_EXP}.py compute  > ./logs/${MY_EXP}.log

#TEMPERING AND LETPF
#Different levels of non-linearity and different sources of non-linearity
MY_EXP=sensitivit_experiment_tempering_addinf_LETPF_R4_D05_Freq8_Hlogaritmic
python -u ./${MY_EXP}.py compute > ./logs/${MY_EXP}.log
MY_EXP=sensitivit_experiment_tempering_addinf_LETPF_R8_D05_Freq8_Hlinear
python -u ./${MY_EXP}.py compute  > ./logs/${MY_EXP}.log
MY_EXP=sensitivit_experiment_tempering_addinf_LETPF_R02_D1_Freq4_Hlinear
python -u ./${MY_EXP}.py compute  > ./logs/${MY_EXP}.log

#TEMPERING HYBRID03
#Different levels of non-linearity and different sources of non-linearity
MY_EXP=sensitivit_experiment_tempering_addinf_HYBRID03_R4_D05_Freq8_Hlogaritmic
python -u ./${MY_EXP}.py compute > ./logs/${MY_EXP}.log
MY_EXP=sensitivit_experiment_tempering_addinf_HYBRID03_R8_D05_Freq8_Hlinear
python -u ./${MY_EXP}.py compute  > ./logs/${MY_EXP}.log
MY_EXP=sensitivit_experiment_tempering_addinf_HYBRID03_R02_D1_Freq4_Hlinear
python -u ./${MY_EXP}.py compute  > ./logs/${MY_EXP}.log

#TEMPERING HYBRID05
#Different levels of non-linearity and different sources of non-linearity
MY_EXP=sensitivit_experiment_tempering_addinf_HYBRID05_R4_D05_Freq8_Hlogaritmic
python -u ./${MY_EXP}.py compute > ./logs/${MY_EXP}.log
MY_EXP=sensitivit_experiment_tempering_addinf_HYBRID05_R8_D05_Freq8_Hlinear
python -u ./${MY_EXP}.py compute  > ./logs/${MY_EXP}.log
MY_EXP=sensitivit_experiment_tempering_addinf_HYBRID05_R02_D1_Freq4_Hlinear
python -u ./${MY_EXP}.py compute  > ./logs/${MY_EXP}.log

#RIP AND LETPF
#Different levels of non-linearity and different sources of non-linearity
MY_EXP=sensitivit_experiment_rip_addinf_LETPF_R4_D05_Freq8_Hlogaritmic
python -u ./${MY_EXP}.py compute > ./logs/${MY_EXP}.log
MY_EXP=sensitivit_experiment_rip_addinf_LETPF_R8_D05_Freq8_Hlinear
python -u ./${MY_EXP}.py compute  > ./logs/${MY_EXP}.log
MY_EXP=sensitivit_experiment_rip_addinf_LETPF_R02_D1_Freq4_Hlinear
python -u ./${MY_EXP}.py compute  > ./logs/${MY_EXP}.log

#RIP AND HYBRID03
#Different levels of non-linearity and different sources of non-linearity
MY_EXP=sensitivit_experiment_rip_addinf_HYBRID03_R4_D05_Freq8_Hlogaritmic
python -u ./${MY_EXP}.py compute > ./logs/${MY_EXP}.log
MY_EXP=sensitivit_experiment_rip_addinf_HYBRID03_R8_D05_Freq8_Hlinear
python -u ./${MY_EXP}.py compute  > ./logs/${MY_EXP}.log
MY_EXP=sensitivit_experiment_rip_addinf_HYBRID03_R02_D1_Freq4_Hlinear
python -u ./${MY_EXP}.py compute  > ./logs/${MY_EXP}.log

#RIP AND HYBRID05
#Different levels of non-linearity and different sources of non-linearity
MY_EXP=sensitivit_experiment_rip_addinf_HYBRID05_R4_D05_Freq8_Hlogaritmic
python -u ./${MY_EXP}.py compute > ./logs/${MY_EXP}.log
MY_EXP=sensitivit_experiment_rip_addinf_HYBRID05_R8_D05_Freq8_Hlinear
python -u ./${MY_EXP}.py compute  > ./logs/${MY_EXP}.log
MY_EXP=sensitivit_experiment_rip_addinf_HYBRID05_R02_D1_Freq4_Hlinear
python -u ./${MY_EXP}.py compute  > ./logs/${MY_EXP}.log

