#!/bin/bash
#PBS -l nodes=1:ppn=16

module load anaconda3/2019.10
source activate daba

cd /home/jruiz/DABA/Lorenz_96/experiments/

#RIP AND GMPF
#Different levels of non-linearity and different sources of non-linearity
MY_EXP=sensitivit_experiment_rip_multinf_gm_R4_D05_Freq8_Hlogaritmic
python -u ./${MY_EXP}.py compute  > ./logs/${MY_EXP}.log


#Tempering AND GMPF
#Different levels of non-linearity and different sources of non-linearity
MY_EXP=sensitivit_experiment_tempering_multinf_gm_R4_D05_Freq8_Hlogaritmic
python -u ./${MY_EXP}.py compute  > ./logs/${MY_EXP}.log

#RIP AND GMPF
#Different levels of non-linearity and different sources of non-linearity
MY_EXP=sensitivit_experiment_rip_multinf_gm_R4_D05_Freq8_Hlogaritmic_ModelError
python -u ./${MY_EXP}.py compute  > ./logs/${MY_EXP}.log

#Tempering AND GMPF
#Different levels of non-linearity and different sources of non-linearity
MY_EXP=sensitivit_experiment_tempering_multinf_gm_R4_D05_Freq8_Hlogaritmic_ModelError
python -u ./${MY_EXP}.py compute  > ./logs/${MY_EXP}.log

#RIP AND GMPF
#Different levels of non-linearity and different sources of non-linearity
MY_EXP=sensitivit_experiment_rip_multinf_gm_R4_D05_Freq8_Hcuadratic
python -u ./${MY_EXP}.py compute  > ./logs/${MY_EXP}.log

#Tempering AND GMPF
#Different levels of non-linearity and different sources of non-linearity
MY_EXP=sensitivit_experiment_tempering_multinf_gm_R4_D05_Freq8_Hcuadratic
python -u ./${MY_EXP}.py compute  > ./logs/${MY_EXP}.log

#RIP AND GMPF
#Different levels of non-linearity and different sources of non-linearity
MY_EXP=sensitivit_experiment_rip_multinf_gm_R8_D05_Freq8_Hlinear
python -u ./${MY_EXP}.py compute  > ./logs/${MY_EXP}.log

#Tempering AND GMPF
#Different levels of non-linearity and different sources of non-linearity
MY_EXP=sensitivit_experiment_tempering_multinf_gm_R8_D05_Freq8_Hlinear
python -u ./${MY_EXP}.py compute  > ./logs/${MY_EXP}.log

#Adaptive Rip AND GM
#Different levels of non-linearity and different sources of non-linearity
MY_EXP=sensitivit_experiment_adrip_multinf_gm_R4_D05_Freq8_Hcuadratic
python -u ./${MY_EXP}.py compute  > ./logs/${MY_EXP}.log

#Adaptive Rip AND GM
#Different levels of non-linearity and different sources of non-linearity
MY_EXP=sensitivit_experiment_adrip_multinf_gm_R4_D05_Freq8_Hlogaritmic
python -u ./${MY_EXP}.py compute  > ./logs/${MY_EXP}.log

#Adaptive Rip AND GM
#Different levels of non-linearity and different sources of non-linearity
MY_EXP=sensitivit_experiment_adrip_multinf_gm_R4_D05_Freq8_Hlogaritmic_ModelError
python -u ./${MY_EXP}.py compute  > ./logs/${MY_EXP}.log

#Adaptive Rip AND GM
#Different levels of non-linearity and different sources of non-linearity
MY_EXP=sensitivit_experiment_adrip_multinf_gm_R8_D05_Freq8_Hlinear
python -u ./${MY_EXP}.py compute  > ./logs/${MY_EXP}.log

#Adaptive tempering AND GM
#Different levels of non-linearity and different sources of non-linearity
MY_EXP=sensitivit_experiment_adtempering_multinf_gm_R4_D05_Freq8_Hcuadratic
python -u ./${MY_EXP}.py compute  > ./logs/${MY_EXP}.log

#Adaptive tempering AND GM
#Different levels of non-linearity and different sources of non-linearity
MY_EXP=sensitivit_experiment_adtempering_multinf_gm_R4_D05_Freq8_Hlogaritmic
python -u ./${MY_EXP}.py compute  > ./logs/${MY_EXP}.log

#Adaptive tempering AND GM
#Different levels of non-linearity and different sources of non-linearity
MY_EXP=sensitivit_experiment_adtempering_multinf_gm_R4_D05_Freq8_Hlogaritmic_ModelError
python -u ./${MY_EXP}.py compute  > ./logs/${MY_EXP}.log

#Adaptive tempering AND GM
#Different levels of non-linearity and different sources of non-linearity
MY_EXP=sensitivit_experiment_adtempering_multinf_gm_R8_D05_Freq8_Hlinear
python -u ./${MY_EXP}.py compute  > ./logs/${MY_EXP}.log


