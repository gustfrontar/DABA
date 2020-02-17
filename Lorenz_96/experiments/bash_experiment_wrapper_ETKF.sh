#!/bin/bash
#PBS -l nodes=1:ppn=40

module load anaconda3/2019.10
source activate daba

cd /home/jruiz/DABA/Lorenz_96/experiments/

#RIP AND LETKF
#Different levels of non-linearity and different sources of non-linearity
MY_EXP=sensitivit_experiment_rip_multinf_LETKF_R4_D05_Freq8_Hlogaritmic
python -u ./${MY_EXP}.py compute  > ./logs/${MY_EXP}.log

#Tempering AND LETKF
#Different levels of non-linearity and different sources of non-linearity
MY_EXP=sensitivit_experiment_tempering_multinf_LETKF_R4_D05_Freq8_Hlogaritmic
python -u ./${MY_EXP}.py compute  > ./logs/${MY_EXP}.log

#RIP AND LETKF
#Different levels of non-linearity and different sources of non-linearity
#MY_EXP=sensitivit_experiment_rip_multinf_LETKF_R4_D05_Freq8_Hlogaritmic_ModelError
#python -u ./${MY_EXP}.py compute  > ./logs/${MY_EXP}.log

#Tempering AND LETKF
#Different levels of non-linearity and different sources of non-linearity
#MY_EXP=sensitivit_experiment_tempering_multinf_LETKF_R4_D05_Freq8_Hlogaritmic_ModelError    #ATENCION!!! FALLA
#python -u ./${MY_EXP}.py compute  > ./logs/${MY_EXP}.log

#RIP AND LETKF
#Different levels of non-linearity and different sources of non-linearity
#MY_EXP=sensitivit_experiment_rip_multinf_LETKF_R4_D05_Freq8_Hcuadratic
#python -u ./${MY_EXP}.py compute  > ./logs/${MY_EXP}.log

#Tempering AND LETKF
#Different levels of non-linearity and different sources of non-linearity
#MY_EXP=sensitivit_experiment_tempering_multinf_LETKF_R4_D05_Freq8_Hcuadratic
#python -u ./${MY_EXP}.py compute  > ./logs/${MY_EXP}.log

#RIP AND LETKF
#Different levels of non-linearity and different sources of non-linearity
#MY_EXP=sensitivit_experiment_rip_multinf_LETKF_R8_D05_Freq8_Hlinear
#python -u ./${MY_EXP}.py compute  > ./logs/${MY_EXP}.log

#Tempering AND LETKF
#Different levels of non-linearity and different sources of non-linearity
#MY_EXP=sensitivit_experiment_tempering_multinf_LETKF_R8_D05_Freq8_Hlinear
#python -u ./${MY_EXP}.py compute  > ./logs/${MY_EXP}.log

#Adaptive Rip AND LETKF
#Different levels of non-linearity and different sources of non-linearity
#MY_EXP=sensitivit_experiment_adrip_multinf_LETKF_R4_D05_Freq8_Hcuadratic
#python -u ./${MY_EXP}.py compute  > ./logs/${MY_EXP}.log

#Adaptive Rip AND LETKF
#Different levels of non-linearity and different sources of non-linearity
MY_EXP=sensitivit_experiment_adrip_multinf_LETKF_R4_D05_Freq8_Hlogaritmic
python -u ./${MY_EXP}.py compute  > ./logs/${MY_EXP}.log

#Adaptive Rip AND LETKF
#Different levels of non-linearity and different sources of non-linearity
#MY_EXP=sensitivit_experiment_adrip_multinf_LETKF_R4_D05_Freq8_Hlogaritmic_ModelError   #ATENCION!!! FALLA
#python -u ./${MY_EXP}.py compute  > ./logs/${MY_EXP}.log

#Adaptive Rip AND LETKF
#Different levels of non-linearity and different sources of non-linearity
#MY_EXP=sensitivit_experiment_adrip_multinf_LETKF_R8_D05_Freq8_Hlinear
#python -u ./${MY_EXP}.py compute  > ./logs/${MY_EXP}.log

#Adaptive tempering AND LETKF
#Different levels of non-linearity and different sources of non-linearity
#MY_EXP=sensitivit_experiment_adtempering_multinf_LETKF_R4_D05_Freq8_Hcuadratic
#python -u ./${MY_EXP}.py compute  > ./logs/${MY_EXP}.log

#Adaptive tempering AND LETKF
#Different levels of non-linearity and different sources of non-linearity
MY_EXP=sensitivit_experiment_adtempering_multinf_LETKF_R4_D05_Freq8_Hlogaritmic
python -u ./${MY_EXP}.py compute  > ./logs/${MY_EXP}.log

#Adaptive tempering AND LETKF
#Different levels of non-linearity and different sources of non-linearity
#MY_EXP=sensitivit_experiment_adtempering_multinf_LETKF_R4_D05_Freq8_Hlogaritmic_ModelError   #ATENCION LO COMENTO PREVENTIVAMENTE
#python -u ./${MY_EXP}.py compute  > ./logs/${MY_EXP}.log

#Adaptive tempering AND LETKF
#Different levels of non-linearity and different sources of non-linearity
#MY_EXP=sensitivit_experiment_adtempering_multinf_LETKF_R8_D05_Freq8_Hlinear
#python -u ./${MY_EXP}.py compute  > ./logs/${MY_EXP}.log

#Adaptive ORip AND LETKF
#Different levels of non-linearity and different sources of non-linearity
#MY_EXP=sensitivit_experiment_orip_multinf_LETKF_R4_D05_Freq8_Hcuadratic
#python -u ./${MY_EXP}.py compute  > ./logs/${MY_EXP}.log

#Adaptive ORip AND LETKF
#Different levels of non-linearity and different sources of non-linearity
MY_EXP=sensitivit_experiment_orip_multinf_LETKF_R4_D05_Freq8_Hlogaritmic
python -u ./${MY_EXP}.py compute  > ./logs/${MY_EXP}.log

#Adaptive ORip AND LETKF
#Different levels of non-linearity and different sources of non-linearity
#MY_EXP=sensitivit_experiment_orip_multinf_LETKF_R4_D05_Freq8_Hlogaritmic_ModelError  #ATENCION LO COMENTO PREVENTIVAMENTE
#python -u ./${MY_EXP}.py compute  > ./logs/${MY_EXP}.log

#Adaptive ORip AND LETKF
#Different levels of non-linearity and different sources of non-linearity
#MY_EXP=sensitivit_experiment_orip_multinf_LETKF_R8_D05_Freq8_Hlinear
#python -u ./${MY_EXP}.py compute  > ./logs/${MY_EXP}.log

#TEMPERING AND LETKF
#Different levels of non-linearity and different sources of non-linearity
MY_EXP=sensitivit_experiment_tempering_multinf_LETKF_R1_D1_Freq4_Hlinear
python -u ./${MY_EXP}.py compute  > ./logs/${MY_EXP}.log

#ADAPTIVE TEMPERING AND LETKF
#Different levels of non-linearity and different sources of non-linearity
MY_EXP=sensitivit_experiment_adtempering_multinf_LETKF_R1_D1_Freq4_Hlinear
python -u ./${MY_EXP}.py compute  > ./logs/${MY_EXP}.log

#RIP AND LETKF
#Different levels of non-linearity and different sources of non-linearity
MY_EXP=sensitivit_experiment_rip_multinf_LETKF_R1_D1_Freq4_Hlinear
python -u ./${MY_EXP}.py compute  > ./logs/${MY_EXP}.log

#ADAPTIVE RIP AND LETKF
#Different levels of non-linearity and different sources of non-linearity
MY_EXP=sensitivit_experiment_adrip_multinf_LETKF_R1_D1_Freq4_Hlinear
python -u ./${MY_EXP}.py compute  > ./logs/${MY_EXP}.log

#ORIGINAL RIP AND LETKF
#Different levels of non-linearity and different sources of non-linearity
MY_EXP=sensitivit_experiment_orip_multinf_LETKF_R1_D1_Freq4_Hlinear
python -u ./${MY_EXP}.py compute  > ./logs/${MY_EXP}.log





















