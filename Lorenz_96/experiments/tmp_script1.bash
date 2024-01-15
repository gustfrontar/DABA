#!/bin/bash                                                     
cd /home/jruiz/Dropbox/DABA/Lorenz_96/experiments                                                    
export OMP_NUM_THREADS=1                                  
python -u ./sensitivity_experiment_ptemp1.8_multinf_LETKF_R1_D1_Freq4_Hradar.py compute  > ./logs/sensitivity_experiment_ptemp1.8_multinf_LETKF_R1_D1_Freq4_Hradar.py.log           
