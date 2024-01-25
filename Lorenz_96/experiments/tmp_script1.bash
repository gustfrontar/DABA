#!/bin/bash                                                     
cd /mnt/data/dropbox/DABA/Lorenz_96/experiments                                                    
export OMP_NUM_THREADS=1                                  
python -u ./sensitivity_experiment_grosserr_multinf_LETKF_R5_D1_Freq4_Hradar.py compute  > ./logs/sensitivity_experiment_grosserr_multinf_LETKF_R5_D1_Freq4_Hradar.py.log           
