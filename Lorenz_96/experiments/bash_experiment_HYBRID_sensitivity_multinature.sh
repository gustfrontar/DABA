#!/bin/bash
#PBS -l nodes=1:ppn=40

module load anaconda3/2019.10
source activate daba

cd /home/jruiz/DABA/Lorenz_96/experiments/

#HYBRID LETKF-LETPF adaptive tempering
MY_EXP=sensitivit_experiment_bridge_multinf_adtemp_hybrid_multinature
python -u ./${MY_EXP}.py  > ./logs/${MY_EXP}.log

#HYBRID LETKF-LETPF adaptive tempered rip
MY_EXP=sensitivit_experiment_bridge_multinf_adrip_hybrid_multinature
python -u ./${MY_EXP}.py  > ./logs/${MY_EXP}.log

#HYBRID LETKF-LETPF tempered rip
MY_EXP=sensitivit_experiment_bridge_multinf_trip_hybrid_multinature
python -u ./${MY_EXP}.py  > ./logs/${MY_EXP}.log

#HYBRID LETKF-LETPF original rip
MY_EXP=sensitivit_experiment_bridge_multinf_orip_hybrid_multinature
python -u ./${MY_EXP}.py  > ./logs/${MY_EXP}.log

#HYBRID LETKF-LETPF adaptive tempering
MY_EXP=sensitivit_experiment_bridge_multinf_temp_hybrid_multinature
python -u ./${MY_EXP}.py  > ./logs/${MY_EXP}.log






