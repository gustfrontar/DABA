#!/bin/bash
#PBS -l nodes=1:ppn=40

module load anaconda3/2019.10
source activate daba

cd /home/jruiz/DABA/Lorenz_96/experiments/

#python -u ./sensitivit_experiment_tempering_addinf_LETKF_R4_D05_Freq8_Hlogaritmic.py > ./bash_experiment.log
#python -u ./sensitivit_experiment_bridging_addinf_R4_D05_Freq8_Hlogaritmic.py > ./bash_experiment_2.log
#python -u ./sensitivit_experiment_bridging_addinf_Ens100_R4_D05_Freq8_Hlogaritmic.py > ./bash_experiment_3.log
#python -u ./sensitivit_experiment_tempering_addinf_LETKF_R8_D05_Freq8_Hlinear.py > ./bash_experiment_4.log
python -u ./sensitivit_experiment_bridging_addinf_R8_D05_Freq8_Hlinear.py > ./bash_experiment_5.log


