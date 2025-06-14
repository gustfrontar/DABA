#!/bin/bash

MY_EXP=sens_exp_multifyloc_HYBRIDF_ptemp2.0.py

NATURE_ARRAY=($(ls ./data/Nature/MultipleNature*.npz))

for str in ${NATURE_ARRAY[@]}; do
  str=$(basename "$str")
  str=${str::-4}  #Remove npz. from the nature name.
  echo "Sending $MY_EXP with nature $str to the queue"
  ./queue_expV2.sh  $MY_EXP $str
  sleep 1
done
