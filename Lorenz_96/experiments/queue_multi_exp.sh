#!/bin/bash

MY_EXP=sens_exp_multifyloc_LEKTF_ptemp2.0.py


NATURE_ARRAY=( 'MultipleNature_Nature_Freq4_Den1.0_Type1_ObsErr1' \
'MultipleNature_Nature_Freq4_Den0.5_Type1_ObsErr1' \
'MultipleNature_Nature_Freq4_Den0.1_Type1_ObsErr1' \
'MultipleNature_Nature_Freq8_Den1.0_Type1_ObsErr1' \
'MultipleNature_Nature_Freq8_Den0.5_Type1_ObsErr1' \
'MultipleNature_Nature_Freq8_Den0.1_Type1_ObsErr1' \
'MultipleNature_Nature_Freq12_Den1.0_Type1_ObsErr1' \
'MultipleNature_Nature_Freq12_Den0.5_Type1_ObsErr1' \
'MultipleNature_Nature_Freq12_Den0.1_Type1_ObsErr1' \
'MultipleNature_Nature_Freq16_Den1.0_Type1_ObsErr1' \
'MultipleNature_Nature_Freq16_Den0.5_Type1_ObsErr1' \
'MultipleNature_Nature_Freq16_Den0.1_Type1_ObsErr1' \
'MultipleNature_Nature_Freq20_Den1.0_Type1_ObsErr1' \
'MultipleNature_Nature_Freq20_Den0.5_Type1_ObsErr1' \
'MultipleNature_Nature_Freq20_Den0.1_Type1_ObsErr1' \
'MultipleNature_Nature_Freq4_Den1.0_Type3_ObsErr1' \
'MultipleNature_Nature_Freq4_Den0.5_Type3_ObsErr1' \
'MultipleNature_Nature_Freq4_Den0.1_Type3_ObsErr1' \
'MultipleNature_Nature_Freq8_Den1.0_Type3_ObsErr1' \
'MultipleNature_Nature_Freq8_Den0.5_Type3_ObsErr1' \
'MultipleNature_Nature_Freq8_Den0.1_Type3_ObsErr1' \
'MultipleNature_Nature_Freq12_Den1.0_Type3_ObsErr1' \
'MultipleNature_Nature_Freq12_Den0.5_Type3_ObsErr1' \
'MultipleNature_Nature_Freq12_Den0.1_Type3_ObsErr1' \
'MultipleNature_Nature_Freq16_Den1.0_Type3_ObsErr1' \
'MultipleNature_Nature_Freq16_Den0.5_Type3_ObsErr1' \
'MultipleNature_Nature_Freq16_Den0.1_Type3_ObsErr1' \
'MultipleNature_Nature_Freq20_Den1.0_Type3_ObsErr1' \
'MultipleNature_Nature_Freq20_Den0.5_Type3_ObsErr1' \
'MultipleNature_Nature_Freq20_Den0.1_Type3_ObsErr1' \
)

for str in ${NATURE_ARRAY[@]}; do
  echo "Sending $MY_EXP with nature $str to the queue"
  ./queue_expV2.sh  $MY_EXP $str
  sleep 1
done
