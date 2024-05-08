dsname=$1

for model in cc_8k cc_merged_v2_8k intramask_cc_8k intramask_cc_merged_v2_8k adamask_cc_merged_v2_8k cc_merged_v2_8k_intrav2cont; do
  bash scripts/eval_ppl/eval.sh $model $dsname ;
done