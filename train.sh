#!/bin/bash

#SBATCH --account=za99
#SBATCH --job-name=ADVISE
#SBATCH --time=168:00:00
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --partition=m3g
#SBATCH --output=ADVISE.%j.out
#SBATCH --constraint=V100-32G

module load anaconda/2019.03-Python3.7-gcc5
source activate /scratch/za99/jchu0050/conda_envs/joEnv

set -x

export PYTHONPATH="$PYTHONPATH:`pwd`"

mkdir -p "log"
mkdir -p "saved_results"

# Train for only 10,000 steps. 
# The results we put in the challenge trained for 200,000 steps.

number_of_steps=200000
#number_of_steps=10000
#number_of_steps=50000

#name="vse++"
#name="vse++_768d"
#name="advise.densecap_0.1.symbol_0.1"
#name="advise.kb"
name="advise.densecap_0.1.symbol_0_768d.1"
#name="advise.kb_768d"

ender="adagrad_mi_rob_0.01_abs_5000p_350g"

export CUDA_VISIBLE_DEVICES=0
python3 "train/train.py" \
    --pipeline_proto="configs/${name}.pbtxt" \
    --train_log_dir="logs/${name}_${ender}/train" \
    --number_of_steps="${number_of_steps}" \
    > "log/${name}_${ender}.train.log" 2>&1 &

# Also specify --restore_from if fine-tune the knowledge branch (advise.kb).
#    --restore_from="logs/advise.densecap_0.1.symbol_0.1/saved_ckpts/model.ckpt-10000" \

python3 "train/eval.py" \
    --pipeline_proto="configs/${name}.pbtxt" \
    --action_reason_annot_path="data/train/QA_Combined_Action_Reason_train.json" \
    --train_log_dir="logs/${name}_${ender}/train" \
    --eval_log_dir="logs/${name}_${ender}/eval" \
    --saved_ckpt_dir="logs/${name}_${ender}/train" \
    --continuous_evaluation="true" \
    --number_of_steps="${number_of_steps}" \
    > "log/${name}_${ender}.eval.log" 2>&1 &

wait

#########################################################
# Export the results for testing.
#########################################################
python3 "train/eval.py" \
    --pipeline_proto="configs/${name}_test.pbtxt" \
    --action_reason_annot_path="data/test/QA_Combined_Action_Reason_test.json" \
    --saved_ckpt_dir="logs/${name}_${ender}/train" \
    --continuous_evaluation="false" \
    --final_results_path="saved_results/${name}_${ender}.json" \
    || exit -1

exit 0
