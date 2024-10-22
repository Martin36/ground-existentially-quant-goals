set -e

export PYTHONPATH=src

# MODEL_ID=$1
# CKPT_NAME=$2
MODEL_FILE=$1

# MODEL_ID='258a4369b85c4f178eba595c127b6517'
# CKPT_NAME='epoch=2579-step=12900'

DATASET_NAME='6v-17m-500_mc'
DOMAIN='blocks'
GOAL_PRED='on'
MAX_NR_OBJS=17
DOMAIN_FILE='data/blocks_1/domain.pddl'

# MODEL_FILE=logs/val_pred/${MODEL_ID}/checkpoints/${CKPT_NAME}.ckpt
DATA_FILE=data/datasets/${DOMAIN}-${GOAL_PRED}/${DATASET_NAME}/test.json
PREDICTIONS_FOLDER=models/val_sub/${DOMAIN}-${GOAL_PRED}/${DATASET_NAME}
PDDL_OUTPUT_FOLDER=eval/val_pred/${DOMAIN}-${GOAL_PRED}/${DATASET_NAME}
PREDICTIONS_FILE=${PREDICTIONS_FOLDER}/predictions.json

# FD_OUT=fd_out_${MODEL_ID}_${MAX_NR_OBJS}
# SAS_FILE=${MODEL_ID}_${MAX_NR_OBJS}.sas
# PLAN_FILE=sas_plan_${MODEL_ID}${MAX_NR_OBJS}
FD_OUT=fd_out
SAS_FILE=output.sas
PLAN_FILE=sas_plan

# Create data if it does not exist
if ! [ -f "${DATA_FILE}" ]; then
  echo "Data file '${DATA_FILE}' does not exist. Generating data ..."
  python src/data_processing/create_state_space_data.py --domain=${DOMAIN} --dataset_size=500 --goal_pred=${GOAL_PRED} --nr_goals=5 --max_nr_objects=${MAX_NR_OBJS} --nr_colors=6 --max_nr_vars=6 --gen_test_data --recolor_states
fi

# Run model on test data
echo 'Making subsitiontions with model'
sub_start_time=`date +%s`
python src/test.py --model_type=val_sub --data_file=${DATA_FILE} --model_path=${MODEL_FILE} --batch_size=1
sub_end_time=`date +%s`
sub_time=`expr $sub_end_time - $sub_start_time`
echo Execution time for substitutions: $sub_time seconds

# Create PDDL files from predictions
echo 'Creating PDDL files from output predictions'
python src/eval_preds.py --predictions_file=${PREDICTIONS_FILE} --domain_file=${DOMAIN_FILE} --output_folder=${PDDL_OUTPUT_FOLDER}

# Run FD on ground problems
echo 'Running FD on the ground PDDL files'
ground_start_time=`date +%s`
bash scripts/run_fast_downward.sh ${PDDL_OUTPUT_FOLDER} 1 ${FD_OUT} ${SAS_FILE} ${PLAN_FILE} 1 30m 1
ground_end_time=`date +%s`
ground_time=`expr $ground_end_time - $ground_start_time`
echo Execution time for grounded goals: $ground_time seconds

total_ground_time=`expr $ground_time + $sub_time`
echo Execution time including the substitutions: $total_ground_time seconds

# Run FD on quantified problems
echo 'Running FD on the quantified PDDL files'
quant_start_time=`date +%s`
bash scripts/run_fast_downward.sh ${PDDL_OUTPUT_FOLDER} 2 ${FD_OUT} ${SAS_FILE} ${PLAN_FILE} 0 30m 1
quant_end_time=`date +%s`
quant_time=`expr $quant_end_time - $quant_start_time`
echo Execution time for quantified goals: $quant_time seconds

# echo Speedup: `expr $quant_time / $ground_time`
echo Speedup: $(echo $quant_time / $ground_time | bc -l)
# echo Speedup including the substitutions: `expr $quant_time / $total_ground_time`
echo Speedup including the substitutions: $(echo $quant_time / $total_ground_time | bc -l)

# Parse the FD output
echo 'Parsing the FD output...'
python src/data_processing/parse_fast_downward_outputs.py --input_folder=${FD_OUT} --output_folder=${PREDICTIONS_FOLDER} --predictions_file=${PREDICTIONS_FILE}
