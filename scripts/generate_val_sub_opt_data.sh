
# Blocks on, optimal test data
PYTHONPATH=src python src/data_processing/create_state_space_data.py \
  --dataset_size=500 \
  --goal_pred=on \
  --nr_goals=5 \
  --domain=blocks \
  --max_nr_objects=9 \
  --max_nr_vars=6 \
  --nr_obj_split=7,8,9 \
  --split_sizes=0,500,0 \
  --nr_colors=6 \
  --recolor_states
