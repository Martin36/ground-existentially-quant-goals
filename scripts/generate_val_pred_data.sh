
# Blocks on, value prediction data
PYTHONPATH=src python src/data_processing/create_supervised_data.py \
  --domain=blocks \
  --dataset_size=40000 \
  --goal_pred=on \
  --nr_goals=3 \
  --max_nr_objects=9 \
  --nr_colors=4 \
  --max_nr_vars=4 \
  --nr_obj_split=7,8,9 \
  --split_sizes=39500,500,0 \
  --max_expanded_states=10000000 \
  --partially_grounded \
  --recolor_states
