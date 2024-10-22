
# Train model on Blocks on 4v-9m-40000_mc
PYTHONPATH=src python src/train.py \
  --model_type=val_pred \
  --model_name=val_pred \
  --data_folder=data/datasets/val_pred/blocks-on/4v-9m-40000_mc \
  --output_dir=models/val_pred/blocks-on/4v-9m-40000_mc \
  --experiment_name=val_pred_p-g_blocks-on_4v-9m-40000_mc \
  --batch_size=1024
