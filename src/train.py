from ast import literal_eval
import comet_ml
import os, torch
import pytorch_lightning as pl
from utils_package import load_json
from torch.utils.data import DataLoader
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping, LearningRateMonitor
from pytorch_lightning.loggers import CometLogger, CSVLogger

from args.training_args import TrainingArgs
from utils import get_domain_from_atoms
from val_pred import *


if __name__ == "__main__":
  args = TrainingArgs().get_args()
  
  train_data = load_json(os.path.join(args.data_folder, "train.json"))
  val_data = load_json(os.path.join(args.data_folder, "dev.json"))
  
  domain_name = get_domain_from_atoms(literal_eval(
    train_data[list(train_data.keys())[0]][0]["init_state"]
  ))

  device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
  callbacks = []
  hyperparams = {
    "model_type": args.model_type,
    "domain_name": domain_name,
    "model_name": args.model_name,
    "batch_size": args.batch_size,
    "dropout": args.dropout,
    "train_size": len(train_data),
    "val_size": len(val_data),
  }

  monitored_metric = "val_loss"

  callbacks.append(MetricsCB(val_data, output_folder=args.output_dir))
  
  if args.patience:
    callbacks.append(EarlyStopping(monitor=monitored_metric, patience=args.patience))
    
  train_dataset = Dataset(
    train_data, domain_name
  )
  val_dataset = Dataset(
    val_data, domain_name, is_val=True
  )
  predicates = train_dataset.predicates
    
  model = Model(
    predicates, use_readout=args.use_readout
  ).to(device)
         
  if not args.debug and os.environ.get("COMET_API_KEY"):
    logger = CometLogger(
      api_key=os.environ["COMET_API_KEY"],
      project_name=args.model_type,
      save_dir="logs",
      experiment_name=args.experiment_name,
    )
    logger.log_hyperparams(hyperparams)
  else:
    logger = CSVLogger("logs", name=args.model_type)

  if not os.path.exists(args.output_dir):
    os.makedirs(args.output_dir)

  if args.model_path:
    model = Model.load_from_checkpoint(args.model_path, map_location=device)

  callbacks.append(ModelCheckpoint(save_top_k=3, monitor=monitored_metric))
  callbacks.append(LearningRateMonitor("epoch"))

  trainer = pl.Trainer(
    accelerator=str(device),
    callbacks=callbacks,
    devices=torch.cuda.device_count() if device == "cuda" else None,
    logger=logger,
    num_sanity_val_steps=0,
    max_epochs=200000,
    gradient_clip_val=0.5 if args.clip_grad else 0,
  )
  
  train_dataloader = DataLoader(
    train_dataset, batch_size=args.batch_size, #shuffle=True,
    collate_fn=collate_fn, num_workers=args.num_workers
  )
  val_dataloader = DataLoader(
    val_dataset, batch_size=args.batch_size,# shuffle=True
    collate_fn=collate_fn, num_workers=args.num_workers
  )
  
  trainer.fit(
    model,
    train_dataloader,
    val_dataloader,
  )

