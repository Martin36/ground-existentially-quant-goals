import os
import torch
import pytorch_lightning as pl
from glob import glob
from torch.utils.data import DataLoader

from args.testing_args import TestingArgs
from utils import (
  get_data, get_default_test_output_folder, get_domain_name, natural_keys
)

def filter_data(data, nr_objects):
  if type(data) == list:
    data = [d for d in data if d["nr_objects"] == nr_objects]
  else:
    for key in data:
      data[key] = [d for d in data[key] if d["nr_objects"] == nr_objects]
  return data


if __name__ == "__main__":
  args = TestingArgs().get_args()
  
  data = get_data(args.data_file, args.data_folder)
  if args.nr_objects:
    data = filter_data(data, args.nr_objects)
  domain_name = get_domain_name(data)
  
  device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

  callbacks = []

  if not args.output_dir:
    if args.data_file:
      output_folder = get_default_test_output_folder(
        args.model_type, args.data_file, nr_objects=args.nr_objects
      )
    else:
      output_folder = get_default_test_output_folder(
        args.model_type, args.data_folder, nr_objects=args.nr_objects
      )
  else:
    output_folder = args.output_dir

  if args.model_type == "val_pred":
    from val_pred import *

    dataset = Dataset(
      data, domain_name, is_val=True
    )

    model_names = None
    if args.models_folder:
      model_paths = glob(os.path.join(args.models_folder, "*.ckpt"))
      model_paths.sort(key=natural_keys)
      model_names = [os.path.basename(path) for path in model_paths]
      model_names = [name.split(".")[0] for name in model_names]
      models = []
      for path in model_paths:
        models.append(
          Model.load_from_checkpoint(path, map_location=device)
        )
    else:
      model = Model.load_from_checkpoint(args.model_path, map_location=device)

    callbacks.append(MetricsCB(
      data, output_folder=output_folder, model_names=model_names
    ))

  if args.model_type == "val_sub" and args.random_sub:
    from val_sub import *
    from rand.model import Model as RandModel
    from metrics_cb import MetricsCB

    model = RandModel(
      domain_name=domain_name, constrained_bindings=args.constrained_bindings
    )

    dataset = Dataset(data, domain_name)

    callbacks.append(
      MetricsCB(
        data, output_folder=output_folder
      )
    )
  elif args.model_type == "val_sub":
    from val_sub import *
    from val_pred import Model as ValPredModel
    from metrics_cb import MetricsCB

    dataset = Dataset(data, domain_name)
    predicates = dataset.predicates

    model_names = None
    if args.models_folder:
      model_paths = glob(os.path.join(args.models_folder, "*.ckpt"))
      if len(model_paths) == 0:
        raise ValueError(f"No model found in {args.models_folder}")
      model_paths.sort(key=natural_keys)
      model_names = [os.path.basename(path) for path in model_paths]
      model_names = [name.split(".")[0] for name in model_names]
      val_pred_models = []
      for path in model_paths:
        val_pred_models.append(
          ValPredModel.load_from_checkpoint(path, map_location=device)
        )
      models = []
      for val_pred_model in val_pred_models:
        models.append(
          Model(predicates, domain_name, val_pred_model, device=device)
        )
    else:
      val_pred_model = ValPredModel.load_from_checkpoint(args.model_path, map_location=device)
      model = Model(predicates, domain_name, val_pred_model, device)
    
    callbacks.append(
      MetricsCB(
        data, output_folder=output_folder, model_names=model_names
      )
    )

  dataloader = DataLoader(
    dataset, batch_size=args.batch_size, collate_fn=collate_fn
  )
  
  if not os.path.exists(output_folder):
    os.makedirs(output_folder)
  
  trainer = pl.Trainer(
    accelerator=str(device),
    auto_lr_find=True,
    callbacks=callbacks,
    devices=torch.cuda.device_count() if device == "cuda" else None,
    num_sanity_val_steps=0,
  )
  
  if args.models_folder:
    for model in models:
      trainer.test(model, dataloader)
  else:
    trainer.test(model, dataloader)
  
  