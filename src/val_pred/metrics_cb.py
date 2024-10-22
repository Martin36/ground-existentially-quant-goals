import math
import os
from typing import Dict, List
import pytorch_lightning as pl
from utils_package import store_json

from utils import get_nr_of_variables_from_tuple
from val_pred.model import Model


class MetricsCB(pl.Callback):
  def __init__(
    self, data: Dict[int, List[dict]], output_folder: str = None, 
    model_names: List[str] = None
  ):
    """A callback that computes metrics after each validation epoch.

    Args:
      data (Dict[int, List[dict]]): The data that was used to validate the predictions. It should contain unique ids for each instance.
      output_folder (str, optional): Where to store the predictions. Defaults to None.
    """
    self.data = self.format_data(data)
    self.output_folder = output_folder
    self.model_names = model_names
    self.curr_model_idx = 0
    # Counter for counting the streak of epochs with 100% accuracy
    self.acc_counter = 0
    self.best_avg_error = math.inf
    self.best_acc = 0
    
  def format_data(self, data):
    # In this class we only care about the instances does not need to
    # group them by cost
    return [d for cost in data for d in data[cost]]
        
  def get_metrics(self, pl_module: Model):
    metrics = {}
    metrics["accuracy"] = self.get_accuracy(pl_module.all_predictions)
    metrics["top 1 accuracy"] = self.get_accuracy(
      pl_module.all_predictions, radius=1
    )
    metrics["top 2 accuracy"] = self.get_accuracy(
      pl_module.all_predictions, radius=2
    )
    metrics["top 5 accuracy"] = self.get_accuracy(
      pl_module.all_predictions, radius=5
    )
    metrics["avg error"] = self.get_avg_error(pl_module)
    metrics["accuracy by nr vars"] = self.get_accuracy_by_nr_vars(
      pl_module.all_predictions
    )
    return metrics

  def get_accuracy(self, predictions: Dict[str, Dict], radius=0):
    if len(predictions) == 0:
      print("No predictions made")
      return 1
    total_acc = 0
    for id in predictions:
      d = [d for d in self.data if id == d["id"]][0]
      pred = predictions[id]
      label = d["cost"]
      if pred >= label - radius and pred <= label + radius:
        total_acc += 1
    return total_acc / len(predictions)
  
  def get_accuracy_by_nr_vars(self, predictions: Dict[str, Dict]):
    result = {}
    radii = [1, 2, 5]
    preds_grouped_by_nr_vars = {}
    for id in predictions:
      d = [d for d in self.data if id == d["id"]][0]
      nr_vars = get_nr_of_variables_from_tuple(eval(d["quant_goal"]))
      if nr_vars not in preds_grouped_by_nr_vars:
        preds_grouped_by_nr_vars[nr_vars] = {}
      pred = predictions[id]
      preds_grouped_by_nr_vars[nr_vars][id] = pred

    for nr_vars in preds_grouped_by_nr_vars:
      if nr_vars not in result:
        result[nr_vars] = {}
      for radius in radii:
        result[nr_vars][radius] = self.get_accuracy(
          preds_grouped_by_nr_vars[nr_vars], radius
        )
    
    return result
      
  def get_avg_error(self, pl_module: Model):
    if len(pl_module.all_predictions) == 0:
      print("No predictions made")
      return 0
    total_error = 0
    for id in pl_module.all_predictions:
      d = [d for d in self.data if id == d["id"]][0]
      pred = pl_module.all_predictions[id]
      label = d["cost"]
      total_error += abs(pred - label)
    return total_error / len(pl_module.all_predictions)
            
  def store_predictions(self, all_predictions: dict):
    if not self.output_folder:
      raise ValueError("Output folder need to be specified to store predictions")
    if self.model_names:
      output_file = os.path.join(self.output_folder, f"{self.model_names[self.curr_model_idx]}/predictions.json")
    else:
      output_file = os.path.join(self.output_folder, "predictions.json")
    pred_data = []
    for pred_id in all_predictions:
      d = [d for d in self.data if pred_id == d["id"]][0]
      pred = all_predictions[pred_id]
      pred_data.append({
        **d,
        "pred_cost": pred,
      })
    store_json(pred_data, output_file)
        
  def store_metrics(self, metrics: dict):
    if not self.output_folder:
      raise ValueError("Output folder need to be specified to store metrics")
    if self.model_names:
      output_file = os.path.join(self.output_folder, f"{self.model_names[self.curr_model_idx]}/metrics.json")
    else:
      output_file = os.path.join(self.output_folder, "metrics.json")
    store_json(metrics, output_file)
    
  def on_validation_epoch_end(self, trainer: pl.Trainer, pl_module: Model):    
    metrics = self.get_metrics(pl_module)
    self.log("val_acc", metrics["accuracy"], on_step=False, 
             on_epoch=True, prog_bar=True, logger=True)
    self.log("val_top_1_acc", metrics["top 1 accuracy"], on_step=False, 
            on_epoch=True, prog_bar=True, logger=True)
    self.log("val_top_2_acc", metrics["top 2 accuracy"], on_step=False, 
            on_epoch=True, prog_bar=True, logger=True)
    self.log("val_top_5_acc", metrics["top 5 accuracy"], on_step=False, 
            on_epoch=True, prog_bar=True, logger=True)
    self.log("val_avg_error", metrics["avg error"], on_step=False,
              on_epoch=True, prog_bar=True, logger=True)

    if metrics["avg error"] < self.best_avg_error:
      self.best_avg_error = metrics["avg error"]
      print(f"New best avg error: {self.best_avg_error}")
      self.store_predictions(pl_module.all_predictions)
      self.store_metrics(metrics)
    
    if metrics["accuracy"] < self.best_acc:
      self.best_acc = metrics["accuracy"]
      print(f"New best accuracy: {self.best_acc}")
    
    streak_len = 10
    if int(metrics["accuracy"]) == 1:
      self.acc_counter += 1
    else:
      self.acc_counter = 0

    if self.acc_counter >= streak_len:
      print(f"Stopping training because of 100% accuracy for {streak_len} epochs")
      trainer.should_stop = True

    pl_module.reset_stored_data()
                          
  def on_test_epoch_end(self, trainer: pl.Trainer, pl_module: Model):
    metrics = self.get_metrics(pl_module)
    self.log("test_acc", metrics["accuracy"], on_step=False, 
             on_epoch=True, prog_bar=True, logger=True)
    self.log("test_top_1_acc", metrics["top 1 accuracy"], on_step=False, 
            on_epoch=True, prog_bar=True, logger=True)
    self.log("test_top_2_acc", metrics["top 2 accuracy"], on_step=False, 
            on_epoch=True, prog_bar=True, logger=True)
    self.log("test_top_5_acc", metrics["top 5 accuracy"], on_step=False, 
            on_epoch=True, prog_bar=True, logger=True)
    self.log("test_avg_error", metrics["avg error"], on_step=False,
              on_epoch=True, prog_bar=True, logger=True)

    self.store_predictions(pl_module.all_predictions)
    self.store_metrics(metrics)

    if self.model_names:
      print("====================================")
      print(f"Metrics for model: '{self.model_names[self.curr_model_idx]}'")
      print("====================================")

      self.curr_model_idx += 1
        
    pl_module.reset_stored_data()
