from collections import defaultdict
import os, random
from typing import Dict, List
import pytorch_lightning as pl
from ast import literal_eval
from utils_package import store_json
from constants import HELPER_PREDICATES

from utils import (
  get_accuracy, get_binding_str, get_color_accuracy, get_smooth_accuracy, 
  get_top_n_accuracy, get_type_accuracy, get_unsolvable_instances
)


class MetricsCB(pl.Callback):
  def __init__(
    self, data: List[Dict],
    print_every: int = None,
    output_folder: str = None, stop_on_acc = False,
    save_preds_every: int = None, model_names: List[str] = None,
    use_curriculum: bool = False, curriculum_threshold: float = 0.9,
  ):
    """A callback that computes metrics after each validation epoch.

    Args:
        data (List[Dict]): A list of the  containing the data
        print_every (int, optional): How often sample predictions should be printed. Defaults to 5.
        output_folder (str, optional): Where to store the predictions. Defaults to None.
        stop_on_acc (bool, optional): If true, training will stop if the accuracy is 1. Defaults to False.
    """
    self.data = data
    self.print_every = print_every
    self.output_folder = output_folder
    self.stop_on_acc = stop_on_acc
    self.save_preds_every = save_preds_every
    self.model_names = model_names
    self.use_curriculum = use_curriculum
    self.curriculum_threshold = curriculum_threshold
    self.no_metrics = data[0].get("binding_costs") is None
    self.curr_model_idx = 0
    # Counter for counting the streak of epochs with 100% accuracy
    self.acc_counter = 0
    self.best_acc = 0

  def get_metrics(self, pl_module):
    metrics = {}
    metrics["accuracy"] = get_accuracy(
      pl_module.all_predictions, self.data
    )
    metrics["color accuracy"] = get_color_accuracy(
      pl_module.all_predictions, self.data
    )
    metrics["type_accuracy"] = get_type_accuracy(
      pl_module.all_predictions, self.data
    )
    metrics["smooth_accuracy"] = get_smooth_accuracy(
      pl_module.all_predictions, self.data
    )
    metrics["top_5_accuracy"] = get_top_n_accuracy(
      pl_module.all_predictions, self.data, n=5
    )
    metrics["unsolvable"] = get_unsolvable_instances(
      pl_module.all_predictions, self.data
    )
    return metrics
  
  def get_incorrect_predictions(self, all_predictions: dict):
    result = []
    for key, pred in all_predictions.items():
      for d in self.data:
        if key == d["id"]:
          if not self.is_correct(pred, d["binding_costs"]):
            result.append({
              **d,
              "pred": pred,
            })
    return result
  
  def print_predictions(self, all_predictions: dict):
    incorrect_predictions = self.get_incorrect_predictions(all_predictions)
    if len(incorrect_predictions) == 0:
      print("No incorrect predictions")
      return
    nr_samples = min(2, len(incorrect_predictions))
    rand_samples = random.choices(incorrect_predictions, k=nr_samples)
    print("====================================")
    print("Printing incorrect predictions")
    print("====================================")
    print()
    for d in rand_samples:
      min_cost_goals = self.get_min_cost_goals(d["binding_costs"])
      print("Prediction: ", d["pred"])
      print("Quant goal: ", d["quant_goal"])
      for i, binding in enumerate(min_cost_goals):
        print(f"Target binding {i+1}: ", binding)
      print("Initial state: ", d["init_state"])
      print("====================================")
      print()
      
  def get_min_cost_goals(self, binding_costs: dict):
    min_cost = min(binding_costs.values())
    return [goal for goal, cost in binding_costs.items() if cost == min_cost]

  def is_correct(self, prediction: dict, binding_costs: dict):
    min_cost = min(binding_costs.values())
    pred_str = get_binding_str(prediction)
    cost = binding_costs.get(pred_str)
    if cost is None:
      return False
    return True if cost == min_cost else False
    
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
      pred_goal = []
      for predicate in literal_eval(d["quant_goal"]):
        ground_pred = []
        for elem in predicate:
          if elem in HELPER_PREDICATES:
            break
          if elem in pred:
            ground_pred.append(pred[elem])
          else:
            ground_pred.append(elem)
        if len(ground_pred) > 0:
          pred_goal.append(tuple(ground_pred))
      pred_str = get_binding_str(pred)
      if self.no_metrics:
        pred_data.append({
          **d,
          "prediction": pred_str,
          "pred_goal": str(pred_goal),
        })
      else:
        pred_cost = d["binding_costs"].get(pred_str)
        min_cost = min(d["binding_costs"].values())
        max_cost = max(d["binding_costs"].values())
        pred_data.append({
          **d,
          "prediction": pred_str,
          "pred_cost": pred_cost,
          "min_cost": min_cost,
          "max_cost": max_cost,
          "pred_goal": str(pred_goal),
          "correct": self.is_correct(pred, d["binding_costs"]),
        })
    store_json(pred_data, output_file)
    if not self.no_metrics:
      print(f"Nr of incorrect predictions: {len([d for d in pred_data if not d['correct']])}")
      self.store_incorrect_predictions(pred_data)
    
  def store_incorrect_predictions(self, pred_data: list):
    if not self.output_folder:
      raise ValueError("Output folder need to be specified to store incorrect predictions")
    if self.model_names:
      output_file = os.path.join(self.output_folder, f"{self.model_names[self.curr_model_idx]}/incorrect_preds.json")
    else:
      output_file = os.path.join(self.output_folder, "incorrect_preds.json")
    incorrect_preds = [d for d in pred_data if not d["correct"]]
    store_json(incorrect_preds, output_file)
        
  def store_metrics(self, all_predictions: dict, metrics: dict):
    if not self.output_folder:
      raise ValueError("Output folder need to be specified to store metrics")
    if self.model_names:
      output_file = os.path.join(self.output_folder, f"{self.model_names[self.curr_model_idx]}/metrics.json")
    else:
      output_file = os.path.join(self.output_folder, "metrics.json")
    
    obj_count = defaultdict(int)
    obj_correct_count = defaultdict(int)
    var_count = defaultdict(int)
    var_correct_count = defaultdict(int)
    
    for pred_id in all_predictions:
      d = [d for d in self.data if pred_id == d["id"]][0]
      pred = all_predictions[pred_id]
      correct = self.is_correct(pred, d["binding_costs"])
      
      nr_objs = d["nr_objects"]
      nr_vars = d["nr_variables"]
      
      obj_count[nr_objs] += 1
      if not nr_objs in obj_correct_count:
        obj_correct_count[nr_objs] = 0
      if correct:
        obj_correct_count[nr_objs] += 1
      var_count[nr_vars] += 1
      if correct:
        var_correct_count[nr_vars] += 1
    
    obj_acc = {k: v / obj_count[k] for k, v in obj_correct_count.items()}
    var_acc = {k: v / var_count[k] for k, v in var_correct_count.items()}

    var_obj_smooth_acc = self.calc_var_obj_smoot_acc(all_predictions)
    
    metrics["obj_acc"] = obj_acc
    metrics["var_acc"] = var_acc
    metrics["obj_count"] = obj_count
    metrics["var_obj_smooth_acc"] = var_obj_smooth_acc
    
    store_json(metrics, output_file)
  
  def calc_var_obj_smoot_acc(self, all_predictions: dict):
    var_obj_count = defaultdict(int)
    var_obj_acc = defaultdict(int)

    for pred_id in all_predictions:
      d = [d for d in self.data if pred_id == d["id"]][0]
      pred = all_predictions[pred_id]
      pred_cost = d["binding_costs"].get(get_binding_str(pred))
      opt_cost = min(d["binding_costs"].values())
      if pred_cost is None:
        acc = 0
      elif pred_cost == opt_cost:
        acc = 1
      else:
        acc = opt_cost / pred_cost
      var_obj_count[(d["nr_objects"], d["nr_variables"])] += 1
      var_obj_acc[(d["nr_objects"], d["nr_variables"])] += acc
    
    return {str(k): v / var_obj_count[k] for k, v in var_obj_acc.items()}

  def on_validation_epoch_end(self, trainer: pl.Trainer, pl_module):
    if self.print_every and \
       trainer.current_epoch % self.print_every == 0 and \
       trainer.current_epoch > 0:
      self.print_predictions(pl_module.all_predictions)

    if self.save_preds_every and \
       trainer.current_epoch % self.save_preds_every == 0 and \
       trainer.current_epoch > 0:
      self.store_predictions(pl_module.all_predictions)
    
    metrics = self.get_metrics(pl_module)
    self.log("val_acc", metrics["accuracy"], on_step=False, 
             on_epoch=True, prog_bar=True, logger=True)
    self.log("val_color_acc", metrics["color accuracy"], on_step=False, 
             on_epoch=True, prog_bar=False, logger=True)
    self.log("val_type_acc", metrics["type_accuracy"], on_step=False,
             on_epoch=True, prog_bar=False, logger=True)
    self.log("val_smooth_acc", metrics["smooth_accuracy"], on_step=False,
             on_epoch=True, prog_bar=False, logger=True)
    self.log("val_top_5_acc", metrics["top_5_accuracy"], on_step=False,
             on_epoch=True, prog_bar=True, logger=True)

    if metrics["accuracy"] > self.best_acc:
      self.best_acc = metrics["accuracy"]
      print(f"New best accuracy: {self.best_acc}")
      self.store_predictions(pl_module.all_predictions)
      self.store_metrics(pl_module.all_predictions, metrics)
    
    pl_module.reset_stored_data()

    streak_len = 10
    if int(metrics["accuracy"]) == 1:
      self.acc_counter += 1
    else:
      self.acc_counter = 0

    if self.acc_counter >= streak_len:
      print(f"Stopping training because of 100% accuracy for {streak_len} epochs")
      trainer.should_stop = True

    pl_module.reset_stored_data()
    
  def on_test_epoch_end(self, trainer: pl.Trainer, pl_module):
    if self.no_metrics:
      self.store_predictions(pl_module.all_predictions)
      pl_module.reset_stored_data()
      self.curr_model_idx += 1
      return
  
    metrics = self.get_metrics(pl_module)

    self.log("test_acc", metrics["accuracy"], on_step=False, 
             on_epoch=True, prog_bar=True, logger=True)
    self.log("test_color_acc", metrics["color accuracy"], on_step=False, 
             on_epoch=True, prog_bar=False, logger=True)
    self.log("test_type_acc", metrics["type_accuracy"], on_step=False,
             on_epoch=True, prog_bar=False, logger=True)
    self.log("test_smooth_acc", metrics["smooth_accuracy"], on_step=False,
             on_epoch=True, prog_bar=False, logger=True)
    self.log("test_top_5_acc", metrics["top_5_accuracy"], on_step=False,
             on_epoch=True, prog_bar=True, logger=True)
    self.log("test_unsolvable", metrics["unsolvable"], on_step=False,
             on_epoch=True, prog_bar=False, logger=True)

    self.store_predictions(pl_module.all_predictions)
    self.store_metrics(pl_module.all_predictions, metrics)

    if self.model_names:
      print("====================================")
      print(f"Metrics for model: '{self.model_names[self.curr_model_idx]}'")
      print("====================================")

      self.curr_model_idx += 1

    pl_module.reset_stored_data()
