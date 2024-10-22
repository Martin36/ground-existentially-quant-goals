import argparse
from collections import defaultdict
import math

from utils_package import load_json, store_json


if __name__ == '__main__':
  arg_parser = argparse.ArgumentParser()
  arg_parser.add_argument(
    "--input_file",
    type=str,
    required=True,
    help="Path to the input file"
  )
  arg_parser.add_argument(
    "--output_file",
    type=str,
    required=True,
    help="Path to the output file"
  )
  
  args = arg_parser.parse_args()
  
  predictions = load_json(args.input_file)
  
  stats = {}
  var_obj_stats = {
    "count": defaultdict(int),
    "pred_cost": defaultdict(int),
    "min_cost": defaultdict(int),
    "max_cost": defaultdict(int),
    "fd_quant_cost": defaultdict(int),
    "fd_ground_cost": defaultdict(int),
    "nr_unsolvable": defaultdict(int),
    "smooth_acc": defaultdict(int),
    "fd_smooth_acc": defaultdict(int),
  }
  nr_bindings_count = {
    "total": defaultdict(int)
  }
  
  for prediction in predictions:
    var_obj_tup_str = str((prediction["nr_variables"], prediction["nr_objects"]))
    var_obj_stats["count"][var_obj_tup_str] += 1
    # TODO: How to handle the costs of the unsolvable cases?
    # Current approach set the cost to infinity
    pred_cost = prediction["pred_cost"] if prediction["pred_cost"] is not None else math.inf
    var_obj_stats["pred_cost"][var_obj_tup_str] += pred_cost
    var_obj_stats["min_cost"][var_obj_tup_str] += prediction["min_cost"]
    var_obj_stats["max_cost"][var_obj_tup_str] += prediction["max_cost"]
    var_obj_stats["fd_quant_cost"][var_obj_tup_str] += prediction["fast_downward"]["quant"]
    # TODO: How to handle the costs of the unsolvable cases?
    # Current approach set the cost to infinity
    fd_ground_cost = prediction["fast_downward"]["ground"] if prediction["fast_downward"]["ground"] is not None else math.inf
    var_obj_stats["fd_ground_cost"][var_obj_tup_str] += fd_ground_cost
    if prediction["pred_cost"] is None:
      var_obj_stats["nr_unsolvable"][var_obj_tup_str] += 1
    
    if prediction["pred_cost"] is None:
      smooth_acc = 0
    elif prediction["pred_cost"] <= prediction["min_cost"]:
      smooth_acc = 1
    else:
      smooth_acc = prediction["min_cost"] / prediction["pred_cost"]
    var_obj_stats["smooth_acc"][var_obj_tup_str] += smooth_acc
    
    fd_quant_cost = prediction["fast_downward"]["quant"]
    if fd_quant_cost is None:
      fd_smooth_acc = 0
    elif fd_quant_cost <= prediction["min_cost"]:
      # TODO: What if the cost is less than the min cost?
      fd_smooth_acc = 1
    else:
      fd_smooth_acc = prediction["min_cost"] / fd_quant_cost
    var_obj_stats["fd_smooth_acc"][var_obj_tup_str] += fd_smooth_acc
    
    nr_bindings = len(prediction["binding_costs"])
    if var_obj_tup_str not in nr_bindings_count:
      nr_bindings_count[var_obj_tup_str] = defaultdict(int)  
    nr_bindings_count[var_obj_tup_str][nr_bindings] += 1
    nr_bindings_count["total"][nr_bindings] += 1
    
  total_count = sum(nr_bindings_count["total"].values())
  average = 0 
  for key in nr_bindings_count["total"]:
    average += key * nr_bindings_count["total"][key]
  average /= total_count
  nr_bindings_count["total"]["average"] = average
  
  # Calculate average binding count for each pair
  for pair in nr_bindings_count:
    if pair == "total":
      continue
    total_count = sum(nr_bindings_count[pair].values())
    average = 0 
    for key in nr_bindings_count[pair]:
      average += key * nr_bindings_count[pair][key]
    average /= total_count
    nr_bindings_count[pair]["average"] = average
    
  # Convert all keys to strings
  for pair in nr_bindings_count:
    nr_bindings_count[pair] = {str(k): v for k, v in nr_bindings_count[pair].items()}
  
  for pair in var_obj_stats["count"]:
    count = var_obj_stats["count"][pair]
    var_obj_stats["pred_cost"][pair] /= count
    var_obj_stats["min_cost"][pair] /= count
    var_obj_stats["max_cost"][pair] /= count
    var_obj_stats["fd_quant_cost"][pair] /= count
    var_obj_stats["fd_ground_cost"][pair] /= count
    var_obj_stats["smooth_acc"][pair] /= count
    var_obj_stats["fd_smooth_acc"][pair] /= count
    
  stats["var_obj_stats"] = var_obj_stats
  stats["nr_bindings_count"] = nr_bindings_count
  
  store_json(stats, args.output_file, sort_keys=True)
    
    
  
  