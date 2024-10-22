import argparse
from ast import literal_eval
from collections import defaultdict
import os
import pandas as pd
from utils_package import load_json, store_json

from constants import DOMAIN
from utils import (
  convert_binding_str_to_dict, get_domain_from_atoms, 
  get_goal_pred_from_quant_goal, is_unsolvable
)

if __name__ == "__main__":
  arg_parser = argparse.ArgumentParser()
  arg_parser.add_argument("--input_file", type=str, required=True, help="Path to the input file")
  # arg_parser.add_argument("--output_file", type=str, required=True, help="Path to the output file")
  # arg_parser.add_argument("--unsolvable_file", type=str, help="Path to the unsolvable file")
  args = arg_parser.parse_args()
  
  predictions = load_json(args.input_file)
  
  folder = os.path.dirname(args.input_file)
  output_file = os.path.join(folder, "paper_stats.csv")
  unsolvable_file = os.path.join(folder, "unsolvable.json")
  
  domain = get_domain_from_atoms(literal_eval(predictions[0]["init_state"]))
  goal_pred = get_goal_pred_from_quant_goal(literal_eval(predictions[0]["quant_goal"]))
  
  nr_unsolvable = 0
  unsolvable = []
  total = 0
  avg_opt_cost = 0
  avg_model_cost = 0
  var_const_unsolvable_counter = defaultdict(int)
  var_cost_total_counter = defaultdict(int)
  
  for prediction in predictions:
    total += 1
    bindings = convert_binding_str_to_dict(prediction["prediction"])
    unsol, reason = is_unsolvable(prediction, bindings, goal_pred)
    var_cost_total_counter[(prediction["nr_variables"], prediction["nr_objects"])] += 1
    if unsol:
      nr_unsolvable += 1
      prediction["unsolvable_reason"] = reason
      unsolvable.append(prediction)
      var_const_unsolvable_counter[(prediction["nr_variables"], prediction["nr_objects"])] += 1
    else:
      # How close to optimal is the binding?
      if "pred_cost" in prediction and \
         "min_cost" in prediction:
        avg_opt_cost += prediction["min_cost"]
        avg_model_cost += prediction["pred_cost"]
  
  avg_opt_cost /= total
  avg_model_cost /= total
  
  if domain == DOMAIN.BLOCKS:
    domain = f"{domain}-{goal_pred}"
    
  stats = {
    "domain": [domain],
    "#": [total],
    "% solvable": [(total - nr_unsolvable) / total],
    "nr_solvable": [total - nr_unsolvable],
    "ref_cost": [avg_opt_cost],
    "quality": avg_model_cost / avg_opt_cost
  }
  
  stats_df = pd.DataFrame(stats)
  stats_df.to_csv(output_file, index=False)
  print("Stored stats in", output_file)
  store_json(unsolvable, unsolvable_file)

  var_const_unsolvable_stats = {
    "var const pair": [],
    "percentage unsolvable": [],
    "total unsolvable": [],
    "total": []
  }

  for key in var_cost_total_counter:
    perc_var_const_unvolvable = var_const_unsolvable_counter[key] / var_cost_total_counter[key]
    var_const_unsolvable_stats["var const pair"].append(key)
    var_const_unsolvable_stats["percentage unsolvable"].append(perc_var_const_unvolvable)
    var_const_unsolvable_stats["total unsolvable"].append(var_const_unsolvable_counter[key])
    var_const_unsolvable_stats["total"].append(var_cost_total_counter[key])
  
  var_const_unsol_file = os.path.join(folder, "var_const_unsolvable.csv")
  var_const_unsolvable_stats_df = pd.DataFrame(var_const_unsolvable_stats)
  var_const_unsolvable_stats_df.sort_values(by="var const pair", inplace=True)
  var_const_unsolvable_stats_df.to_csv(var_const_unsol_file, index=False)
  print("Stored var const unsolvable stats in", var_const_unsol_file)

