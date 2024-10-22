import argparse
from collections import defaultdict
import os
import pandas as pd
from utils_package import load_json

if __name__ == "__main__":
  arg_parser = argparse.ArgumentParser()
  arg_parser.add_argument("--input_file", type=str, required=True, help="Path to the input file")
  args = arg_parser.parse_args()
  
  predictions = load_json(args.input_file)
  folder = os.path.dirname(args.input_file)
  output_file = os.path.join(folder, "paper_stats_fd.csv")
      
  const_stats = {}
  
  for prediction in predictions:
    nr_objs = prediction["nr_objects"]
    if nr_objs not in const_stats:
      const_stats[nr_objs] = defaultdict(int)
    const_stats[nr_objs]["#"] += 1
    
    if prediction["fast_downward"]["quant"]["cost"] is not None:
      const_stats[nr_objs]["fd coverage"] += 1
      const_stats[nr_objs]["ref cost"] += prediction["fast_downward"]["quant"]["cost"]
      const_stats[nr_objs]["fd solvable"] += 1
    if prediction["fast_downward"]["ground"]["cost"] is not None:
      const_stats[nr_objs]["coverage"] += 1
      const_stats[nr_objs]["quality"] += prediction["fast_downward"]["ground"]["cost"]
      const_stats[nr_objs]["cost"] += prediction["fast_downward"]["ground"]["cost"]
      const_stats[nr_objs]["model solvable"] += 1
  
  for key in const_stats:
    assert const_stats[key]["coverage"] == const_stats[key]["model solvable"]
    # Need to normalize the ref cost and quality
    const_stats[key]["ref cost"] /= const_stats[key]["fd solvable"]
    const_stats[key]["cost"] /= const_stats[key]["model solvable"]
    if not const_stats[key]["ref cost"]:
      const_stats[key]["quality"] = 0.0
    else:
      const_stats[key]["quality"] /= const_stats[key]["model solvable"]
      const_stats[key]["quality"] /= const_stats[key]["ref cost"]
       
  const_stats_csv = {
    "nr objects": [],
    "#": [],
    "coverage": [],
    "fd coverage": [],
    "ref cost": [],
    "cost": [],
    "quality": []
  }

  for key in const_stats:
    const_stats_csv["nr objects"].append(key)
    const_stats_csv["#"].append(const_stats[key]["#"])
    const_stats_csv["coverage"].append(const_stats[key]["coverage"])
    const_stats_csv["fd coverage"].append(const_stats[key]["fd coverage"])
    const_stats_csv["ref cost"].append(const_stats[key]["ref cost"])
    const_stats_csv["cost"].append(const_stats[key]["cost"])
    const_stats_csv["quality"].append(const_stats[key]["quality"])

  const_stats_df = pd.DataFrame(const_stats_csv)
  const_stats_df.sort_values(by="nr objects", inplace=True)
  const_stats_df.to_csv(output_file, index=False)
  print("Stored const stats in", output_file)

  # Calculate (var, const) stats
  var_const_stats = {}

  for prediction in predictions:
    nr_vars = prediction["nr_variables"]
    nr_objs = prediction["nr_objects"]
    if (nr_vars, nr_objs) not in var_const_stats:
      var_const_stats[(nr_vars, nr_objs)] = defaultdict(int)
    var_const_stats[(nr_vars, nr_objs)]["#"] += 1
    
    if prediction["fast_downward"]["quant"]["cost"] is not None:
      var_const_stats[(nr_vars, nr_objs)]["fd coverage"] += 1
      var_const_stats[(nr_vars, nr_objs)]["ref cost"] += prediction["fast_downward"]["quant"]["cost"]
      var_const_stats[(nr_vars, nr_objs)]["fd solvable"] += 1
    if prediction["fast_downward"]["ground"]["cost"] is not None:
      var_const_stats[(nr_vars, nr_objs)]["coverage"] += 1
      var_const_stats[(nr_vars, nr_objs)]["quality"] += prediction["fast_downward"]["ground"]["cost"]
      var_const_stats[(nr_vars, nr_objs)]["cost"] += prediction["fast_downward"]["ground"]["cost"]
      var_const_stats[(nr_vars, nr_objs)]["model solvable"] += 1

  for key in var_const_stats:
    assert var_const_stats[key]["coverage"] == var_const_stats[key]["model solvable"]
    # Need to normalize the ref cost and quality
    if var_const_stats[key]["fd solvable"] == 0:
      var_const_stats[key]["ref cost"] = None
    else:
      var_const_stats[key]["ref cost"] /= var_const_stats[key]["fd solvable"]
    if var_const_stats[key]["model solvable"] == 0:
      var_const_stats[key]["cost"] = None
    else:
      var_const_stats[key]["cost"] /= var_const_stats[key]["model solvable"]
    if not var_const_stats[key]["ref cost"]:
      var_const_stats[key]["quality"] = None
    else:
      var_const_stats[key]["quality"] /= var_const_stats[key]["model solvable"]
      var_const_stats[key]["quality"] /= var_const_stats[key]["ref cost"]

  var_const_stats_csv = {
    "nr variables": [],
    "nr objects": [],
    "#": [],
    "coverage": [],
    "fd coverage": [],
    "ref cost": [],
    "cost": [],
    "quality": []
  }

  for key in var_const_stats:
    var_const_stats_csv["nr variables"].append(key[0])
    var_const_stats_csv["nr objects"].append(key[1])
    var_const_stats_csv["#"].append(var_const_stats[key]["#"])
    var_const_stats_csv["coverage"].append(var_const_stats[key]["coverage"])
    var_const_stats_csv["fd coverage"].append(var_const_stats[key]["fd coverage"])
    var_const_stats_csv["ref cost"].append(var_const_stats[key]["ref cost"])
    var_const_stats_csv["cost"].append(var_const_stats[key]["cost"])
    var_const_stats_csv["quality"].append(var_const_stats[key]["quality"])

  var_const_stats_df = pd.DataFrame(var_const_stats_csv)
  var_const_stats_df.sort_values(by=["nr variables", "nr objects"], inplace=True)
  var_const_stats_file = os.path.join(folder, "var_const_stats_fd.csv")
  var_const_stats_df.to_csv(var_const_stats_file, index=False)
  print("Stored var const stats in", var_const_stats_file)