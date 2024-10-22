import argparse
import os
from glob import glob
from typing import Dict, List
from utils_package import store_json, load_json
  
  
def get_smooth_acc(predictions: List[Dict]):
  total_acc = 0
  for prediction in predictions:
    min_cost = prediction["fast_downward"]["quant"]["cost"]
    if min_cost is None:
      # This means that LAMA did not terminate within the time limit
      continue
    ground = prediction["fast_downward"].get("ground")
    if ground is None:
      acc = 0
      continue
    cost = ground.get("cost")
    if cost is None:
      acc = 0
    elif cost <= min_cost:
      acc = 1
    else:
      acc = min_cost / cost
    total_acc += acc
  total_acc /= len(predictions)
  return total_acc

  
if __name__ == "__main__":
  arg_parser = argparse.ArgumentParser()
  arg_parser.add_argument(
    "--input_folder",
    type=str,
    required=True,
    help="Path to the folder containing the Fast Downward outputs"
  )
  arg_parser.add_argument(
    "--output_folder",
    type=str,
    required=True,
    help="Path to the output folder"
  )
  arg_parser.add_argument(
    "--predictions_file",
    type=str,
    help="Path to the file containing the predictions"
  )
  args = arg_parser.parse_args()

  input_files = glob(os.path.join(args.input_folder, "*.txt"))
  result = {}
  for file in input_files:
    file_name = os.path.basename(file).split(".")[0]
    file_id = file_name.split("_")[0]
    is_quant = "quant" in file_name
    
    with open(file, "r") as f:
      lines = f.readlines()
      cost = None
      reached_time_limit = False
      reached_memory_limit = False
      for line in lines:
        if "Plan cost:" in line:
          cost = int(line.split(" ")[-1])
          break
        if "hit the time limit" in line:
          reached_time_limit = True
          break
        if "exit code: 21" in line:
          reached_time_limit = True
          break
        if "exit code: 23" in line:
          reached_time_limit = True
          break
        if "exit code: 20" in line:
          # Translate out of memory
          reached_memory_limit = True
        if "exit code: 22" in line:
          # Search out of memory
          reached_memory_limit = True
      
    if file_id not in result:
      result[file_id] = {
        "quant": {},
        "ground": {},
      }
    
    if is_quant:
      result[file_id]["quant"]["cost"] = cost
      result[file_id]["quant"]["reached_time_limit"] = reached_time_limit
      result[file_id]["quant"]["reached_memory_limit"] = reached_memory_limit
    else:
      result[file_id]["ground"]["cost"] = cost
      result[file_id]["ground"]["reached_time_limit"] = reached_time_limit
      result[file_id]["ground"]["reached_memory_limit"] = reached_memory_limit
  
  # If the ground file is missing, set the cost to None
  for id in result:
    if not result[id]["ground"]:
      result[id]["ground"]["cost"] = None
      result[id]["ground"]["reached_time_limit"] = False
      result[id]["ground"]["reached_memory_limit"] = False
  
  if not os.path.exists(args.output_folder):
    os.makedirs(args.output_folder)
    
  store_json(result, os.path.join(args.output_folder, "fd_outputs.json"))
  
  if args.predictions_file:
    predictions = load_json(args.predictions_file)
    if not any([pred["id"] in result for pred in predictions]):
      raise ValueError("The predictions file and FD outputs are probably on different instances")
    for prediction in predictions:
      if prediction["id"] not in result:
        # TODO: This should not happen since the quantified instance should 
        # always be generated
        result[prediction["id"]] = {
          "ground": {
            "cost": None,
            "reached_time_limit": False,
          },
          "quant": {
            "cost": None,
            "reached_time_limit": False,
          }
        }
      prediction["fast_downward"] = result[prediction["id"]]
    new_predictions_file = os.path.join(args.output_folder, "predictions_w_fd.json")
    store_json(predictions, new_predictions_file)
  
    # Calculate metrics
    smooth_acc = None
    nr_objs_acc = {}
    nr_vars_acc = {}
    if "quant" in predictions[0]["fast_downward"]:
      smooth_acc = get_smooth_acc(predictions)
      nr_objs_set = set([d["nr_objects"] for d in predictions])
      for nr_objs in nr_objs_set:
        subset = [d for d in predictions if d["nr_objects"] == nr_objs]
        smooth_acc = get_smooth_acc(subset)
        nr_objs_acc[nr_objs] = smooth_acc
      
      nr_vars_set = set([d["nr_variables"] for d in predictions])
      for nr_vars in nr_vars_set:
        subset = [d for d in predictions if d["nr_variables"] == nr_vars]
        smooth_acc = get_smooth_acc(subset)
        nr_vars_acc[nr_vars] = smooth_acc    

    # Calculate average ground cost
    total_ground_cost = 0
    total_solvable = 0
    total_quant_cost = 0
    total_lama_quant_time_limit = 0
    total_lama_ground_time_limit = 0
    total_lama_quant_memory_limit = 0
    total_lama_ground_memory_limit = 0

    for prediction in predictions:
      if prediction["fast_downward"]["ground"]["cost"] is not None:
        total_ground_cost += prediction["fast_downward"]["ground"]["cost"]
        total_solvable += 1
      if prediction["fast_downward"]["quant"]["cost"] is not None:
        total_quant_cost += prediction["fast_downward"]["quant"]["cost"]
      if prediction["fast_downward"]["quant"]["reached_time_limit"]:
        total_lama_quant_time_limit += 1
      if prediction["fast_downward"]["ground"]["reached_time_limit"]:
        total_lama_ground_time_limit += 1
      if prediction["fast_downward"]["quant"]["reached_memory_limit"]:
        total_lama_quant_memory_limit += 1
      if prediction["fast_downward"]["ground"]["reached_memory_limit"]:
        total_lama_ground_memory_limit += 1

    perc_solvable = total_solvable / len(predictions)
    avg_ground_cost = total_ground_cost / total_solvable
    avg_quant_cost = total_quant_cost / len(predictions)
    
    ground_quant_cost_ratio = None
    perc_from_opt = None
    if avg_quant_cost > 0:
      ground_quant_cost_ratio = avg_ground_cost / avg_quant_cost  
      perc_from_opt = (total_ground_cost - total_quant_cost) / total_quant_cost

    metrics = {
      "total": {
        "smooth_acc": smooth_acc      
      },
      "nr_objs": nr_objs_acc,
      "nr_vars": nr_vars_acc,
      "percentage_solvable": perc_solvable,
      "average_ground_cost": avg_ground_cost,
      "average_quant_cost": avg_quant_cost if avg_quant_cost > 0 else None,
      "ground_quant_cost_ratio": ground_quant_cost_ratio,
      "percentage_from_optimal": perc_from_opt,
      "num_total": len(predictions),
      "num_solvable": total_solvable,
      "num_lama_quant_time_limit": total_lama_quant_time_limit,
      "num_lama_ground_time_limit": total_lama_ground_time_limit,
      "num_lama_quant_memory_limit": total_lama_quant_memory_limit,
      "num_lama_ground_memory_limit": total_lama_ground_memory_limit,
    }
    store_json(metrics, os.path.join(args.output_folder, "metrics.json"))