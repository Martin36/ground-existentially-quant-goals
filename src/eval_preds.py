import argparse
from ast import literal_eval
import os
import shutil
from typing import List, Tuple
import tarski
from tarski.io import PDDLReader, FstripsWriter
from tarski.syntax import land, exists
from tqdm import tqdm
from utils_package import load_json
from tarski.errors import SortMismatch

from utils import filter_helper_preds, get_vars_from_goal


class PDDLConverter:
  def __init__(self, domain_file: str, output_folder: str, var_type: str):
    self.domain_file = domain_file
    self.output_folder = output_folder
    self.var_type = var_type

  def convert_predictions(self, predictions):
    if os.path.exists(self.output_folder):
      shutil.rmtree(self.output_folder)
    os.makedirs(self.output_folder)

    for prediction in tqdm(predictions): 
      reader = PDDLReader(raise_on_error=True)
      reader.parse_domain(self.domain_file)

      problem = reader.problem
      # problem.name = f"prob{i+1}"
      problem.name = prediction["id"]
      lang = problem.language

      state_tuples = literal_eval(prediction["init_state"])
      quant_goal_tuples = literal_eval(prediction["quant_goal"])
      ground_goal_tuples = literal_eval(prediction["pred_goal"])

      var_names = get_vars_from_goal(quant_goal_tuples)
      obj_type_map = prediction["objects"]
  
      state_tuples = filter_helper_preds(state_tuples)
      quant_goal_tuples = filter_helper_preds(quant_goal_tuples)
      ground_goal_tuples = filter_helper_preds(ground_goal_tuples)

      init, all_objs = self.get_init_state(state_tuples, lang, obj_type_map)
      problem.init = init

      writer = FstripsWriter(problem)

      quant_goal, vars = self.get_quant_goal(
        quant_goal_tuples, lang, all_objs, var_names
      )
      quant_goal = land(*quant_goal, flat=True)
      problem.goal = exists(*vars, quant_goal)

      quant_problem_file = os.path.join(self.output_folder, f"{problem.name}_quant.pddl")
      writer.write_instance(quant_problem_file)

      try:
        goal = self.get_ground_goal(ground_goal_tuples, lang, all_objs)
      except SortMismatch:
        # This means that the ground goal is incorrectly grounded to wrong types
        # TODO: Is there a way to generate the ground problem for this instance?
        continue
      
      problem.goal = land(*goal, flat=True)
  
      ground_problem_file = os.path.join(self.output_folder, f"{problem.name}_ground.pddl")
      writer.write_instance(ground_problem_file)

    out_domain_file = os.path.join(self.output_folder, "domain.pddl")
    writer.write_domain(out_domain_file)

  def get_init_state(
    self, state_tuples: Tuple[Tuple[str,...]], 
    lang, obj_type_map: dict
  ):
    init = tarski.model.create(lang)
    all_objs = []
    for atom in state_tuples:
      predicate = [p for p in lang.predicates if p.name == atom[0]][0]
      curr_objs = []
      for obj_name in atom[1:]:
        if len([o for o in all_objs if o.name == obj_name]) == 0:
          obj_type = obj_type_map[obj_name]
          obj = lang.constant(obj_name, obj_type)
          all_objs.append(obj)
          curr_objs.append(obj)
        else:
          obj = [o for o in all_objs if o.name == obj_name][0]
          curr_objs.append(obj)
      init.add(predicate(*curr_objs))
    return init, all_objs

  def get_ground_goal(self, ground_goal_tuples: Tuple[Tuple[str,...]], lang, all_objs):
    goal = []
    for atom in ground_goal_tuples:
      predicate = [p for p in lang.predicates if p.name == atom[0]][0]
      curr_objs = []
      for obj_name in atom[1:]:
        obj = [o for o in all_objs if o.name == obj_name][0]
        curr_objs.append(obj)
      goal.append(predicate(*curr_objs))
    return goal
  
  def get_quant_goal(
    self, quant_goal_tuples: Tuple[Tuple[str,...]], lang, all_objs, 
    var_names: List[str]
  ):
    quant_goal = []
    vars = []
    for atom in quant_goal_tuples:
      predicate = [p for p in lang.predicates if p.name == atom[0]][0]
      curr_objs = []
      for obj_name in atom[1:]:
        if obj_name in var_names:
          if len([o for o in vars if o.symbol == obj_name]) == 0:
            obj = lang.variable(obj_name, self.var_type)
            vars.append(obj)
            curr_objs.append(obj)
          else:
            obj = [o for o in vars if o.symbol == obj_name][0]
            curr_objs.append(obj)
        else:
          obj = [o for o in all_objs if o.name == obj_name][0]
          curr_objs.append(obj)
      quant_goal.append(predicate(*curr_objs))
    return quant_goal, vars
  

if __name__ == "__main__":
  arg_parser = argparse.ArgumentParser()
  arg_parser.add_argument(
    "--predictions_file",
    type=str,
    required=True,
    help="Path to the file containing the predictions"
  )
  arg_parser.add_argument(
    "--domain_file",
    type=str,
    required=True,
    help="Path to the domain file"
  )
  arg_parser.add_argument(
    "--output_folder",
    type=str,
    required=True,
    help="Path to the output folder"
  )
  # TODO: Suboptimal solution, as it only works for single type domains
  arg_parser.add_argument(
    "--var_type",
    type=str,
    default="object",
    help="Specifying which type the variables are, only required for typed domains"
  )
  
  args = arg_parser.parse_args()

  predictions = load_json(args.predictions_file)
  converter = PDDLConverter(args.domain_file, args.output_folder, args.var_type)
  converter.convert_predictions(predictions)
  print(f"PDDL files stored in '{args.output_folder}'")
