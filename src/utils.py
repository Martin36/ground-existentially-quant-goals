import os, re, torch
from ast import literal_eval
from typing import Dict, List, Tuple, Union
from utils_package import load_json

from constants import COLOR_PREDICATES, DOMAIN, HELPER_PREDICATES

def convert_binding_str_to_dict(binding_str: str):
  """Converts a binding string to a dictionary

  Args:
      binding_str (str): The binding string to convert

  Returns:
      dict: The converted binding dictionary
  """
  binding_dict = {}
  binding_str = binding_str.split(",")
  for b in binding_str:
    b = b.split(":")
    binding_dict[b[0].strip()] = b[1].strip()
  return binding_dict

def filter_helper_preds(atoms: List[Tuple[str,...]]):
  """Filters out the helper predicates from the atoms

  Args:
      atoms (List[Tuple[str,...]]): A list of tuples containing the atoms

  Returns:
      List[Tuple[str,...]]: A list of tuples containing the atoms without the helper predicates
  """
  return [atom for atom in atoms if atom[0] not in HELPER_PREDICATES]

def get_accuracy(predictions: Dict[str, Dict], 
                 mul_tgt_data: List[dict]):
  """Gets the accuracy of the predictions

  Args:
      predictions (Dict[str, Dict]): The predictions in the format {id: {var1: const1, var2: const2, ...}}
      mul_tgt_data (List[dict]): The data

  Returns:
      float: The accuracy of the predictions
  """
  total_accuracy = 0
  for pred_id in predictions:
    pred_map = predictions[pred_id]
    d = [d for d in mul_tgt_data if pred_id == d['id']][0]
    min_cost = min(list(d["binding_costs"].values()))
    min_cost_mappings = [k for k,v in d["binding_costs"].items() if v == min_cost]
    pred_str = get_binding_str(pred_map)
    if pred_str in min_cost_mappings:
      total_accuracy += 1
  total_accuracy /= len(predictions)
  return total_accuracy


def get_binding_str(binding: dict):
  """Gets the string representation of a variable grounding

  Args:
      binding (dict): The variable grounding map in the format {var1: const1, var2: const2, ...}

  Returns:
      str: String representation of the variable grounding
  """
  result = ""
  # Sort the binding by key
  binding = {k: v for k, v in sorted(binding.items(), key=lambda item: item[0])}
  for k,v in binding.items():
    result += f"{k}:{v},"
  return result[:-1]


def get_color_accuracy(predictions: Dict[str, Dict], mul_tgt_data: List[dict]):
  """Returns the accuracy for the color predicate predictions

  Args:
      predictions (Dict[str, Dict]): The predictions
      mul_tgt_data (List[dict]): The data

  Returns:
      float: The accuracy for the color predicate predictions
  """
  total_correct = 0
  total = 0
  for pred_id in predictions:
    pred_map = predictions[pred_id]
    d = [d for d in mul_tgt_data if pred_id == d['id']][0]
    quant_goal = literal_eval(d['quant_goal'])
    init_state = literal_eval(d['init_state'])
    for predicate in quant_goal:
      if is_color_predicate(predicate):
        ground_predicate = (predicate[0], pred_map[predicate[1]])
        if ground_predicate in init_state:
          total_correct += 1
        total += 1
  if total == 0:
    # If there are no color predicates, just give a perfect score
    return 1.0
  return total_correct / total


def get_color_map_from_tuples(atoms: List[Tuple[str,...]]):
  """Gets the color map from the atoms

  Args:
      atoms (List[Tuple[str,...]]): The list of atoms

  Returns:
      Dict[str,str]: A dictionary of the object names: colors
  """
  result = {}
  for atom in atoms:
    if atom[0] in COLOR_PREDICATES:
      result[atom[1]] = atom[0]
    if atom[0] == "color":
      result[atom[1]] = atom[2]
  return result

def get_colors_from_state(state: List[Tuple[str]]):
  """Gets the color objects from the state

  Args:
      state (List[Tuple[str]]): The state

  Returns:
      List[str]: The colors
  """
  colors = []
  for pred in state:
    if pred[0] == "color":
      colors.append(pred[2])
  return list(set(colors))


def get_consts_from_state(state: List[Tuple[str]]):
  """Gets the constants from the state

  Args:
      state (List[Tuple[str]]): The state

  Returns:
      List[str]: The constants
  """
  consts = []
  for pred in state:
    if pred[0] == "constant":
      consts.append(pred[1])
  return consts


def get_data(data_file = None, data_folder = None):
  """Gets the data from either the data file or the data folder.
  Either data_file or data_folder should be provided

  Args:
    data_file (str, optional): The path to the data file. Defaults to None.
    data_folder (str, optional): The path to the data folder. Defaults to None.
    
  Raises:
    ValueError: If neither data_file nor data_folder is provided
    ValueError: If both data_file and data_folder is provided

  Returns:
    dict: The data dictionary
  """
  if not data_file and not data_folder:
    raise ValueError("Either 'data_file' or 'data_folder' should be provided")
  if data_file and data_folder:
    raise ValueError("Only one of 'data_file' or 'data_folder' should be provided")
  if data_file:
    data = load_json(data_file)
  else:
    # data = load_json(os.path.join(args.data_folder, "test.json"))
    data = load_json(os.path.join(data_folder, "dev.json"))
  return data


def get_default_data_gen_output_folder(
  nr_vars: int, domain: DOMAIN, nr_objs: int, ds_size: int, 
  goal_pred: str = None, val_pred: bool = False, 
  use_colors: bool = True, use_neq_preds: bool = False,
  recolor_states: bool = False
):
  """Gets the default output folder for the data generation
  which is on the form "data/dataset/<domain>/<nr_vars>v-<nr_objs>m-<ds_size>"

  Args:
    nr_vars (int): The max number of variables
    domain (DOMAIN): The domain
    nr_objs (int): The max number of constants
    ds_size (int): The size of the dataset
    goal_pred (str): The goal predicate. Default None
    val_pred (bool): Should be set True for the value prediction datasets. Default False
    recolor_states (bool): Whether to recolor the states. Default False
    
  Returns:
    str: The default output folder, for the given parameters
  """
  base_path = "data/datasets"
  if val_pred:
    base_path += "/val_pred"
    
  if domain == DOMAIN.BLOCKS:
    if not goal_pred:
      raise ValueError("The goal predicate is required for the BLOCKS domain")
    base_path = f"{base_path}/{domain}-{goal_pred}/{nr_vars}v-{nr_objs}m-{ds_size}"
  else:  
    base_path = f"{base_path}/{domain}/{nr_vars}v-{nr_objs}m-{ds_size}"
      
  if not use_colors:
    base_path += "_ncol"
  
  if use_neq_preds:
    base_path += "_neq"
  
  if recolor_states:
    base_path += "_mc"  # For "multi color"
      
  return base_path
  
def get_default_goal_pred(domain: DOMAIN):
  """Gets the default goal predicate for the domain

  Args:
      domain (DOMAIN): The domain

  Returns:
      str: The default goal predicate
  """
  if domain == DOMAIN.BLOCKS:
    raise ValueError("The goal predicate is required for the BLOCKS domain. It can be either 'clear' or 'on'")
  if domain in [DOMAIN.DELIVERY, DOMAIN.GRIPPER]:
    return "at"
  if domain == DOMAIN.VISITALL:
    return "visited"
  raise ValueError(f"Domain {domain} not supported")

def get_default_input_folder(domain: DOMAIN):
  """Gets the default input folder, containing the PDDL file for the 
  given domain 

  Args:
    domain (DOMAIN): The domain
    
  Raises:
    ValueError: If the domain is not supported
      
  Returns:
    str: The default input folder
  """
  if domain == DOMAIN.BLOCKS:
    return "data/blocks_1"
  if domain == DOMAIN.DELIVERY:
    return "data/delivery"
  if domain == DOMAIN.GRIPPER:
    return "data/gripper-adj-3r"
  if domain == DOMAIN.VISITALL:
    return "data/visitall_3"
  raise ValueError(f"Domain {domain} not supported")

def get_default_test_output_folder(
  model_type: str, data_folder: str, nr_objects: int = None
):
  """Gets the default output folder for the testing output
  which is on the form "models/<model_type>/<domain>/<name_of_dataset_folder>"

  Args:
    model_type (str): The type of the model
    data_folder (str): The path to the data folder
    nr_objects (int): The number of objects. Default None
    
  Returns:
    str: The default output folder, for the given parameters
  """
  base_path = f"models/{model_type}"
  
  if os.path.isdir(data_folder):
    # Get the last two folders in the path
    variable_path = "/".join(data_folder.split("/")[-2:])
  else:
    # Then the last item in the path is a file
    variable_path = "/".join(data_folder.split("/")[-3:-1])
  
  if nr_objects:
    return f"{base_path}/{variable_path}/{nr_objects}"
  
  return f"{base_path}/{variable_path}"

def get_default_var_type(domain: DOMAIN):
  """Gets the default variable type for the domain

  Args:
      domain (DOMAIN): The domain

  Returns:
      str: The default variable type
  """
  if domain == DOMAIN.BLOCKS:
    return None
  if domain == DOMAIN.GRIPPER:
    return "ball"
  if domain == DOMAIN.DELIVERY:
    return "cell"
  if domain == DOMAIN.VISITALL:
    return None
  raise ValueError(f"Domain {domain} not supported")


def get_domain_name(data: Union[dict, list]) -> str:
  """Gets the domain name from the data

  Args:
      data (Union[dict, list]): The data

  Returns:
      str: The domain name
  """
  if type(data) == list:
    domain_name = get_domain_from_atoms(literal_eval(data[0]["init_state"]))
  else:
    domain_name = get_domain_from_atoms(literal_eval(
      data[list(data.keys())[0]][0]["init_state"]
    ))
  return domain_name


def get_domain_from_atoms(state_atoms: Tuple[str,...]):
  """Gets the domain name from the state atoms

  Args:
      state_atoms (Tuple[str,...]): The state atoms to get the domain from

  Raises:
      ValueError: If the domain cannot be determined from the state atoms

  Returns:
      DOMAIN: The domain
  """
  if any([atom[0] == "on" for atom in state_atoms]) or \
     any([atom[0] == "clear" for atom in state_atoms]):
    return DOMAIN.BLOCKS
  if any([atom[0] == "at" for atom in state_atoms]):
    # In this case, the domain is either "gripper" or "delivery"
    if any([atom[0] == "at-robby" for atom in state_atoms]):
      # the "at-robby" is only in the gripper domain
      return DOMAIN.GRIPPER
    return DOMAIN.DELIVERY
  if any([atom[0] == "at-robot" for atom in state_atoms]):
    return DOMAIN.VISITALL
  raise ValueError("Could not determine domain from the atoms")
  

def get_domain_predicates(domain: DOMAIN):
  """Gets the predicate specifications for the domain

  Args:
      domain (DOMAIN): The domain

  Raises:
      ValueError: If the domain is not supported

  Returns:
      Dict: A dictionary containing the narity and index of the predicates
  """
  if domain in [DOMAIN.BLOCKS, DOMAIN.BLOCKS_CLEAR, DOMAIN.BLOCKS_ON]:
    return load_json("src/predicates/blocks.json")
  elif domain == DOMAIN.GRIPPER:
    return load_json("src/predicates/gripper.json")
  elif domain == DOMAIN.DELIVERY:
    return load_json("src/predicates/delivery.json")
  elif domain == DOMAIN.VISITALL:
    return load_json("src/predicates/visitall.json")
  else:
    raise ValueError(f"Domain {domain} not supported")


def get_goal_pred_from_quant_goal(atoms: List[Tuple[str, ...]]):
  """Gets the goal predicate from the quantified goal.
  Assumes that there is only one kind of goal predicate in the quantified goal

  Args:
      atoms (List[Tuple[str, ...]]): The quantified goal atoms

  Raises:
      ValueError: If the goal predicate is not found in the quantified goal

  Returns:
      str: The goal predicate name
  """
  for atom in atoms:
    if atom[0] not in HELPER_PREDICATES and \
       atom[0] != "color" and atom[0] != "neq" and \
       atom[0] != "same-color" and atom[0] not in COLOR_PREDICATES:
      return atom[0]
  raise ValueError("Goal predicate not found in quant goal")
      

def get_max_nr_vars(data: List[dict]):
  """Gets the maximum number of variables of the dataset

  Args:
      data (List[dict]): A list of the data samples

  Returns:
      int: The maximum number of variables of any data instance
  """
  max_nr_vars = 0
  for d in data:
    quant_goal = literal_eval(d['quant_goal'])
    nr_vars = 0
    for atom in quant_goal:
      if "variable" in atom:
        nr_vars += 1
    if nr_vars > max_nr_vars:
      max_nr_vars = nr_vars
  return max_nr_vars


def get_nr_of_variables_from_tuple(quant_goal: Tuple[Tuple[str, ...]]):
  """Gets the number of variables from the quantified goal 
  when it is represented as a tuple. It assumes that the quant goal 
  contains "variable" predicates for the variables

  Args:
      quant_goal (Tuple[Tuple[str, ...]]): The quantified goal

  Returns:
      int: The number of variables
  """
  var_atoms = [atom for atom in quant_goal if atom[0] == "variable"]
  return len(var_atoms)


def get_object_names_from_state_str(state: str):
  """Gets the objects from a state string

  Args:
      state (str): The state string

  Returns:
      List: A list with the object names
  """
  if "constant" in state:
    result = []
    state = literal_eval(state)
    for atom in state:
      if atom[0] == "constant":
        result.append(atom[1])
  else:
    # TODO: How to handle this case?
    raise NotImplementedError("The state string needs to contain constant predicates for this function to work")
  return result


def get_object_names_from_state_tuple(state: List[Tuple[str]]):
  """Gets the object names from the state, as represented by a list of tuples

  Args:
      state (List[Tuple[str]]): The list of tuples representing the state

  Raises:
      ValueError: If there are no "constant" atoms in the state

  Returns:
      List[str]: A list of the object names
  """
  if not any([atom[0] == "constant" for atom in state]):
    raise ValueError("The state needs to contain constant predicates for this function to work")
  result = set()
  for atom in state:
    if atom[0] == "constant":
      result.add(atom[1])
  return list(result)


def get_smooth_accuracy(predictions: Dict[str, Dict], 
                 mul_tgt_data: List[dict]):
  """Gets the smooth accuracy of the predictions. Compared to the "get_accuracy" this type of 
  accuracy give a percentage of the optimal cost. For example, if the optimal cost of the grounding
  is 5 and the prediction has the cost of 10, then this accuracy gives 
  a score of 5/10=0.5 for this prediction.

  Args:
      predictions (Dict[str, Dict]): The predictions in the format {id: {var1: const1, var2: const2, ...}}
      mul_tgt_data (List[Dict]): The multi target data to get gold data from

  Returns:
      float: The accuracy of the predictions
  """
  total_accuracy = 0
  for pred_id, pred in predictions.items():
    d = [d for d in mul_tgt_data if pred_id == d["id"]][0]
    min_cost = min(d["binding_costs"].values())
    pred_str = get_binding_str(pred)
    cost = d["binding_costs"].get(pred_str)
    if cost == 0:
      acc = 1
    elif cost is None:
      acc = 0
    else:
      acc = min_cost / cost
    total_accuracy += acc
  total_accuracy /= len(predictions)
  return total_accuracy


def get_top_n_accuracy(predictions: Dict[str, Dict], mul_tgt_data: List[dict], n: int):
  total_correct = 0
  total = 0
  for pred_id in predictions:
    pred_map = predictions[pred_id]
    pred_str = get_binding_str(pred_map)
    d = [d for d in mul_tgt_data if pred_id == d['id']][0]
    costs = d["binding_costs"].values()
    highest_cost = sorted(costs)[min(n-1, len(costs)-1)]
    # Get all quant_to_ground_maps that have a cost less than or equal to the highest cost
    top_n = {k:v for k,v in d["binding_costs"].items() if v <= highest_cost}
    if pred_str in top_n:
      total_correct += 1
    total += 1

  if total == 0:
    # If there are no predictions, just give a perfect score
    return 1.0
  return total_correct / total


def get_type_accuracy(predictions: Dict[str, Dict], mul_tgt_data: List[dict]):
  total_correct = 0
  total = 0
  for pred_id in predictions:
    pred_map = predictions[pred_id]
    d = [d for d in mul_tgt_data if pred_id == d['id']][0]
    # Can take arbitrary one since the constant types should are all be the same
    var_type_map = get_var_type_map(d["binding_costs"], d["objects"])
    for var,const in pred_map.items():
      if var_type_map[var] == d["objects"][const]:
        total_correct += 1
      total += 1      
  if total == 0:
    # If there are no predictions, just give a perfect score
    return 1.0
  return total_correct / total


def get_unsolvable_instances(predictions: Dict[str, Dict], data: List[dict]):
  """Gets the percentage of unsolvable instances

  Args:
      predictions (Dict[str, Dict]): The predictions

  Returns:
      float: The percentage of unsolvable instances
  """
  total = 0
  unsolvable = 0
  for pred_id in predictions:
    d = [d for d in data if pred_id == d['id']][0]
    pred_cost = d["binding_costs"].get(get_binding_str(predictions[pred_id]))
    total += 1
    if pred_cost == None:
      unsolvable += 1
  if total == 0:
    return 0.0
  return unsolvable / total


def get_vars_from_goal(goal: List[Tuple]):
  """Gets the variables from the goal. 
  Assumes that there are "variable" predicates in the goal

  Args:
    goal (List[Tuple]): The goal to get the variables from

  Returns:
    List[str]: The list of variables in the goal
  """
  vars = []
  for pred in goal:
    if pred[0] == "variable":
      vars.append(pred[1])
  return vars


def get_var_type_map(binding_costs: Dict[str,int], objects: Dict[str,str]):
  """Gets the variable type map from the binding costs and objects

  Args:
      binding_costs (Dict[str,int]): Dictionary mapping the binding string to the cost
      objects (Dict[str,str]): Dictionary mapping the objects to the type

  Returns:
      Dict[str,str]: Dictionary mapping the variables to the types
  """
  binding_str = list(binding_costs.keys())[0]
  binding_dict = convert_binding_str_to_dict(binding_str)
  return {v: objects[c] for v,c in binding_dict.items()}


def is_color_predicate(atom: Tuple[str,...]):
  """Checks if the atom is a color predicate

  Args:
    atom Tuple[str,...]: The atom to check

  Returns:
    bool: Whether the predicate name is a color predicate
  """
  return atom[0] in COLOR_PREDICATES


def is_unsolvable(
  prediction: Dict[str, object], bindings: Dict[str, str] = {},
  goal_pred: str = None):
  """Checks if a prediction is unsolvable. This could be 
  because the predicted goal contains "on" atoms with the same block
  in both arguments, or if the predicted goal has objects of the wrong color.

  Args:
      prediction (Dict[str, object]): The prediction
      bindings (Dict[str, str]): The variable bindings
      goal_pred (str): The goal predicate

  Returns:
      (bool, str): Whether the prediction is unsolvable, and the reason why
  """
  
  if "pred_goal" in prediction:
    goal_atoms = literal_eval(prediction["pred_goal"])
  else:
    # This is the case for the value predictions
    goal_atoms = literal_eval(prediction["quant_goal"])
    # Remove all the variables from the goal
    vars = get_vars_from_goal(goal_atoms)
    goal_atoms = [atom for atom in goal_atoms 
                  if not any([arg in vars for arg in atom])]
    if len(goal_atoms) == 0:
      return False, None
    
  state_atoms = literal_eval(prediction["init_state"])
  
  for goal_atom in goal_atoms:
    # If the predicted goal contains "on" atoms with the same block
    # in both arguments, then the goal is unsolvable
    if goal_atom[0] == "on" and goal_atom[1] == goal_atom[2]:
      return True, "Same block in both arguments"
    if goal_atom[0] == "at":
      # Check that the types are correct
      # It is either "at(ball, room)" if the domain is gripper
      # or "at(package/truck, cell)" if the domain is delivery
      arg1_type = prediction["objects"][goal_atom[1]]
      if arg1_type not in ["ball", "package", "truck"]:
        return True, "Wrong object type"

    # If the predicted goal contains "on" atoms with the same block
    # at two positions in the tower
    # TODO: How to do this?
    # A block cannot be in two top positions
    # if goal_atom[0] == "on" and \
    #    any([atom[0] == "on" and atom[1] == goal_atom[1] for atom in goal_atoms]):
    #   return True, "Same block in multiple top positions"
    # # A block cannot be in two bottom positions
    # if goal_atom[0] == "on" and \
    #    any([atom[0] == "on" and atom[2] == goal_atom[2] for atom in goal_atoms]):
    #   return True, "Same block in multiple bottom positions"
    # HACK: For blocks on we know that there is only one tower and if two
    # variables are grounded to the same constant if will fail.
    # But then we need to know the groundings. 
    if bindings and goal_pred == "on":
      constants = [const for const in bindings.values()]
      if len(set(constants)) != len(constants):
        return True, "Same block in multiple positions"
    # If the predicted goal has objects of the wrong color
    # then it is unsolvable
    if goal_atom[0] == "color" and goal_atom[1] != goal_atom[2]:
      return True, "Wrong color"
    if is_color_predicate(goal_atom):
      if goal_atom not in state_atoms:
        return True, "Wrong color"
    if goal_atom[0] == "at":
      # A ball cannot be at two positions
      if any([atom[0] == "at" and atom[1] == goal_atom[1] and \
              atom[2] != goal_atom[2] for atom in goal_atoms]):
        return True, "Same object at multiple positions"  
    if goal_atom[0] == "neq" and \
       goal_atom[1] == goal_atom[2]:
      return True, "Violating not equal constraint"

  if "pred_cost" in prediction:
    return prediction["pred_cost"] is None, "Invalid goal state"

  return False, None


def merge_gnn_input(gnn_inputs: List[Tuple[Dict, torch.Tensor]], device: torch.device = torch.device('cpu')):
  """Merges the GNN inputs into a single input
  
  Args:
    gnn_inputs (List[Tuple[Dict, torch.Tensor]]): The GNN inputs to merge
  
  Returns:
    Tuple[Dict, torch.Tensor]: The merged GNN input
  """
  predicates = {}
  nr_objs_list = []
  start_idx = 0
  for state, nr_objs in gnn_inputs:
    for key in state:
      if key not in predicates:
        predicates[key] = []
      curr_pred = state[key] + start_idx * torch.ones(state[key].shape, dtype=torch.long).to(device)
      predicates[key].append(curr_pred)
    start_idx += nr_objs.item()
    nr_objs_list.append(nr_objs)
  for key in predicates:
    predicates[key] = torch.cat(predicates[key], dim=0).to(device)
  nr_objs = torch.cat(nr_objs_list, dim=0)
  return (predicates, nr_objs)


def natural_keys(text):
  '''
  Taken from: https://stackoverflow.com/questions/5967500/how-to-correctly-sort-a-string-with-a-number-inside
  alist.sort(key=natural_keys) sorts in human order. Sorting done in place
  http://nedbatchelder.com/blog/200712/human_sorting.html
  (See Toothy's implementation in the comments)
  '''
  def atoi(text):
    return int(text) if text.isdigit() else text
  return [ atoi(c) for c in re.split(r'(\d+)', text) ]

def replace_grounded_var(
  goal_atoms, grounded_var, grounded_const, remove_ground_col_atoms=False
):
  """Replaces the grounded variable in the goal with a constant
  
  Args:
    goal_atoms (List[Tuple[str, ...]]): The goal atoms
    grounded_var (str): The grounded variable
    grounded_const (str): The grounded constant
      
  Returns:
    List[Tuple[str, ...]]: The goal atoms with the grounded variable replaced with the constant
  """
  next_goal_atoms = []
  for atom in goal_atoms:
    if grounded_var in atom:
      new_atom = [d if d != grounded_var else grounded_const for d in atom]
      if new_atom[0] in HELPER_PREDICATES:
        # Remove helper predicates
        continue
        # Replace variable predicates with constants
        # new_atom[0] = "constant"
      if remove_ground_col_atoms and new_atom[0] in COLOR_PREDICATES:
        continue
      next_goal_atoms.append(tuple(new_atom))
    else:
      next_goal_atoms.append(atom)
  return next_goal_atoms
