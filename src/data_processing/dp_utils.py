import random
import os, re, pymimir
from collections import defaultdict
from glob import glob
from typing import Dict, List, Set, Tuple

from constants import COLOR_PREDICATES, DOMAIN, MIMIR_VAR_TO_VAR

def add_const_preds(state: Tuple, objects: Dict[str,str]):
  """Adds the constant predicates to the state

  Args:
      state (Tuple): The state to add the constant predicates to
      objects (Dict[str,str]): Dictionary of object type maps

  Returns:
      Tuple: The state with the constant predicates added
  """
  state = list(state)
  for obj in objects:
    # Don't add const preds for colors 
    if objects[obj] == "color":
      continue
    state.append(("constant", obj))
  return tuple(state)

def add_neq_preds_to_goal(
  quant_goal: List[pymimir.Atom], predicates: List[pymimir.Predicate]
):
  """Adds the neq predicates to the goal

  Args:
      quant_goal (List[pymimir.Atom]): The goal to add the neq predicates to
      predicates (List[pymimir.Predicate]): The list of predicates

  Returns:
      List[pymimir.Atom]: The goal with the neq predicates added
  """
  neq_pred = [pred for pred in predicates if pred.name == "neq"][0]
  vars = get_vars_from_mimir_goal(quant_goal)
  for var1 in vars:
    for var2 in vars:
      if var1.name != var2.name:
        neq_atom = neq_pred.as_atom()
        neq_atom = neq_atom.replace_term(0, var1)
        neq_atom = neq_atom.replace_term(1, var2)
        quant_goal.append(neq_atom)
  return quant_goal

def add_obj_types_from_preds(objects: Dict[str,str], init_state: List[pymimir.Atom]):
  """Adds the object types to the objects taken from the atoms in the initial
  state. This is to be used if the domain has the types specified as predicates

  Args:
    objects (Dict[str,str]): A dict of the object names: types
    static_atoms (List[pymimir.Atom]): A list of the atoms in the initial state

  Returns:
    dict: A dictionary containing the objects and their types
  """
  for atom in init_state:
    if atom.predicate.name == "ball":
      objects[atom.terms[0].name] = "ball"
    elif atom.predicate.name == "room":
      objects[atom.terms[0].name] = "room"
    elif atom.predicate.name == "gripper":
      objects[atom.terms[0].name] = "gripper"
  return objects

def add_var_preds(state: Tuple, variables: Set[str]):
  """Adds the variable predicates to the state

  Args:
      state (Tuple): The state to add the variable predicates to
      objects (Set[str]): Set of the variables

  Returns:
      Tuple: The state with the variable predicates added
  """
  state = list(state)
  for var in variables:
    state.append(("variable", var))
  return tuple(state)

def convert_atom_list_to_str(
  atoms: List[pymimir.Atom], objects: Dict[str,str], 
  incl_const_preds=False, incl_var_preds=False
):
  """Converts a list of Mimir atoms to a string of tuple representations

  Args:
      atoms (List[pymimir.Atom]): The list of atoms
      objects (Dict[str,str]): Dictionary of object type maps
      incl_const_preds (bool, optional): If "constant" predicates e.g. ("constant", "a") should be added. Defaults to False.
      incl_var_preds (bool, optional): If "variable" predicates e.g. ("variable", "x1") should be added. Defaults to False.

  Returns:
      str: A string representation of the list of atoms
  """
  result = []
  vars = []
  for atom in atoms:
    vars.extend([MIMIR_VAR_TO_VAR[term.name] for term in atom.terms 
                 if term.is_variable()])
    atom_tup = convert_atom_to_tuple(atom)
    result.append(atom_tup)
  if incl_const_preds:
    result = add_const_preds(result, objects)
  elif incl_var_preds:
    vars = set(vars)
    result = add_var_preds(result, vars)
  # Filter duplicates
  # result = list(set(result))
  return str(tuple(sorted(result)))

def convert_atom_to_tuple(atom: pymimir.Atom):
  """Converts a Mimir atom to a tuple

  Args:
      atom (pymimir.Atom): The Mimir atom

  Returns:
      Tuple[str,...]: The tuple representation of the atom
  """
  result = []
  result.append(atom.predicate.name)
  for term in atom.terms:
    if term.is_variable():
      # Need to convert the variable to x1, x2 etc
      result.append(MIMIR_VAR_TO_VAR[term.name])
    else:
      result.append(term.name)
  return tuple(result)

def convert_mimir_binding_to_dict(binding: List[Tuple[str, pymimir.Object]]):
  """Converts the mimir binding to a dictionary

  Args:
      binding (List[Tuple[str, pymimir.Object]]): The mimir binding

  Returns:
      Dict[str, str]: The dictionary representation of the binding
  """
  result = {}
  for var, obj in binding:
    # var_name = MIMIR_VAR_TO_VAR[var.name]
    var_name = MIMIR_VAR_TO_VAR[var]
    result[var_name] = obj.name
  return result

def convert_tuple_to_mimir_state(
  state: Tuple, predicates: List[pymimir.Predicate], 
  objects: List[pymimir.Object]
):
  """Converts a tuple representation of a state to a Mimir state

  Args:
    state (Tuple): Tuple of atom tuples in the state
    predicates (List[pymimir.Predicate]): List of the predicates
    objects (List[pymimir.Object]): List of the objects

  Returns:
    List[pymimir.Atom]: A list of the atoms in the state
  """
  # Remove constant preds
  state = [atom_tup for atom_tup in state if atom_tup[0] != "constant"]
  # Remove variable preds
  state = [atom_tup for atom_tup in state if atom_tup[0] != "variable"]
  result = []
  for atom_tup in state:
    pred = [pred for pred in predicates if pred.name == atom_tup[0]][0]
    atom = pred.as_atom()
    for i, arg in enumerate(atom_tup[1:]):
      obj = [obj for obj in objects if obj.name == arg][0]
      atom = atom.replace_term(i, obj)
    result.append(atom)
  return result

def convert_tuple_to_mimir_goal(
  goal: Tuple, problem: pymimir.Problem, domain: pymimir.Domain
):
  """Converts a tuple representation of a goal to a Mimir goal

  Args:
      goal (Tuple): The tuple representation of the goal
      problem (pymimir.Problem): The Mimir problem
      domain (pymimir.Domain): The Mimir domain

  Returns:
      List[pymimir.Atom]: The list of the atoms in the goal
  """
  # Remove "variable" atoms from the goal
  goal = [atom_tup for atom_tup in goal if atom_tup[0] != "variable"]
  result = []
  var_names = []
  var_idx = 0
  vars = []
  for atom_tup in goal:
    pred = [pred for pred in domain.predicates if pred.name == atom_tup[0]][0]
    atom = pred.as_atom()
    for i, arg in enumerate(atom_tup[1:]):
      obj = [obj for obj in problem.objects if obj.name == arg]
      if len(obj) == 0:
        # Then the argument is a variable
        if arg in var_names:
          # This means that the variable has already been created
          var = vars[var_names.index(arg)]
        else:
          var_type = pred.parameters[i].type
          var = pymimir.Object(var_idx, f"?{arg}", var_type)
          var_idx += 1
          var_names.append(arg)
          vars.append(var)
        atom = atom.replace_term(i, var)
      else:
        # If the argument is a constant, find the object and update the atom
        obj = obj[0]
        atom = atom.replace_term(i, obj)
    result.append(atom)
  return result

def create_color_atom(problem: pymimir.Problem, color: str, obj_name: str):
  """Creates a new color atom from the given color and object name

  Args:
      problem (pymimir.Problem): The problem
      color (str): The color name
      obj_name (str): The object name

  Raises:
      ValueError: If there is no color predicate for the given color

  Returns:
      pymimir.Atom: A Mimir atom of the form "color(obj_name)"
  """
  domain = problem.domain
  col_pred = [pred for pred in domain.predicates if pred.name == color]
  if len(col_pred) == 0:
    raise ValueError(f"Color predicate '{color}' not found in domain predicates. Make sure that the color is added as a predicate to the domain file")
  col_pred = col_pred[0]
  obj = [obj for obj in problem.objects if obj.name == obj_name][0]
  col_atom = col_pred.as_atom()
  col_atom = col_atom.replace_term(0, obj)
  return col_atom
    
def filter_fully_quant_data(data: Dict[int, List[dict]]):
  """Filters out the nr_objects and nr_variables from the fully quant data

  Args:
      data (List[dict]): The data

  Returns:
      List[dict]: The filtered data
  """
  for cost in data:
    for d in data[cost]:
      if "nr_objects" in d:
        del d["nr_objects"]
      if "nr_variables" in d:
        del d["nr_variables"]
  return data

def filter_files(
  files: List[str], max_nr_objects: int, domain: DOMAIN, 
  nr_vars: int = None, remove_too_small_problems = False,
  domain_file: str = None
):
  """Filters the files based on the number of objects

  Args:
    files (List[str]): List of the file paths
    max_nr_objects (int): The maximum number of objects allowed for the instances
    domain (DOMAIN): The domain of the instances
    nr_vars (int): The number of variables, to determine if the problem instance is too small or not. Only require if remove_too_small_problems is True.
    remove_too_small_problems (bool, optional): If True, will remove problems with less than or equal the amount of constants as variables. Defaults to False.
    domain_file (str, optional): The domain file. Only required for the gripper and visitall domains. Defaults to None.
    
  Raises:
    ValueError: If "remove_too_small_problems" is True, but "nr_vars" is not provided

  Returns:
    List[str]: List of the filtered file paths
  """
  if not nr_vars and remove_too_small_problems:
    raise ValueError("Arguemnt nr_vars is required if remove_too_small_problems is True")

  filtered_files = []
  for file in files:
    if "domain" in file:
      continue
    nr_objects = get_nr_objs_from_file(file, domain, domain_file)
    if nr_objects <= max_nr_objects:
      if remove_too_small_problems and \
        is_too_small(nr_objects, nr_vars):
        continue
      filtered_files.append(file)
  return filtered_files

def get_colorable_objects_visitall(objects: Dict[str, str]):
  # In visitall we want to have some cells uncolored
  # and at least one cell needs to be uncolored, which
  # is the cell that the robot starts in
  # And at least two cells needs to have colors
  colorable_objects = {}
  obj_names = list(objects.keys())
  random.shuffle(obj_names)
  nr_uncolored = random.randint(1, len(obj_names) - 2)
  for i, obj_name in enumerate(obj_names):
    if i >= nr_uncolored:
      colorable_objects[obj_name] = objects[obj_name]
  return colorable_objects


def get_color_map(atoms: List[pymimir.Atom]):
  """Gets the color map from the atoms

  Args:
      atoms (List[pymimir.Atom]): The list of atoms

  Returns:
      Dict[str,str]: A dictionary of the object names: colors
  """
  result = {}
  if any([atom.predicate.name == "color" for atom in atoms]):
    return get_color_map_from_binary_color_preds(atoms)
  for atom in atoms:
    if atom.predicate.name in COLOR_PREDICATES:
      result[atom.terms[0].name] = atom.predicate.name
  return result  

def get_color_map_from_binary_color_preds(atoms: List[pymimir.Atom]):
  """Gets the color map from the binary color predicates

  Args:
      atoms (List[pymimir.Atom]): The list of atoms

  Returns:
      Dict[str,str]: A dictionary of the object names: colors
  """
  result = {}
  for atom in atoms:
    if atom.predicate.name == "color":
      result[atom.terms[0].name] = atom.terms[1].name
  return result

def get_domain_and_problem_files(
  input_folder: str, max_nr_objects: int = None, domain: DOMAIN = None,
  nr_vars: int = None, remove_too_small_problems = False,
  nr_colors: int = None 
):
  """Gets the domain and problem files from a given folder

  Args:
    input_folder (str): The path to the folder
    max_nr_objects (int, optional): The maximum number of objects allowed for the instances. Defaults to None, which means no limit.
    domain (DOMAIN, optional): The domain of the instances. Defaults to None.
    nr_vars (int, optional): The max number of variables in the goal. Defaults to None.
    remove_too_small_problems (bool, optional): If True, will remove problems with less than or equal the amount of constants as variables. Defaults to False.
    nr_colors (int, optional): The number of colors to use. Defaults to None.
    
  Returns:
      Tuple[str, List[str]]: A tuple of the domain file and a list of the problem files
  """
  files = glob(os.path.join(input_folder, "*.pddl"))
  domain_file = [file for file in files if "domain" in file]
  if len(domain_file) == 0:
    raise ValueError(f"Folder '{input_folder}' does not contain any domain file. Make sure to name the domain 'domain.pddl'")
  if len(domain_file) > 1:
    raise ValueError(f"Folder '{input_folder}' contains more than one domain file.")
  domain_file = domain_file[0]
  problem_files = [file for file in files if "domain" not in file]

  if max_nr_objects or nr_colors:
    if not domain:
      raise ValueError("Argument domain is required if max_nr_objects is specified")
    problem_files = filter_files(
      problem_files, max_nr_objects, domain, nr_vars, remove_too_small_problems,
      domain_file
    )
  
  problem_files.sort(key=natural_keys)

  return domain_file, problem_files

def get_objects(problem: pymimir.Problem, domain: DOMAIN = None):
  """Gets the objects from a given Mimir problem

  Args:
    problem (mimir.Problem): The Mimir problem
    domain (DOMAIN, optional): The domain name of the problem. Defaults to None.
    
  Returns:
    dict: A dictionary with the object names as keys and types as values
  """
  result = {}
  for obj in problem.objects:
    result[obj.name] = obj.type.name
  if domain == DOMAIN.GRIPPER:
    # This domain has the types specified as predicates which means that
    # the objects will be of type "object" by default
    result = add_obj_types_from_preds(result, problem.initial)
  return result

def get_random_initial_pos_atoms(
  color_preds: List[pymimir.Atom], objects: Dict[str, str],
  problem: pymimir.Problem, predicates: List[pymimir.Predicate]
) -> List[pymimir.Atom]:
  """Gets the initial state atoms for the robot, with a new
  random starting position, that is not colored.
  This is only used for the Visitall domain where we want
  to have the robot start in a position that is not colored

  Args:
      color_preds (List[pymimir.Atom]): The list of color predicates
      objects (Dict[str, str]): The object name to type mapping
      problem (pymimir.Problem): The problem
      predicates (List[pymimir.Predicate]): The list of predicates

  Returns:
      List[pymimir.Atom]: The list of initial state atoms
  """
  colored_places = [atom.terms[0].name for atom in color_preds]
  non_colored_places = [place for place in objects if place not in colored_places]
  robot_place = random.choice(non_colored_places)
  at_robot_pred = [pred for pred in predicates if pred.name == "at-robot"][0]
  robot_place_obj = [obj for obj in problem.objects if obj.name == robot_place][0]
  at_robot_atom = at_robot_pred.as_atom().replace_term(0, robot_place_obj)
  visited_pred = [pred for pred in predicates if pred.name == "visited"][0]
  visited_atom = visited_pred.as_atom().replace_term(0, robot_place_obj)
  initial_pos_preds = [at_robot_atom, visited_atom]
  # Remove the old initial pos atoms
  initial_atoms = [atom for atom in problem.initial if atom.predicate.name not in ["at-robot", "visited"]]
  initial_atoms.extend(initial_pos_preds)
  return initial_atoms

def get_random_object_of_type(
  type: str, objects: List[pymimir.Object], used_objs: List[str] = []
):
  """Gets a random object of the given type

  Args:
      type (str): The type of the object
      objects (List[pymimir.Object]): The list of the mimir objects
      used_objs (List[str], optional): Contains the objects that should not be returned. Defaults to [].

  Raises:
      ValueError: If there are no objects found of the given type

  Returns:
      pymimir.Object: The sampled object
  """
  objects_of_type = [obj for obj in objects if obj.type.name == type]
  if len(objects_of_type) == 0:
    # Then the type might be of the parent type
    objects_of_type = [obj for obj in objects if obj.type.base.name == type]
  if len(used_objs) > 0:
    # Filter out these objects from the list
    objects_of_type = [obj for obj in objects_of_type if obj.name not in used_objs]
  if len(objects_of_type) == 0:
    raise ValueError(f"No usable objects of type '{type}' found")
  return random.choice(objects_of_type)

def get_supervised_stats(
  train_data: List[dict], val_data: List[dict], test_data: List[dict], 
):
  """Gets the stats from the fully quantified data

  Args:
    train_data (List[dict]): The training data
    val_data (List[dict]): The validation data
    test_data (List[dict]): The test data

  Returns:
    Dict: A dictionary of the stats
  """
  stats = defaultdict(int)
  stats["nr vars count"] = {
    "total": defaultdict(int),
    "train": defaultdict(int),
    "val": defaultdict(int),
    "test": defaultdict(int),
  }
  stats["nr objects count"] = {
    "total": defaultdict(int),
    "train": defaultdict(int),
    "val": defaultdict(int),
    "test": defaultdict(int),
  }
  stats["nr colors count"] = {
    "total": defaultdict(int),
    "train": defaultdict(int),
    "val": defaultdict(int),
    "test": defaultdict(int),
  }

  cost_dist = {
    "total": defaultdict(int),
    "train": defaultdict(int),
    "val": defaultdict(int),
    "test": defaultdict(int),
  }

  cost = list(train_data.keys())[0]
  d = train_data[cost][0]
  contains_nr_vars = False
  if "nr_variables" in d:
    contains_nr_vars = True
  
  for cost in train_data:
    cost_dist["total"][cost] += len(train_data[cost])
    cost_dist["train"][cost] += len(train_data[cost])

    for d in train_data[cost]:
      if contains_nr_vars:
        stats["nr vars count"]["train"][d["nr_variables"]] += 1
        stats["nr vars count"]["total"][d["nr_variables"]] += 1
      stats["nr objects count"]["train"][d["nr_objects"]] += 1
      stats["nr objects count"]["total"][d["nr_objects"]] += 1
      stats["nr colors count"]["train"][d["nr_colors"]] += 1
      stats["nr colors count"]["total"][d["nr_colors"]] += 1
  
  for cost in val_data:
    cost_dist["total"][cost] += len(val_data[cost])
    cost_dist["val"][cost] += len(val_data[cost])

    for d in val_data[cost]:
      if contains_nr_vars:
        stats["nr vars count"]["val"][d["nr_variables"]] += 1
        stats["nr vars count"]["total"][d["nr_variables"]] += 1
      stats["nr objects count"]["val"][d["nr_objects"]] += 1
      stats["nr objects count"]["total"][d["nr_objects"]] += 1
      stats["nr colors count"]["val"][d["nr_colors"]] += 1
      stats["nr colors count"]["total"][d["nr_colors"]] += 1
      
  for cost in test_data:
    cost_dist["total"][cost] += len(test_data[cost])
    cost_dist["test"][cost] += len(test_data[cost])

    for d in test_data[cost]:
      if contains_nr_vars:
        stats["nr vars count"]["test"][d["nr_variables"]] += 1
        stats["nr vars count"]["total"][d["nr_variables"]] += 1
      stats["nr objects count"]["test"][d["nr_objects"]] += 1
      stats["nr objects count"]["total"][d["nr_objects"]] += 1
      stats["nr colors count"]["test"][d["nr_colors"]] += 1
      stats["nr colors count"]["total"][d["nr_colors"]] += 1

  stats["cost dist"] = cost_dist
  
  var_cost_dist = {
    "total": defaultdict(int),
    "train": defaultdict(int),
    "val": defaultdict(int),
    "test": defaultdict(int),
  }

  if contains_nr_vars:
    for cost in train_data:
      for d in train_data[cost]:
        var_cost_dist["total"][str((d["nr_variables"], d["cost"]))] += 1
        var_cost_dist["train"][str((d["nr_variables"], d["cost"]))] += 1
    for cost in val_data:
      for d in val_data[cost]:
        var_cost_dist["total"][str((d["nr_variables"], d["cost"]))] += 1
        var_cost_dist["val"][str((d["nr_variables"], d["cost"]))] += 1
    for cost in test_data:
      for d in test_data[cost]:
        var_cost_dist["total"][str((d["nr_variables"], d["cost"]))] += 1
        var_cost_dist["test"][str((d["nr_variables"], d["cost"]))] += 1
  stats["var cost dist"] = var_cost_dist
      
  return stats

def get_max_nr_vars(max_nr_vars: int = None, nr_vars_per_type: str = None):
  """Gets the max nr of variables, either from the max_nr_vars argument or the nr_vars_per_type argument

  Args:
      max_nr_vars (int, optional): The upper limit of variables. Defaults to None.
      nr_vars_per_type (str, optional): Comma separated string of the number of variables per type. Defaults to None.

  Returns:
      int: The max number of variables
  """
  max_nr_vars = None
  if max_nr_vars:
    max_nr_vars = max_nr_vars
  elif nr_vars_per_type:
    max_nr_vars = sum([int(nr) for nr in nr_vars_per_type.split(",")])
  return max_nr_vars

def get_nr_colors(atoms: List[pymimir.Atom]):
  """Gets the number of colors from the atoms in the initial state

  Args:
      atoms (List[pymimir.Atom]): The initial state atoms

  Returns:
      int: The number of colors in the initial state
  """
  binary_color_preds = [atom for atom in atoms
                        if atom.predicate.name == "color"]
  if len(binary_color_preds) > 0:
    unique_colors = set([atom.terms[1].name for atom in binary_color_preds])
    return len(unique_colors)
  color_preds = [atom.predicate.name for atom in atoms
                  if atom.predicate.name in COLOR_PREDICATES]
  unique_color_preds = set(color_preds)
  return len(unique_color_preds)

def get_nr_objs_from_file(file: str, domain: DOMAIN, domain_file: str = None):
  """Gets the number of objects, differently based on the domain

  Args:
    file (str): The file path
    domain (DOMAIN): The domain of the problem file
    domain_file (str, optional): The domain file. Defaults to None.

  Raises:
    ValueError: If the domain is not supported
    ValueError: If the domain file is required, but not provided

  Returns:
    int: The number of objects in the problem file
  """
  if domain == DOMAIN.BLOCKS:
    # Assuing that the problem files are names "p-x-0.pddl", where x is the number of objects
    nr_objects = int(file.split("/")[-1].split("-")[1])
  elif domain == DOMAIN.DELIVERY:
    # Nr of objects in the delivery domain is considered to be the grid size
    # e.g. if the grid is 4x4, then there are 16 objects
    if not domain_file:
      raise ValueError("Argument domain_file is required for the delivery domain")
    nr_objects = get_nr_objs_from_pddl(domain_file, file)
  elif domain == DOMAIN.GRIPPER:
    if not domain_file:
      raise ValueError("Argument domain_file is required for the gripper domain")
    # The file names for gripper does not have a natural way for filtering
    # Instead, we read the file to get the number of objects
    nr_objects = get_nr_objs_from_pddl(domain_file, file)
  elif domain == DOMAIN.VISITALL:
    if not domain_file:
      raise ValueError("Argument domain_file is required for the visitall domain")
    # Same as for gripper here
    nr_objects = get_nr_objs_from_pddl(domain_file, file)
  else:
    raise ValueError(f"Unsupported domain: {domain}")
  return nr_objects
 
def get_nr_objs_from_pddl(dom_file: str, prob_file: str):
  """Reads the files to look at how many objects it contains

  Args:
    dom_file (str): The domain file
    prob_file (str): The problem file
    
  Returns:
    int: The number of objects in the problem file
  """
  domain = pymimir.DomainParser(dom_file).parse()
  problem = pymimir.ProblemParser(prob_file).parse(domain)
  return len(problem.objects)

def get_nr_of_packages_from_state_space(state_space: pymimir.StateSpace):
  """Gets the number of packages from the state space

  Args:
      state_space (pymimir.StateSpace): The state space

  Returns:
      int: The number of packages
  """
  nr_packages = 0
  for obj in state_space.problem.objects:
    if obj.type.name == "package":
      nr_packages += 1
  return nr_packages

def get_nr_of_variables(goal: List[pymimir.Atom]):
  """Gets the number of variables in the goal

  Args:
      goal (List[pymimir.Atom]): The goal

  Returns:
      int: The number of variables
  """
  return len(get_vars_from_mimir_goal(goal))

def get_split_sizes(
  split_sizes: str = None, dataset_size: int = None, split_ratio: str = None
):
  """Get the split sizes for the train, val and test sets

  Args:
    split_sizes (str): Comma separated string of the split sizes. If set, will override the split_ratio argument
    dataset_size (int): The size of the dataset
    split_ratio (str): Commaseparated string of the split ratio

  Returns:
    Tuple(int,int,int): Tuple of the split sizes
  """
  if not split_sizes and not split_ratio:
    raise ValueError("Either split_sizes or split_ratio must be specified")
  if split_sizes:
    split_sizes = [int(nr) for nr in split_sizes.split(",")]
    train_size, val_size, test_size = split_sizes
  else:
    split_ratio = split_ratio.split(",")
    split_ratio = [float(r) for r in split_ratio]
    if not sum(split_ratio) == 1.0:
      raise ValueError(f"Sum of split ratio ({sum(split_ratio)}) must be equal to 1.0")
    split_sizes = [int(dataset_size * r) for r in split_ratio]
    train_size, val_size, test_size = split_sizes
  return train_size, val_size, test_size

def get_stats(train_data: List[dict], val_data: List[dict], test_data: List[dict], 
              incl_cost_dist: bool = False):
  """Gets the stats from the data

  Args:
      train_data (List[dict]): The training data
      val_data (List[dict]): The validation data
      test_data (List[dict]): The test data
      incl_cost_dist (bool, optional): If True, will include the cost distribution. Defaults to False.

  Returns:
      Dict: A dictionary of the stats
  """
  stats = defaultdict(int)
  stats["nr vars count"] = {
    "total": defaultdict(int),
    "train": defaultdict(int),
    "val": defaultdict(int),
    "test": defaultdict(int),
  }
  stats["nr objects count"] = {
    "total": defaultdict(int),
    "train": defaultdict(int),
    "val": defaultdict(int),
    "test": defaultdict(int),
  }

  if incl_cost_dist:
    cost_dist = {
      "total": defaultdict(int),
      "train": defaultdict(int),
      "val": defaultdict(int),
      "test": defaultdict(int),
    }
  
  for d in train_data:
    stats["nr vars count"]["train"][d["nr_variables"]] += 1
    stats["nr objects count"]["train"][d["nr_objects"]] += 1
    stats["nr vars count"]["total"][d["nr_variables"]] += 1
    stats["nr objects count"]["total"][d["nr_objects"]] += 1
    if incl_cost_dist:
      cost_dist["total"][d["cost"]] += 1
      cost_dist["train"][d["cost"]] += 1
  
  for d in val_data:
    stats["nr vars count"]["val"][d["nr_variables"]] += 1
    stats["nr objects count"]["val"][d["nr_objects"]] += 1
    stats["nr vars count"]["total"][d["nr_variables"]] += 1
    stats["nr objects count"]["total"][d["nr_objects"]] += 1
    if incl_cost_dist:
      cost_dist["total"][d["cost"]] += 1
      cost_dist["val"][d["cost"]] += 1
    
  for d in test_data:
    stats["nr vars count"]["test"][d["nr_variables"]] += 1
    stats["nr objects count"]["test"][d["nr_objects"]] += 1
    stats["nr vars count"]["total"][d["nr_variables"]] += 1
    stats["nr objects count"]["total"][d["nr_objects"]] += 1
    if incl_cost_dist:
      cost_dist["total"][d["cost"]] += 1
      cost_dist["test"][d["cost"]] += 1  
  
  if incl_cost_dist:
    stats["cost dist"] = cost_dist
    
  return stats

def get_var_type_count_map(var_types: str, nr_vars_per_type: str, max_nr_vars: int):
  """Gets the variable type count map

  Args:
      var_types (str): Comma separated string of the variable types
      nr_vars_per_type (str): Comma separated string of the number of variables per type
      max_nr_vars (int): The maximum number of variables in the goal

  Raises:
      ValueError: If the number of variables per type is not specified
      ValueError: If the number of variables per type is not compatible with the max number of variables
      ValueError: If the number of variable types is not compatible with the number of variables per type

  Returns:
      dict: A dictionary mapping the variable type to the number of variables for that type
  """
  if var_types:
    if not nr_vars_per_type:
      raise ValueError("Missing argument: --nr_vars_per_type")
    var_types = var_types.split(",")
    nr_vars_per_type = [int(nr) for nr in nr_vars_per_type.split(",")]
    if max_nr_vars:
      if sum(nr_vars_per_type) != max_nr_vars:
        raise ValueError(f"Invalid nr_vars_per_type argument: {nr_vars_per_type}, it is not compatible with the max_nr_vars argument: {max_nr_vars}")
    if len(var_types) != len(nr_vars_per_type):
      raise ValueError(f"Invalid nr_vars_per_type argument: {nr_vars_per_type}")
    return {var_type: nr_vars for var_type, nr_vars in zip(var_types, nr_vars_per_type)}
  else:
    return None
  
def get_var_type_map_from_goal(goal: List[pymimir.Atom]):
  """Gets the variable type map from the goal

  Args:
    goal (List[pymimir.Atom]): The goal
      
  Returns:
    Dict[str,str]: A dictionary of the variable names: types
  """
  result = {}
  for atom in goal:
    for term in atom.terms:
      if term.is_variable():
        result[term.name] = term.type
  return result
  
def get_vars_from_mimir_goal(goal: List[pymimir.Atom]) -> List[pymimir.Object]:
  """Gets the variables from the goal

  Args:
      goal (List[pymimir.Atom]): The goal atoms

  Returns:
      List[pymimir.Object]: The list of the variables
  """
  result = set()
  for atom in goal:
    for term in atom.terms:
      if term.is_variable():
        result.add(term)
  return list(result)

def get_var_names_from_mimir_goal(goal: List[pymimir.Atom]):
  """Gets the variables from the goal

  Args:
      goal (List[pymimir.Atom]): The goal

  Returns:
      List[str]: A list of the variable names
  """
  result = set()
  for atom in goal:
    for term in atom.terms:
      if term.is_variable():
        result.add(term.name)
  return result

def is_too_small(nr_objects: int, nr_variables: int):
  # If the number of objects is equal to the number of variables, 
  # then there is only one possible grounding, which is not interesting
  # For Blocks On the number of variables could be the same as the 
  # number of objects, because different groundings will have different
  # costs here
  # This can be solved by not passing the 'remove_too_small_problems' arg
  return nr_objects <= nr_variables

# DUPLICATE OF THE FUNCTION IN utils.py
# NEEDS TO BE HERE BECAUSE OF THE PROBLEMS WITH torch and pymimir imports
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

def replace_vars(
  quant_goal: List[pymimir.Atom], domain_name: DOMAIN, goal_pred: str = None
):
  """Replaces the variables in the quantified goal with new variables.
  The new variables will be different from each other and will depend
  on the domain and the goal predicate

  Args:
    quant_goal (List[pymimir.Atom]): The quantified goal
    domain_name (DOMAIN): The domain
    goal_pred (str, optional): The goal predicate. Defaults to None.

  Raises:
    ValueError: If the goal predicate is not supported for the domain

  Returns:
    List[pymimir.Atom]: A list of atoms in the quantified goal, with the variables replaced
  """
  if domain_name == DOMAIN.BLOCKS: 
    if goal_pred == "on":
      quant_goal = replace_blocks_on_vars(quant_goal)
    elif goal_pred == "clear":
      quant_goal = replace_blocks_clear_vars(quant_goal)
    else:
      raise ValueError(f"Predicate '{goal_pred}' not supported for domain '{domain_name}'")
  elif domain_name == DOMAIN.GRIPPER:
    if goal_pred == "at":
      quant_goal = replace_gripper_at_vars(quant_goal)
    else:
      raise ValueError(f"Predicate '{goal_pred}' not supported for domain '{domain_name}'")
  elif domain_name == DOMAIN.DELIVERY:
    if goal_pred == "at":
      quant_goal = replace_delivery_at_vars(quant_goal)
    else:
      raise ValueError(f"Predicate '{goal_pred}' not supported for domain '{domain_name}'")
  elif domain_name == DOMAIN.VISITALL:
    if goal_pred == "visited":
      quant_goal = replace_visitall_visited_vars(quant_goal)
    else:
      raise ValueError(f"Predicate '{goal_pred}' not supported for domain '{domain_name}'")
  else:
    raise ValueError(f"Domain '{domain_name}' not supported")
  return quant_goal

def replace_blocks_clear_vars(quant_goal: List[pymimir.Atom]):
  """Replaces the variables in the blocks clear predicate with new variables
  Each variable in the new goal will be different

  Args:
      quant_goal (List[pymimir.Atom]): A list of atoms in the quantified goal

  Returns:
      List[pymimir.Atom]: A list of atoms in the quantified goal, with the variables replaced
  """
  new_goal = []
  var_idx = 1
  for _, atom in enumerate(quant_goal):
    for j, term in enumerate(atom.terms):
      new_term = pymimir.Object(var_idx, f"?x{var_idx}", term.type)
      atom = atom.replace_term(j, new_term)
      var_idx += 1
    new_goal.append(atom)
  return new_goal

def replace_blocks_on_vars(quant_goal: List[pymimir.Atom]):
  """Replaces the variables in the blocks on predicates with new variables.
  The first variable in the second atom will be the same as the second variable
  in the first atom. The same is true for the second and third atom, and so on.
  The result will be a stack of blocks.

  Args:
      quant_goal (List[pymimir.Atom]): A list of atoms in the quantified goal

  Returns:
      List[pymimir.Atom]: A list of atoms in the quantified goal, with the variables replaced
  """
  new_goal = []
  var_idx = 1
  for i, atom in enumerate(quant_goal):
    for j, term in enumerate(atom.terms):
      if i > 0 and j == 0:
        # Then we want to use the second argument of the previous atom
        prev_atom = new_goal[i-1]
        prev_term = prev_atom.terms[1]
        atom = atom.replace_term(j, prev_term)
      else:
        # Otherwise, create a new term and add this
        new_term = pymimir.Object(var_idx, f"?x{var_idx}", term.type)
        atom = atom.replace_term(j, new_term)
        var_idx += 1
    new_goal.append(atom)
  return new_goal

# TODO: Refactor these are that are the same for most domains
def replace_delivery_at_vars(quant_goal: List[pymimir.Atom]):
  """Replaces the variables in the quant goal with new variables
  The second arguement is a cell, which should be replaced by a 
  unique variable

  Args:
    quant_goal (List[pymimir.Atom]): The quantified goal

  Returns:
    List[pymimir.Atom]: The quantified goal with variables replaced
  """
  new_goal = []
  var_idx = 1
  for atom in quant_goal:
    new_atom = atom
    # Replace all variable terms with unique variable names
    for j, term in enumerate(atom.terms):
      # Do not replace the constants
      if term.is_variable():
        new_term = pymimir.Object(var_idx, f"?x{var_idx}", term.type)
        new_atom = new_atom.replace_term(j, new_term)
        var_idx += 1
    new_goal.append(new_atom)
  return new_goal

def replace_gripper_at_vars(quant_goal: List[pymimir.Atom]):
  """Replaces the variables in the quant goal with new variables
  The first arguement is a ball, which should be replaced by a 
  unique variable

  Args:
      quant_goal (List[pymimir.Atom]): The quantified goal with variables replaced
  """
  new_goal = []
  var_idx = 1
  for atom in quant_goal:
    new_atom = atom
    # Replace all variable terms with unique variable names
    for j, term in enumerate(atom.terms):
      # Do not replace the constants
      if term.is_variable():
        new_term = pymimir.Object(var_idx, f"?x{var_idx}", term.type)
        new_atom = new_atom.replace_term(j, new_term)
        var_idx += 1
    new_goal.append(new_atom)
  return new_goal

def replace_visitall_visited_vars(quant_goal: List[pymimir.Atom]):
  """Replaces each variable in the visited predicate with a new variable.
  The new variables will be different from each other

  Args:
    quant_goal (List[pymimir.Atom]): A list of atoms in the quantified goal

  Returns:
    List[pymimir.Atom]: A list of atoms in the quantified goal, with the variables replaced
  """
  new_goal = []
  var_idx = 1
  for atom in quant_goal:
    for j, term in enumerate(atom.terms):
      new_term = pymimir.Object(var_idx, f"?x{var_idx}", term.type)
      atom = atom.replace_term(j, new_term)
      var_idx += 1
    new_goal.append(atom)
  return new_goal

def split_files(
  files: List[str], domain: DOMAIN, domain_file: str, nr_obj_splits: List[int], 
  max_nr_objects: int = None
):
  """Splits the files into train, dev and test sets

  Args:
      files (List[str]): List of problem files
      domain (DOMAIN): The domain
      domain_file (str): The domain file
      nr_obj_splits (List[int]): How many objects to keep in each split
      max_nr_objects (int, optional): How many objects to keep at maximum. Defaults to None.

  Raises:
      ValueError: If there are no files with number of objects <= nr_obj_splits[0]
      ValueError: If there are no files with number of objects > nr_obj_splits[0] and <= nr_obj_splits[1]
      ValueError: If there are no files with number of objects > nr_obj_splits[1] and <= max_nr_objects
      ValueError: If there are no files with number of objects > nr_obj_splits[1] and <= max_nr_objects

  Returns:
      Tuple[List[str],List[str],List[str]]: _description_
  """
  train_files = [file for file in files 
                  if get_nr_objs_from_file(file, domain, domain_file) <= nr_obj_splits[0]]
  dev_files = [file for file in files 
                if get_nr_objs_from_file(file, domain, domain_file) > nr_obj_splits[0] and \
                get_nr_objs_from_file(file, domain, domain_file) <= nr_obj_splits[1]]
  test_files = [file for file in files 
                if get_nr_objs_from_file(file, domain, domain_file) > nr_obj_splits[1]]

  if len(train_files) == 0:
    raise ValueError(f"No training files found. There are no files with number of objects equal or below '{nr_obj_splits[0]}'. Please try another object split.")
  if len(dev_files) == 0:
    raise ValueError(f"No valiation files found. There are no files with number of objects larger than '{nr_obj_splits[0]}' and less than or equal to '{nr_obj_splits[1]}'. Please try another object split.")
  if len(test_files) == 0:
    if not max_nr_objects:
      raise ValueError(f"No test files found. There are no files with number of objects > '{nr_obj_splits[1]}'. Please try another object split.")
    raise ValueError(f"No test files found. There are no files with number of objects larger than '{nr_obj_splits[1]}' and less than or equal to '{max_nr_objects}'. Please try another object split.")

  return train_files, dev_files, test_files

