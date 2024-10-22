import os, uuid, random, pymimir
from typing import Dict, List
from tqdm import tqdm
from utils_package import store_json

from constants import COLOR_PREDICATES, DOMAIN
from data_processing.data_generation_args import DataGenerationArgs
from data_processing.data_generator import DataGenerator
from data_processing.dp_utils import (
  convert_atom_list_to_str, filter_fully_quant_data,  
  get_domain_and_problem_files, get_nr_colors, get_nr_of_variables, 
  get_random_initial_pos_atoms, 
  get_supervised_stats, get_var_type_count_map, get_objects
)
from utils import (
  get_default_data_gen_output_folder, get_default_goal_pred, 
  get_default_input_folder, get_nr_of_variables_from_tuple
)

class SupervisedDataGenerator(DataGenerator):
  def __init__(
    self, domain_name: DOMAIN, domain_file: str, files: List[str],
    data_size: int, nr_goals: int, goal_pred: str, 
    max_nr_vars: int, max_nr_objects: int = None, 
    var_type_count_map: Dict[str,int] = None, color_types: str = None, 
    nr_colors: int = None, nr_obj_split: str = None, split_sizes: str = None,
    split_ratio: str = None, max_expanded_states: int = 1000000, 
    partially_grounded: bool = False, recolor_states: bool = False,
    use_neq_preds: bool = False
  ):
    super().__init__(
      domain_file, files, domain_name, nr_goals, goal_pred,
      data_size, max_nr_vars, var_type_count_map=var_type_count_map,
      color_types=color_types, nr_colors=nr_colors, 
      max_expanded_states=max_expanded_states, 
      partially_grounded=partially_grounded, nr_obj_split=nr_obj_split,
      split_sizes=split_sizes, split_ratio=split_ratio,
      max_nr_objects=max_nr_objects, recolor_states=recolor_states,
      use_neq_preds=use_neq_preds
    )    
    self.max_cost = 0
      
  def get_train_data(self):
    train_data = self.get_data_from_files(self.train_files, self.train_size)
    for cost in train_data:
      if not all([d["nr_objects"] <= self.nr_obj_splits[0] for d in train_data[cost]]):
        raise ValueError(f"Train data contains instances with more than {self.nr_obj_splits[0]} objects")
    train_data_list = [d for cost in train_data for d in train_data[cost]]
    if len(train_data_list) != self.train_size:
      print(f"Warning: Size of train_data '{len(train_data_list)}' does not match train split size '{self.train_size}'")
    return train_data
  
  def get_val_data(self):
    val_data = self.get_data_from_files(self.val_files, self.val_size)
    for cost in val_data:
      if not all([d["nr_objects"] > self.nr_obj_splits[0] and d["nr_objects"] <= self.nr_obj_splits[1] for d in val_data[cost]]):
        raise ValueError(f"Validation data contains instances with less than {self.nr_obj_splits[0]} or more than {self.nr_obj_splits[1]} objects")
    val_data_list = [d for cost in val_data for d in val_data[cost]]
    if len(val_data_list) != self.val_size:
      print(f"Warning: Size of val_data '{len(val_data_list)}' does not match dev split size '{self.val_size}'")
    return val_data
  
  def get_test_data(self):
    test_data = self.get_data_from_files(self.test_files, self.test_size)
    for cost in test_data:
      if not all([d["nr_objects"] > self.nr_obj_splits[1] for d in test_data[cost]]):
        raise ValueError(f"Test data contains instances with less than {self.nr_obj_splits[1]} objects")
    test_data_list = [d for cost in test_data for d in test_data[cost]]
    if len(test_data_list) != self.test_size:
      print(f"Warning: Size of test_data '{len(test_data_list)}' does not match test split size '{self.test_size}'")
    return test_data
  
  def get_data_from_files(self, files: List[str], data_size: int = None):
    if data_size == 0:
      return {}
    data_size = data_size if data_size else self.data_size
    
    # Expand state spaces
    state_spaces = self.get_state_spaces(files)
    
    result: Dict[int, List] = {}
    for _ in tqdm(range(data_size)):
      # Sample state space
      state_space_index = random.randint(0, len(state_spaces) - 1)
      state_space = state_spaces[state_space_index]
      # Sample initial state
      init_state = random.choice(state_space.get_states())
      objects = get_objects(state_space.problem, self.domain_name)

      if self.recolor_states:
        init_state_atoms = self.get_recolored_state(
          init_state, objects, state_space.problem
        )
        problem = state_space.problem.replace_initial(init_state_atoms)
        successor_generator = pymimir.GroundedSuccessorGenerator(problem)
        state_space = pymimir.StateSpace.new(
          problem, successor_generator, max_expanded=self.max_expanded_states
        )
        init_state = state_space.get_initial_state()
        init_state_atoms = init_state.get_atoms()
        
      if self.domain_name == DOMAIN.VISITALL:
        # For the visitall domain we need to remove all visited predicates
        # and make sure that the robot starts in an uncolored cell
        init_state_atoms = init_state.get_atoms()
        # Remove the visited predicates
        init_state_atoms = [atom for atom in init_state_atoms
                            if atom.predicate.name != "visited"]
        color_preds = [atom for atom in init_state_atoms
                        if atom.predicate.name in COLOR_PREDICATES]
        init_state_atoms = get_random_initial_pos_atoms(
          color_preds, objects, state_space.problem, self.predicates
        )
        problem = state_space.problem.replace_initial(init_state_atoms)
        successor_generator = pymimir.GroundedSuccessorGenerator(problem)
        state_space = pymimir.StateSpace.new(
          problem, successor_generator, max_expanded=self.max_expanded_states
        )
        init_state = state_space.get_initial_state()
        
      quant_goal = self.get_quant_goal(objects, state_space.problem)
      
      if type(quant_goal) == tuple:
        # This means that the goal is unsolvable, so the 
        # cost should be the max, or -1
        cost = -1
        nr_vars = get_nr_of_variables_from_tuple(quant_goal)
        quant_goal = str(tuple(sorted(quant_goal)))
      else:
        cost = self.get_cost(init_state, quant_goal)
        nr_vars = get_nr_of_variables(quant_goal)
        quant_goal = convert_atom_list_to_str(
          quant_goal, objects, incl_var_preds=True
        )

      if cost not in result:
        result[cost] = []
      if cost > self.max_cost:
        self.max_cost = cost
              
      res_item = {}
      res_item["quant_goal"] = quant_goal
      res_item["nr_objects"] = len(objects)
      res_item["nr_variables"] = nr_vars
      res_item["init_state"] = convert_atom_list_to_str(
        init_state.get_atoms(), objects, incl_const_preds=True
      )
      res_item["cost"] = cost
      res_item["id"] = str(uuid.uuid4())
      res_item["nr_colors"] = get_nr_colors(init_state.get_atoms())
      
      result[cost].append(res_item)

    result = self.remove_negative_costs(result)
    
    return result
              
  def remove_negative_costs(self, result: Dict[int, List[Dict]]):
    if -1 in result:
      result[self.max_cost*2] = []
      for d in result[-1]:
        d["cost"] = self.max_cost*2
        result[self.max_cost*2].append(d)
      del result[-1]
    assert all([d["cost"] >= 0 for cost in result for d in result[cost]]), "Cost cannot be negative"
    return result
    
  def get_cost(self, init_state: pymimir.State, quant_goal: List[pymimir.Atom]):
    problem = init_state.get_problem()
    init_state_atoms = init_state.get_atoms()
    problem = problem.replace_initial(init_state_atoms)
    successor_generator = pymimir.GroundedSuccessorGenerator(problem)
    state_space = pymimir.StateSpace.new(
      problem, successor_generator, max_expanded=self.max_expanded_states
    )
    if not state_space:
      print(f"WARNING: State space for problem '{problem.name}' is larger than {self.max_expanded_states} states. Increase max_expanded_states to include this file.")
      return
    goal_matcher = pymimir.GoalMatcher(state_space)
    _, cost = goal_matcher.best_match(quant_goal)
    return cost

if __name__ == "__main__":
  args = DataGenerationArgs().get_args()
  
  if not args.input_folder:
    input_folder = get_default_input_folder(args.domain)
    print(f"No input folder specified. Using default folder '{input_folder}'")
  else:
    input_folder = args.input_folder
    
  if not args.goal_pred:
    goal_pred = get_default_goal_pred(args.domain)
  else:
    goal_pred = args.goal_pred

  if not args.output_folder:
    output_folder = get_default_data_gen_output_folder(
      args.max_nr_vars, args.domain, args.max_nr_objects, args.dataset_size,
      goal_pred, val_pred=True, 
      use_colors=True if args.nr_colors else False,
      use_neq_preds=args.use_neq_preds, recolor_states=args.recolor_states
    )
    print(f"No output folder specified. Writing to default folder '{output_folder}'")
  else:
    output_folder = args.output_folder
    
  domain_file, files = get_domain_and_problem_files(
    input_folder, args.max_nr_objects, args.domain, 
    args.max_nr_vars, args.remove_too_small_problems, args.nr_colors
  )
  var_type_count_map = get_var_type_count_map(
    args.var_types, args.nr_vars_per_type, args.max_nr_vars
  )
  generator = SupervisedDataGenerator(
    args.domain, domain_file, files, args.dataset_size, args.nr_goals, 
    goal_pred, args.max_nr_vars, args.max_nr_objects, 
    var_type_count_map, args.color_types, args.nr_colors, args.nr_obj_split, 
    args.split_sizes, args.split_ratio, args.max_expanded_states,
    args.partially_grounded, args.recolor_states,
    args.use_neq_preds
  )
    
  if args.nr_obj_split:
    train_data = generator.get_train_data()
    val_data = generator.get_val_data()
    test_data = generator.get_test_data()
  else:
    data = generator.get_data_from_files(files, args.dataset_size)

  stats = get_supervised_stats(train_data, val_data, test_data)

  # Filter out nr_objects and nr_variables from the data
  train_data = filter_fully_quant_data(train_data)
  val_data = filter_fully_quant_data(val_data)
  test_data = filter_fully_quant_data(test_data)
  
  # store_json(result, os.path.join(output_folder, "mul_tgt_data.json"))
  store_json(train_data, os.path.join(output_folder, "train.json"))
  store_json(val_data, os.path.join(output_folder, "dev.json"))
  store_json(test_data, os.path.join(output_folder, "test.json"))
  store_json(stats, os.path.join(output_folder, "stats.json"), sort_keys=True)
  
    