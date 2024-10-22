from collections import defaultdict
import os, random, uuid, pymimir
from typing import Dict, List
from datetime import datetime
from tqdm import tqdm
from utils_package import store_json
from math import factorial

from constants import COLOR_PREDICATES, DOMAIN
from data_processing.data_generation_args import DataGenerationArgs
from data_processing.data_generator import DataGenerator
from data_processing.dp_utils import (
  convert_atom_list_to_str, convert_mimir_binding_to_dict, 
  get_domain_and_problem_files, get_max_nr_vars, get_nr_colors, 
  get_nr_of_packages_from_state_space, get_nr_of_variables, get_random_initial_pos_atoms, get_stats, 
  get_var_type_count_map, get_objects
)
from utils import (
  get_binding_str, get_default_goal_pred, get_default_input_folder, 
  get_default_data_gen_output_folder
)
class StateSpaceDataCreator(DataGenerator):
  
  def __init__(
    self, domain_name: DOMAIN, domain_file: str, files: List[str], 
    data_size: int, nr_goals: int, goal_pred: str, 
    max_nr_vars: int, var_type_count_map = None, color_types: str = None, 
    nr_colors: int = None, max_expanded_states: int = 1000000,
    nr_obj_split: str = None, max_nr_objects: int = None, split_sizes: str = None,
    split_ratio: str = None, nr_goal_split: str = None,
    recolor_states: bool = False, use_neq_preds: bool = False
  ):
    super().__init__(
      domain_file, files, domain_name, nr_goals, goal_pred,
      data_size, max_nr_vars, var_type_count_map=var_type_count_map,
      color_types=color_types, nr_colors=nr_colors, 
      max_expanded_states=max_expanded_states,
      nr_obj_split=nr_obj_split,
      max_nr_objects=max_nr_objects, split_sizes=split_sizes,
      split_ratio=split_ratio, nr_goal_split=nr_goal_split,
      recolor_states=recolor_states, use_neq_preds=use_neq_preds
    )
    self.var_count = defaultdict(int)
    self.nr_packages_state_space_map = None
    # This is for determining how many possible bindings there are
    # "with_replacement" should be true, if there can be multiple 
    # variable bindings to the same constant. This is true for all the
    # domains that have multiple variables of the same color in the goal
    # except for blocks-on and there are no "neq" predicates
    self.with_replacement = True
    if self.domain_name == DOMAIN.BLOCKS and self.goal_pred == "on":
      self.with_replacement = False
    if self.use_neq_preds:
      self.with_replacement = False
    # This is the way to tell that variables can have the same color 
    if not self.recolor_states:
      self.with_replacement = False

  def get_train_data(self):
    train_data = self.get_data_from_files(self.train_files, self.train_size, split="train")
    if not all([d["nr_objects"] <= self.nr_obj_splits[0] for d in train_data]):
      raise ValueError(f"Train data contains instances with more than {self.nr_obj_splits[0]} objects")
    if len(train_data) != self.train_size:
      print(f"Warning: Size of train_data '{len(train_data)}' does not match train split size '{self.train_size}'")
    return train_data
  
  def get_val_data(self):
    val_data = self.get_data_from_files(self.val_files, self.val_size, split="val")
    if not all([d["nr_objects"] > self.nr_obj_splits[0] and d["nr_objects"] <= self.nr_obj_splits[1] for d in val_data]):
      raise ValueError(f"Validation data contains instances with less than {self.nr_obj_splits[0]} or more than {self.nr_obj_splits[1]} objects")
    if len(val_data) != self.val_size:
      print(f"Warning: Size of val_data '{len(val_data)}' does not match dev split size '{self.val_size}'")
    return val_data
  
  def get_test_data(self):
    test_data = self.get_data_from_files(self.test_files, self.test_size, split="test")
    if not all([d["nr_objects"] > self.nr_obj_splits[1] for d in test_data]):
      raise ValueError(f"Test data contains instances with less than {self.nr_obj_splits[1]} objects")
    if len(test_data) != self.test_size:
      print(f"Warning: Size of test_data '{len(test_data)}' does not match test split size '{self.test_size}'")
    return test_data
  
  def get_unlabelled_data(self):
    problems = [pymimir.ProblemParser(file).parse(self.domain) for file in self.files]
    if self.add_colors:
      problems = [self.add_color_preds(problem) for problem in problems]
    if self.use_neq_preds:
      problems = [self.add_neq_preds(problem) for problem in problems]

    result = []
    for _ in tqdm(range(self.data_size)):
      problem = random.choice(problems)

      if self.recolor_states:
        state = problem.create_state(problem.initial)
        objects = get_objects(problem, self.domain_name)
        initial_atoms = self.get_recolored_state(state, objects, problem)
        problem = problem.replace_initial(initial_atoms)

      if self.domain_name == DOMAIN.VISITALL:
        # For the visitall domain we need to remove all visited predicates
        # and make sure that the robot starts in an uncolored cell
        init_state_atoms = problem.initial
        # Remove the visited predicates
        init_state_atoms = [atom for atom in init_state_atoms
                            if atom.predicate.name != "visited"]
        color_preds = [atom for atom in init_state_atoms
                        if atom.predicate.name in COLOR_PREDICATES]
        init_state = get_random_initial_pos_atoms(
          color_preds, objects, problem, self.predicates
        )
      else:
        init_state = self.get_init_state_by_random_walk(problem)

      objects = get_objects(problem, self.domain_name)
      quant_goal = self.get_quant_goal(objects, problem)

      res_item = {}
      res_item["quant_goal"] = convert_atom_list_to_str(
        quant_goal, objects, incl_var_preds=True
      )
      res_item["nr_objects"] = len(objects)
      res_item["nr_variables"] = get_nr_of_variables(quant_goal)
      res_item["init_state"] = convert_atom_list_to_str(
        init_state, objects, incl_const_preds=True
      )
      res_item["id"] = str(uuid.uuid4())
      res_item["objects"] = objects
      res_item["nr_goal_colors"] = get_nr_colors(quant_goal)
      res_item["nr_state_colors"] = get_nr_colors(init_state)

      result.append(res_item)

    return result
  
  def get_data_from_files(self, files: List[str], data_size: int = None, split: str = None):
    if data_size == 0:
      return []
    data_size = data_size if data_size else self.data_size
        
    state_spaces = self.get_state_spaces(files)
    self.nr_packages_state_space_map = None
    
    result = []
    for _ in tqdm(range(data_size)):
      if self.domain_name == DOMAIN.DELIVERY:
        state_space = self.sample_state_space_delivery(state_spaces)
      else:
        state_space_index = random.randint(0, len(state_spaces) - 1)
        state_space = state_spaces[state_space_index]
      init_state = random.choice(state_space.get_states())
      init_state_atoms = init_state.get_atoms()
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

      problem = state_space.problem.replace_initial(init_state_atoms)
      successor_generator = pymimir.GroundedSuccessorGenerator(problem)
      state_space = pymimir.StateSpace.new(
        problem, successor_generator, max_expanded=self.max_expanded_states
      )

      quant_goal = self.get_quant_goal(objects, problem, split=split)
      
      non_color_objects = {obj: t for obj, t in objects.items() if t != "color"}
      binding_cost_map = self.get_binding_cost_map(
        quant_goal, state_space, non_color_objects
      )
      if len(binding_cost_map) == 0:
        raise ValueError("No valid bindings found for this state")          
      
      init_state_str = convert_atom_list_to_str(
        init_state_atoms, objects, incl_const_preds=True
      )

      res_item = {}
      res_item["quant_goal"] = convert_atom_list_to_str(
        quant_goal, objects, incl_var_preds=True
      )
      res_item["nr_objects"] = len(non_color_objects)
      res_item["nr_variables"] = get_nr_of_variables(quant_goal)
      res_item["init_state"] = init_state_str
      res_item["binding_costs"] = binding_cost_map
      res_item["id"] = str(uuid.uuid4())
      res_item["objects"] = objects
      res_item["nr_colors"] = get_nr_colors(init_state.get_atoms())

      result.append(res_item)

    return result
                
  def sample_state_space_delivery(self, state_spaces: List[pymimir.StateSpace]):
    # For delivery we need to bias the sampling towards state spaces
    # with more packages, since this is the limiting factor for the number
    # of goals. If we don't do this, there is a high probability that no 
    # instances with 5 and 6 goals are generated.
    if self.nr_packages_state_space_map is None:
      self.nr_packages_state_space_map = {}
      for state_space in state_spaces:
        nr_packages = get_nr_of_packages_from_state_space(state_space)
        if nr_packages not in self.nr_packages_state_space_map:
          self.nr_packages_state_space_map[nr_packages] = []
        self.nr_packages_state_space_map[nr_packages].append(state_space)
    nr_packages = random.choice(list(self.nr_packages_state_space_map.keys()))
    return random.choice(self.nr_packages_state_space_map[nr_packages])
      
  def get_binding_cost_map(
    self, quant_goal: List[pymimir.Atom], state_space: pymimir.StateSpace, 
    objects: Dict[str, str]
  ):
    problem = state_space.problem
    color_map = self.get_color_map_from_problem(problem)
    states = state_space.get_states()
    binding_cost_map = {}
    nr_pos_bindings = self.get_nr_pos_bindings(objects, color_map, quant_goal)
    state_costs = []
    literal_grounder = pymimir.LiteralGrounder(problem, quant_goal)
    for state in states:
      cost = state_space.get_distance_from_initial_state(state)
      if len(state_costs) > 0 and cost < state_costs[-1]:
        raise ValueError("States are not in order of distance from initial state")
      state_costs.append(cost)
      bindings = literal_grounder.ground(state)
      for binding in bindings:
        binding_dict = convert_mimir_binding_to_dict(binding[1])
        binding_str = get_binding_str(binding_dict)
        if binding_str not in binding_cost_map:
          binding_cost_map[binding_str] = cost
          if len(binding_cost_map) == nr_pos_bindings:
            return binding_cost_map
    return binding_cost_map

  def get_color_map_from_problem(self, problem: pymimir.Problem):
    color_map = {}
    for atom in problem.initial:
      if atom.predicate.name in COLOR_PREDICATES:
        color_map[atom.terms[0].name] = atom.predicate.name
      if atom.predicate.name == "color":
        color_map[atom.terms[0].name] = atom.terms[1].name
    return color_map

  def get_nr_pos_bindings(
    self, objects: Dict[str, str], color_map: Dict[str, str],
    quant_goal: List[pymimir.Atom]
  ):
    if self.add_colors:
      return self.get_nr_pos_bindings_with_colors(objects, color_map)
    # The number of possible bindings is the number of ways to sample
    # nr_vars from nr_objs without replacement
    nr_vars = get_nr_of_variables(quant_goal)
    nr_objs = len(objects)
    return int(factorial(nr_objs) / factorial(nr_objs - nr_vars))
  
  def get_nr_pos_bindings_with_colors(
    self, objects: Dict[str, str], color_map: Dict[str, str]
  ):
    result = 0
    for color in self.colors:
      if self.color_types:
        nr_objs = len([obj for obj in objects
                        if objects[obj] in self.color_types and \
                           color_map[obj] == color])
        nr_vars = self.get_nr_vars_with_color_and_var_type(color)
      else:
        nr_objs = len([obj for obj in objects 
                      if obj in color_map and color_map[obj] == color])
        nr_vars = self.get_nr_vars_with_color(color)
      if not result:
        if self.with_replacement:
          result = nr_objs ** nr_vars
        else:
          result = int(factorial(nr_objs) / factorial(nr_objs - nr_vars))          
      else:
        if self.with_replacement:
          result *= nr_objs ** nr_vars
        else:
          result *= int(factorial(nr_objs) / factorial(nr_objs - nr_vars))
    return result
  
  def get_nr_vars_with_color(self, color: str):
    vars = set()
    for var in self.var_color_map:
      if self.var_color_map[var] == color:
        vars.add(var)
    return len(vars)
  
  def get_nr_vars_with_color_and_var_type(self, color: str):
    vars = set()
    for var in self.var_color_map:
      if self.var_color_map[var] == color and var in self.var_type_map:
        vars.add(var)
    return len(vars)
      
  def get_var_type_count_map(self):
    max_nr_vars = sum(self.var_type_count_map.values())
    # Keep at least one variable of each type
    min_nr_vars = len(self.var_type_count_map)
    var_type_count_map = {t: 1 for t in self.var_type_count_map}
    # For all remaining variables, "Hand out" for each type until,
    # the number of variables stated in self.var_type_count_map
    # has been reached
    nr_vars = random.randint(min_nr_vars, max_nr_vars)
    vars_left = nr_vars - min_nr_vars
    while vars_left > 0:
      var_type = random.choice(list(self.var_type_count_map.keys()))
      if var_type_count_map[var_type] < self.var_type_count_map[var_type]:
        var_type_count_map[var_type] += 1
        vars_left -= 1
    return var_type_count_map
  
  def get_init_state_by_random_walk(self, problem: pymimir.Problem):
    successor_generator = pymimir.GroundedSuccessorGenerator(problem)
    nr_steps = random.randint(1, 10)
    state = problem.create_state(problem.initial)
    for _ in range(nr_steps):
      applicable_actions = successor_generator.get_applicable_actions(state)
      action = random.choice(applicable_actions)
      next_state = action.apply(state)
      state = next_state
    return state.get_atoms()


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
      args.max_nr_vars, args.domain, args.max_nr_objects, 
      args.dataset_size, goal_pred, 
      use_colors=True if args.nr_colors else False, use_neq_preds=args.use_neq_preds,
      recolor_states=args.recolor_states
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
  generator = StateSpaceDataCreator(
    args.domain, domain_file, files, args.dataset_size, 
    args.nr_goals, goal_pred, args.max_nr_vars, 
    var_type_count_map=var_type_count_map,
    color_types=args.color_types, nr_colors=args.nr_colors,
    max_expanded_states=args.max_expanded_states,
    nr_obj_split=args.nr_obj_split,
    max_nr_objects=args.max_nr_objects, split_sizes=args.split_sizes,
    split_ratio=args.split_ratio, 
    recolor_states=args.recolor_states, 
    use_neq_preds=args.use_neq_preds,
  )
      
  if args.gen_test_data:
    data = generator.get_unlabelled_data()
    stats = get_stats([], [], data)
    store_json(data, os.path.join(output_folder, "test.json"))
    store_json(stats, os.path.join(output_folder, "stats.json"))
    exit()

  if args.nr_obj_split:
    train_data = generator.get_train_data()
    val_data = generator.get_val_data()
    test_data = generator.get_test_data()
    result = train_data + val_data + test_data
  else:
    result = generator.get_data_from_files(files)

  if args.dataset_size and not len(result) == args.dataset_size:
    print(f"Warning: Expected {args.dataset_size} instances, but got {len(result)}") 

  if not os.path.exists(output_folder):
    os.makedirs(output_folder)   

  stats = get_stats(train_data, val_data, test_data)

  max_nr_vars = get_max_nr_vars(args.max_nr_vars, args.nr_vars_per_type)
  metadata = {
    "Domain": args.domain,
    "Goal predicate": args.goal_pred,
    "Size": args.dataset_size,
    "# Goals": args.nr_goals,
    "# Vars": max_nr_vars,
    "Max # consts": args.max_nr_objects,
    "Obj split": args.nr_obj_split,
    "Split sizes": args.split_sizes,
    "Datetime": str(datetime.now())
  }
  
  store_json(result, os.path.join(output_folder, "mul_tgt_data.json"))
  store_json(train_data, os.path.join(output_folder, "train.json"))
  store_json(val_data, os.path.join(output_folder, "dev.json"))
  store_json(test_data, os.path.join(output_folder, "test.json"))
  store_json(stats, os.path.join(output_folder, "stats.json"))
  store_json(metadata, os.path.join(output_folder, "metadata.json"))    
    