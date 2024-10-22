import random, pymimir
from typing import Dict, List

from constants import COLOR_PREDICATES, DOMAIN, MIMIR_VAR_TO_VAR
from data_processing.dp_utils import (
  add_neq_preds_to_goal, add_var_preds, create_color_atom, 
  get_color_map, get_colorable_objects_visitall, get_objects, get_random_initial_pos_atoms, 
  get_random_object_of_type, get_split_sizes, get_var_type_map_from_goal, 
  get_var_names_from_mimir_goal, replace_vars, 
  split_files
)

class DataGenerator:
  """Base class for data generators
  """

  def __init__(
    self, domain_file: str, files: List[str], domain_name: DOMAIN, nr_goals: int, 
    goal_pred: str, data_size: int, max_nr_vars: int, 
    var_type_count_map: Dict[str,int] = None,
    color_types: str = None, nr_colors: int = None, max_expanded_states: int = 1000000,
    partially_grounded: bool = False, nr_obj_split: str = None, 
    split_sizes: str = None, split_ratio: str = None, max_nr_objects: int = None, 
    nr_goal_split: str = None, recolor_states: bool = False, 
    use_neq_preds: bool = False
  ):
    self.domain_file = domain_file
    self.files = files
    self.domain_name = domain_name
    self.nr_goals = nr_goals
    self.goal_pred = goal_pred
    self.nr_colors = nr_colors
    self.max_expanded_states = max_expanded_states
    self.partially_grounded = partially_grounded
    self.data_size = data_size
    self.max_nr_vars = max_nr_vars
    self.var_type_count_map = var_type_count_map
    self.recolor_states = recolor_states
    self.use_neq_preds = use_neq_preds

    self.add_colors = True if nr_colors else False
    self.domain = pymimir.DomainParser(self.domain_file).parse()
    self.predicates: List[pymimir.Predicate] = self.domain.predicates

    self.color_types = None
    self.color_map = None
    self.var_type_map = None
    if color_types:
      self.color_types = color_types.split(",")

    if nr_obj_split:
      self.nr_obj_splits = [int(nr) for nr in nr_obj_split.split(",")]

      self.train_files, self.val_files, self.test_files = split_files(
        files, domain_name, domain_file, self.nr_obj_splits, max_nr_objects
      )
      self.train_size, self.val_size, self.test_size = get_split_sizes(
        split_sizes, data_size, split_ratio
      )

    self.nr_goal_splits = None
    if nr_goal_split:
      self.nr_goal_splits = [int(nr) for nr in nr_goal_split.split(",")]

    if self.add_colors:
      self.colors = COLOR_PREDICATES[:self.nr_colors]
      
    self.nr_goals_count = {}
    for i in range(1, self.nr_goals + 1):
      self.nr_goals_count[i] = 0
                    
  def get_color_predicates(
      self, problem: pymimir.Problem, objects: Dict[str, str], 
      include_uncolored=False
      ):
    """Gets the color predicates for the objects in the initial state

    Args:
      problem (mimir.Problem): The problem instance
      objects (Dict[str, str]): A dictionary of the objects in the problem instance.
      include_uncolored (bool, optional): If True, will incude objects that are of a color type but are not colored e.g. for the Visitall domain. Defaults to False.

    Returns:
      List[Atom]: A list of the color atoms, represented as pymimir Atoms
    """
    color_preds = []
    self.color_map = {}
    remaining_colors = self.get_remaining_colors(objects)
    # TODO: Have this as a hyperparameter?
    uncolored_threshold = 0.8
    obj_names = list(objects.keys())
    random.shuffle(obj_names)
    
    for obj in obj_names:
      if self.color_types and not objects[obj] in self.color_types:
        # Don't add color predicates for these objects
        continue
      # Add colors to the objects in the initial state
      if len(remaining_colors) > 0:
        color = random.choice(remaining_colors)
        remaining_colors.remove(color)
      else:
        if include_uncolored:
          prob = random.random()
          if prob < uncolored_threshold:
            # Then we skip coloring this object
            continue
        color = random.choice(self.colors)          
      color_atom = create_color_atom(problem, color, obj)
      color_preds.append(color_atom)
      self.color_map[obj] = color

    return color_preds
      
  def get_remaining_colors(self, objects: Dict[str, str]):
    """Gets the colors that remains to be added to the objects.
    
    Args:
      objects (Dict[str, str]): A dictionary of the objects in the problem instance.

    Raises:
      ValueError: If the minimum number of colors is larger than the number of objects

    Returns:
      List[str]: A list of the remaining colors
    """
    if self.domain_name == DOMAIN.VISITALL:
      # Then we want to keep at least one cell without a color
      nr_objs = len(objects)
      if len(self.colors) > nr_objs - 1:
        return self.colors[:nr_objs - 1].copy()
    # If there is no quantified goal, then we just
    # return all the selected colors
    return self.colors.copy()
    
  def add_color_preds(self, problem: pymimir.Problem):
    include_uncolored = self.domain_name == DOMAIN.VISITALL
    objects  = get_objects(problem)
    color_preds = self.get_color_predicates(
      problem, objects, include_uncolored=include_uncolored
    )
    if self.domain_name == DOMAIN.VISITALL:
      initial_atoms = get_random_initial_pos_atoms(
        color_preds, objects, problem, self.predicates
      )
      problem = problem.replace_initial(initial_atoms + color_preds)
    else:
      problem = problem.replace_initial(problem.initial + color_preds)
    return problem
    
  def get_nr_goals(self, objects: Dict[str, str], split: str = None):
    objects = {obj: type for obj, type in objects.items() if type != "color"}
    nr_objects = len(objects)
    
    if self.nr_goal_splits and not split:
      raise ValueError("To be able to use 'nr_goal_splits' the 'split' argument needs to be provided")
    
    if self.nr_goal_splits and split:
      if split == "train":
        nr_goals = self.nr_goal_splits[0]
      elif split == "val":
        nr_goals = self.nr_goal_splits[1]
      elif split == "test":
        nr_goals = self.nr_goal_splits[2]
    else:
      nr_goals = self.nr_goals

    if self.goal_pred == "clear" and nr_goals > nr_objects:
      # Clear goals needs one object per goal
      return nr_objects
    elif self.goal_pred == "on" and nr_goals >= nr_objects:
      # On goals needs nr_goals + 1 objects
      return nr_objects - 1
    elif self.goal_pred == "at":
      if self.domain_name == DOMAIN.DELIVERY:
        # There needs to be at least one package for each goal
        packages = {obj for obj, type in objects.items() if type == "package"}
        if nr_goals > len(packages):
          return len(packages)
      elif self.domain_name == DOMAIN.GRIPPER:
        # There needs to be at least one ball for each goal
        balls = {obj for obj, type in objects.items() if type == "ball"}
        if nr_goals > len(balls):
          return len(balls)
    elif self.goal_pred == "visited":
      # There needs to be at least one place for each goal
      # And there needs to be at least one uncolored place
      # which means that the maximum nr of goals is the number of places - 1
      if nr_goals > len(objects) - 1:
        return len(objects) - 1
        
    return nr_goals

  def get_state_spaces(self, files: List[str]) -> List[pymimir.StateSpace]:
    state_spaces = []
    for file in files:
      problem = pymimir.ProblemParser(file).parse(self.domain)
      if self.add_colors:
        problem = self.add_color_preds(problem)
      if self.use_neq_preds:
        problem = self.add_neq_preds(problem)
      successor_generator = pymimir.GroundedSuccessorGenerator(problem)
      state_space = pymimir.StateSpace.new(
        problem, successor_generator, max_expanded=self.max_expanded_states
      )
      if not state_space:
        print(f"WARNING: State space is larger than {self.max_expanded_states} states. Skipping problem file '{file}'. Increase max_expanded_states to include this file.")
        continue
      state_spaces.append(state_space)
    return state_spaces
  
  def add_neq_preds(self, problem: pymimir.Problem):
    objects = problem.objects
    state_atoms = problem.initial
    neq_pred = [pred for pred in self.predicates if pred.name == "neq"][0]
    for obj1 in objects:
      for obj2 in objects:
        if obj1.name != obj2.name:
          neq_atom = neq_pred.as_atom()
          neq_atom = neq_atom.replace_term(0, obj1)
          neq_atom = neq_atom.replace_term(1, obj2)
          state_atoms.append(neq_atom)
    problem = problem.replace_initial(state_atoms)
    return problem
  
  def get_quant_goal_predicates(self):
    result = []
    for pred in self.domain.predicates:
      if pred.name == self.goal_pred:
        result.append(pred)
    return result
      
  def add_quant_goal_color_atoms(
    self, quant_goal: List[pymimir.Atom], color_map: Dict[str, str],
    problem: pymimir.Problem
  ):
    # Get a color atom for each variable in the quantified goal
    remaining_var_colors = [color for color in color_map.values()]
    unique_colors = list(set(remaining_var_colors))
    new_atoms = []
    self.var_color_map = {}
    for atom in quant_goal:
      for term in atom.terms:
        if term.is_constant():
          # Then it is already colored
          continue
        # Check if term is colorable
        if not self.color_types or term.type.name in self.color_types:
          if term.name in self.var_color_map:
            # Then this variable has already been assigned a color
            continue
          if len(unique_colors) > 0:
            # Make sure that there is at least one variable of each color
            col = random.choice(unique_colors)
            unique_colors.remove(col)
          else:
            col = random.choice(remaining_var_colors)
          # Create a color predicate with the sample color
          color_atom = self.get_ground_color_atom(term, col, problem)
          new_atoms.append(color_atom)
          self.var_color_map[term.name] = col
          remaining_var_colors.remove(col)          
    return new_atoms + quant_goal
  
  def get_ground_color_atom(
    self, term: pymimir.Object, color: str, problem: pymimir.Problem
  ):
    color_pred = [pred for pred in self.predicates if pred.name == color][0]
    color_atom = color_pred.as_atom()
    color_atom = color_atom.replace_term(0, term)
    return color_atom 

  def get_random_atoms(self, objects: Dict[str, str], split: str = None):
    nr_goals = self.get_nr_goals(objects, split=split)
    goal_preds = self.get_quant_goal_predicates()
    def get_random_atom():
      return goal_preds[random.randint(0, len(goal_preds) - 1)].as_atom()
    # As there will be more state spaces that can have fewer variables
    # we want to prefer to have more atoms for these instances, if 
    # there are more instances with fewer variables
    if self.is_lowest_nr_goals(nr_goals):    
      nr_atoms = nr_goals
    else:
      nr_atoms = random.randint(1, nr_goals)
    self.nr_goals_count[nr_atoms] += 1
    quant_goal = [get_random_atom() for _ in range(nr_atoms)]
    return quant_goal
  
  def is_lowest_nr_goals(self, nr_goals: int):
    if len(self.nr_goals_count) == 0:
      return False
    min_nr_goals_count = min(self.nr_goals_count.values())
    return self.nr_goals_count[nr_goals] == min_nr_goals_count    
      
  def get_quant_goal(
    self, objects: Dict[str, str], problem: pymimir.Problem, split: str = None
  ):
    quant_goal = self.get_random_atoms(objects, split=split)
    # Replace the non-var type variables with constants
    if self.domain_name == DOMAIN.DELIVERY:
      quant_goal = self.replace_non_var_types_delivery(quant_goal, problem)
    else:
      quant_goal = self.replace_non_var_types(quant_goal, problem)
    quant_goal = replace_vars(quant_goal, self.domain_name, self.goal_pred)
    # The colors need to be added before variables are replaced with constants
    # Otherwise there will be no states where the variables are grounded to
    # incorrect colors, and the model will not learn how to predict these states
    if self.add_colors:
      color_map = get_color_map(problem.initial)
      if self.domain_name == DOMAIN.VISITALL:
        # If the domain is visitall, then it could be the case that there are not 
        # enough colored cells for the current quantified goal.
        # Therefore, the goal predicates needs to be reduced
        if len(color_map) < len(quant_goal):
          quant_goal = quant_goal[:len(color_map)]
      quant_goal = self.add_quant_goal_color_atoms(quant_goal, color_map, problem)
    if self.use_neq_preds:
      quant_goal = add_neq_preds_to_goal(quant_goal, self.predicates)
    if self.partially_grounded:
      quant_goal = self.replace_vars_with_consts(quant_goal, problem)
    if self.var_type_count_map and type(quant_goal) != tuple:
      self.var_type_map = get_var_type_map_from_goal(quant_goal)
    return quant_goal
      
  def replace_non_var_types(
    self, quant_goal: List[pymimir.Atom], problem: pymimir.Problem
  ):
    # Replace the non-var type variables with constants
    if not self.var_type_count_map:
      return quant_goal
    objects = problem.objects
    new_goal = []
    var_types = list(self.var_type_count_map.keys())
    for atom in quant_goal:
      new_atom = atom
      for i, term in enumerate(atom.terms):
        if term.is_variable() and term.type.name not in var_types:
          obj = get_random_object_of_type(term.type.name, objects)
          new_atom = new_atom.replace_term(i, obj)
      new_goal.append(new_atom)
    return new_goal

  def replace_non_var_types_delivery(
    self, quant_goal: List[pymimir.Atom], problem: pymimir.Problem
  ):
    # This domain needs to be handled separately, since 
    # the variables are cells and the constants in the "at" predicate
    # are packages, where we can only have one package at one location
    # so we need to make sure that ther is only a single package at each
    # location
    if not self.var_type_count_map:
      return quant_goal
    objects = problem.objects
    new_goal = []
    used_objs = []
    var_types = list(self.var_type_count_map.keys())
    for atom in quant_goal:
      new_atom = atom
      for i, term in enumerate(atom.terms):
        if term.is_variable() and term.type.name not in var_types:
          obj = get_random_object_of_type(term.type.name, objects, used_objs)
          used_objs.append(obj.name)
          new_atom = new_atom.replace_term(i, obj)
      new_goal.append(new_atom)
    return new_goal          
    
  def replace_vars_with_consts(
    self, quant_goal: List[pymimir.Atom], problem: pymimir.Problem
  ):
    # 1. sample nr of variables to change 
    # 2. sample one object for each variable (with replacement)
    # 3. replace the variables with the sampled objects
    mimir_objs = problem.objects
    # Randomly replace the variables with constants
    vars = get_var_names_from_mimir_goal(quant_goal)
    nr_vars_to_replace = random.randint(0, len(vars))
    if nr_vars_to_replace == 0:
      return quant_goal
    
    repl_vars = random.sample(vars, k=nr_vars_to_replace)
    repl_objs = random.choices(mimir_objs, k=nr_vars_to_replace)
    var_obj_map = {var: obj for var, obj in zip(repl_vars, repl_objs)}
    new_goal = self.replace_vars_from_var_cost_map(quant_goal, var_obj_map)
    return new_goal
  
  def replace_vars_from_var_cost_map(
    self, quant_goal: List[pymimir.Atom], var_obj_map: Dict[str, pymimir.Object]
  ):
    new_goal = []
    for atom in quant_goal:
      new_atom = atom
      for i, term in enumerate(atom.terms):
        if term.name in var_obj_map:
          # Using the fact that any term with the wrong type will lead to 
          # an unsolvable goal
          if term.type.name != var_obj_map[term.name].type.name:
            # Then one way would be to just create the tuple goal directly
            # since we know that for this goal the cost is the max cost
            return self.create_tuple_goal(quant_goal, var_obj_map)
          new_atom = new_atom.replace_term(i, var_obj_map[term.name])
      new_goal.append(new_atom)
    return new_goal

  def create_tuple_goal(
    self, quant_goal: List[pymimir.Atom], var_obj_map: Dict[str, pymimir.Object]
  ):
    # Creates the tuple goal directly
    # This is for the cases where at least one variable is 
    # grounded to an object of the wrong type
    # Then this case cannot be handled by Mimir
    tuple_goal = []
    vars = []
    self.var_type_map = {}
    for atom in quant_goal:
      tup_atom = []
      tup_atom.append(atom.predicate.name)
      for term in atom.terms:
        if term.name in var_obj_map:
          tup_atom.append(var_obj_map[term.name].name)
        else:
          if term.is_constant():
            tup_atom.append(term.name)
          else:
            tup_atom.append(MIMIR_VAR_TO_VAR[term.name])
          if term.is_variable():
            vars.append(MIMIR_VAR_TO_VAR[term.name])
            self.var_type_map[MIMIR_VAR_TO_VAR[term.name]] = term.type.name
      tuple_goal.append(tuple(tup_atom))
    vars = set(vars)
    # Might be duplicate atoms, filter these out
    tuple_goal = list(set(tuple_goal))
    tuple_goal = add_var_preds(tuple_goal, vars)
    return tuple_goal

  def get_recolored_state(
    self, init_state: pymimir.State, objects: Dict[str, str],
    problem: pymimir.Problem
  ) -> List[pymimir.Atom]:
    # Remove the old color predicates
    init_state_atoms = init_state.get_atoms()
    init_state_atoms = [atom for atom in init_state_atoms 
                        if atom.predicate.name not in COLOR_PREDICATES \
                          and atom.predicate.name != "color"]
    
    color_map = self.get_color_map(objects)
    
    predicates = problem.domain.predicates
    for obj_name, color in color_map.items():
      pred = [p for p in predicates if p.name == color][0]
      obj = [o for o in problem.objects if o.name == obj_name][0]
      atom = pred.as_atom()
      atom = atom.replace_term(0, obj)
      init_state_atoms.append(atom)
    return init_state_atoms

  def get_color_map(self, objects: Dict[str, str]):
    color_map = {}

    colorable_objects = self.get_colorable_objects(objects)
    color_obj_names = [name for name, t in objects.items()
                       if t == "color"]
    # Sample the number of colors to use
    # nr_colors = random.randint(2, self.nr_colors)
    nr_colors = random.randint(1, self.nr_colors)
    nr_colors = min(nr_colors, len(colorable_objects))

    # if nr_colors <= 1:
    #   raise ValueError("The number of colors should be at least 2")

    obj_names = list(colorable_objects.keys())
    random.shuffle(obj_names)

    if len(color_obj_names) != 0:
      remaining_colors = random.sample(color_obj_names, nr_colors)
    else:
      remaining_colors = COLOR_PREDICATES[:nr_colors]
    colors = remaining_colors.copy()
    for obj_name in obj_names:
      if remaining_colors:
        # Make sure that there is at least one object of each color
        color = random.choice(remaining_colors)
        color_map[obj_name] = color
        remaining_colors.remove(color)
      else:
        color = random.choice(colors)
        color_map[obj_name] = color
    return color_map

  def get_colorable_objects(self, objects: Dict[str, str]):
    colorable_objects = {}
    # Remove the color objects, if any
    colorable_objects = {n: t for n, t in objects.items() if t != "color"}
    if self.domain_name == DOMAIN.VISITALL:
      return get_colorable_objects_visitall(colorable_objects)
    if not self.color_types:
      return colorable_objects
    colorable_objects = {n: t for n, t in colorable_objects.items() 
                         if t in self.color_types}
    return colorable_objects
