import random
from typing import Dict, List, Tuple
import pytorch_lightning as pl

from constants import COLOR_PREDICATES, DOMAIN
from utils import get_consts_from_state, get_vars_from_goal, replace_grounded_var


class Model(pl.LightningModule):
  def __init__(self, domain_name: DOMAIN, constrained_bindings=False):
    super().__init__()

    self.constrained_bindings = constrained_bindings
    self.var_type = self.get_default_var_type(domain_name)

    self.all_predictions: Dict[str, dict] = {}
    self.all_goal_scores = {}

  def get_default_var_type(self, domain_name: DOMAIN):
    if domain_name == DOMAIN.BLOCKS:
      return None
    if domain_name == DOMAIN.GRIPPER:
      return "ball"
    if domain_name == DOMAIN.DELIVERY:
      return "cell"
    if domain_name == DOMAIN.VISITALL:
      return "place"
    raise ValueError(f"Unknown domain: {domain_name}")

  def get_obj_color(self, obj: str, atoms: List[Tuple[str,...]]):
    for atom in atoms:
      if atom[0] in COLOR_PREDICATES and atom[1] == obj:
        return atom[0]
    return None
  
  def is_valid_binding(
    self, var: str, const: str, state_atoms: List[Tuple[str,...]], 
    goal_atoms: List[Tuple[str,...]]):
    # The const needs to be the same type as the var
    # and the same color
    
    # Check that the color is correct
    var_col = self.get_obj_color(var, goal_atoms)
    const_col = self.get_obj_color(const, state_atoms)
    if not var_col == const_col:
      return False
    
    if any(atom[0] == "neq" for atom in goal_atoms):
      neq_atoms = [atom for atom in goal_atoms if atom[0] == "neq"]
      for neq_atom in neq_atoms:
        if neq_atom[1] == var:
          # Then the second argument should not be the constant
          if neq_atom[2] == const:
            return False
        if neq_atom[2] == var:
          # Then the first argument should not be the constant
          if neq_atom[1] == const:
            return False

    # Check that the type is correct
    if self.var_type is None:
      return True
    # TODO: This is not easy to determine from the state and goal,
    # since there are no types present there
    # It is only determinable with additional information about
    # the domain, which we would rather not hardcode in
    return True

  def get_next_goals(self, goal_atoms: List[Tuple[str,...]], state_atoms: List[Tuple[str,...]]):
    # Here we want to limit the next goals to only those where
    # the variables is grounded to a constant of the 
    # correct type
    next_goals = []
    bindings = []
    vars = get_vars_from_goal(goal_atoms)
    consts = get_consts_from_state(state_atoms)
    for var in vars:
      for const in consts:
        if self.constrained_bindings and \
           self.is_valid_binding(var, const, state_atoms, goal_atoms):
          next_goals += [tuple(replace_grounded_var(
            goal_atoms, var, const
          ))]
          bindings += [{var: const}]
        elif not self.constrained_bindings:
          next_goals += [tuple(replace_grounded_var(
            goal_atoms, var, const
          ))]
          bindings += [{var: const}]
          
    return next_goals, bindings


  def test_step(self, batch, _):
    # Assuming that the batch size is 1
    if not len(batch["id"]) == 1:
      raise ValueError("Batch size must be 1")
    next_goal = batch["goal_atoms"][0]
    state_atoms = batch["state_atoms"][0]
    id = batch["id"][0]
    
    for id in batch["id"]:
      self.all_predictions[id] = {}
      self.all_goal_scores[id] = {}
    
    while True:
      next_goals, next_goals_bindings = self.get_next_goals(next_goal, state_atoms)
      if len(next_goals) == 0:
        break

      # Get a random next goal
      next_goal_idx = random.randint(0, len(next_goals) - 1)                        
      next_goal = next_goals[next_goal_idx]
      binding = next_goals_bindings[next_goal_idx]
      self.all_predictions[id].update(binding)

      next_goal = [atom for atom in next_goal if atom[0] != "same-color"]
      if len(get_vars_from_goal(next_goal)) == 0:
        break
    
  def reset_stored_data(self):
    self.all_predictions = {}
