from typing import Dict, Tuple

import torch
from constants import COLOR_PREDICATES, DOMAIN
from utils import (
  get_color_map_from_tuples, get_domain_predicates, 
  get_object_names_from_state_str, get_object_names_from_state_tuple, 
  get_vars_from_goal
)


class BaseDataConverter:
  """Base class for data converters. 
  """
  def __init__(
    self, dom_name: DOMAIN, device="cpu"
  ):
    self.dom_name = dom_name
    self.device = device
    self.state_predicates, self.goal_predicates = self.get_predicates()
    self.pred_to_idx = {pred["name"]: pred["idx"] 
                        for pred in self.state_predicates}
    self.goal_pred_to_idx = {pred["name"]: pred["idx"] 
                             for pred in self.goal_predicates}
    self.obj_idxs = {}
  
  def get_predicates(self):
    state_predicates = get_domain_predicates(self.dom_name)
    
    last_pred_idx = state_predicates[-1]["idx"]
    state_predicates.extend([
      {"name": color, "arity": 1, "idx": i + 1 + last_pred_idx} 
      for i, color in enumerate(COLOR_PREDICATES)
    ])

    last_pred_idx = state_predicates[-1]["idx"]
    goal_predicates = [
      {"name": d["name"], "arity": d["arity"], "idx": i + 1 + last_pred_idx} 
      for i, d in enumerate(state_predicates)
    ]
    
    last_pred_idx = goal_predicates[-1]["idx"]
    goal_predicates.append(
      {"name": "R", "arity": 2, "idx": 1 + last_pred_idx} 
    )
        
    return state_predicates, goal_predicates

  def convert_atom_list_to_dict(self, atom_list, goal=False, obj_idxs=None):
    state_dict: Dict[int, list] = {}
    for atom in atom_list:
      pred_name = atom[0]
      if goal or pred_name in ["color", "="]:
        pred_idx = self.goal_pred_to_idx[pred_name]
      else:
        pred_idx = self.pred_to_idx[pred_name]
      
      if not pred_idx in state_dict:
        state_dict[pred_idx] = []
      
      curr_obj_idxs = []
      for obj in atom[1:]:
        if obj_idxs:
          curr_obj_idxs.append(obj_idxs[obj])
        else:
          if obj not in self.obj_idxs:
            self.obj_idxs[obj] = len(self.obj_idxs)
          curr_obj_idxs.append(self.obj_idxs[obj])
      state_dict[pred_idx].extend(curr_obj_idxs)

    for key in state_dict:
      state_dict[key] = torch.LongTensor(state_dict[key]).to(self.device)

    return state_dict
  
  def get_state_dict(self, state_atoms, goal_atoms, obj_idxs=None):
    """Converts the list of state and goal atoms to dictionary format
    for input to the GNN.

    Args:
        state_atoms (List[Tuple]): A list of tuples representing the state atoms
        goal_atoms (List[Tuple]): A list of tuples representing the goal atoms

    Returns:
        dict: The dicitionary containting the state and goal atoms
    """
    state_dict = self.convert_atom_list_to_dict(state_atoms, obj_idxs=obj_idxs)
    goal_dict = self.convert_atom_list_to_dict(goal_atoms, goal=True, obj_idxs=obj_idxs)
    state_dict.update(goal_dict)
    return state_dict

  def get_rel_atoms(self, variables, constants):
    rel_atoms = []
    for var in variables:
      for const in constants:
        rel_atoms.append(("R", var, const))
    return rel_atoms

  def add_same_color_preds_to_goal(
    self, init_state: Tuple[Tuple[str,...]], goal_atoms: Tuple[Tuple[str, ...]],
    vars_only = False
  ):
    state_col_map = get_color_map_from_tuples(init_state)
    goal_col_map = get_color_map_from_tuples(goal_atoms)
    if vars_only:
      vars = get_vars_from_goal(goal_atoms)
      if len(goal_col_map) == 0:
        return self.add_same_color_preds_no_colors(init_state, goal_atoms)
      goal_col_map = {var: goal_col_map[var] for var in vars}
    same_col_atoms = self.get_same_col_atoms(goal_col_map, state_col_map)
    goal_atoms += tuple(same_col_atoms)
    return goal_atoms
  
  def get_same_col_atoms(
    self, goal_col_map: Dict[str, str], state_col_map: Dict[str, str]
  ):
    same_col_atoms = []
    for goal_obj in goal_col_map:
      for state_obj in state_col_map:
        if goal_col_map[goal_obj] == state_col_map[state_obj] and \
            state_col_map[state_obj] != state_obj:   # Don't add same-color predicates for the colors
          same_col_atoms.append(("same-color", goal_obj, state_obj))
    return same_col_atoms
    
  def add_same_color_preds_no_colors(
    self, init_state: Tuple[Tuple[str,...]], goal_atoms: Tuple[Tuple[str, ...]]
  ):
    """Adds the same color predicates between all the objects.
    To be used when there are no colors in the goal or state
    """
    vars = get_vars_from_goal(goal_atoms)
    objs = get_object_names_from_state_tuple(init_state) 
    result = []
    for var in vars:
      for obj in objs:
        result.append(("same-color", var, obj))
    return goal_atoms + tuple(result)

  def add_rel_preds_to_goal(self, init_state: str, goal_atoms: Tuple[Tuple[str, ...]]):
    obj_names = get_object_names_from_state_str(init_state)
    vars = get_vars_from_goal(goal_atoms)
    rel_atoms = self.get_rel_atoms(vars, obj_names)
    goal_atoms += tuple(rel_atoms)
    return goal_atoms

  def get_obj_idxs(self, state_atoms, goal_atoms):
    obj_idxs = {}
    for atom in state_atoms + goal_atoms:
      for obj in atom[1:]:
        if obj not in obj_idxs:
          obj_idxs[obj] = len(obj_idxs)
    return obj_idxs