import random
from typing import Dict, List, Tuple
import torch

from base_data_converter import BaseDataConverter
from constants import COLOR_PREDICATES, DOMAIN
from utils import (
  get_colors_from_state, get_consts_from_state, get_vars_from_goal
)

class DataConverter(BaseDataConverter):
  
  def __init__(
      self, dom_name: DOMAIN, device="cpu"
  ):
    super().__init__(
      dom_name, device
    )
    
  def convert(self, data: Dict[int, List[dict]]):
    result: Dict[int, List[dict]] = {}
    predicates = [(p["idx"], p["arity"]) for p in self.state_predicates]
    predicates += [(p["idx"], p["arity"]) for p in self.goal_predicates]

    for cost in data:
      for d in data[cost]:
        state_atoms = eval(d['init_state'])
        goal_atoms = eval(d['quant_goal'])
        
        state_atoms = tuple(set(state_atoms))
        goal_atoms = tuple(set(goal_atoms))

        state_atoms, goal_atoms = self.get_permuted_colors(
          state_atoms, goal_atoms
        )
           
        goal_atoms = self.add_rel_preds_to_goal(d["init_state"], goal_atoms)
          
        init_state_dict = self.get_state_dict(state_atoms, goal_atoms)              
        nr_var_and_const = self.get_nr_var_and_const(state_atoms, goal_atoms)
        
        if cost not in result:
          result[cost] = []
        
        result[cost].append({
          "gnn_input": (init_state_dict, nr_var_and_const),
          "id": d["id"],
          "state_atoms": state_atoms,
          "goal_atoms": goal_atoms,
          "cost": d["cost"],
        })
        self.obj_idxs = {}

    return result, predicates
        
  def get_nr_var_and_const(
    self, state_atoms: List[Tuple[str,...]], goal_atoms: List[Tuple[str,...]]
  ):
    nr_consts = len(get_consts_from_state(state_atoms))
    nr_colors = len(get_colors_from_state(state_atoms))
    nr_vars = len(get_vars_from_goal(goal_atoms))
    return torch.LongTensor([nr_vars + nr_consts + nr_colors])
    
  def get_permuted_colors(self, state_atoms, goal_atoms):
    col_idxs = [i for i, _ in enumerate(COLOR_PREDICATES)]
    new_col_idxs = random.sample(col_idxs, len(col_idxs))
    idx_to_col = {i: col for i, col in enumerate(COLOR_PREDICATES)}
    col_to_idx = {col: i for i, col in enumerate(COLOR_PREDICATES)}

    new_state_atoms = []
    for atom in state_atoms:
      if atom[0] in COLOR_PREDICATES:
        col_idx = col_to_idx[atom[0]]
        new_idx = new_col_idxs[col_idx]
        new_state_atoms.append((idx_to_col[new_idx], atom[1]))
      elif atom[0] == "color":
        col_idx = col_to_idx[atom[2]]
        new_idx = new_col_idxs[col_idx]
        new_state_atoms.append((atom[0], atom[1], idx_to_col[new_idx]))
      elif atom[0] == "=" and atom[1] in COLOR_PREDICATES:
        col_idx = col_to_idx[atom[1]]
        new_idx = new_col_idxs[col_idx]
        col = idx_to_col[new_idx]
        new_state_atoms.append(("=", col, col))
      else:
        new_state_atoms.append(atom)

    new_goal_atoms = []
    for atom in goal_atoms:
      if atom[0] in COLOR_PREDICATES:
        col_idx = col_to_idx[atom[0]]
        new_idx = new_col_idxs[col_idx]
        new_goal_atoms.append((idx_to_col[new_idx], atom[1]))
      elif atom[0] == "color":
        col_idx = col_to_idx[atom[2]]
        new_idx = new_col_idxs[col_idx]
        new_goal_atoms.append((atom[0], atom[1], idx_to_col[new_idx]))
      else:
        new_goal_atoms.append(atom)
    
    return tuple(new_state_atoms), tuple(new_goal_atoms)