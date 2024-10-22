import torch
from typing import Dict, List, Tuple
import pytorch_lightning as pl

from constants import COLOR_PREDICATES, DOMAIN
from val_sub.converter import DataConverter
from val_pred.model import Model as ValPredModel
from utils import (
  get_consts_from_state, get_vars_from_goal, replace_grounded_var, 
  merge_gnn_input
)

class Model(pl.LightningModule):
  
  def __init__(
    self, predicates: List[Tuple[int, int]], domain: DOMAIN, 
    val_func: ValPredModel, device="cpu"
  ):
    super().__init__()
        
    self.predicates = predicates
    self.domain = domain
    self.val_func = val_func
    
    self.save_hyperparameters(logger=False)

    self.converter = DataConverter(domain, device)
    
    self.all_predictions: Dict[str, dict] = {}
                
  def get_next_goals(self, goal_atoms_list, state_atoms_list):
    """Gets the next goal states for a batch of states. The next goal states are 
    the goal states where a variable has been grounded with a constant.    

    Args:
        batch (dict): The batch
    """     
    next_goals_lists = []
    bindings_lists = []
    for goal_atoms, state_atoms in zip(goal_atoms_list, state_atoms_list):
      next_goals = []
      bindings = []
      vars = get_vars_from_goal(goal_atoms)
      if self.domain == "grid-coloring":
        # Then we only want to ground variables to colors
        consts = ["red", "blue"]
      else:
        consts = get_consts_from_state(state_atoms)        
      for var in vars:
        for const in consts:
          next_goals += [tuple(replace_grounded_var(
            goal_atoms, var, const, # remove_ground_col_atoms=True
          ))]
          bindings += [{var: const}]
      # for i in range(len(next_goals)):
      #   # Remove duplicate atoms from goal
      #   next_goals[i] = tuple(sorted(list(set(next_goals[i]))))
      next_goals_lists.append(next_goals)
      bindings_lists.append(bindings)
    return next_goals_lists, bindings_lists
  
  def get_best_next_goal(
      self, next_goals: List[Tuple[str,...]], state_atoms: List[Tuple[str,...]], 
      obj_idxs: dict
  ):
    gnn_inputs = []
    for goal in next_goals:
      gnn_state = self.converter.get_state_dict(state_atoms, goal, obj_idxs=obj_idxs)
      gnn_inputs.append((gnn_state, torch.LongTensor([len(obj_idxs)]).to(self.device)))
    gnn_input = merge_gnn_input(gnn_inputs, self.device)
    vals = self.val_func.mlp.forward(gnn_input)
    idx = torch.argmin(vals)
    goal_scores = {}
    for i in range(len(next_goals)):
      goal_scores[next_goals[i]] = vals[i].item()
    return next_goals[idx], goal_scores

  def test_step(self, batch, _):
    batch_next_goals = batch["goal_atoms"]
    batch_next_obj_idxs = batch["obj_idxs"]
    
    for id in batch["id"]:
      self.all_predictions[id] = {}
    
    while True:
      batch_potential_next_goals, batch_next_goals_bindings = self.get_next_goals(batch_next_goals, batch["state_atoms"])
      batch_next_goals = []
      fully_grounded = []
      for i in range(len(batch["id"])):
        next_goals = batch_potential_next_goals[i]
        if len(next_goals) == 0:
          fully_grounded.append(True)
          batch_next_goals.append([])
          continue
        
        next_goal_bindings = batch_next_goals_bindings[i]
        state_atoms = batch["state_atoms"][i]
        obj_idxs = batch_next_obj_idxs[i]
        id = batch["id"][i]
                                
        next_goal, goal_scores = self.get_best_next_goal(
          next_goals, state_atoms, obj_idxs
        )
        goal_idx = next_goals.index(next_goal)
        binding = next_goal_bindings[goal_idx]
        next_goal = [atom for atom in next_goal if atom[0] != "same-color"]
        batch_next_goals.append(next_goal)
        self.all_predictions[id].update(binding)
        if len(get_vars_from_goal(next_goal)) == 0:
          fully_grounded.append(True)
        else:
          fully_grounded.append(False)
      if all(fully_grounded):
        break
                    
  def reset_stored_data(self):
    self.all_predictions = {}

  def move_to_device(self, device):
    self.converter.device = device
  