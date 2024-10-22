import random
from typing import Dict, List
import torch
from torch.utils.data import Dataset as TorchDataset
from constants import DOMAIN

from utils import merge_gnn_input
from val_pred.converter import DataConverter


class Dataset(TorchDataset):
  
  def __init__(
    self, data: Dict[int, List[dict]], dom_name: DOMAIN,  
    device="cpu", is_val=False
  ):
    self.dom_name = dom_name
    self.device = device
    self.is_val = is_val
    self.num_items = self.get_num_items(data)
    
    self.converter = DataConverter(
      dom_name, device
    )
    self.data, self.predicates = self.converter.convert(data)
    
    if self.is_val:
      self.data_list = []
      for cost in self.data:
        for d in self.data[cost]:
          self.data_list.append(d)
    
  def __len__(self):
    return self.num_items
  
  def __getitem__(self, idx):
    if self.is_val:
      # Then we want to return the instance at "idx", since we want
      # to evaluate the model on the whole evaluation set
      return self.data_list[idx]
    # 1. Randomize the cost
    # 2. Pick a random instance with that cost
    cost_idx = random.randint(0, len(self.data) - 1)
    cost = list(self.data.keys())[cost_idx]
    sampled_instance = random.choice(self.data[cost])
    return sampled_instance
      
  def get_num_items(self, data):
    return sum([len(data[cost]) for cost in data])

def collate_fn(batch):
  gnn_input = merge_gnn_input([b["gnn_input"] for b in batch])
  ids = [b["id"] for b in batch]
  state_atoms = [b["state_atoms"] for b in batch]
  goal_atoms = [b["goal_atoms"] for b in batch]
  cost = torch.FloatTensor([b["cost"] for b in batch])

  return {
    "gnn_input": gnn_input,
    "id": ids,
    "state_atoms": state_atoms,
    "goal_atoms": goal_atoms,
    "cost": cost,
  }
