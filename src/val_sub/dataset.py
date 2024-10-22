from torch.utils.data import Dataset as TorchDataset
from constants import DOMAIN

from val_sub.converter import DataConverter


class Dataset(TorchDataset):
  
  def __init__(self, data: list, dom_name: DOMAIN, device="cpu"):
    converter = DataConverter(dom_name, device=device)
    self.states, self.predicates = converter.convert(data)
      
  def __len__(self):
    return len(self.states)
  
  def __getitem__(self, idx):
    return self.states[idx]
    
def collate_fn(batch):
  ids = [b["id"] for b in batch]
  obj_idxs = [b["obj_idxs"] for b in batch]
  state_atoms = [b["state_atoms"] for b in batch]
  goal_atoms = [b["goal_atoms"] for b in batch]
  binding_costs = [b["binding_costs"] for b in batch]
  nr_consts = [b["nr_consts"] for b in batch]
  nr_vars = [b["nr_vars"] for b in batch]
  
  return {
    "id": ids,
    "obj_idxs": obj_idxs,
    "state_atoms": state_atoms,
    "goal_atoms": goal_atoms,
    "binding_costs": binding_costs,
    "nr_consts": nr_consts,
    "nr_vars": nr_vars,
  }
