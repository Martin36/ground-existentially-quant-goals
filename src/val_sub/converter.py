from base_data_converter import BaseDataConverter

from constants import DOMAIN


class DataConverter(BaseDataConverter):
  
  def __init__(self, dom_name: DOMAIN, device="cpu"):
    super().__init__(dom_name, device)

  def convert(self, data):
    instances = []
    predicates = [(p["idx"], p["arity"]) for p in self.state_predicates]
    predicates += [(p["idx"], p["arity"]) for p in self.goal_predicates]

    for d in data:
      state_atoms = eval(d['init_state'])
      goal_atoms = eval(d['quant_goal'])

      goal_atoms = self.add_rel_preds_to_goal(d["init_state"], goal_atoms)
      
      instances.append({
        "nr_consts": d["nr_objects"],
        "nr_vars": d["nr_variables"],
        "id": d["id"],
        "obj_idxs": self.get_obj_idxs(state_atoms, goal_atoms),
        "state_atoms": state_atoms,
        "goal_atoms": goal_atoms,
        "binding_costs": d["binding_costs"] if "binding_costs" in d else None,
      })
      
    return instances, predicates
    
