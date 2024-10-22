from typing import List, Tuple
from torch import nn

from lib.pddl_gnn.max_base import MaxModelBase
from lib.pddl_gnn.max_readout_base import MaxReadoutModelBase


class MLP(nn.Module):
  def __init__(
    self, embedding_size: int, predicates: List[Tuple[int, int]],
    use_readout: bool = False
  ):
    super().__init__()
    
    gnn_layers = 30
    if use_readout:
      self.gnn = MaxReadoutModelBase(predicates, embedding_size, gnn_layers, True)
    else:
      self.gnn = MaxModelBase(
        predicates, embedding_size, gnn_layers, output_graph_state=True
      )
       
  def forward(self, gnn_input):
    state = gnn_input
    output = self.gnn(state)
    return output