from typing import List, Tuple
import torch
import pytorch_lightning as pl

from val_pred.mlp import MLP

class Model(pl.LightningModule):
  
  def __init__(
    self, predicates: List[Tuple[int, int]], use_readout=False, 
  ):
    super().__init__()
    
    # MODEL HYPERPARAMETERS
    embedding_size = 32

    self.save_hyperparameters(logger=False)
    
    self.mlp = MLP(
      embedding_size, predicates, use_readout=use_readout
    )
    
    self.loss = torch.nn.MSELoss()

    self.all_predictions = {}
    # The classification threshold when using BCEWithLogitsLoss
    self.threshold = 0.5
              
  def training_step(self, batch, _):
    output = self.mlp(batch["gnn_input"]).squeeze()
    labels = batch["cost"]
    loss = self.loss(output, labels)
    return loss

  def validation_step(self, batch, _):
    output = self.mlp(batch["gnn_input"]).squeeze()
    labels = batch["cost"]
    loss = self.loss(output, labels)
    for id, prediction in zip(batch["id"], output):
      self.all_predictions[id] = prediction.item()
    self.log("val_loss", loss, prog_bar=True, batch_size=len(batch["id"]))
    return loss
  
  def test_step(self, batch, _):
    output = self.mlp(batch["gnn_input"]).squeeze()
    for id, prediction in zip(batch["id"], output):
      self.all_predictions[id] = prediction.item()
                        
  def configure_optimizers(self):
    lr = 0.0001
    # optimizer = torch.optim.Adam(self.parameters(), lr=lr)
    optimizer = torch.optim.Adam(self.mlp.parameters(), lr=lr)
    return optimizer
  
  def reset_stored_data(self):
    self.all_predictions = {}

  def move_to_device(self, device):
    self.mlp = self.mlp.to(device)
  