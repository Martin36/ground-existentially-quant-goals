import torch
import torch.nn as nn
import pytorch_lightning as pl

# Imports related to type annotations
from typing import List, Dict, Tuple
from torch.nn.functional import Tensor, mish


class MLP(nn.Module):
    def __init__(self, input_size: int, output_size: int):
        super().__init__()
        self.input_size = input_size
        self.output_size = output_size
        self._residual = nn.Linear(input_size, input_size, True)
        self._output = nn.Linear(input_size, output_size, True)

    def forward(self, input):
        return self._output(input + mish(self._residual(input)))


class RelationMessagePassing(nn.Module):
    def __init__(self, relations: List[Tuple[int, int]], embedding_size: int):
        super().__init__()
        self._embedding_size = embedding_size
        self._relation_mlps = nn.ModuleList()
        for relation, arity in relations:
            assert relation == len(self._relation_mlps)
            input_size = arity * embedding_size
            output_size = arity * embedding_size
            if (input_size > 0) and (output_size > 0):
                mlp = MLP(input_size, output_size)
            else:
                mlp = None
            self._relation_mlps.append(mlp)
        self._update_mlp = MLP(2 * embedding_size, embedding_size)
        self._dummy = nn.Parameter(torch.empty(0))

    def get_device(self):
        return self._dummy.device

    def forward(self, object_embeddings: Tensor, relations: Dict[int, Tensor]) -> Tensor:
        # Compute an aggregated message for each recipient
        outputs = []
        for relation, module in enumerate(self._relation_mlps):
            if (module is not None) and (relation in relations):
                values = relations[relation]
                input = torch.index_select(object_embeddings, 0, values).view(-1, module.input_size)
                output = module(input).view(-1, self._embedding_size)
                object_indices = values.view(-1, 1).expand(-1, self._embedding_size)
                outputs.append((output, object_indices))

        include_self = False
        exps_max = torch.zeros_like(object_embeddings, device=self.get_device())
        for output, indices in outputs:
            exps_max.scatter_reduce_(0, indices, output, "amax", include_self=include_self)
            include_self = True
        exps_max = exps_max.detach()

        exps_sum = torch.full_like(object_embeddings, 1E-16, device=self.get_device())
        for output, indices in outputs:
            max_offsets = exps_max.gather(0, indices).detach()
            exps = torch.exp(12.0 * (output - max_offsets))
            exps_sum.scatter_add_(0, indices, exps)

        # Update states with aggregated messages
        max_msg = ((1.0 / 12.0) * torch.log(exps_sum)) + exps_max
        next_object_embeddings = self._update_mlp(torch.cat([max_msg, object_embeddings], dim=1))
        return next_object_embeddings


class Readout(nn.Module):
    def __init__(self, input_size: int, output_size: int):
        super().__init__()
        self._value = MLP(input_size, output_size)

    def forward(self, batch_num_objects: List[int], object_embeddings: Tensor) -> Tensor:
        cumsum_indices = torch.tensor(batch_num_objects, device=object_embeddings.device).cumsum(0) - 1
        cumsum_states = object_embeddings.cumsum(0).index_select(0, cumsum_indices)
        aggregated_embeddings = torch.cat((cumsum_states[0].view(1, -1), cumsum_states[1:] - cumsum_states[0:-1]))
        return self._value(aggregated_embeddings)


class RelationMessagePassingModel(nn.Module):
    def __init__(self, relations: list, embedding_size: int, num_layers: int):
        super().__init__()
        self._embedding_size = embedding_size
        self._num_layers = num_layers
        self._relation_network = RelationMessagePassing(relations, embedding_size)
        self._global_readout = Readout(embedding_size, embedding_size)
        self._readout_update = MLP(2 * embedding_size, embedding_size)
        self._dummy = nn.Parameter(torch.empty(0))

    def get_device(self):
        return self._dummy.device

    def forward(self, states: Tuple[Dict[int, Tensor], List[int]]) -> Tensor:
        object_embeddings = self._initialize_nodes(sum(states[1]))
        object_embeddings = self._pass_messages(object_embeddings, states[0], states[1])
        return object_embeddings

    def _pass_messages(self, object_embeddings: Tensor, relations: Dict[int, Tensor], batch_num_objects: List[int]) -> Tensor:
        for _ in range(self._num_layers):
            object_embeddings = self._relation_network(object_embeddings, relations)
            readout = self._global_readout(batch_num_objects, object_embeddings)
            readout_msg = torch.cat([readout[index].expand(num_objects, -1) for index, num_objects in enumerate(batch_num_objects)], dim=0)
            update_msg = torch.cat((object_embeddings, readout_msg), dim=1)
            object_embeddings = self._readout_update(update_msg)
        return object_embeddings

    def _initialize_nodes(self, num_objects: int) -> Tensor:
        init_zeroes = torch.zeros((num_objects, (self._embedding_size // 2) + (self._embedding_size % 2)), dtype=torch.float, device=self.get_device())
        init_random = torch.randn((num_objects, self._embedding_size // 2), device=self.get_device())
        init_nodes = torch.cat([init_zeroes, init_random], dim=1)
        return init_nodes


class MaxReadoutModelBase(pl.LightningModule):
    def __init__(self, predicates: list, embedding_size: int, num_layers: int,
                 output_graph_state=False):
        super().__init__()
        self.output_graph_state = output_graph_state
        self.save_hyperparameters()
        self._model = RelationMessagePassingModel(predicates, embedding_size, num_layers)
        self._readout = Readout(embedding_size, 1)

    def forward(self, states: Tuple[Dict[int, Tensor], List[int]]) -> Tensor:
        object_embeddings = self._model(states)
        if self.output_graph_state:
            return torch.abs(self._readout(states[1], object_embeddings))
        else:
            return object_embeddings
