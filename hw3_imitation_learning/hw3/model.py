"""Model definitions for SO-100 imitation policies."""

from __future__ import annotations

import abc
from typing import Literal, TypeAlias

import torch
from torch import nn
import torch.nn.functional as F


class BasePolicy(nn.Module, metaclass=abc.ABCMeta):
    """Base class for action chunking policies."""

    def __init__(self, state_dim: int, action_dim: int, chunk_size: int) -> None:
        super().__init__()
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.chunk_size = chunk_size

    @abc.abstractmethod
    def compute_loss(
        self, state: torch.Tensor, action_chunk: torch.Tensor
    ) -> torch.Tensor:
        """Compute training loss for a batch."""

    @abc.abstractmethod
    def sample_actions(
        self,
        state: torch.Tensor,
    ) -> torch.Tensor:
        """Generate a chunk of actions with shape (batch, chunk_size, action_dim)."""


# TODO: Students implement ObstaclePolicy here.
class ObstaclePolicy(BasePolicy):
    """Predicts action chunks with an MSE loss.

    A simple MLP that maps a state vector to a flat action chunk
    (chunk_size * action_dim) and reshapes to (B, chunk_size, action_dim).
    """

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        chunk_size: int,
        hidden_sizes: list[int] = [512, 512, 512],
        **kwargs
    ) -> None:
        super().__init__(state_dim=state_dim, action_dim=action_dim, chunk_size=chunk_size)
        self.chunk_size = chunk_size
        self.action_dim = action_dim

        layers = []
        input_dim = state_dim
        for h in hidden_sizes:
            layers.append(nn.Linear(input_dim, h))
            layers.append(nn.ReLU())
            input_dim = h
        layers.append(nn.Linear(input_dim, chunk_size * action_dim))

        self.mlp = nn.Sequential(*layers)

    def forward(
        self,
        state: torch.Tensor
    ) -> torch.Tensor:
        """Return predicted action chunk of shape (B, chunk_size, action_dim)."""
        flat_pred = self.mlp(state)  # shape (B, chunk_size * action_dim)
        B = state.shape[0]
        return flat_pred.view(B, self.chunk_size, self.action_dim)

    def compute_loss(
        self,
        state: torch.Tensor,
        action_chunk: torch.Tensor
    ) -> torch.Tensor:
        pred = self.forward(state)
        return F.mse_loss(pred, action_chunk)
        

    def sample_actions(
        self,
        state: torch.Tensor,
    ) -> torch.Tensor:
        return self.forward(state)



# TODO: Students implement MultiTaskPolicy here.
class MultiTaskPolicy(BasePolicy):
    """Goal-conditioned policy for the multicube scene."""

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        chunk_size: int,
        dim_ff: int = 512,
        dropout: float=0.1,
        state_mean: list[float] | torch.Tensor = None,
        state_std: list[float] | torch.Tensor = None,
        action_mean: list[float] | torch.Tensor = None,
        action_std: list[float] | torch.Tensor = None,
        **kwargs
    ) -> None:
        super().__init__(state_dim=state_dim, action_dim=action_dim, chunk_size=chunk_size)
        self.chunk_size = chunk_size 
        self.action_dim = action_dim
        self.ee_params=4
        self.environment_params=12

        #End effector network
        self.ee_net = nn.Sequential(
            nn.Linear(self.ee_params, dim_ff//2),
            nn.LayerNorm(dim_ff//2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(dim_ff//2, dim_ff//4),
            nn.LayerNorm(dim_ff//4),
            nn.ReLU(),
            nn.Dropout(dropout)

        )

        #Environment network
        self.environment_net = nn.Sequential(
            nn.Linear(self.environment_params, dim_ff),
            nn.LayerNorm(dim_ff),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(dim_ff, dim_ff//2),
            nn.LayerNorm(dim_ff//2),
            nn.ReLU(),
            nn.Dropout(dropout)
        )

        #Processing
        self.final = nn.Sequential(
            nn.Linear(3*dim_ff//4, dim_ff),
            nn.LayerNorm(dim_ff),
            nn.ReLU(),
            nn.Linear(dim_ff, dim_ff),
            nn.LayerNorm(dim_ff),
            nn.ReLU(),
            nn.Linear(dim_ff, dim_ff//2),
            nn.LayerNorm(dim_ff//2),
            nn.ReLU(),
            nn.Linear(dim_ff//2, dim_ff//2),
            nn.LayerNorm(dim_ff//2),
            nn.ReLU(),
            nn.Linear(dim_ff//2, chunk_size * action_dim)
        )


        s_mean = torch.tensor(state_mean) if state_mean is not None else torch.zeros(state_dim)
        s_std = torch.tensor(state_std) if state_std is not None else torch.ones(state_dim)
        a_mean = torch.tensor(action_mean) if action_mean is not None else torch.zeros(action_dim)
        a_std = torch.tensor(action_std) if action_std is not None else torch.ones(action_dim)

        s_std = torch.clamp(s_std, min=1e-5)
        a_std = torch.clamp(a_std, min=1e-5)

        self.register_buffer('state_mean', s_mean)
        self.register_buffer('state_std', s_std)
        self.register_buffer('action_mean', a_mean)
        self.register_buffer('action_std', a_std)


    def compute_loss(
        self,
        state: torch.Tensor,
        action_chunk: torch.Tensor
    ) -> torch.Tensor:
        state_norm = (state - self.state_mean) / self.state_std
        action_chunk_norm = (action_chunk - self.action_mean) / self.action_std

        # Noise injection 
        noise = torch.randn_like(state_norm) * 0.02 
        noise[:, 3:4] = 0.0    #Do not affect gripper and encoding
        noise[:, 13:16] = 0.0
        state_norm = state_norm + noise

        pred_norm = self.forward(state_norm)
    
        pos_loss = F.mse_loss(pred_norm[..., :3], action_chunk_norm[..., :3])
        grip_loss = F.mse_loss(pred_norm[..., 3:], action_chunk_norm[..., 3:])

        return pos_loss + 2.0 * grip_loss

    def sample_actions(
        self,
        state: torch.Tensor,
    ) -> torch.Tensor:
        state_norm = (state - self.state_mean) / self.state_std
        
        pred_norm = self.forward(state_norm)
        
        action_real = (pred_norm * self.action_std) + self.action_mean
        return action_real

    def forward(
        self,
        state: torch.Tensor
    ) -> torch.Tensor:
        """Return predicted action chunk of shape (B, chunk_size, action_dim)."""
        ee=state[:,0:3]
        gripper=state[:,3:4]
        red_cube=state[:,4:7]
        green_cube=state[:,7:10]
        blue_cube=state[:,10:13]
        encoding=state[:,13:16]
        goal=state[:,16:]

        target = torch.einsum('bi,bij->bj', encoding, torch.stack([red_cube, green_cube, blue_cube], dim=1))
        
        rel_target_pos=target-ee
        goal_target = goal - target
        rel_bin_pos=goal-ee

        output_ee=self.ee_net(torch.cat([ee,gripper],dim=1))
        output_environment=self.environment_net(torch.cat([rel_target_pos,goal_target,rel_bin_pos,encoding],dim=1))

        x = torch.cat([output_ee, output_environment], dim=1)
        out = self.final(x)
        return out.view(-1, self.chunk_size, self.action_dim)
        

PolicyType: TypeAlias = Literal["obstacle", "multitask"]


def build_policy(
    policy_type: PolicyType,
    *,
    state_dim: int,
    action_dim: int,
    chunk_size: int,
    **kwargs
) -> BasePolicy:
    if policy_type == "obstacle":
        return ObstaclePolicy(
            action_dim=action_dim,
            state_dim=state_dim,
            chunk_size=chunk_size,
            **kwargs
            # TODO: Build with your chosen specifications
        )
    if policy_type == "multitask":
        return MultiTaskPolicy(
            action_dim=action_dim,
            state_dim=state_dim,
            chunk_size=chunk_size,
            **kwargs
            # TODO: Build with your chosen specifications
        )
    raise ValueError(f"Unknown policy type: {policy_type}")