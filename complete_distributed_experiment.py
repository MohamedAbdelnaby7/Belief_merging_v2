#!/usr/bin/env python3
"""
Complete Standalone Distributed Belief Merging Experiment
Modified to support multiple grid sizes, agent numbers, and NEW MERGE METHODS.
Includes all original classes + distributed execution framework
No external dependencies on your original file
FIXED: Consistent seed generation for proper checkpointing
REVERTED: Using MPC instead of MCTS for better memory management
UPDATED: Using Gurobi for Optimization
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import matplotlib.patches as patches
from matplotlib.animation import FuncAnimation
import json
import time
import pickle
import hashlib
from datetime import datetime
import os
import sys
from pathlib import Path
import logging
from scipy.stats import entropy, pearsonr, spearmanr
# from scipy.optimize import minimize # Scipy minimize replaced by Gurobi
import pandas as pd
import seaborn as sns
from typing import Dict, List, Tuple, Any, Union, Optional
import itertools
import warnings
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor, as_completed
import psutil
import signal
import fcntl
import tempfile
import shutil
from dataclasses import dataclass, asdict, field
import argparse
import socket
import subprocess
import math
import gc
import functools

# Gurobi Import
try:
    import gurobipy as gp # type: ignore
    from gurobipy import GRB # type: ignore
except ImportError:
    print("WARNING: gurobipy not found. Optimization methods will fall back to analytical solutions.")

warnings.filterwarnings('ignore')


# ===================================================================
# TRUST DECAY — sensor-model mutual information and Gaussian prior
# ===================================================================

@functools.lru_cache(maxsize=256)
def compute_I_step(alpha: float, beta: float, n_points: int = 200) -> float:
    """
    Expected mutual information (bits) per step between two agents' binary
    observations of the same target, averaged uniformly over all prior
    probabilities in (0, 1).

    Depends only on sensor parameters alpha and beta.  No observation
    sharing required.  Result is cached per unique (alpha, beta) pair.

    Note on N > 2 agents: this function computes the pairwise quantity
    between exactly two agents.  The calling code scales by (N - 1) to
    account for the number of other agents each agent was disconnected from.
    See merge_beliefs_trust_decay for the scaling.

    Args:
        alpha:    False positive rate.
        beta:     False negative rate.
        n_points: Number of prior values to average over (more = more accurate).

    Returns:
        Average expected mutual information in bits per step.

    Notes:
        - Perfect sensors (alpha=0, beta=0) yield the maximum I_step (1 bit at
          p=0.5), producing the steepest temporal decay.
        - Random sensors (alpha=beta=0.5) yield I_step=0; no decay applied.
        - Result is cached via lru_cache so it is computed at most once per
          unique (alpha, beta) pair during a run.
    """
    EPS = 1e-12

    def h2(p: float) -> float:
        p = max(EPS, min(1.0 - EPS, p))
        return -p * math.log2(p) - (1.0 - p) * math.log2(1.0 - p)

    def h_joint(p11, p10, p01, p00) -> float:
        total = 0.0
        for v in (p11, p10, p01, p00):
            v = max(v, EPS)
            total -= v * math.log2(v)
        return total

    mis = []
    for k in range(1, n_points):
        p = k / n_points

        p11 = p * (1.0 - beta) ** 2 + (1.0 - p) * alpha ** 2
        p10 = p * (1.0 - beta) * beta  + (1.0 - p) * alpha * (1.0 - alpha)
        p01 = p10                        # symmetric sensor model
        p00 = p * beta ** 2             + (1.0 - p) * (1.0 - alpha) ** 2

        p_z1 = p * (1.0 - beta) + (1.0 - p) * alpha

        mi = h2(p_z1) + h2(p_z1) - h_joint(p11, p10, p01, p00)
        mis.append(max(0.0, mi))

    return float(np.mean(mis)) if mis else 0.0


def compute_gaussian_prior(grid_size: tuple, center_cell: int,
                           sigma: float) -> np.ndarray:
    """
    Initialise a belief as a 2D isotropic Gaussian centred on center_cell.

    Models the scenario where the target's last known location is
    approximately known (rescue mission, last GPS fix).  Setting sigma to
    a large value (e.g. grid_size[0]) recovers the uniform distribution.

    Args:
        grid_size:   (rows, cols) tuple.
        center_cell: flat cell index of the distribution centre.
        sigma:       standard deviation in grid cells.

    Returns:
        Normalised belief array of shape (rows * cols,).
    """
    rows, cols = grid_size
    cy, cx = divmod(center_cell, cols)
    n_states = rows * cols
    belief = np.zeros(n_states)
    two_sigma_sq = 2.0 * max(sigma, 1e-6) ** 2
    for s in range(n_states):
        sy, sx = divmod(s, cols)
        dist_sq = float((sy - cy) ** 2 + (sx - cx) ** 2)
        belief[s] = math.exp(-dist_sq / two_sigma_sq)
    total = float(np.sum(belief))
    return belief / total if total > 1e-15 else np.ones(n_states) / n_states


# ===================================================================
# SEED FIX: Consistent seed generation for proper checkpointing
# ===================================================================

def generate_consistent_seed(grid_size, n_agents, pattern, trial_id):
    """
    Generate a consistent seed that will be the same across Python sessions
    Replaces the problematic hash() function with hashlib for consistency
    
    Args:
        grid_size: (rows, cols) tuple
        n_agents: number of agents
        pattern: movement pattern string
        trial_id: trial identifier
    
    Returns:
        Consistent integer seed
    """
    seed_string = f"{grid_size[0]}x{grid_size[1]}_{n_agents}agents_{pattern}_trial{trial_id}"
    hash_object = hashlib.sha256(seed_string.encode())
    hash_hex = hash_object.hexdigest()
    seed = int(hash_hex[:8], 16) % (2**31 - 1)  # Keep it within int32 range
    return seed


# ===================================================================
# ORIGINAL BELIEF MERGING CLASSES
# ===================================================================

class UnifiedBeliefMergingFramework:
    """
    Framework for different belief merging approaches
    """
    _shared_env = None 
    
    @classmethod
    def get_gurobi_env(cls):
        """Initialize one Gurobi environment per worker process."""
        if cls._shared_env is None:
            try:
                cls._shared_env = gp.Env(empty=True)
                cls._shared_env.setParam('OutputFlag', 0)
                cls._shared_env.setParam('Threads', 1)
                cls._shared_env.start()
            except Exception:
                cls._shared_env = False  # Mark as failed so we don't spam retries
        
        # Return the env, or None if it failed to initialize
        return cls._shared_env if cls._shared_env is not False else None
        
    def __init__(self, grid_size=(20, 20), n_agents=4):
        self.grid_size = grid_size
        self.total_states = grid_size[0] * grid_size[1]
        self.n_agents = n_agents
        
    def merge_beliefs_average(self, beliefs, agent_weights=None):
        """Simple averaging of beliefs (Arithmetic Mean) - Analytical"""
        if agent_weights is None:
            agent_weights = np.ones(len(beliefs)) / len(beliefs)
        
        merged = np.zeros_like(beliefs[0])
        for i, belief in enumerate(beliefs):
            merged += agent_weights[i] * belief
        
        return merged / np.sum(merged)

    def merge_beliefs_geometric(self, beliefs, agent_weights=None):
        """Geometric Mean (Logarithmic Opinion Pool) - Analytical"""
        # This corresponds to Reverse KL Minimization (Analytical Solution)
        if len(beliefs) == 1: return beliefs[0].copy()
        
        # Log-space summation to avoid underflow
        # log(prod(p_i)) = sum(log(p_i))
        log_beliefs = [np.log(np.clip(b, 1e-10, 1)) for b in beliefs]
        
        # If weights are provided, multiply logs by weights (weighted geometric mean)
        if agent_weights is not None:
             weighted_log_sum = np.sum([w * lb for w, lb in zip(agent_weights, log_beliefs)], axis=0)
        else:
             weighted_log_sum = np.sum(log_beliefs, axis=0)
             
        merged = np.exp(weighted_log_sum)
        return merged / np.sum(merged)
    
    def merge_beliefs_kl(self, beliefs, agent_weights=None):
        """KL divergence-based merging (Standard/Forward KL) using Gurobi"""
        # Objective: min Sum w_i KL(P_i || Q)
        # Equivalent to: max Sum_x (Sum_i w_i P_i(x)) * log(Q(x))
        
        if agent_weights is None:
            agent_weights = np.ones(len(beliefs))
        
        n_states = beliefs[0].shape[0]
        # Calculate linear coefficients C[x] = Sum_i w_i P_i(x)
        beliefs_flat = [b.flatten() for b in beliefs]
        C = np.zeros(n_states)
        for i, b in enumerate(beliefs_flat):
            C += agent_weights[i] * b
            
        try:
            env = self.get_gurobi_env()
            with gp.Model("forward_kl", env=env) as model:
                
                # Variables Q(x)
                q = model.addVars(n_states, lb=1e-9, ub=1.0, name="q")
                
                # Variables for log(Q(x))
                log_q = model.addVars(n_states, lb=-float('inf'), name="log_q")
                
                # Constraint: Sum Q(x) = 1
                model.addConstr(q.sum() == 1, "sum_prob")
                
                # General Constraints: log_q[i] = ln(q[i])
                for i in range(n_states):
                    model.addGenConstrLog(q[i], log_q[i])
                    
                # Objective: Maximize Sum C[i] * log_q[i]
                obj_expr = gp.LinExpr()
                for i in range(n_states):
                    if C[i] > 1e-12: # Skip negligible terms to speed up setup
                        obj_expr += C[i] * log_q[i]
                
                model.setObjective(obj_expr, GRB.MAXIMIZE)
                model.optimize()
                
                if model.status == GRB.OPTIMAL:
                    merged = np.array([q[i].X for i in range(n_states)])
                    model.dispose()
                    # Normalize to be safe, though constraint handles it
                    return merged.reshape(beliefs[0].shape) / np.sum(merged)
                else:
                    print(f"Gurobi Failed with status {model.status}")
                    model.dispose()
                    # Fallback to analytical mean if solver fails
                    return self.merge_beliefs_average(beliefs, agent_weights)
                        
        except Exception as e:
            print(f"Gurobi Failed: {e}")
            # Fallback if Gurobi fails or not installed
            return self.merge_beliefs_average(beliefs, agent_weights)

    def merge_beliefs_reverse_kl(self, beliefs, agent_weights=None):
        """Reverse KL divergence merging (Optimization based) using Gurobi"""
        # Objective: min Sum w_i KL(Q || P_i)
        # Equivalent to: min Sum_x Q(x) log Q(x) - Sum_x Q(x) * (Sum_i w_i log P_i(x))
        
        if agent_weights is None:
            agent_weights = np.ones(len(beliefs))
            
        n_states = beliefs[0].shape[0]
        beliefs_flat = [b.flatten() for b in beliefs]
        
        # Calculate D[x] = Sum_i w_i log P_i(x)
        D = np.zeros(n_states)
        for i, b in enumerate(beliefs_flat):
            b_safe = np.clip(b, 1e-12, 1.0)
            D += agent_weights[i] * np.log(b_safe)
            
        try:
            env = self.get_gurobi_env()
            with gp.Model("reverse_kl", env=env) as model:
                
                # Variable Q(x)
                q = model.addVars(n_states, lb=1e-9, ub=1.0, name="q")
                
                # Constraint: Sum Q(x) = 1
                model.addConstr(q.sum() == 1, "sum_prob")
                
                # Objective: Sum (q_i log q_i - D_i q_i)
                # We use Piecewise Linear Approximation for x log x
                # Define sampling points
                x_pts = np.linspace(1e-9, 1.0, 100)
                y_pts = x_pts * np.log(x_pts)
                
                for i in range(n_states):
                    # Combine convex entropy term and linear term
                    # cost = (q log q) - (D[i] * q)
                    y_pts_combined = y_pts - (D[i] * x_pts)
                    model.setPWLObj(q[i], x_pts, y_pts_combined)
                
                model.modelSense = GRB.MINIMIZE
                model.optimize()
                
                if model.status == GRB.OPTIMAL:
                    merged = np.array([q[i].X for i in range(n_states)])
                    model.dispose()
                    return merged.reshape(beliefs[0].shape) / np.sum(merged)
                else:
                    print(f"Gurobi Failed with status {model.status}")
                    model.dispose()
                    # Fallback to analytical geometric mean
                    return self.merge_beliefs_geometric(beliefs, agent_weights)
                        
        except Exception as e:
            print(f"Gurobi Failed: {e}")
            return self.merge_beliefs_geometric(beliefs, agent_weights)
    
    def merge_beliefs_visit_weighted(self, beliefs, visit_counts):
        """Analytical solution for state-weighted KL divergence"""
        if len(beliefs) == 1:
            return beliefs[0].copy()
        
        # 1. Calculate state-specific weights (using your +1 smoothing logic)
        visits = np.array(visit_counts) + 1.0
        
        # Normalize so weights sum to 1 across agents for EACH state
        sum_visits = np.sum(visits, axis=0)
        weights_matrix = visits / sum_visits 
        
        # 2. Calculate the weighted arithmetic mean per state
        beliefs_matrix = np.array(beliefs)
        merged = np.sum(weights_matrix * beliefs_matrix, axis=0)
        
        # 3. Normalize to ensure valid probability distribution
        return merged / np.sum(merged)

    def merge_beliefs_trust_decay(self, beliefs, visit_counts,
                                  blackout_steps, alpha, beta,
                                  b_prior, n_agents,
                                  current_entropy=None, h_max=None,
                                  lambda_decay=1.0):
        """
        Temporally-discounted visit-weighted belief merge.

        The decay factor is:

            gamma = exp(-lambda * k * log2(N) * I_step(alpha,beta) * w_entropy)

        where w_entropy = 1 - H_ratio  (entropy weight).

        When current_entropy is None (or h_max is None), w_entropy = 1 and
        the method degrades to fixed-gamma trust decay (trust_decay_kl
        behaviour).  When entropy information is supplied, the method
        activates as trust_decay_kl_adaptive: decay is near-zero during
        exploration (high entropy) and strongest during exploitation (peaked
        belief), aligned with the Bizyaeva bifurcation framework.

        Floor: b_prior — the consensus merged belief from the previous
        communication event.  NOT the uniform distribution.  Reverting to
        b_prior preserves accumulated spatial knowledge; reverting to uniform
        destroys it.

        Args:
            beliefs:          List of per-agent belief arrays.
            visit_counts:     List of per-agent visit-count arrays.
            blackout_steps:   Steps since last merge.
            alpha:            Sensor false positive rate.
            beta:             Sensor false negative rate.
            b_prior:          Consensus belief at last disconnection.
            n_agents:         Number of agents in the team.
            current_entropy:  Entropy of current merged belief (optional).
            h_max:            log2(|S|), maximum possible entropy (optional).
            lambda_decay:     Tunable decay rate.

        Returns:
            Normalised merged belief array.
        """
        if len(beliefs) == 1:
            return beliefs[0].copy()

        # Spatial component: visit-weighted arithmetic mean (unchanged)
        visit_merged = self.merge_beliefs_visit_weighted(beliefs, visit_counts)

        i_step      = compute_I_step(alpha, beta)
        n_scale     = math.log2(max(float(n_agents), 2.0))

        if current_entropy is not None and h_max is not None and h_max > 1e-15:
            h_ratio       = float(current_entropy) / float(h_max)
            entropy_weight = max(0.0, 1.0 - h_ratio)
        else:
            entropy_weight = 1.0

        gamma   = float(np.exp(-lambda_decay * blackout_steps * n_scale
                               * i_step * entropy_weight))
        blended = gamma * visit_merged + (1.0 - gamma) * b_prior
        total   = float(np.sum(blended))
        if total < 1e-15:
            return b_prior.copy()
        return blended / total
    
    def jensen_shannon_divergence(self, p, q):
        """Calculate Jensen-Shannon divergence between two distributions"""
        p = np.clip(p, 1e-10, 1)
        q = np.clip(q, 1e-10, 1)
        m = 0.5 * (p + q)
        return 0.5 * (np.sum(p * np.log(p / m)) + np.sum(q * np.log(q / m)))


class TargetMovementPolicy:
    """
    Target movement policy - simple patterns without MPC/MDP
    """
    def __init__(self, grid_size, movement_pattern='random'):
        self.grid_size = grid_size
        self.rows, self.cols = grid_size
        self.movement_pattern = movement_pattern
        self.step_count = 0
        
    def get_next_position(self, current_pos, step=None):
        """Get next position based on movement pattern"""
        if step is not None:
            self.step_count = step
            
        if self.movement_pattern == 'stationary':
            return current_pos
        
        r, c = divmod(current_pos, self.cols)
        
        if self.movement_pattern == 'random':
            # Random walk with 0.8 prob of moving, 0.2 of staying
            if np.random.random() < 0.2:
                return current_pos
                
            moves = []
            if r > 0: moves.append(current_pos - self.cols)
            if r < self.rows-1: moves.append(current_pos + self.cols)
            if c > 0: moves.append(current_pos - 1)
            if c < self.cols-1: moves.append(current_pos + 1)
            
            return np.random.choice(moves) if moves else current_pos
            
        elif self.movement_pattern == 'evasive':
            # Try to move away from center
            center_r, center_c = self.rows // 2, self.cols // 2
            
            # Calculate direction away from center
            dr = 1 if r > center_r else -1 if r < center_r else 0
            dc = 1 if c > center_c else -1 if c < center_c else 0
            
            # Preferred moves
            preferred = []
            if 0 <= r + dr < self.rows:
                preferred.append(current_pos + dr * self.cols)
            if 0 <= c + dc < self.cols:
                preferred.append(current_pos + dc)
                
            if preferred and np.random.random() < 0.7:
                return np.random.choice(preferred)
            else:
                # Random move
                moves = []
                if r > 0: moves.append(current_pos - self.cols)
                if r < self.rows-1: moves.append(current_pos + self.cols)
                if c > 0: moves.append(current_pos - 1)
                if c < self.cols-1: moves.append(current_pos + 1)
                return np.random.choice(moves) if moves else current_pos
                
        elif self.movement_pattern == 'patrol':
            # Circular patrol pattern
            corners = [
                0,  # top-left
                self.cols - 1,  # top-right
                (self.rows - 1) * self.cols + self.cols - 1,  # bottom-right
                (self.rows - 1) * self.cols  # bottom-left
            ]
            
            # Find closest corner
            min_dist = float('inf')
            target_corner = corners[0]
            for corner in corners:
                corner_r, corner_c = divmod(corner, self.cols)
                dist = abs(r - corner_r) + abs(c - corner_c)
                if dist < min_dist and dist > 0:  # Don't stay at current corner
                    min_dist = dist
                    target_corner = corner
                    
            # Move towards target corner
            target_r, target_c = divmod(target_corner, self.cols)
            dr = 1 if target_r > r else -1 if target_r < r else 0
            dc = 1 if target_c > c else -1 if target_c < c else 0
            
            new_r = r + dr
            new_c = c + dc
            
            if 0 <= new_r < self.rows and 0 <= new_c < self.cols:
                return new_r * self.cols + new_c
                
        return current_pos


# --- MPC IMPLEMENTATION (REPLACING MCTS) ---

class MultiAgentMPC:
    """
    Model Predictive Control for multi-agent information gathering.
    Evaluates joint actions to maximize entropy reduction over a finite horizon.
    """
    def __init__(self, grid_size, n_agents, horizon=2, alpha=0.1, beta=0.2):
        self.grid_size = grid_size
        self.rows, self.cols = grid_size
        self.n_agents = n_agents
        self.horizon = horizon
        self.alpha = alpha
        self.beta = beta
        self.n_states = grid_size[0] * grid_size[1]
        
        # 8-Direction offsets (row_change, col_change)
        self.move_offsets = [
            (-1, 0), (1, 0), (0, -1), (0, 1),   # Cardinal
            (-1, -1), (-1, 1), (1, -1), (1, 1), # Diagonal
            (0, 0) # Stay
        ]

    def get_joint_action(self, beliefs: Union[np.ndarray, List[np.ndarray]], 
                        agent_positions: List[int], 
                        fast_mode: bool = False,
                        random_walk_mode: bool = False) -> List[int]: 
        
        if random_walk_mode:
             return self._get_random_joint_action(agent_positions)

        # Determine mode
        if isinstance(beliefs, np.ndarray):
             root_beliefs = [beliefs.copy()] 
             shared_mode = True
        else:
             root_beliefs = [b.copy() for b in beliefs]
             shared_mode = False

        # Generate candidate joint actions
        # To avoid combinatorial explosion, we sample diverse joint actions
        # If fast_mode is True, we sample fewer actions or use a greedy heuristic
        samples = 10 if fast_mode else 50
        candidate_actions = self._generate_candidate_actions(agent_positions, samples=samples)
        
        best_value = -float('inf')
        best_action = candidate_actions[0]
        
        for action in candidate_actions:
            # Evaluate action (Horizon 1 rollout for speed/memory efficiency)
            # Full horizon MPC would require recursive tree search, similar to MCTS but breadth-first
            # Here we do a 1-step lookahead + heuristic for subsequent steps if horizon > 1
            
            value = self._evaluate_action(action, root_beliefs, agent_positions, shared_mode)
            
            if value > best_value:
                best_value = value
                best_action = action
                
        return list(best_action)

    def _generate_candidate_actions(self, current_positions, samples=20):
        """Generate a subset of legal joint actions"""
        agent_legal_moves = []
        for pos in current_positions:
            r, c = divmod(pos, self.grid_size[1])
            moves = []
            for dr, dc in self.move_offsets:
                nr, nc = r + dr, c + dc
                if 0 <= nr < self.grid_size[0] and 0 <= nc < self.grid_size[1]:
                    moves.append(nr * self.grid_size[1] + nc)
            agent_legal_moves.append(moves)
        
        # Random sampling from cartesian product to ensure diversity
        joint_actions = []
        # Attempt to get 'samples' unique joint actions
        attempts = 0
        while len(joint_actions) < samples and attempts < samples * 3:
            action = tuple(np.random.choice(m) for m in agent_legal_moves)
            if action not in joint_actions:
                joint_actions.append(action)
            attempts += 1
            
        return list(joint_actions)

    def _evaluate_action(self, joint_action, beliefs, current_positions, shared_mode):
        """Evaluate a joint action based on expected entropy reduction"""
        # Simulate step
        new_positions = list(joint_action)
        expected_entropy_reduction = 0
        
        # We assume no detection (most likely outcome) to update belief for planning
        # This is the "Most Likely Observation" assumption standard in fast MPC
        
        if shared_mode:
            # One belief, multiple agents observing
            temp_belief = beliefs[0].copy()
            entropy_before = entropy(temp_belief)
            
            for pos in new_positions:
                # Update belief assuming NO detection (0)
                temp_belief = self._update_belief_single(temp_belief, pos, 0)
            
            expected_entropy_reduction = entropy_before - entropy(temp_belief)
            
        else:
            # Independent beliefs
            total_reduction = 0
            for i, b in enumerate(beliefs):
                entropy_before = entropy(b)
                pos = new_positions[i]
                # Update belief assuming NO detection
                new_b = self._update_belief_single(b, pos, 0)
                total_reduction += (entropy_before - entropy(new_b))
            
            expected_entropy_reduction = total_reduction
            
        return expected_entropy_reduction

    def _update_belief_single(self, belief: np.ndarray, 
                            position: int, 
                            observation: int) -> np.ndarray:
        """Update single agent's belief"""
        likelihood = np.ones(self.n_states)
        if observation == 1:
            likelihood[position] = 1 - self.beta
            likelihood[np.arange(self.n_states) != position] = self.alpha
        else:
            likelihood[position] = self.beta
            likelihood[np.arange(self.n_states) != position] = 1 - self.alpha
        
        posterior = belief * likelihood
        return posterior / (np.sum(posterior) + 1e-10)

    def _get_random_joint_action(self, agent_positions: List[int]) -> List[int]:
        """Simple random walk"""
        joint_action = []
        for pos in agent_positions:
            r, c = divmod(pos, self.cols)
            moves = []
            for dr, dc in self.move_offsets:
                nr, nc = r + dr, c + dc
                if 0 <= nr < self.rows and 0 <= nc < self.cols:
                    moves.append(nr * self.cols + nc)
            joint_action.append(np.random.choice(moves))
        return joint_action


class ControlledMergingExperiment:
    """
    Main experiment class with proper MPC implementation
    """
    
    def __init__(self, grid_size=(20, 20), n_agents=4, alpha=0.1, beta=0.2, horizon=2,
                 prior_type='uniform', prior_sigma=5.0, prior_sigma_fraction=0.0):
        self.grid_size = grid_size
        self.n_agents = n_agents
        self.alpha = alpha
        self.beta  = beta
        # Compute effective sigma: fraction overrides absolute when > 0.
        # This ensures consistent prior concentration across all grid sizes.
        if prior_sigma_fraction > 0.0:
            self.prior_sigma = prior_sigma_fraction * min(grid_size[0], grid_size[1])
        else:
            self.prior_sigma = prior_sigma
        self.prior_type  = prior_type   # 'uniform' or 'gaussian'
        self.prior_sigma = prior_sigma  # std dev in grid cells for Gaussian prior
        self.merger  = UnifiedBeliefMergingFramework(grid_size, n_agents)
        self.planner = MultiAgentMPC(grid_size, n_agents, horizon, alpha, beta)
        
    def _run_single_experiment(self, trial_config, merge_interval, max_steps,
                               fast_mode=False, random_walk_mode=False,
                               merge_method='standard_kl',
                               comm_model='fixed', trial_seed=0):
        """Run a single experiment with specified merge interval and method"""
        # Special case for full communication
        if merge_interval == 0:
            return self._run_centralized_full_communication(trial_config, max_steps, fast_mode, random_walk_mode)
        
        # For other strategies
        start_time = time.time()
        
        # Initialize agent beliefs from configured prior distribution
        n_states = self.grid_size[0] * self.grid_size[1]
        H_MAX = math.log2(n_states)

        initial_target = trial_config['target_trajectory'][0]
        if self.prior_type == 'gaussian':
            agent_beliefs = [
                compute_gaussian_prior(self.grid_size, initial_target, self.prior_sigma)
                for _ in range(self.n_agents)
            ]
        else:
            agent_beliefs = [
                np.ones(n_states) / n_states
                for _ in range(self.n_agents)
            ]
        
        agent_positions = trial_config['initial_positions'].copy()
        agent_trajectories = [[pos] for pos in agent_positions]
        
        # Tracking variables
        target_found = False
        first_discovery_step = max_steps
        discovery_count = 0
        entropy_history = []
        divergence_history = []
        merge_events = []

        # Track how many times each agent visits each cell
        n_states = self.grid_size[0] * self.grid_size[1]
        agent_visit_counts = [np.zeros(n_states) for _ in range(self.n_agents)]

        # ── Communication schedule ────────────────────────────────────────────
        # For 'fixed':   merge every merge_interval steps (original behaviour)
        # For 'poisson': pre-generate event times drawn from an exponential
        #                distribution with mean = merge_interval.
        #                Using a separate RNG seeded from trial_seed ensures
        #                reproducibility while keeping comm events independent
        #                of observation randoms.
        if comm_model == 'poisson' and merge_interval not in (0, float('inf')):
            _comm_rng = np.random.RandomState((trial_seed * 1_000_003 + 7) % (2 ** 31 - 1))
            _intervals = _comm_rng.exponential(scale=float(merge_interval),
                                               size=max_steps)
            _times     = np.cumsum(_intervals)
            poisson_events = set(int(t) for t in _times if 0 < t < max_steps)
        else:
            poisson_events = None

        # Tracks the last step at which communication occurred.
        # Used to compute the actual blackout duration passed to trust decay.
        last_comm_step = 0
        # Initialised to the same prior as agent beliefs so the floor is
        # always consistent with the starting belief.
        b_prior_at_last_merge = agent_beliefs[0].copy()

        # Belief map quality tracking
        # Q_t = mean probability assigned to true target location across agents
        belief_quality_history = []
        
        # Mark initial positions as visited
        for i, pos in enumerate(agent_positions):
            agent_visit_counts[i][pos] += 1
        
        # Run simulation
        for step in range(max_steps):
            # Target position for this step
            target_pos = trial_config['target_trajectory'][step]
            
            # Calculate and store metrics BEFORE actions
            entropies = [entropy(b) for b in agent_beliefs]
            entropy_history.append({
                'mean': np.mean(entropies),
                'std': np.std(entropies),
                'max': np.max(entropies),
                'min': np.min(entropies)
            })
            
            # Calculate divergence between agents
            divergences = []
            for i in range(len(agent_beliefs)):
                for j in range(i+1, len(agent_beliefs)):
                    div = self.merger.jensen_shannon_divergence(agent_beliefs[i], agent_beliefs[j])
                    divergences.append(div)
            
            divergence_history.append({
                'mean': np.mean(divergences) if divergences else 0,
                'std': np.std(divergences) if divergences else 0,
                'max': np.max(divergences) if divergences else 0
            })
            
            # ── Communication event check ─────────────────────────────────────
            # Fixed model  : fire every merge_interval steps exactly.
            # Poisson model: fire when step is in the pre-generated event set.
            if comm_model == 'poisson' and poisson_events is not None:
                should_communicate = (step in poisson_events)
            elif merge_interval != float('inf') and merge_interval != 0:
                should_communicate = (step > 0 and step % merge_interval == 0)
            else:
                should_communicate = False

            if should_communicate:
                # Actual blackout duration — varies per event under Poisson model
                actual_blackout = max(1, step - last_comm_step)
                beliefs_before  = [b.copy() for b in agent_beliefs]
                current_entropy = float(np.mean([entropy(b) for b in agent_beliefs]))

                if merge_method == 'geometric_mean':
                    merged_belief = self.merger.merge_beliefs_geometric(agent_beliefs)
                elif merge_method == 'arithmetic_mean':
                    merged_belief = self.merger.merge_beliefs_average(agent_beliefs)
                elif merge_method == 'reverse_kl':
                    merged_belief = self.merger.merge_beliefs_reverse_kl(agent_beliefs)
                elif merge_method == 'standard_kl':
                    merged_belief = self.merger.merge_beliefs_kl(agent_beliefs)
                elif merge_method == 'weighted_visits_kl':
                    # Cumulative visit counts — original CDC paper behaviour
                    merged_belief = self.merger.merge_beliefs_visit_weighted(
                        agent_beliefs, agent_visit_counts)
                elif merge_method == 'weighted_visits_kl_reset':
                    # Same formula, but visit counts reset after merge (see below)
                    merged_belief = self.merger.merge_beliefs_visit_weighted(
                        agent_beliefs, agent_visit_counts)
                elif merge_method == 'trust_decay_kl':
                    # Fixed-gamma trust decay (entropy weight = 1 always)
                    merged_belief = self.merger.merge_beliefs_trust_decay(
                        agent_beliefs, agent_visit_counts,
                        blackout_steps=actual_blackout,
                        alpha=self.alpha, beta=self.beta,
                        b_prior=b_prior_at_last_merge,
                        n_agents=self.n_agents)
                elif merge_method == 'trust_decay_kl_adaptive':
                    # Entropy-weighted gamma: decay activates during exploitation
                    merged_belief = self.merger.merge_beliefs_trust_decay(
                        agent_beliefs, agent_visit_counts,
                        blackout_steps=actual_blackout,
                        alpha=self.alpha, beta=self.beta,
                        b_prior=b_prior_at_last_merge,
                        n_agents=self.n_agents,
                        current_entropy=current_entropy,
                        h_max=H_MAX)
                else:
                    merged_belief = self.merger.merge_beliefs_kl(agent_beliefs)

                for i in range(self.n_agents):
                    agent_beliefs[i] = merged_belief.copy()

                # Update prior and last communication step
                b_prior_at_last_merge = merged_belief.copy()
                last_comm_step        = step

                # Reset visit counts for methods that use per-blackout counting
                if merge_method in ('weighted_visits_kl_reset',
                                    'trust_decay_kl',
                                    'trust_decay_kl_adaptive'):
                    for i in range(self.n_agents):
                        agent_visit_counts[i] = np.zeros(n_states)
                        agent_visit_counts[i][agent_positions[i]] += 1

                entropy_after = entropy(merged_belief)
                merge_events.append({
                    'step':            step,
                    'blackout_steps':  actual_blackout,
                    'entropy_before':  float(np.mean([entropy(b) for b in beliefs_before])),
                    'entropy_after':   float(entropy_after),
                    'entropy_reduction': float(np.mean([entropy(b) for b in beliefs_before])) - float(entropy_after),
                    'h_ratio':         float(entropy_after / H_MAX),
                })
            
            # Get joint action using MPC
            joint_action = self.planner.get_joint_action(
                agent_beliefs,
                agent_positions,
                fast_mode=fast_mode,
                random_walk_mode=random_walk_mode
            )
            
            # Belief map quality: mean probability mass at true target location
            # This continuous metric reflects epistemic progress regardless of
            # whether the binary discovery threshold has been crossed.
            belief_quality_history.append(
                float(np.mean([b[target_pos] for b in agent_beliefs]))
            )

            # Make observations BEFORE moving
            for i, pos in enumerate(agent_positions):
                obs_rand = trial_config['observation_randoms'][step, i]
                
                if pos == target_pos:
                    observation = 1 if obs_rand > self.beta else 0
                    if observation == 1 and not target_found:
                        target_found = True
                        first_discovery_step = step
                    if observation == 1:
                        discovery_count += 1
                else:
                    observation = 1 if obs_rand < self.alpha else 0
                
                # Update individual belief using helper in planner
                agent_beliefs[i] = self.planner._update_belief_single(
                    agent_beliefs[i], pos, observation
                )
            
            # Execute joint action
            agent_positions = joint_action
            for i, new_pos in enumerate(joint_action):
                agent_trajectories[i].append(new_pos)
                #keep a tracker for number of times agent i visited cell at new_pos
                agent_visit_counts[i][new_pos] += 1
        
        # Final merge for no_comm strategy
        if merge_interval == float('inf'):
            final_beliefs = agent_beliefs
            # For final metric, we can use the requested method to see the theoretical consensus
            if merge_method == 'geometric_mean' or merge_method == 'reverse_kl':
                final_merged = self.merger.merge_beliefs_geometric(final_beliefs)
            else:
                final_merged = self.merger.merge_beliefs_kl(final_beliefs)
        else:
            final_beliefs = agent_beliefs
            final_merged = np.mean(final_beliefs, axis=0)
            final_merged = final_merged / np.sum(final_merged)
        
        # Calculate final metrics
        final_target_pos = trial_config['target_trajectory'][-1]
        
        # Performance metrics
        prob_at_true_target = final_merged[final_target_pos]
        
        # Find position with highest belief
        predicted_pos = np.argmax(final_merged)
        pred_r, pred_c = divmod(predicted_pos, self.grid_size[1])
        true_r, true_c = divmod(final_target_pos, self.grid_size[1])
        prediction_error = np.sqrt((pred_r - true_r)**2 + (pred_c - true_c)**2)
        
        elapsed_time = time.time() - start_time
        
        return {
            'target_found':           target_found,
            'first_discovery_step':   first_discovery_step,
            'discovery_count':        discovery_count,
            'elapsed_time':           elapsed_time,
            'belief_quality_history': belief_quality_history,
            'avg_belief_quality':     float(np.mean(belief_quality_history)) if belief_quality_history else 0.0,
            'final_belief_quality':   belief_quality_history[-1] if belief_quality_history else 0.0,
            'final_merged_belief':    final_merged,
            'final_entropy':          entropy(final_merged),
            'entropy_history':        entropy_history,
            'divergence_history': divergence_history,
            'merge_events': merge_events,
            'total_merges': len(merge_events),
            'prob_at_true_target': prob_at_true_target,
            'prediction_error': prediction_error,
            'agent_trajectories': agent_trajectories
        }
    
    def _run_centralized_full_communication(self, trial_config, max_steps, fast_mode=False, random_walk_mode=False):
        """Run true full communication - single shared belief, coordinated MPC"""
        start_time = time.time()
        
        # Single shared belief for all agents
        shared_belief = np.ones(self.grid_size[0] * self.grid_size[1]) / (self.grid_size[0] * self.grid_size[1])
        
        # Agent positions
        agent_positions = trial_config['initial_positions'].copy()
        agent_trajectories = [[pos] for pos in agent_positions]
        
        # Tracking variables
        target_found = False
        first_discovery_step = max_steps
        discovery_count = 0
        entropy_history = []
        
        # Add artificial communication overhead
        COMM_OVERHEAD_PER_STEP = 0.001 * self.n_agents * (self.n_agents - 1) / 2
        
        # Run simulation
        for step in range(max_steps):
            # Target position for this step
            target_pos = trial_config['target_trajectory'][step]
            
            # Calculate and store metrics
            entropy_val = entropy(shared_belief)
            entropy_history.append({
                'mean': entropy_val,
                'std': 0,  # No variance - single belief
                'max': entropy_val,
                'min': entropy_val
            })
            
            # Get joint action using MPC with shared belief (REPLACED MCTS WITH MPC)
            joint_action = self.planner.get_joint_action(
                shared_belief, 
                agent_positions, 
                fast_mode=fast_mode,
                random_walk_mode=random_walk_mode
            )
            
            # ALL agents make observations BEFORE moving
            for i, pos in enumerate(agent_positions):
                obs_rand = trial_config['observation_randoms'][step, i]
                
                if pos == target_pos:
                    observation = 1 if obs_rand > self.beta else 0
                    if observation == 1 and not target_found:
                        target_found = True
                        first_discovery_step = step
                    if observation == 1:
                        discovery_count += 1
                else:
                    observation = 1 if obs_rand < self.alpha else 0
                
                # Update SHARED belief using helper in planner
                shared_belief = self.planner._update_belief_single(
                    shared_belief, pos, observation
                )
            
            # Execute joint action
            agent_positions = joint_action
            for i, new_pos in enumerate(joint_action):
                agent_trajectories[i].append(new_pos)
            
            # Add communication overhead
            time.sleep(COMM_OVERHEAD_PER_STEP)
        
        # Final metrics
        final_target_pos = trial_config['target_trajectory'][-1]
        
        # Performance metrics
        prob_at_true_target = shared_belief[final_target_pos]
        
        # Find position with highest belief
        predicted_pos = np.argmax(shared_belief)
        pred_r, pred_c = divmod(predicted_pos, self.grid_size[1])
        true_r, true_c = divmod(final_target_pos, self.grid_size[1])
        prediction_error = np.sqrt((pred_r - true_r)**2 + (pred_c - true_c)**2)
        
        elapsed_time = time.time() - start_time
        
        return {
            'target_found': target_found,
            'first_discovery_step': first_discovery_step,
            'discovery_count': discovery_count,
            'elapsed_time': elapsed_time,
            'final_merged_belief': shared_belief,
            'final_entropy': entropy(shared_belief),
            'entropy_history': entropy_history,
            'divergence_history': [{'mean': 0, 'std': 0, 'max': 0} for _ in range(max_steps)],
            'merge_events': [],  # No merge events - always together
            'total_merges': 0,
            'prob_at_true_target': prob_at_true_target,
            'prediction_error': prediction_error,
            'agent_trajectories': agent_trajectories
        }


# ===================================================================
# DISTRIBUTED EXECUTION FRAMEWORK 
# ===================================================================

@dataclass
class ExperimentConfig:
    """Configuration for the experiment - now supports multiple grid sizes and agent numbers"""
    grid_sizes: List[Tuple[int, int]] = field(default_factory=lambda: [(20, 20)])
    n_agents_list: List[int] = field(default_factory=lambda: [4])
    alpha: float = 0.1
    beta:  float = 0.2
    horizon: int = 2
    n_trials: int = 30
    max_steps: int = 1000
    merge_intervals: List[Union[int, float]] = None
    target_patterns: List[str] = None
    fast_mode: bool = False
    random_walk_mode: bool = False
    merge_methods: List[str] = None
    prior_type: str = 'gaussian'   # 'uniform' or 'gaussian'
    # Absolute sigma in grid cells.  Use this when the prior uncertainty has
    # a fixed physical interpretation (e.g. GPS accuracy in metres).
    prior_sigma: float = 5.0
    # Relative sigma as a fraction of min(rows, cols).  When > 0 this
    # OVERRIDES prior_sigma so that concentration is consistent across all
    # grid sizes.  Example: 0.15 gives sigma = 1.5 cells on a 10x10 grid
    # and 7.5 cells on a 50x50 grid.  Use this for simulation comparisons
    # that sweep grid sizes so the prior is equally informative at every scale.
    prior_sigma_fraction: float = 0.15
    # Communication model.
    # 'fixed'   : merge every merge_interval steps exactly (original behaviour).
    # 'poisson' : communication events fire at Poisson-process times with
    #             mean inter-event duration = merge_interval.  All agents
    #             participate in each event.  The actual blackout length
    #             therefore varies per event, making trust decay genuinely
    #             discriminative within a trial.
    comm_model: str = 'poisson' # 'fixed' or 'poisson'

    def __post_init__(self):
        if self.merge_intervals is None:
            self.merge_intervals = [0, 10, 25, 50, 100, 200, 500, float('inf')]
        if self.target_patterns is None:
            self.target_patterns = ['stationary', 'random', 'evasive', 'patrol']
        if self.merge_methods is None:
            self.merge_methods = [
                'standard_kl', 'reverse_kl', 'geometric_mean', 'arithmetic_mean',
                'weighted_visits_kl', 'weighted_visits_kl_reset',
                'trust_decay_kl', 'trust_decay_kl_adaptive'
            ]
    
    def to_dict(self):
        d = asdict(self)
        # Convert tuples to lists for JSON serialization
        d['grid_sizes'] = [[r, c] for r, c in self.grid_sizes]
        return d
    
    @classmethod
    def from_dict(cls, d):
        # Convert lists back to tuples for grid sizes
        if 'grid_sizes' in d:
            d['grid_sizes'] = [(r, c) for r, c in d['grid_sizes']]
        return cls(**d)


@dataclass
class TrialTask:
    """Individual trial task for parallel execution"""
    grid_size: Tuple[int, int]
    n_agents: int
    pattern: str
    trial_id: int
    merge_interval: Union[int, float]
    merge_method: str
    config: ExperimentConfig
    trial_seed: int
    checkpoint_dir: str

    def get_task_id(self):
        """Generate unique task ID including grid size, n_agents, and comm model."""
        grid_str = f"{self.grid_size[0]}x{self.grid_size[1]}"
        return (
            f"grid{grid_str}_agents{self.n_agents}_{self.pattern}"
            f"_{self.merge_method}_trial{self.trial_id}"
            f"_interval{self.merge_interval}"
            f"_{self.config.comm_model}"
            f"_seed{self.trial_seed}"
        )

    def get_checkpoint_path(self):
        """Get checkpoint file path for this task."""
        return os.path.join(self.checkpoint_dir, f"{self.get_task_id()}.pkl")


class DistributedExperimentManager:
    """Manages distributed execution with checkpointing - now supports multiple configurations"""
    
    def __init__(self, config: ExperimentConfig, checkpoint_dir: str = "checkpoints", 
                 results_dir: str = "results", max_workers: int = None):
        self.config = config
        self.checkpoint_dir = Path(checkpoint_dir)
        self.results_dir = Path(results_dir)
        self.checkpoint_dir.mkdir(exist_ok=True)
        self.results_dir.mkdir(exist_ok=True)
        
        # Determine optimal number of workers
        if max_workers is None:
            # Use 80% of available CPUs, but cap at 500 to avoid overwhelming
            max_workers = min(int(psutil.cpu_count() * 0.8), 500)
        self.max_workers = max_workers
        
        # Setup logging
        self.setup_logging()
        
        # Progress tracking
        self.total_tasks = 0
        self.completed_tasks = 0
        self.failed_tasks = 0
        
    def setup_logging(self):
        """Setup comprehensive logging"""
        log_dir = self.results_dir / "logs"
        log_dir.mkdir(exist_ok=True)
        
        # Main log file
        log_file = log_dir / f"experiment_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
        
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_file),
                logging.StreamHandler(sys.stdout)
            ]
        )
        
        self.logger = logging.getLogger(__name__)
        self.logger.info(f"Experiment started on {socket.gethostname()}")
        self.logger.info(f"Using {self.max_workers} workers")
        self.logger.info(f"Config: {self.config}")
    
    def generate_all_tasks(self) -> List[TrialTask]:
        """Generate all trial tasks for multiple grid sizes and agent numbers"""
        tasks = []
        
        for grid_size in self.config.grid_sizes:
            for n_agents in self.config.n_agents_list:
                for pattern in self.config.target_patterns:
                    for trial_id in range(self.config.n_trials):
                        # FIXED: Use consistent seed generation instead of hash()
                        trial_seed = generate_consistent_seed(grid_size, n_agents, pattern, trial_id)
                        
                        for merge_interval in self.config.merge_intervals:
                            for merge_method in self.config.merge_methods:
                                task = TrialTask(
                                    grid_size=grid_size,
                                    n_agents=n_agents,
                                    pattern=pattern,
                                    trial_id=trial_id,
                                    merge_interval=merge_interval,
                                    merge_method=merge_method,
                                    config=self.config,
                                    trial_seed=trial_seed,
                                    checkpoint_dir=str(self.checkpoint_dir)
                                )
                                tasks.append(task)
        
        self.total_tasks = len(tasks)
        self.logger.info(f"Generated {self.total_tasks} total tasks")
        self.logger.info(f"Grid sizes: {self.config.grid_sizes}")
        self.logger.info(f"Agent numbers: {self.config.n_agents_list}")
        return tasks
    
    def filter_incomplete_tasks(self, tasks: List[TrialTask]) -> List[TrialTask]:
        """Filter out already completed tasks"""
        incomplete_tasks = []
        
        for task in tasks:
            checkpoint_path = task.get_checkpoint_path()
            if not os.path.exists(checkpoint_path):
                incomplete_tasks.append(task)
            else:
                # Verify checkpoint integrity
                try:
                    with open(checkpoint_path, 'rb') as f:
                        result = pickle.load(f)
                    if self.validate_result(result):
                        self.completed_tasks += 1
                        continue
                    else:
                        self.logger.warning(f"Invalid checkpoint found: {checkpoint_path}")
                        os.remove(checkpoint_path)
                except Exception as e:
                    self.logger.warning(f"Corrupted checkpoint: {checkpoint_path}, error: {e}")
                    if os.path.exists(checkpoint_path):
                        os.remove(checkpoint_path)
                
                incomplete_tasks.append(task)
        
        self.logger.info(f"Found {self.completed_tasks} completed tasks")
        self.logger.info(f"Remaining tasks: {len(incomplete_tasks)}")
        return incomplete_tasks
    
    def validate_result(self, result: Dict) -> bool:
        """Validate that a result contains all required fields"""
        required_fields = [
            'target_found', 'first_discovery_step', 'discovery_count',
            'elapsed_time', 'final_merged_belief', 'final_entropy',
            'entropy_history', 'divergence_history', 'prediction_error'
        ]
        
        return all(field in result for field in required_fields)
    
    def run_distributed_experiment(self):
        """Run the complete distributed experiment"""
        self.logger.info("Starting distributed experiment")
        start_time = time.time()
        
        # Generate all tasks
        all_tasks = self.generate_all_tasks()
        
        # Filter out completed tasks
        remaining_tasks = self.filter_incomplete_tasks(all_tasks)
        
        if not remaining_tasks:
            self.logger.info("All tasks already completed!")
            return self.collect_results()
        
        # Setup signal handlers for graceful shutdown
        self.setup_signal_handlers()
        
        # Run tasks in parallel
        self.logger.info(f"Starting parallel execution with {self.max_workers} workers")
        
        try:
           # Switch to multiprocessing.Pool to enable worker recycling
            # maxtasksperchild=10 forces the worker to die and respawn after 10 tasks,
            # completely wiping any Gurobi or NumPy memory fragmentation.
            with mp.Pool(processes=self.max_workers, maxtasksperchild=10) as pool:
                
                # imap_unordered is highly memory efficient for massive iterables
                results_iterator = pool.imap_unordered(run_single_trial_task, remaining_tasks)
                
                for task_id in results_iterator:
                    if task_id is not None:
                        self.completed_tasks += 1
                        self.logger.info(
                            f"Completed {task_id} "
                            f"({self.completed_tasks}/{self.total_tasks})"
                        )
                    else:
                        self.failed_tasks += 1
                        self.logger.error("A task failed (check error logs in checkpoint dir)")
                    
                    # Progress update
                    if self.completed_tasks % 50 == 0:
                        elapsed = time.time() - start_time
                        rate = self.completed_tasks / elapsed
                        eta = (self.total_tasks - self.completed_tasks) / rate if rate > 0 else 0
                        self.logger.info(
                            f"Progress: {self.completed_tasks}/{self.total_tasks} "
                            f"({100*self.completed_tasks/self.total_tasks:.1f}%), "
                            f"Rate: {rate:.2f} tasks/sec, ETA: {eta/3600:.1f} hours"
                        )
                        
        except KeyboardInterrupt:
            self.logger.info("Received interrupt signal, shutting down gracefully...")
            pool.terminate()
            pool.join()
            return None
        
        total_time = time.time() - start_time
        self.logger.info(
            f"Experiment completed in {total_time/3600:.2f} hours. "
            f"Completed: {self.completed_tasks}, Failed: {self.failed_tasks}"
        )
        
        # Collect and analyze results
        return self.collect_results()
    
    def setup_signal_handlers(self):
        """Setup signal handlers for graceful shutdown"""
        def signal_handler(signum, frame):
            self.logger.info(f"Received signal {signum}, initiating graceful shutdown...")
            sys.exit(0)
        
        signal.signal(signal.SIGINT, signal_handler)
        signal.signal(signal.SIGTERM, signal_handler)
    
    def collect_results(self):
        """Skip full consolidation to prevent OOM errors on large experiments."""
        self.logger.info("Skipping full consolidation to save memory.")
        self.logger.info(f"All individual results are safely stored in: {self.checkpoint_dir}")
        return {"status": "completed", "checkpoints_dir": str(self.checkpoint_dir)}
        
        # all_results = {}
        
        # for grid_size in self.config.grid_sizes:
        #     grid_key = f"{grid_size[0]}x{grid_size[1]}"
        #     all_results[grid_key] = {}
            
        #     for n_agents in self.config.n_agents_list:
        #         agent_key = f"{n_agents}_agents"
        #         all_results[grid_key][agent_key] = {}
                
        #         for pattern in self.config.target_patterns:
        #             pattern_results = {}
                    
        #             for merge_interval in self.config.merge_intervals:
        #                 interval_results = {}
                        
        #                 # Initialize for methods
        #                 for method in self.config.merge_methods:
        #                     interval_results[method] = []
                        
        #                 for trial_id in range(self.config.n_trials):
        #                     # FIXED: Use consistent seed generation
        #                     trial_seed = generate_consistent_seed(grid_size, n_agents, pattern, trial_id)
                            
        #                     for merge_method in self.config.merge_methods:
        #                         task = TrialTask(
        #                             grid_size=grid_size,
        #                             n_agents=n_agents,
        #                             pattern=pattern,
        #                             trial_id=trial_id,
        #                             merge_interval=merge_interval,
        #                             merge_method=merge_method,
        #                             config=self.config,
        #                             trial_seed=trial_seed,
        #                             checkpoint_dir=str(self.checkpoint_dir)
        #                         )
                                
        #                         checkpoint_path = task.get_checkpoint_path()
                                
        #                         if os.path.exists(checkpoint_path):
        #                             try:
        #                                 with open(checkpoint_path, 'rb') as f:
        #                                     result = pickle.load(f)
        #                                 interval_results[merge_method].append(result)
        #                             except Exception as e:
        #                                 self.logger.error(f"Failed to load {checkpoint_path}: {e}")
                        
        #                 # Store results with proper key naming
        #                 if merge_interval == 0:
        #                     key = 'full_comm'
        #                 elif merge_interval == float('inf'):
        #                     key = 'no_comm'
        #                 else:
        #                     key = f'interval_{merge_interval}'
                        
        #                 pattern_results[key] = interval_results
                    
        #             all_results[grid_key][agent_key][pattern] = pattern_results
        
        # # Save consolidated results
        # consolidated_path = self.results_dir / f"consolidated_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pkl"
        # with open(consolidated_path, 'wb') as f:
        #     pickle.dump(all_results, f)
        
        # self.logger.info(f"Consolidated results saved to {consolidated_path}")
        # return all_results


def run_single_trial_task(task: TrialTask) -> Optional[Dict]:
    """Run a single trial task - designed for parallel execution"""
    
    # Check if already completed
    checkpoint_path = task.get_checkpoint_path()
    if os.path.exists(checkpoint_path):
        try:
            with open(checkpoint_path, 'rb') as f:
                result = pickle.load(f)
            return result
        except:
            # Corrupted checkpoint, will regenerate
            if os.path.exists(checkpoint_path):
                os.remove(checkpoint_path)
    
    try:
        # Create experiment instance with specific grid size and n_agents
        experiment = ControlledMergingExperiment(
            grid_size=task.grid_size,
            n_agents=task.n_agents,
            alpha=task.config.alpha,
            beta=task.config.beta,
            horizon=task.config.horizon,
            prior_type=task.config.prior_type,
            prior_sigma=task.config.prior_sigma,
            prior_sigma_fraction=task.config.prior_sigma_fraction
        )
        
        # Generate trial configuration
        np.random.seed(task.trial_seed)
        
        # Generate target trajectory
        target_policy = TargetMovementPolicy(task.grid_size, task.pattern)
        initial_target = np.random.randint(0, task.grid_size[0] * task.grid_size[1])
        
        target_trajectory = [initial_target]
        current_pos = initial_target
        for step in range(task.config.max_steps):
            current_pos = target_policy.get_next_position(current_pos, step)
            target_trajectory.append(current_pos)
        
        # Generate initial agent positions
        total_states = task.grid_size[0] * task.grid_size[1]
        available_positions = list(range(total_states))
        if initial_target in available_positions:
            available_positions.remove(initial_target)
        
        # Handle case where we have more agents than available positions
        if task.n_agents > len(available_positions):
            raise ValueError(f"Cannot place {task.n_agents} agents in grid of size {task.grid_size} with target at {initial_target}")
        
        initial_positions = np.random.choice(
            available_positions, 
            task.n_agents, 
            replace=False
        ).tolist()
        
        # Pre-generate observation random numbers
        observation_randoms = np.random.random((task.config.max_steps, task.n_agents))
        
        # Create trial configuration
        trial_config = {
            'trial_id': task.trial_id,
            'seed': task.trial_seed,
            'target_trajectory': target_trajectory,
            'initial_positions': initial_positions,
            'observation_randoms': observation_randoms,
            'target_pattern': task.pattern,
            'grid_size': task.grid_size,
            'n_agents': task.n_agents
        }
        
        # Run the experiment with MPC (replacing MCTS logic)
        result = experiment._run_single_experiment(
            trial_config,
            task.merge_interval,
            task.config.max_steps,
            task.config.fast_mode,
            task.config.random_walk_mode,
            task.merge_method,
            comm_model=task.config.comm_model,
            trial_seed=task.trial_seed
        )
        
        # Add metadata
        result['task_metadata'] = {
            'grid_size':      task.grid_size,
            'n_agents':       task.n_agents,
            'pattern':        task.pattern,
            'trial_id':       task.trial_id,
            'merge_interval': task.merge_interval,
            'merge_method':   task.merge_method,
            'comm_model':     task.config.comm_model,
            'trial_seed':     task.trial_seed,
            'hostname':       socket.gethostname(),
            'pid':            os.getpid(),
            'completion_time':datetime.now().isoformat()
        }
        
        # Atomic save to checkpoint
        save_result_atomic(result, checkpoint_path)

        # Explicitly delete large variables to help GC
        del experiment
        del trial_config
        del result
        
        # FORCE GARBAGE COLLECTION
        gc.collect()
        
        return task.get_task_id()
        
    except Exception as e:
        # Log error but don't crash the worker
        error_msg = f"Task {task.get_task_id()} failed: {str(e)}"
        print(f"ERROR: {error_msg}")
        
        # Save error info
        error_path = checkpoint_path.replace('.pkl', '_ERROR.txt')
        with open(error_path, 'w') as f:
            f.write(f"{datetime.now().isoformat()}: {error_msg}\n")
            f.write(f"Exception type: {type(e).__name__}\n")
            f.write(f"Exception details: {str(e)}\n")
            import traceback
            f.write(f"Traceback:\n{traceback.format_exc()}\n")
        
        return None


def save_result_atomic(result: Dict, filepath: str):
    """Atomically save result to avoid corruption"""
    temp_path = filepath + '.tmp'
    try:
        with open(temp_path, 'wb') as f:
            pickle.dump(result, f)
        
        # Atomic move
        os.rename(temp_path, filepath)
    except Exception as e:
        if os.path.exists(temp_path):
            os.remove(temp_path)
        raise e


def main():
    """Main execution function"""
    parser = argparse.ArgumentParser(description='Complete Distributed Belief Merging Experiment')
    parser.add_argument('--config-file', type=str, help='JSON config file path')
    parser.add_argument('--max-workers', type=int, help='Maximum number of workers')
    parser.add_argument('--checkpoint-dir', type=str, default='checkpoints', help='Checkpoint directory')
    parser.add_argument('--results-dir', type=str, default='results', help='Results directory')
    
    args = parser.parse_args()
    
    # Load configuration
    if args.config_file and os.path.exists(args.config_file):
        with open(args.config_file, 'r') as f:
            config_dict = json.load(f)
        config = ExperimentConfig.from_dict(config_dict)
    else:
        # Default configuration with multiple grid sizes and agent numbers
        config = ExperimentConfig(
            grid_sizes=[(10, 10), (20, 20), (30, 30)],  # Multiple grid sizes
            n_agents_list=[2, 3, 4],  # Multiple agent numbers
            alpha=0.1,
            beta=0.2,
            horizon=3,  # Full MPC horizon
            n_trials=50,
            max_steps=1000,
            merge_intervals=[0, 10, 25, 50, 100, 200, 500, float('inf')],
            target_patterns=['stationary', 'random', 'evasive', 'patrol'],
            fast_mode=False,  # TRUE MPC for computational accuracy
            random_walk_mode=False, # Active search by default
            merge_methods=['standard_kl', 'reverse_kl', 'geometric_mean', 'arithmetic_mean', 'weighted_visits_kl']        
        )
    
    print("="*80)
    print("COMPLETE DISTRIBUTED BELIEF MERGING EXPERIMENT")
    print("="*80)
    print(f"Configuration: {config}")
    print(f"Grid sizes: {config.grid_sizes}")
    print(f"Agent numbers: {config.n_agents_list}")
    print(f"TRUE MPC Mode: {not config.fast_mode}")
    print(f"Random Walk Mode: {config.random_walk_mode}")
    print(f"Methods: {config.merge_methods}")
    
    # Create and run experiment manager
    manager = DistributedExperimentManager(
        config=config,
        checkpoint_dir=args.checkpoint_dir,
        results_dir=args.results_dir,
        max_workers=args.max_workers
    )
    
    # Run the experiment
    results = manager.run_distributed_experiment()
    
    if results is not None:
        print("\nExperiment completed successfully!")
        print(f"Results saved in: {manager.results_dir}")
    else:
        print("\nExperiment was interrupted.")


if __name__ == "__main__":
    main()