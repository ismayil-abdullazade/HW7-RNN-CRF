#!/usr/bin/env python3

# CS465 at Johns Hopkins University.
# Awesome CRF with multiple improvements for extra credit

from __future__ import annotations
import logging
import torch.nn as nn
import torch.nn.functional as F
from math import inf, log, exp
from pathlib import Path
from typing import Optional
from typing_extensions import override
from typeguard import typechecked

import torch
from torch import Tensor, cuda
from jaxtyping import Float

import logsumexp_safe

from corpus import IntegerizedSentence, Sentence, Tag, TaggedCorpus, Word, BOS_WORD, EOS_WORD
from integerize import Integerizer
from crf_neural import ConditionalRandomFieldNeural
from crf_backprop import TorchScalar

logger = logging.getLogger(Path(__file__).stem)

torch.manual_seed(1337)
cuda.manual_seed(69_420)

class ConditionalRandomFieldAwesome(ConditionalRandomFieldNeural):
    """
    Awesome CRF with multiple improvements:
    1. Trainable word embeddings (tune during training)
    2. Efficient single-pass architecture (compute all potentials at once)
    3. Layer normalization for stability
    4. Better initialization strategy
    """

    awesome = True
    
    @override
    def __init__(self, 
                 tagset: Integerizer[Tag],
                 vocab: Integerizer[Word],
                 lexicon: Tensor,
                 rnn_dim: int,
                 unigram: bool = False,
                 init_scale: Optional[float] = None,
                 tune_embeddings: bool = True,
                 use_layernorm: bool = True):
        
        self.tune_embeddings = tune_embeddings
        self.use_layernorm = use_layernorm
        
        # Call parent constructor
        super().__init__(tagset, vocab, lexicon, rnn_dim, unigram, init_scale)
        
        logger.info(f"Awesome CRF: tune_embeddings={tune_embeddings}, layernorm={use_layernorm}")

    @override
    def init_params(self) -> None:
        """Initialize parameters with improvements"""
        
        if self.init_scale is not None:
            scale = self.init_scale
            logger.info(f"Using init_scale={scale} (user-specified)")
        else:
            is_toy = (self.V <= 5 and self.k <= 5)
            scale = 1.0 if is_toy else 0.1
            logger.info(f"Auto-detected {'toy' if is_toy else 'real'} dataset; using init_scale={scale}")

        # === RNN Parameters ===
        self.M = nn.Parameter(torch.empty(self.rnn_dim, self.e))
        nn.init.xavier_uniform_(self.M)
        
        self.M_prime = nn.Parameter(torch.empty(self.rnn_dim, self.e))
        nn.init.xavier_uniform_(self.M_prime)
        
        # === IMPROVEMENT 1: Make word embeddings trainable ===
        if self.tune_embeddings:
            # Convert fixed embeddings to parameters
            self.E = nn.Parameter(self.E.clone())
            logger.info(f"Word embeddings are TRAINABLE ({self.E.shape[0]} x {self.E.shape[1]})")
        else:
            logger.info(f"Word embeddings are FIXED ({self.E.shape[0]} x {self.E.shape[1]})")
        
        # === IMPROVEMENT 2: Layer normalization ===
        if self.use_layernorm:
            self.layer_norm = nn.LayerNorm(2 * self.rnn_dim)
            logger.info("Using LayerNorm after biRNN")
        
        # === IMPROVEMENT 3: Efficient single-pass architecture ===
        # Instead of computing each tag pair separately, we use a projection layer
        # that computes all transition logits at once
        
        # Context dimension: forward + backward RNN + previous tag embedding
        context_dim = 2 * self.rnn_dim + self.k
        
        # Hidden layer for transitions (optional, adds capacity)
        self.transition_hidden_dim = min(128, 2 * self.rnn_dim)
        self.transition_hidden = nn.Parameter(torch.empty(self.transition_hidden_dim, context_dim))
        nn.init.xavier_uniform_(self.transition_hidden)
        self.transition_hidden.data *= scale
        
        # Output layer: maps hidden to k^2 transition logits
        self.transition_output = nn.Parameter(torch.empty(self.k * self.k, self.transition_hidden_dim))
        nn.init.xavier_uniform_(self.transition_output)
        self.transition_output.data *= scale
        
        self.transition_bias = nn.Parameter(torch.zeros(self.k * self.k))
        
        # Emission parameters: single pass for all tags
        # Context: forward + backward RNN + word embedding
        emission_context_dim = 2 * self.rnn_dim + self.e
        
        # Hidden layer for emissions
        self.emission_hidden_dim = min(128, 2 * self.rnn_dim)
        self.emission_hidden = nn.Parameter(torch.empty(self.emission_hidden_dim, emission_context_dim))
        nn.init.xavier_uniform_(self.emission_hidden)
        self.emission_hidden.data *= scale
        
        # Output layer: maps hidden to k emission logits
        self.emission_output = nn.Parameter(torch.empty(self.k, self.emission_hidden_dim))
        nn.init.xavier_uniform_(self.emission_output)
        self.emission_output.data *= scale
        
        self.emission_bias = nn.Parameter(torch.zeros(self.k))
        
        # Precompute tag embeddings (one-hot)
        self.tag_embeddings = torch.eye(self.k)
        
        logger.info(f"Parameters: {sum(p.numel() for p in self.parameters())} total")
        logger.info(f"  RNN: {self.M.numel() + self.M_prime.numel()}")
        logger.info(f"  Transitions: {self.transition_hidden.numel() + self.transition_output.numel()}")
        logger.info(f"  Emissions: {self.emission_hidden.numel() + self.emission_output.numel()}")
        if self.tune_embeddings:
            logger.info(f"  Word embeddings: {self.E.numel()}")

    @override
    @typechecked
    def A_at(self, position, sentence) -> Tensor:
        """
        Efficient computation of k x k transition potential matrix.
        Uses single forward pass instead of computing each tag pair separately.
        """
        j = position
        
        # Get biRNN context
        h_j = self.h[j]
        h_prime_j = self.h_prime[j]
        context = torch.cat([h_j, h_prime_j])  # (2*rnn_dim,)
        
        # Apply layer norm if enabled
        if self.use_layernorm:
            context = self.layer_norm(context)
        
        # Create feature vector: context + zero padding for tag (we'll broadcast)
        # For transitions, we condition on context but not on specific previous tag
        # (More efficient than original per-tag-pair computation)
        features = torch.cat([context, torch.zeros(self.k)])  # (context_dim,)
        
        # Single pass through network
        hidden = torch.tanh(features @ self.transition_hidden.t())  # (hidden_dim,)
        logits = hidden @ self.transition_output.t() + self.transition_bias  # (k^2,)
        
        # Reshape to k x k matrix
        logits_matrix = logits.view(self.k, self.k)
        
        # Apply structural zeros
        logits_matrix[:, self.bos_t] = float('-inf')
        logits_matrix[self.eos_t, :] = float('-inf')
        
        # Exponentiate to get potentials
        phi_A = torch.exp(logits_matrix)
        
        return phi_A
    
    @override
    @typechecked
    def B_at(self, position, sentence) -> Tensor:
        """
        Efficient computation of k x V emission potential matrix.
        Uses single forward pass to compute all tag logits at once.
        """
        j = position
        w_j, _ = sentence[j]
        
        # Get biRNN context
        h_j = self.h[j]
        h_prime_j = self.h_prime[j]
        context = torch.cat([h_j, h_prime_j])  # (2*rnn_dim,)
        
        # Apply layer norm if enabled
        if self.use_layernorm:
            context = self.layer_norm(context)
        
        # Get word embedding
        if w_j < len(self.E):
            word_emb = self.E[w_j]  # (e,)
        else:
            word_emb = torch.zeros(self.e)
        
        # Single pass: context + word -> hidden -> k logits
        features = torch.cat([context, word_emb])  # (emission_context_dim,)
        hidden = torch.tanh(features @ self.emission_hidden.t())  # (emission_hidden_dim,)
        logits = hidden @ self.emission_output.t() + self.emission_bias  # (k,)
        
        # Apply structural zeros
        logits[self.eos_t] = float('-inf')
        logits[self.bos_t] = float('-inf')
        
        # Exponentiate to get potentials
        potentials = torch.exp(logits)  # (k,)
        
        # Return k x V matrix with potentials for word w_j
        phi_B = torch.zeros(self.k, self.V)
        
        if w_j < self.V:
            phi_B[:, w_j] = potentials
        
        return phi_B
    
    def get_param_groups(self, base_lr: float = 0.1):
        """
        Return parameter groups with different learning rates for better training.
        RNNs train slower, embeddings faster, potentials in between.
        """
        param_groups = []
        
        # Group 1: RNN parameters (slower learning)
        rnn_params = [self.M, self.M_prime]
        param_groups.append({
            'params': rnn_params,
            'lr': base_lr * 0.5,  # Half speed for RNNs
            'name': 'rnn'
        })
        
        # Group 2: Transition and emission networks (normal speed)
        network_params = [
            self.transition_hidden, self.transition_output, self.transition_bias,
            self.emission_hidden, self.emission_output, self.emission_bias
        ]
        param_groups.append({
            'params': network_params,
            'lr': base_lr,
            'name': 'networks'
        })
        
        # Group 3: Word embeddings (faster learning if trainable)
        if self.tune_embeddings:
            param_groups.append({
                'params': [self.E],
                'lr': base_lr * 2.0,  # Double speed for embeddings
                'name': 'embeddings'
            })
        
        logger.info(f"Created {len(param_groups)} parameter groups with base_lr={base_lr}")
        for group in param_groups:
            logger.info(f"  {group['name']}: lr={group['lr']}, {sum(p.numel() for p in group['params'])} params")
        
        return param_groups
