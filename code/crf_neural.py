#!/usr/bin/env python3

# CS465 at Johns Hopkins University.

# Subclass ConditionalRandomFieldBackprop to get a biRNN-CRF model.

from __future__ import annotations
import logging
import torch.nn as nn
import torch.nn.functional as F
from math import inf, log, exp
from pathlib import Path
from typing_extensions import override
from typeguard import typechecked

import torch
from torch import Tensor, cuda
from jaxtyping import Float

# Import safe logsumexp to handle -inf in backprop
import logsumexp_safe  # patches torch.logsumexp to handle -inf correctly

from corpus import IntegerizedSentence, Sentence, Tag, TaggedCorpus, Word, BOS_WORD, EOS_WORD
from integerize import Integerizer
from crf_backprop import ConditionalRandomFieldBackprop, TorchScalar

logger = logging.getLogger(Path(__file__).stem)  # For usage, see findsim.py in earlier assignment.
    # Note: We use the name "logger" this time rather than "log" since we
    # are already using "log" for the mathematical log!

# Set the seed for random numbers in torch, for replicability
torch.manual_seed(1337)
cuda.manual_seed(69_420)  # No-op if CUDA isn't available

class ConditionalRandomFieldNeural(ConditionalRandomFieldBackprop):
    """A CRF that uses a biRNN to compute non-stationary potential
    matrices.  The feature functions used to compute the potentials
    are now non-stationary, non-linear functions of the biRNN
    parameters."""

    neural = True    # class attribute that indicates that constructor needs extra args
    
    @override
    def __init__(self, 
                 tagset: Integerizer[Tag],
                 vocab: Integerizer[Word],
                 lexicon: Tensor,
                 rnn_dim: int,
                 unigram: bool = False,
                 init_scale: Optional[float] = None):
        # [doctring inherited from parent method]

        if unigram:
            raise NotImplementedError("Not required for this homework")

        self.rnn_dim = rnn_dim
        self.e = lexicon.size(1) # dimensionality of word's embeddings
        self.E = lexicon
        self.init_scale = init_scale

        nn.Module.__init__(self)  
        super().__init__(tagset, vocab, unigram)


    @override
    def init_params(self) -> None:

        """
            Initialize all the parameters you will need to support a bi-RNN CRF
            This will require you to create parameters for M, M', U_a, U_b, theta_a
            and theta_b. Use xavier uniform initialization for the matrices and 
            normal initialization for the vectors. 
        """
        if self.init_scale is not None:
            scale = self.init_scale
            logger.info(f"Using init_scale={scale} (user-specified)")
        else:
            is_toy = (self.V <= 5 and self.k <= 5)
            scale = 1.0 if is_toy else 0.3
            logger.info(f"Auto-detected {'toy' if is_toy else 'real'} dataset; using init_scale={scale}")

        # See the "Parameterization" section of the reading handout to determine
        # what dimensions all your parameters will need.

        # M: maps word embeddings (e-dim) to forward RNN hidden states (rnn_dim)
        self.M = nn.Parameter(torch.empty(self.rnn_dim, self.e))
        nn.init.xavier_uniform_(self.M)
        
        # M': maps word embeddings (e-dim) to backward RNN hidden states (rnn_dim)
        self.M_prime = nn.Parameter(torch.empty(self.rnn_dim, self.e))
        nn.init.xavier_uniform_(self.M_prime)
        
        # U_a: maps concatenated biRNN context (2*rnn_dim) and tag pair (k+k) to transition potentials
        # Output dimension should be k^2 for the k x k transition matrix
        self.U_a = nn.Parameter(torch.empty(self.k * self.k, 2 * self.rnn_dim + self.k + self.k))
        nn.init.xavier_uniform_(self.U_a)
        # Scale down to prevent explosion
        self.U_a.data *= scale
        
        # theta_a: bias for transition potentials (k^2 dimensional)
        self.theta_a = nn.Parameter(torch.empty(self.k * self.k))
        nn.init.zeros_(self.theta_a)
        
        # U_b: maps concatenated biRNN context (2*rnn_dim), tag (k), and word embedding (e) to emission potentials
        # Output dimension should be k for a single word emission
        self.U_b = nn.Parameter(torch.empty(self.k, 2 * self.rnn_dim + self.k + self.e))
        nn.init.xavier_uniform_(self.U_b)
        # Scale down to prevent explosion
        self.U_b.data *= scale
        
        # theta_b: bias for emission potentials (k dimensional)
        self.theta_b = nn.Parameter(torch.empty(self.k))
        nn.init.zeros_(self.theta_b)
        
        # Precompute constants that don't change during training (use register_buffer so they move to GPU)
        # One-hot tag embeddings (identity matrix) - computed once
        self.register_buffer('tag_embeddings', torch.eye(self.k))
        
        # Precompute expanded tag embeddings for A_at() to avoid repeated expansions
        # tag_s: (k, k, k) - for each (s,t) pair, embed(s)
        # tag_t: (k, k, k) - for each (s,t) pair, embed(t)
        tag_s = self.tag_embeddings.unsqueeze(1).expand(-1, self.k, -1)
        tag_t = self.tag_embeddings.unsqueeze(0).expand(self.k, -1, -1)
        self.register_buffer('tag_s_expanded', tag_s)
        self.register_buffer('tag_t_expanded', tag_t)
        
        # Precompute masks for structural zeros (avoid log(0))
        # A_mask: can't transition TO bos or FROM eos
        A_mask = torch.ones(self.k, self.k)
        A_mask[:, self.bos_t] = 1e-10
        A_mask[self.eos_t, :] = 1e-10
        self.register_buffer('A_mask', A_mask)
        
        # B_mask: eos and bos can't emit regular words
        B_mask = torch.ones(self.k)
        B_mask[self.eos_t] = 1e-10
        B_mask[self.bos_t] = 1e-10
        self.register_buffer('B_mask', B_mask)
        
        # Cache for RNN hidden states
        self.h = None
        self.h_prime = None
        self._rnn_cache_key = None
        
    @override
    def init_optimizer(self, lr: float, weight_decay: float) -> None:
        # [docstring will be inherited from parent]
    
        # Use SGD optimizer for consistency with parent class
        # AdamW can cause issues with the way we accumulate gradients
        self.optimizer = torch.optim.SGD( 
            params=self.parameters(),       
            lr=lr, weight_decay=weight_decay
        )            
       
    @override
    def updateAB(self) -> None:
        # Nothing to do - self.A and self.B are not used in non-stationary CRFs
        pass

    @override
    def setup_sentence(self, isent: IntegerizedSentence) -> None:
        """Pre-compute the biRNN prefix and suffix contextual features (h and h'
        vectors) at all positions, as defined in the "Parameterization" section
        of the reading handout.  They can then be accessed by A_at() and B_at().
        
        Optimized: Uses batched matrix operations where possible.
        """

        # Cache based on sentence identity to avoid recomputation
        cache_key = tuple(w for w, _ in isent)
        if hasattr(self, '_rnn_cache_key') and self._rnn_cache_key == cache_key:
            return  # Already computed for this sentence
        
        n = len(isent)
        
        # Get all word embeddings at once (batch operation)
        word_indices = torch.tensor([w for w, _ in isent], dtype=torch.long)
        
        # Create embedding matrix: use E for known words, zeros for unknown
        embeddings = torch.zeros(n, self.e)
        valid_mask = word_indices < len(self.E)
        if valid_mask.any():
            embeddings[valid_mask] = self.E[word_indices[valid_mask]]
        
        # Forward RNN: Still need sequential computation due to recurrence
        # But we can optimize the loop
        h = [torch.zeros(self.rnn_dim) for _ in range(n)]
        for j in range(1, n):
            h[j] = torch.tanh(self.M @ embeddings[j] + h[j-1])
        
        # Backward RNN: Also sequential
        h_prime = [torch.zeros(self.rnn_dim) for _ in range(n)]
        for j in range(n-2, -1, -1):
            h_prime[j] = torch.tanh(self.M_prime @ embeddings[j] + h_prime[j+1])
        
        # Store for use by A_at() and B_at()
        self.h = h
        self.h_prime = h_prime
        self._rnn_cache_key = cache_key

    @override
    def accumulate_logprob_gradient(self, sentence: Sentence, corpus: TaggedCorpus) -> None:
        isent = self._integerize_sentence(sentence, corpus)
        super().accumulate_logprob_gradient(sentence, corpus)

    @override
    @typechecked
    def A_at(self, position, sentence) -> Tensor:
        """Computes non-stationary k x k transition potential matrix using biRNN 
        contextual features and tag embeddings (one-hot encodings)."""
        
        j = position
        
        # Get precomputed biRNN context at position j
        h_j = self.h[j]
        h_prime_j = self.h_prime[j]
        context = torch.cat([h_j, h_prime_j])  # (2*rnn_dim,)
        
        # Expand context to (k, k, 2*rnn_dim) - same for all pairs
        context_exp = context.view(1, 1, -1).expand(self.k, self.k, -1)
        
        # Use precomputed tag embeddings (already expanded in constructor)
        # Concatenate all features: [context, tag_s, tag_t]
        features = torch.cat([context_exp, self.tag_s_expanded, self.tag_t_expanded], dim=2)  # (k, k, feature_dim)
        
        # Reshape to (k^2, feature_dim) for batch processing
        features_flat = features.reshape(self.k * self.k, -1)
        
        # Compute log-potentials (unbounded)
        logits = (self.U_a * features_flat).sum(dim=1) + self.theta_a  # (k^2,)
        
        # Reshape to k x k matrix and apply structural zeros in log-space
        logits_matrix = logits.view(self.k, self.k)
        logits_matrix[:, self.bos_t] = float('-inf')
        logits_matrix[self.eos_t, :] = float('-inf')
        
        # Exponentiate to get potentials
        phi_A = torch.exp(logits_matrix)
        
        return phi_A
        
    @override
    @typechecked
    def B_at(self, position, sentence) -> Tensor:
        """Computes non-stationary k x V emission potential matrix using biRNN 
        contextual features, tag embeddings (one-hot encodings), and word embeddings."""
        
        j = position
        w_j, _ = sentence[j]
        
        # Get precomputed biRNN context at position j
        h_j = self.h[j]
        h_prime_j = self.h_prime[j]
        context = torch.cat([h_j, h_prime_j])  # (2*rnn_dim,)
        
        # Get word embedding for w_j
        if w_j < len(self.E):
            word_emb = self.E[w_j]  # (e,)
        else:
            word_emb = torch.zeros(self.e)
        
        # Create feature vectors for all k tags efficiently
        # For each tag t: [context, embed(t), word_emb]
        
        # Expand context to (k, 2*rnn_dim)
        context_exp = context.unsqueeze(0).expand(self.k, -1)
        
        # Expand word embedding to (k, e)
        word_emb_exp = word_emb.unsqueeze(0).expand(self.k, -1)
        
        # Concatenate: use precomputed tag_embeddings instead of torch.eye()
        features = torch.cat([context_exp, self.tag_embeddings, word_emb_exp], dim=1)  # (k, feature_dim)
        
        # Compute log-potentials (unbounded)
        logits = (self.U_b * features).sum(dim=1) + self.theta_b  # (k,)
        
        # Apply structural zeros in log-space
        logits[self.eos_t] = float('-inf')
        logits[self.bos_t] = float('-inf')
        
        # Exponentiate to get potentials
        potentials = torch.exp(logits)  # (k,)
        
        # Return k x V matrix with potentials for word w_j
        phi_B = torch.zeros(self.k, self.V)
        
        # Only set potentials if w_j is in the regular vocabulary
        if w_j < self.V:
            phi_B[:, w_j] = potentials
        
        return phi_B
