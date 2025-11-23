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

from corpus import IntegerizedSentence, Sentence, Tag, TaggedCorpus, Word
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
                 unigram: bool = False):
        # [doctring inherited from parent method]

        if unigram:
            raise NotImplementedError("Not required for this homework")

        self.rnn_dim = rnn_dim
        self.e = lexicon.size(1) # dimensionality of word's embeddings
        self.E = lexicon

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
        self.U_a.data *= 0.1
        
        # theta_a: bias for transition potentials (k^2 dimensional)
        self.theta_a = nn.Parameter(torch.empty(self.k * self.k))
        nn.init.zeros_(self.theta_a)
        
        # U_b: maps concatenated biRNN context (2*rnn_dim), tag (k), and word embedding (e) to emission potentials
        # Output dimension should be k for a single word emission
        self.U_b = nn.Parameter(torch.empty(self.k, 2 * self.rnn_dim + self.k + self.e))
        nn.init.xavier_uniform_(self.U_b)
        # Scale down to prevent explosion
        self.U_b.data *= 0.1
        
        # theta_b: bias for emission potentials (k dimensional)
        self.theta_b = nn.Parameter(torch.empty(self.k))
        nn.init.zeros_(self.theta_b)
        
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
        
        Make sure to call this method from the forward_pass, backward_pass, and
        Viterbi_tagging methods of HiddenMarkovMOdel, so that A_at() and B_at()
        will have correct precomputed values to look at!"""

        # Cache based on sentence identity to avoid recomputation
        # Use tuple of word indices as cache key (tags don't affect RNN computation)
        cache_key = tuple(w for w, _ in isent)
        if hasattr(self, '_rnn_cache_key') and self._rnn_cache_key == cache_key:
            return  # Already computed for this sentence
        
        n = len(isent)
        
        # Initialize h vectors (forward RNN) - use zeros that support gradients
        h = [torch.zeros(self.rnn_dim) for _ in range(n)]
        
        # Forward pass: compute h_j from h_{j-1} and word embedding at position j
        for j in range(1, n):
            w_j, _ = isent[j]
            # Get word embedding
            if w_j < len(self.E):
                x_j = self.E[w_j]
            else:
                x_j = torch.zeros(self.e)
            
            # h_j = tanh(M @ x_j + h_{j-1})
            h[j] = torch.tanh(self.M @ x_j + h[j-1])
        
        # Initialize h' vectors (backward RNN)
        h_prime = [torch.zeros(self.rnn_dim) for _ in range(n)]
        
        # Backward pass: compute h'_j from h'_{j+1} and word embedding at position j
        for j in range(n-2, -1, -1):
            w_j, _ = isent[j]
            # Get word embedding
            if w_j < len(self.E):
                x_j = self.E[w_j]
            else:
                x_j = torch.zeros(self.e)
            
            # h'_j = tanh(M' @ x_j + h'_{j+1})
            h_prime[j] = torch.tanh(self.M_prime @ x_j + h_prime[j+1])
        
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
        contextual features and tag embeddings (one-hot encodings). Output should 
        be ϕA from the "Parameterization" section in the reading handout."""

        j = position
        
        # Get biRNN context at position j
        h_j = self.h[j]
        h_prime_j = self.h_prime[j]
        context = torch.cat([h_j, h_prime_j])  # (2*rnn_dim,)
        
        # Use one-hot tag embeddings (identity matrix)
        tag_embeddings = torch.eye(self.k)  # (k, k)
        
        # For each pair (s, t), concatenate [context, embed(s), embed(t)]
        # This creates a k x k matrix
        # We need to compute this efficiently using broadcasting
        
        # Expand context to (k, k, 2*rnn_dim) by repeating
        context_expanded = context.unsqueeze(0).unsqueeze(0).expand(self.k, self.k, -1)
        
        # Expand tag embeddings: s varies over rows, t varies over columns
        tag_s = tag_embeddings.unsqueeze(1).expand(-1, self.k, -1)  # (k, k, k) - embed(s) for each (s,t)
        tag_t = tag_embeddings.unsqueeze(0).expand(self.k, -1, -1)  # (k, k, k) - embed(t) for each (s,t)
        
        # Concatenate all features: [context, tag_s, tag_t]
        features = torch.cat([context_expanded, tag_s, tag_t], dim=2)  # (k, k, 2*rnn_dim + k + k)
        
        # Reshape for matrix multiplication
        features_flat = features.view(self.k * self.k, -1)  # (k^2, 2*rnn_dim + k + k)
        
        # Compute potentials: sigmoid(U_a @ features + theta_a)
        logits = features_flat @ self.U_a.t() + self.theta_a  # (k^2, k^2) @ (k^2, features) -> (k^2,)
        # Wait, U_a is (k^2, features), so U_a @ features^T gives us (k^2,) for each of k^2 pairs
        # Actually, we want: for each (s,t) pair, compute one potential value
        
        # Let me reconsider: U_a should map features to a single scalar per (s,t) pair
        # U_a: (k^2, feature_dim) - NO, that's wrong
        # U_a should be (1, feature_dim) to map features -> scalar, but we have k^2 pairs
        # OR U_a maps each feature vector to a scalar, so we need U_a to be applied k^2 times
        
        # Actually, from the handout equation 45: we compute one potential per (s,t) pair
        # So U_a should give us k^2 outputs from k^2 input feature vectors
        # This means U_a maps feature_dim -> 1, and we apply it k^2 times
        
        # Let me fix this: U_a should be (1, feature_dim) OR we use linear layer
        # Actually, each (s,t) pair gets its own row in U_a
        
        # Reinterpret: U_a is (k^2, feature_dim), theta_a is (k^2,)
        # For input features (k^2, feature_dim), we compute:
        # output[i] = U_a[i] @ features[i] + theta_a[i]
        
        logits = torch.sum(self.U_a * features_flat, dim=1) + self.theta_a  # (k^2,)
        potentials = torch.sigmoid(logits)  # (k^2,)
        
        # Reshape to k x k matrix
        phi_A = potentials.view(self.k, self.k)
        
        # Enforce structural zeros by creating a mask and multiplying
        # Add small epsilon to avoid exact zeros which can cause log(0) = -inf issues
        mask = torch.ones(self.k, self.k)
        mask[:, self.bos_t] = 1e-10
        mask[self.eos_t, :] = 1e-10
        phi_A = phi_A * mask
        
        return phi_A
        
    @override
    @typechecked
    def B_at(self, position, sentence) -> Tensor:
        """Computes non-stationary k x V emission potential matrix using biRNN 
        contextual features, tag embeddings (one-hot encodings), and word embeddings. 
        Output should be ϕB from the "Parameterization" section in the reading handout."""

        j = position
        w_j, _ = sentence[j]
        
        # Get biRNN context at position j
        h_j = self.h[j]
        h_prime_j = self.h_prime[j]
        context = torch.cat([h_j, h_prime_j])  # (2*rnn_dim,)
        
        # Get word embedding for w_j
        if w_j < len(self.E):
            word_emb = self.E[w_j]  # (e,)
        else:
            word_emb = torch.zeros(self.e)
        
        # Use one-hot tag embeddings
        tag_embeddings = torch.eye(self.k)  # (k, k)
        
        # For each tag t, concatenate [context, embed(t), word_emb]
        # This creates a k-dimensional vector (one potential per tag for this specific word)
        
        # Expand context to (k, 2*rnn_dim)
        context_expanded = context.unsqueeze(0).expand(self.k, -1)
        
        # Expand word embedding to (k, e)
        word_emb_expanded = word_emb.unsqueeze(0).expand(self.k, -1)
        
        # Concatenate features for each tag: [context, tag_emb[t], word_emb]
        features = torch.cat([context_expanded, tag_embeddings, word_emb_expanded], dim=1)  # (k, 2*rnn_dim + k + e)
        
        # Compute potentials: sigmoid(U_b @ features + theta_b)
        # U_b: (k, feature_dim), features: (k, feature_dim)
        # For each tag t: U_b[t] @ features[t] + theta_b[t]
        logits = torch.sum(self.U_b * features, dim=1) + self.theta_b  # (k,)
        potentials = torch.sigmoid(logits)  # (k,)
        
        # Enforce structural zeros by masking before creating the full matrix
        # Use small epsilon instead of exact zero to avoid log(0) issues
        mask = torch.ones(self.k)
        mask[self.eos_t] = 1e-10
        mask[self.bos_t] = 1e-10
        potentials = potentials * mask
        
        # We need to return a k x V matrix, but we only computed potentials for one word
        # The CRF code expects B[t, w] format, so we create a full matrix with this column
        phi_B = torch.zeros(self.k, self.V)
        if w_j < self.V:
            phi_B[:, w_j] = potentials
        
        return phi_B
