#!/usr/bin/env python3

# CS465 at Johns Hopkins University.

# Subclass ConditionalRandomFieldBackprop to get a biRNN-CRF model.

from __future__ import annotations
import logging
import torch.nn as nn
import torch.nn.functional as F
from math import inf
from pathlib import Path
from typing_extensions import override
from typeguard import typechecked

import torch
from torch import Tensor, cuda

from corpus import Sentence, Tag, TaggedCorpus, Word, IntegerizedSentence
from integerize import Integerizer
from crf_backprop import ConditionalRandomFieldBackprop

logger = logging.getLogger(Path(__file__).stem)

torch.manual_seed(1337)
cuda.manual_seed(69_420)

class ConditionalRandomFieldNeural(ConditionalRandomFieldBackprop):
    """A CRF that uses a biRNN to compute non-stationary potential
    matrices.  The feature functions used to compute the potentials
    are now non-stationary, non-linear functions of the biRNN
    parameters."""

    neural = True    
    
    @override
    def __init__(self, 
                 tagset: Integerizer[Tag],
                 vocab: Integerizer[Word],
                 lexicon: Tensor,
                 rnn_dim: int,
                 unigram: bool = False):
        
        if unigram:
            raise NotImplementedError("Not required for this homework")

        self.rnn_dim = rnn_dim
        self.E = lexicon
        self.e = lexicon.size(1)
        self.num_words = len(vocab)

        # Hidden feature dimension (size of vector f in Eq 47/48)
        # Not specified in PDF, choosing reasonable size
        self.phi_dim = 32 

        nn.Module.__init__(self)  
        super().__init__(tagset, vocab, unigram)


    @override
    def init_params(self) -> None:
        """
        Initialize all the parameters for M, M', U_a, U_b, theta_a and theta_b.
        Following docstring: xavier uniform for matrices, normal for vectors.
        """
        
        # 1. RNN Parameters (Eq 46)
        # M: Maps [1, h_{i-1}, w_i] -> h_i (Dimension d)
        # Input size: 1 (bias) + d (rnn_dim) + e (emb_dim)
        self.M = nn.Parameter(torch.empty(self.rnn_dim, 1 + self.rnn_dim + self.e))
        
        # M': Maps [1, w_i, h'_{i+1}] -> h'_i (Dimension d)
        # Input size: 1 + e + d
        self.M_prime = nn.Parameter(torch.empty(self.rnn_dim, 1 + self.e + self.rnn_dim))
        
        # 2. Transition Parameters (Eq 47, 45)
        # U_A: Maps [1, h_{i-2}, s, t, h'_i] -> f_A (Dimension phi_dim)
        # Input: 1 + d + k + k + d
        dim_UA = 1 + 2*self.rnn_dim + 2*self.k
        self.U_A = nn.Parameter(torch.empty(self.phi_dim, dim_UA))
        
        # Theta_A: Dot product with f_A -> scalar
        self.Theta_A = nn.Parameter(torch.empty(self.phi_dim))
        
        # 3. Emission Parameters (Eq 48, 45)
        # U_B: Maps [1, h_{i-1}, t, w, h'_i] -> f_B (Dimension phi_dim)
        # Input: 1 + d + k + e + d
        dim_UB = 1 + 2*self.rnn_dim + self.k + self.e
        self.U_B = nn.Parameter(torch.empty(self.phi_dim, dim_UB))
        
        # Theta_B: Dot product with f_B -> scalar
        self.Theta_B = nn.Parameter(torch.empty(self.phi_dim))
        
        # Initialization Logic
        for p in [self.M, self.M_prime, self.U_A, self.U_B]:
            nn.init.xavier_uniform_(p)
        for p in [self.Theta_A, self.Theta_B]:
            nn.init.normal_(p, mean=0, std=0.1)

    @override
    def init_optimizer(self, lr: float, weight_decay: float) -> None:
        self.optimizer = torch.optim.AdamW( 
            params=self.parameters(),       
            lr=lr, weight_decay=weight_decay
        )                                   
        self.scheduler = None            
       
    @override
    def updateAB(self) -> None:
        pass

    @override
    def setup_sentence(self, isent: IntegerizedSentence) -> None:
        """
        Pre-compute RNN states using Eq 46 manually.
        """
        device = next(self.parameters()).device
        if self.E.device != device: self.E = self.E.to(device)
        
        word_indices = torch.tensor([w for w, t in isent], dtype=torch.long, device=device)
        
        # Get embeddings (Shape: L x e)
        embs = F.embedding(word_indices, self.E)
        L = len(isent)
        d = self.rnn_dim
        
        # --- Forward Scan (Eq 46 left) ---
        # h_j = sigmoid( M [1; h_{j-1}; w_j] )
        # We collect [h_0, ... h_{L-1}] corresponding to indices of isent.
        # Actually isent indices are 0..L-1. 
        # Logic: h[j] captures w_0...w_j
        # Base case: h_{-1} = 0
        
        h_curr = torch.zeros(d, device=device) # h_{-1}
        h_forward_list = []
        
        # Bias is 1. We can perform Matrix mult M @ vec.
        # M is (d, 1+d+e).
        # Splitting M for efficiency: Bias=M[:,0], WeightsH=M[:,1:d+1], WeightsW=M[:,d+1:]
        # h_new = sigmoid(Bias + W_H*h + W_W*w)
        m_bias = self.M[:, 0]
        m_h = self.M[:, 1:1+d]
        m_w = self.M[:, 1+d:]
        
        for j in range(L):
            w_j = embs[j]
            # Scan operation
            # Note: using matmul (@) for vectors
            activation = m_bias + (m_h @ h_curr) + (m_w @ w_j)
            h_curr = torch.sigmoid(activation)
            h_forward_list.append(h_curr)
            
        self.ctx_forward = torch.stack(h_forward_list)
        
        # --- Backward Scan (Eq 46 right) ---
        # h'_j-1 = sigmoid( M' [1; w_j; h'_j] )
        # Logic: h'[j] captures w_{n}...w_{j+1}. 
        # Wait, recursion is h'_{j-1} depends on h'_j. 
        # Base case: h'_L (aka h'_{n}) = 0 (just after last word).
        
        h_prime_curr = torch.zeros(d, device=device) # h'_{n}
        h_backward_list = [torch.zeros(d, device=device)] * L
        
        # M': (d, 1+e+d)
        mp_bias = self.M_prime[:, 0]
        mp_w = self.M_prime[:, 1:1+e]
        mp_hp = self.M_prime[:, 1+e:]
        
        # Scan backwards from L-1 down to 0. 
        # At step j, we compute h'_{j-1} using w_j and h'_j
        # We need to store h'_i to be accessed later. 
        # Code convention: ctx_backward[i] should store h'_i
        # Base case h'_L is 0.
        
        # We iterate j from L-1 down to 0.
        # Eq: h'_{j-1} uses w_j and h'_j. 
        # The result h'_{j-1} will be context for step j-1.
        
        # We need an array where index 'i' gives h'_i.
        # Valid i are 0..L.
        
        h_backward_dense = torch.zeros((L + 1, d), device=device)
        # h_backward_dense[L] is already 0.
        
        for j in range(L - 1, -1, -1):
            w_j = embs[j]
            hp_prev = h_backward_dense[j+1] # This is h'_j in the formula if 'prev' means processed before?
            # Formula: h'_{j-1} <-- (w_j, h'_j). 
            # Actually, notation index logic in reading: h'_{j-1} is embedding *after* w_{j-1} but *before* w_j?
            # Text says: h'_{j-1} = suffix w_j...w_n.
            # So at index j-1 (between w_{j-1} and w_j), the vector is derived from w_j and h'_j.
            
            activation = mp_bias + (mp_w @ w_j) + (mp_hp @ hp_prev)
            h_backward_dense[j] = torch.sigmoid(activation)
            
        self.ctx_backward = h_backward_dense

    @override
    def accumulate_logprob_gradient(self, sentence: Sentence, corpus: TaggedCorpus) -> None:
        isent = self._integerize_sentence(sentence, corpus)
        super().accumulate_logprob_gradient(sentence, corpus)

    @override
    @typechecked
    def A_at(self, position: int, sentence) -> Tensor:
        """Computes dynamic A using Eq 47 and U_A parameter."""
        i = position # Index of tag t
        device = self.embedded_sentence.device
        
        # Eq 47: prefix h_{i-2}, s, t, suffix h'_i
        
        if i >= 2: h_p = self.ctx_forward[i-2]
        else:      h_p = torch.zeros(self.rnn_dim, device=device)
        
        # h'_i
        h_s = self.ctx_backward[i] 
        # (ctx_backward has size L+1, so i is valid up to L)
        
        # Prepare Broadcasting
        # Dim order for Linear(dim_in) input: [..., dim_in]
        
        # Split U_A weights for efficiency: bias + dot(U_parts, components)
        # U_A dim: 1 (bias-like col) + d + k + k + d
        u_bias = self.U_A[:, 0] 
        u_h1   = self.U_A[:, 1:1+self.rnn_dim]
        u_s    = self.U_A[:, 1+self.rnn_dim : 1+self.rnn_dim+self.k]
        u_t    = self.U_A[:, 1+self.rnn_dim+self.k : 1+self.rnn_dim+2*self.k]
        u_h2   = self.U_A[:, 1+2*self.rnn_dim+2*self.k:]
        
        # Pre-calculate context contributions (constant for this i)
        # Shapes: (phi_dim,)
        val_h1 = u_h1 @ h_p 
        val_h2 = u_h2 @ h_s
        base_act = u_bias + val_h1 + val_h2 
        
        # Add Tag contributions
        # We need (K, K) output of phi_dim vectors. 
        # U_s @ s_vector(one-hot-row) -> Just columns of U_s
        # S contribution varies over Rows (K, 1, phi)
        # T contribution varies over Cols (1, K, phi)
        
        # u_s is (phi, k). Transpose to (k, phi) for broadcasting
        eff_s = u_s.T.unsqueeze(1) # (K, 1, phi)
        eff_t = u_t.T.unsqueeze(0) # (1, K, phi)
        
        # Broadcast base
        eff_base = base_act.view(1, 1, -1)
        
        # Sigmoid Input
        # Shape (K, K, phi_dim)
        activations = eff_base + eff_s + eff_t
        f = torch.sigmoid(activations)
        
        # Final Dot Product with Theta
        # Shape (K, K). theta is (phi_dim).
        # sum(f * theta, dim=-1)
        log_A = (f * self.Theta_A.view(1,1,-1)).sum(dim=-1)
        
        return torch.exp(log_A)
        
    @override
    @typechecked
    def B_at(self, position: int, sentence) -> Tensor:
        """Computes dynamic B for observed word w using Eq 48 and U_B."""
        i = position
        device = self.embedded_sentence.device
        w_idx = sentence[i][0]

        # If boundary index (e.g. calling len(sent)), return zeros
        if i >= len(self.embedded_sentence):
             return torch.zeros(self.k, w_idx+2, device=device)

        # Eq 48: h_{i-1}, t, w, h'_i
        if i >= 1: h_p = self.ctx_forward[i-1]
        else:      h_p = torch.zeros(self.rnn_dim, device=device)
        
        h_s = self.ctx_backward[i]
        w_vec = self.embedded_sentence[i]
        
        # Split U_B: bias + d + k + e + d
        # d = rnn_dim
        curr = 1
        u_bias = self.U_B[:, 0]
        u_hp = self.U_B[:, curr : curr+self.rnn_dim]; curr += self.rnn_dim
        u_t  = self.U_B[:, curr : curr+self.k];       curr += self.k
        u_w  = self.U_B[:, curr : curr+self.e];       curr += self.e
        u_hs = self.U_B[:, curr :]
        
        # Constant parts for this position
        val_hp = u_hp @ h_p
        val_w  = u_w  @ w_vec
        val_hs = u_hs @ h_s
        
        base_act = u_bias + val_hp + val_w + val_hs
        
        # Tag Part: u_t (phi, k) -> (k, phi)
        # We want Output size K (one score per tag).
        eff_t = u_t.T 
        
        # Broadcast
        # (K, phi) + (1, phi)
        activations = eff_t + base_act.unsqueeze(0)
        f = torch.sigmoid(activations)
        
        # Dot product theta
        log_scores = (f * self.Theta_B.view(1, -1)).sum(dim=-1)
        scores = torch.exp(log_scores)
        
        # Create Result (K, valid_size)
        valid_V = max(w_idx+1, self.num_words+2)
        B_out = torch.zeros((self.k, valid_V), device=device)
        B_out[:, w_idx] = scores
        
        return B_out