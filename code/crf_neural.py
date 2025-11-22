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
    """A CRF that uses a biRNN to compute non-stationary potential matrices."""
    neural = True    
    
    @override
    def __init__(self, tagset: Integerizer[Tag], vocab: Integerizer[Word], lexicon: Tensor, rnn_dim: int, unigram: bool = False):
        if unigram: raise NotImplementedError("Not required for this homework")

        self.rnn_dim = rnn_dim
        self.E = lexicon
        self.e = lexicon.size(1)
        self.num_words = len(vocab)
        self.phi_dim = 64  # Dimension for feature vectors f

        nn.Module.__init__(self)  
        super().__init__(tagset, vocab, unigram)

    @override
    def init_params(self) -> None:
        """Initialize parameters for M, M', U_a, U_b, theta_a and theta_b."""
        
        # 1. Manual RNN Parameters (Handout Eq 46)
        # M: Maps [1, h, w] -> h
        self.M = nn.Parameter(torch.empty(self.rnn_dim, 1 + self.rnn_dim + self.e))
        # M': Maps [1, w, h'] -> h'
        self.M_prime = nn.Parameter(torch.empty(self.rnn_dim, 1 + self.e + self.rnn_dim))
        
        # 2. Transition Feature MLP (Handout Eq 47)
        # U_A maps [1, h, s, t, h'] -> f (phi_dim)
        dim_UA = 1 + 2*self.rnn_dim + 2*self.k
        self.U_A = nn.Parameter(torch.empty(self.phi_dim, dim_UA))
        self.Theta_A = nn.Parameter(torch.empty(self.phi_dim))
        
        # 3. Emission Feature MLP (Handout Eq 48)
        # U_B maps [1, h, t, w, h'] -> f (phi_dim)
        dim_UB = 1 + 2*self.rnn_dim + self.k + self.e
        self.U_B = nn.Parameter(torch.empty(self.phi_dim, dim_UB))
        self.Theta_B = nn.Parameter(torch.empty(self.phi_dim))
        
        # Init: Xavier Uniform for Matrices, Normal for Thetas
        for p in [self.M, self.M_prime, self.U_A, self.U_B]:
            nn.init.xavier_uniform_(p)
        for p in [self.Theta_A, self.Theta_B]:
            nn.init.normal_(p, mean=0, std=0.1)

    @override
    def init_optimizer(self, lr: float, weight_decay: float) -> None:
        self.optimizer = torch.optim.AdamW(params=self.parameters(), lr=lr, weight_decay=weight_decay)       
       
    @override
    def updateAB(self) -> None:
        pass # A and B are dynamic in this subclass

    @override
    def setup_sentence(self, isent: IntegerizedSentence) -> None:
        """Pre-compute manual RNN states h and h'."""
        
        dev = next(self.parameters()).device
        if self.E.device != dev: self.E = self.E.to(dev)
        
        word_idx = torch.tensor([w for w, t in isent], dtype=torch.long, device=dev)
        embs = F.embedding(word_idx, self.E) # L x e
        
        L = len(isent)
        d = self.rnn_dim
        
        # 1. Forward Scan (M)
        # h_j depends on w_j and h_{j-1}.
        m_bias, m_h, m_w = self.M[:,0], self.M[:, 1:d+1], self.M[:, d+1:]
        
        h = torch.zeros(d, device=dev) # h_{-1}
        h_fw = []
        for j in range(L):
            act = m_bias + (m_h @ h) + (m_w @ embs[j])
            h = torch.sigmoid(act)
            h_fw.append(h)
        self.ctx_forward = torch.stack(h_fw) # indices 0..L-1 maps to h_0..h_{L-1}

        # 2. Backward Scan (M')
        # h'_{j-1} depends on w_j and h'_j. 
        # Storage: index i in ctx_backward maps to h'_i. Size L+1.
        mp_bias, mp_w, mp_hp = self.M_prime[:,0], self.M_prime[:, 1:e+1], self.M_prime[:, e+1:]
        e = self.e
        mp_bias = self.M_prime[:, 0]
        mp_w = self.M_prime[:, 1:e+1]
        mp_hp = self.M_prime[:, e+1:]

        h_bw_dense = torch.zeros((L + 1, d), device=dev) # initialized with h'_L = 0
        
        for j in range(L - 1, -1, -1):
            # Computing h'_{j}? No, formula is h'_{j-1} based on h'_j.
            # h_bw_dense[j+1] holds h'_{j+1} ? No notation is tricky.
            # Let's stick to reading convention:
            # h'_j (suffix starting at w_{j+1}) depends on w_{j+1} and h'_{j+1}.
            # My loop j is decreasing.
            # Let's say we compute vector just BEFORE w_j. That is h'_{j-1} in reading.
            # It depends on w_j and vector AFTER w_j (which is h'_j).
            
            hp_next = h_bw_dense[j+1] # This represents vector after w_j
            w_vec = embs[j]
            
            act = mp_bias + (mp_w @ w_vec) + (mp_hp @ hp_next)
            h_bw_dense[j] = torch.sigmoid(act)
            
        self.ctx_backward = h_bw_dense

    @override
    def accumulate_logprob_gradient(self, sentence: Sentence, corpus: TaggedCorpus) -> None:
        isent = self._integerize_sentence(sentence, corpus)
        super().accumulate_logprob_gradient(sentence, corpus)

    @override
    @typechecked
    def A_at(self, position: int, sentence) -> Tensor:
        i = position; dev = self.E.device
        
        # Inputs
        h_p = self.ctx_forward[i-2] if i >= 2 else torch.zeros(self.rnn_dim, device=dev)
        h_s = self.ctx_backward[i] # valid up to L
        
        # Decompose Weights for Broadcasting: Bias + Dot(W, Input)
        d = self.rnn_dim
        k = self.k
        u_bias = self.U_A[:, 0]
        u_hp = self.U_A[:, 1:d+1]
        u_s  = self.U_A[:, d+1 : d+1+k]
        u_t  = self.U_A[:, d+1+k : d+1+2*k]
        u_hs = self.U_A[:, d+1+2*k:]
        
        # Constants
        base = u_bias + (u_hp @ h_p) + (u_hs @ h_s) # (phi,)
        
        # Tag Parts
        # u_s (phi, k).T -> (k, phi). View (k, 1, phi)
        # u_t (phi, k).T -> (k, phi). View (1, k, phi)
        term_s = u_s.T.unsqueeze(1)
        term_t = u_t.T.unsqueeze(0)
        
        f = torch.sigmoid(base.view(1,1,-1) + term_s + term_t) # KxKxPhi
        log_A = (f * self.Theta_A.view(1,1,-1)).sum(dim=-1)
        return torch.exp(log_A)

    @override
    @typechecked
    def B_at(self, position: int, sentence) -> Tensor:
        i = position; dev = self.E.device
        w_idx = sentence[i][0]
        if i >= len(self.ctx_forward): return torch.zeros(self.k, w_idx+2, device=dev)

        h_p = self.ctx_forward[i-1] if i>=1 else torch.zeros(self.rnn_dim, device=dev)
        h_s = self.ctx_backward[i]
        w_vec = self.embedded_sentence[i] # precomputed in setup
        
        d=self.rnn_dim; k=self.k; e=self.e
        # U_B splits: bias(1) + hp(d) + t(k) + w(e) + hs(d)
        curr = 1
        u_bias = self.U_B[:, 0]
        u_hp   = self.U_B[:, curr:curr+d]; curr+=d
        u_t    = self.U_B[:, curr:curr+k]; curr+=k
        u_w    = self.U_B[:, curr:curr+e]; curr+=e
        u_hs   = self.U_B[:, curr:]
        
        base = u_bias + (u_hp @ h_p) + (u_w @ w_vec) + (u_hs @ h_s)
        term_t = u_t.T # (k, phi)
        
        f = torch.sigmoid(base.unsqueeze(0) + term_t)
        scores = torch.exp( (f * self.Theta_B.view(1, -1)).sum(dim=-1) )
        
        valid_V = max(w_idx+1, self.num_words+2)
        B_out = torch.zeros((self.k, valid_V), device=dev)
        B_out[:, w_idx] = scores
        return B_out