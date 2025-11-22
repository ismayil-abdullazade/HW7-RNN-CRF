#!/usr/bin/env python3
from __future__ import annotations
import logging
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path
from typing_extensions import override
from typeguard import typechecked

import torch
from torch import Tensor
from crf_backprop import ConditionalRandomFieldBackprop

logger = logging.getLogger(Path(__file__).stem)
torch.manual_seed(1337)

class ConditionalRandomFieldNeural(ConditionalRandomFieldBackprop):
    neural = True    
    
    @override
    def __init__(self, tagset, vocab, lexicon, rnn_dim, unigram=False):
        if unigram: raise NotImplementedError("BiRNN-CRF requires bigram structure")
        self.rnn_dim = rnn_dim
        self.E = lexicon
        self.e = lexicon.size(1)
        self.num_words = len(vocab)
        self.phi_dim = 64 

        nn.Module.__init__(self)  
        super().__init__(tagset, vocab, unigram)

    @override
    def init_params(self) -> None:
        # BiRNN Params
        self.M = nn.Parameter(torch.empty(self.rnn_dim, 1 + self.rnn_dim + self.e))
        self.M_prime = nn.Parameter(torch.empty(self.rnn_dim, 1 + self.e + self.rnn_dim))
        
        # Potential Network Params
        dim_UA = 1 + 2*self.rnn_dim + 2*self.k
        self.U_A = nn.Parameter(torch.empty(self.phi_dim, dim_UA))
        self.Theta_A = nn.Parameter(torch.empty(self.phi_dim))
        
        dim_UB = 1 + 2*self.rnn_dim + self.k + self.e
        self.U_B = nn.Parameter(torch.empty(self.phi_dim, dim_UB))
        self.Theta_B = nn.Parameter(torch.empty(self.phi_dim))
        
        # Initialize (Gain < 1 to keep starting activations controlled)
        for p in [self.M, self.M_prime, self.U_A, self.U_B]:
            nn.init.xavier_uniform_(p, gain=0.5)
        for p in [self.Theta_A, self.Theta_B]:
            nn.init.normal_(p, std=0.01)

    @override
    def init_optimizer(self, lr: float, weight_decay: float) -> None:
        # AdamW is standard for Neural Nets
        self.optimizer = torch.optim.AdamW(params=self.parameters(), lr=lr, weight_decay=weight_decay)       
       
    @override
    def updateAB(self) -> None:
        pass # Dynamic calculation in A_at/B_at

    @override
    def setup_sentence(self, isent) -> None:
        dev = next(self.parameters()).device
        if self.E.device != dev: self.E = self.E.to(dev)
        
        word_idx = torch.tensor([w for w, t in isent], dtype=torch.long, device=dev)
        embs = F.embedding(word_idx, self.E) 
        L, d = len(isent), self.rnn_dim
        
        # Forward BiRNN
        h = torch.zeros(d, device=dev)
        h_fw = []
        m_bias, m_h, m_w = self.M[:,0], self.M[:, 1:d+1], self.M[:, d+1:]
        for j in range(L):
            h = torch.sigmoid(m_bias + (m_h @ h) + (m_w @ embs[j]))
            h_fw.append(h)
        self.ctx_forward = torch.stack(h_fw) 

        # Backward BiRNN
        h_bw = torch.zeros((L + 1, d), device=dev)
        mp_bias, mp_w, mp_hp = self.M_prime[:,0], self.M_prime[:, 1:self.e+1], self.M_prime[:, self.e+1:]
        for j in range(L - 1, -1, -1):
            h_bw[j] = torch.sigmoid(mp_bias + (mp_w @ embs[j]) + (mp_hp @ h_bw[j+1]))
        self.ctx_backward = h_bw
        self.embedded_sentence = embs

    @override
    def accumulate_logprob_gradient(self, sentence, corpus) -> None:
        isent = self._integerize_sentence(sentence, corpus)
        self.setup_sentence(isent) 
        super().accumulate_logprob_gradient(sentence, corpus)

    @override
    @typechecked
    def A_at(self, position: int, sentence) -> Tensor:
        i = position
        dev = self.E.device
        
        h_p = self.ctx_forward[i-2] if i >= 2 else torch.zeros(self.rnn_dim, device=dev)
        h_s = self.ctx_backward[i] if i < len(self.ctx_backward) else torch.zeros(self.rnn_dim, device=dev)
        
        # Weights decomposition
        d, k = self.rnn_dim, self.k
        curr = 0
        u_bias, curr = self.U_A[:,0], 1
        u_hp, curr = self.U_A[:, curr:curr+d], curr+d
        u_s,  curr = self.U_A[:, curr:curr+k], curr+k
        u_t,  curr = self.U_A[:, curr:curr+k], curr+k
        u_hs, curr = self.U_A[:, curr:], curr+d
        
        # MLP Base
        base = u_bias + (u_hp @ h_p) + (u_hs @ h_s)
        # Tensor broadcast mixing
        f = torch.sigmoid(base.view(1,1,-1) + u_s.T.unsqueeze(1) + u_t.T.unsqueeze(0))
        
        log_A = (f * self.Theta_A.view(1,1,-1)).sum(dim=-1)
        # Max log clamp (avoid exp explosion)
        log_A = torch.clamp(log_A, max=20.0)
        return torch.exp(log_A)

    @override
    @typechecked
    def B_at(self, position: int, sentence) -> Tensor:
        i, dev = position, self.E.device
        w_idx = sentence[i][0]
        if i >= len(self.embedded_sentence): return torch.zeros(self.k, self.num_words, device=dev)
        
        h_p = self.ctx_forward[i-1] if i>=1 else torch.zeros(self.rnn_dim, device=dev)
        h_s = self.ctx_backward[i]
        w_vec = self.embedded_sentence[i] 
        
        d, k, e = self.rnn_dim, self.k, self.e
        curr = 1
        u_hp, curr = self.U_B[:, curr:curr+d], curr+d
        u_t,  curr = self.U_B[:, curr:curr+k], curr+k
        u_w,  curr = self.U_B[:, curr:curr+e], curr+e
        u_hs, curr = self.U_B[:, curr:], curr+d
        
        base = self.U_B[:,0] + (u_hp @ h_p) + (u_w @ w_vec) + (u_hs @ h_s)
        
        # Emission potentials (optimized: just calculate column for observed word?)
        # Note: The parent implementation likely wants a B matrix, but usually checks only B[:, w].
        # B_at typically returns Shape (K, V). 
        # Returning zeros everywhere except column w_idx saves computation but is strictly specific.
        
        f = torch.sigmoid(base.unsqueeze(0) + u_t.T)
        log_val = (f * self.Theta_B.view(1,-1)).sum(dim=-1)
        log_val = torch.clamp(log_val, max=20.0)
        scores = torch.exp(log_val)
        
        # Initialize with safe epsilon for unused entries
        B_out = torch.full((self.k, self.num_words), 1e-20, device=dev)
        B_out[:, w_idx] = scores
        return B_out