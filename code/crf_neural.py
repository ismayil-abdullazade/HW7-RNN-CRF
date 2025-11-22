#!/usr/bin/env python3

# CS465 at Johns Hopkins University.
# Subclass ConditionalRandomFieldBackprop to get a biRNN-CRF model.

from __future__ import annotations
import logging
import torch.nn as nn
import torch.nn.functional as F
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
        if unigram: raise NotImplementedError("BiRNN-CRF requires bigram features (A_st)")

        self.rnn_dim = rnn_dim
        self.E = lexicon # Fixed embeddings (NumWords x EmbSize)
        self.e = lexicon.size(1)
        self.num_words = len(vocab)
        
        # Feature mixing dimensionality (internal hidden layer size for potentials)
        self.phi_dim = 64

        nn.Module.__init__(self)  
        # Note: super().__init__ will call init_params(), so self.E etc must be set before.
        super().__init__(tagset, vocab, unigram)

    @override
    def init_params(self) -> None:
        """Initialize parameters for M, M', U_a, U_b, theta_a and theta_b.
        See Reading Handout Eq 46, 47, 48."""
        
        # 1. Bi-Directional RNN Parameters (Handout Eq 46)
        # We manually implement the RNN cells to control the inputs [1; h; w] exactly.
        
        # M: Maps [1, h, w] -> h (Forward)
        # dim: output x input = d x (1 + d + e)
        self.M = nn.Parameter(torch.empty(self.rnn_dim, 1 + self.rnn_dim + self.e))
        
        # M': Maps [1, w, h'] -> h' (Backward)
        # dim: output x input = d x (1 + e + d)
        self.M_prime = nn.Parameter(torch.empty(self.rnn_dim, 1 + self.e + self.rnn_dim))
        
        # 2. Transition Feature MLP (Handout Eq 47)
        # U_A maps [1, h_{i-2}, s, t, h'_i] -> f_A (phi_dim)
        # Inputs: Bias(1) + h_prev(d) + tag_s(k) + tag_t(k) + h_next(d)
        # Note: We treat tags as one-hot, effectively embedding them.
        dim_UA = 1 + 2*self.rnn_dim + 2*self.k
        self.U_A = nn.Parameter(torch.empty(self.phi_dim, dim_UA))
        
        # Weight vector Theta_A to project f_A -> log potential
        self.Theta_A = nn.Parameter(torch.empty(self.phi_dim))
        
        # 3. Emission Feature MLP (Handout Eq 48)
        # U_B maps [1, h_{i-1}, t, w, h'_i] -> f_B (phi_dim)
        # Inputs: Bias(1) + h_prev(d) + tag_t(k) + word(e) + h_next(d)
        dim_UB = 1 + 2*self.rnn_dim + self.k + self.e
        self.U_B = nn.Parameter(torch.empty(self.phi_dim, dim_UB))
        
        # Weight vector Theta_B to project f_B -> log potential
        self.Theta_B = nn.Parameter(torch.empty(self.phi_dim))
        
        # Init strategy: Xavier Uniform for Weights, Normal for Thetas
        for p in [self.M, self.M_prime, self.U_A, self.U_B]:
            nn.init.xavier_uniform_(p)
        for p in [self.Theta_A, self.Theta_B]:
            nn.init.normal_(p, mean=0, std=0.1)

    @override
    def init_optimizer(self, lr: float, weight_decay: float) -> None:
        # Per instructions Step 0 notes / Reading: Neural Nets usually prefer AdamW
        self.optimizer = torch.optim.AdamW(params=self.parameters(), lr=lr, weight_decay=weight_decay)       
       
    @override
    def updateAB(self) -> None:
        # We don't update static A/B matrices in the neural model. 
        # A_at and B_at calculate them dynamically.
        pass 

    @override
    def setup_sentence(self, isent: IntegerizedSentence) -> None:
        """Pre-compute the manual RNN states h (forward) and h' (backward) for the sentence.
        This 'Eager' approach (Instructions Step 4) prevents O(L^2) re-computation."""
        
        dev = next(self.parameters()).device
        if self.E.device != dev: self.E = self.E.to(dev)
        
        word_indices = torch.tensor([w for w, t in isent], dtype=torch.long, device=dev)
        word_embs = F.embedding(word_indices, self.E) # L x e
        
        L = len(isent)
        d = self.rnn_dim
        
        # --- 1. Forward Scan (M) ---
        # Computes h_0 ... h_{L-1} where h_j uses prefix w_0...w_j
        # Eq 46: h_j = sigma(M @ [1; h_{j-1}; w_j])
        # We initialize h_{-1} as 0
        
        m_bias = self.M[:, 0]           # (d,)
        m_h    = self.M[:, 1:d+1]       # (d, d)
        m_w    = self.M[:, d+1:]        # (d, e)
        
        h = torch.zeros(d, device=dev)  # h_{-1}
        h_fw = []
        for j in range(L):
            # Activation: bias + Mh * h + Mw * w
            act = m_bias + (m_h @ h) + (m_w @ word_embs[j])
            h = torch.sigmoid(act)
            h_fw.append(h)
        self.ctx_forward = torch.stack(h_fw) # Size (L, d). index j -> h_j

        # --- 2. Backward Scan (M') ---
        # Computes h'_0 ... h'_{L-1} where h'_{j-1} depends on w_j, h'_j
        # Note: We store this such that accessing index i gives h'_i (the vector *after* word i)
        # Wait - let's align with Eq 46 exactly: h'_{j-1} = sigma(M' @ [1; w_j; h'_j])
        # Base case: h'_n = 0 (represented as state L in our dense storage)
        
        mp_bias = self.M_prime[:, 0]        # (d,)
        mp_w    = self.M_prime[:, 1:e+1]    # (d, e)
        mp_hp   = self.M_prime[:, e+1:]     # (d, d)
        e = self.e

        h_bw_dense = torch.zeros((L + 1, d), device=dev) # index L = h'_L = 0
        
        for j in range(L - 1, -1, -1):
            # Calculating h' just before word w_j (i.e. h'_{j-1})
            # It uses w_j and h' just after word w_j (h'_j)
            
            hp_next = h_bw_dense[j+1] # This corresponds to h'_j
            w_vec = word_embs[j]
            
            act = mp_bias + (mp_w @ w_vec) + (mp_hp @ hp_next)
            
            # Store result at index j (effectively h'_{j-1}) 
            # Careful: convention says h'_i is suffix after word i. 
            # The calculation loop generates "Vector left of w_j" based on "Vector right of w_j".
            h_bw_dense[j] = torch.sigmoid(act)
            
        self.ctx_backward = h_bw_dense
        # Map: ctx_backward[i] gives h'_i. 
        # Check: ctx_backward[0] was computed using w_0 and ctx_backward[1]. 
        # This represents vector left of w_0. i.e. h'_{-1}.
        # So accessing [i] gives vector *left* of word i.
        self.embedded_sentence = word_embs

    @override
    def accumulate_logprob_gradient(self, sentence: Sentence, corpus: TaggedCorpus) -> None:
        # Ensure lexicon operations are on the right device if moved
        isent = self._integerize_sentence(sentence, corpus)
        # Precompute biRNN vectors for this specific sentence
        self.setup_sentence(isent) 
        # Call parent to run forward pass
        super().accumulate_logprob_gradient(sentence, corpus)

    @override
    @typechecked
    def A_at(self, position: int, sentence) -> Tensor:
        """Compute non-stationary transition matrix A at position i."""
        i = position; dev = self.E.device
        
        # Inputs according to Eq 47: h_{i-2} and h'_i
        # Be careful with indices. Position i in Viterbi usually means transitioning to tag at i.
        # (Tag sequence indices usually 1-based in math, but 0-based in list).
        # h_{i-2}: if i < 2, use zero.
        
        h_p = self.ctx_forward[i-2] if i >= 2 else torch.zeros(self.rnn_dim, device=dev)
        
        # h'_i: Context AFTER the bigram? 
        # If bigram is t_{i-1} -> t_i. This occurs across words w_{i-1}, w_i.
        # We need context to the right of word i? Or right of i-1?
        # Footnote 32: "prefix... h_{i-2}... suffix h'_i".
        # Our backward array stores "vector left of word k" at index k.
        # So we need vector to the right of word i-1. Which is left of word i.
        # That is ctx_backward[i]. 
        if i < len(self.ctx_backward):
            h_s = self.ctx_backward[i] 
        else:
            h_s = torch.zeros(self.rnn_dim, device=dev)

        # U_A layout: bias | h_p | s | t | h_s
        d = self.rnn_dim; k = self.k
        u_bias = self.U_A[:, 0]                 # (phi,)
        u_hp   = self.U_A[:, 1:d+1]             # (phi, d)
        u_s    = self.U_A[:, d+1 : d+1+k]       # (phi, k)
        u_t    = self.U_A[:, d+1+k : d+1+2*k]   # (phi, k)
        u_hs   = self.U_A[:, d+1+2*k:]          # (phi, d)
        
        # Compute constant part (doesn't depend on s,t tags)
        base = u_bias + (u_hp @ h_p) + (u_hs @ h_s) # Shape (phi,)
        
        # Tag Parts: Use Broadcasting to compute for all (s, t) pairs at once
        # u_s: weights for tag s. Shape (phi, k). Transpose -> (k, phi). Reshape (k, 1, phi)
        term_s = u_s.T.unsqueeze(1) 
        
        # u_t: weights for tag t. Shape (phi, k). Transpose -> (k, phi). Reshape (1, k, phi)
        term_t = u_t.T.unsqueeze(0)
        
        # Add everything: (1,1,phi) + (k,1,phi) + (1,k,phi) -> (k, k, phi)
        f_A = torch.sigmoid(base.view(1,1,-1) + term_s + term_t) 
        
        # Dot product with Theta_A: (k,k,phi) * (1,1,phi) -> sum last dim -> (k,k)
        log_A = (f_A * self.Theta_A.view(1,1,-1)).sum(dim=-1)
        
        # Exponentiate to get potential (unnormalized prob)
        return torch.exp(log_A)

    @override
    @typechecked
    def B_at(self, position: int, sentence) -> Tensor:
        """Compute non-stationary emission matrix B at position i.
        Optimized: only computes column for the actual observed word w."""
        
        i = position; dev = self.E.device
        w_idx = sentence[i][0]
        
        # Eq 48 inputs: h_{i-1}, w_vec, h'_i
        h_p = self.ctx_forward[i-1] if i >= 1 else torch.zeros(self.rnn_dim, device=dev)
        h_s = self.ctx_backward[i] if i < len(self.ctx_backward) else torch.zeros(self.rnn_dim, device=dev)
        w_vec = self.embedded_sentence[i]
        
        d=self.rnn_dim; k=self.k; e=self.e
        
        # U_B layout: bias | hp | t | w | hs
        curr = 0
        u_bias = self.U_B[:, curr]; curr += 1
        u_hp   = self.U_B[:, curr : curr+d]; curr += d
        u_t    = self.U_B[:, curr : curr+k]; curr += k
        u_w    = self.U_B[:, curr : curr+e]; curr += e
        u_hs   = self.U_B[:, curr:]
        
        # Constant parts (dep on context and word, not tag)
        # bias + hp@h_p + w@w_vec + hs@h_s
        base = u_bias + (u_hp @ h_p) + (u_w @ w_vec) + (u_hs @ h_s) # (phi,)
        
        # Tag part: (phi, k)
        term_t = u_t.T # (k, phi)
        
        # Sigmoid(Base + term_t). Broadcasting (1, phi) + (k, phi)
        f_B = torch.sigmoid(base.unsqueeze(0) + term_t) # (k, phi)
        
        # Theta dot product
        log_scores = (f_B * self.Theta_B.view(1, -1)).sum(dim=-1) # (k,)
        scores = torch.exp(log_scores)
        
        # Construct B matrix (K x V). Only the column for w_idx is filled.
        # Instructions allow B_at to effectively return potentials for the specific position.
        B_out = torch.zeros((self.k, self.num_words), device=dev)
        # B_out is huge/sparse? Just fill the column for the observed word.
        # This implies the parent class calls B[:, word] at some point.
        # Or checks A_at/B_at
        B_out[:, w_idx] = scores
        return B_out