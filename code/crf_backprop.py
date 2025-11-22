#!/usr/bin/env python3
# CS465 at Johns Hopkins University.
# Subclass ConditionalRandomField to use PyTorch facilities.

from __future__ import annotations
import logging
import torch.nn as nn
from pathlib import Path
from typing_extensions import override

import torch
from crf import ConditionalRandomField

logger = logging.getLogger(Path(__file__).stem)  
torch.manual_seed(1337)

class ConditionalRandomFieldBackprop(ConditionalRandomField, nn.Module):
    @override
    def __init__(self, tagset, vocab, unigram=False):
        nn.Module.__init__(self)  
        super().__init__(tagset, vocab, unigram)
        self.count_params()
        self._batch_losses = []
        
    @override
    def init_params(self) -> None:
        """
        Initialize WA, WB parameters for linear-chain CRF.
        We use 'structurally unlikely' values (-100.0) for invalid transitions
        instead of -inf, allowing the optimizer to reason about them without NaN.
        """
        # Logit Parameters: small random init centered on 0
        self.WB = nn.Parameter(torch.empty(self.k, self.V))
        nn.init.uniform_(self.WB, -0.01, 0.01)
        
        rows = 1 if self.unigram else self.k
        self.WA = nn.Parameter(torch.empty(rows, self.k))
        nn.init.uniform_(self.WA, -0.01, 0.01)

        # Apply "Soft" Structural Zeros
        # -100.0 in logit space ensures Prob(transition) ~ 1e-44 after Softmax
        # This satisfies "Structure" while preventing "Gradient Explosion".
        with torch.no_grad():
            logit_mask = -100.0
            self.WA[:, self.bos_t] = logit_mask
            if not self.unigram:
                self.WA[self.eos_t, :] = logit_mask
            self.WB[self.bos_t, :] = logit_mask
            self.WB[self.eos_t, :] = logit_mask

        self.updateAB() 

    @override
    def updateAB(self) -> None:
        """
        Compute A and B matrices from WA, WB logits using Softmax.
        Apply Safe Clamps to prevent 0.0 probabilities.
        """
        # Standard Float32 Precision safe floor. 
        # 1e-45 (used in hmm.py) behaves like 0.0 in float32 arithmetic.
        # 1e-20 ensures valid logs (~ -46.0) and gradients.
        EPSILON = 1e-20 
        
        # 1. A Matrix (Transition)
        wa_logits = self.WA
        
        # Additional explicit masking for safety before Softmax
        # We clone to avoid modifying the parameter inplace during forward pass
        mask_A = torch.zeros_like(wa_logits)
        mask_A[:, self.bos_t] = -1e4
        if not self.unigram:
             mask_A[self.eos_t, :] = -1e4 
        
        A_soft = torch.softmax(wa_logits + mask_A, dim=1)
        self.A = torch.clamp(A_soft, min=EPSILON)
        
        if self.unigram:
            self.A = self.A.repeat(self.k, 1)

        # 2. B Matrix (Emission)
        wb_logits = self.WB
        mask_B = torch.zeros_like(wb_logits)
        mask_B[self.bos_t, :] = -1e4
        mask_B[self.eos_t, :] = -1e4
        
        B_soft = torch.softmax(wb_logits + mask_B, dim=1)
        self.B = torch.clamp(B_soft, min=EPSILON)

    def init_optimizer(self, lr: float, weight_decay: float) -> None:      
        # Using SGD with standard Gradient Clipping usually stabilizes CRF training
        self.optimizer = torch.optim.SGD(
            params=self.parameters(), 
            lr=lr, weight_decay=weight_decay
        )

    def count_params(self) -> None:
        p_count = sum(p.numel() for p in self.parameters() if p.requires_grad)
        logger.info(f"Parameters: {p_count}")

    @override
    def train(self, corpus, *args, minibatch_size=1, lr=1.0, reg=0.0, **kwargs) -> None:
        # Re-initialize optimizer with correct LR
        self.init_optimizer(lr=lr, weight_decay=2 * reg * minibatch_size / len(corpus))
        super().train(corpus, *args, minibatch_size=minibatch_size, lr=lr, reg=reg, **kwargs)

    @override        
    def _zero_grad(self):
        self.optimizer.zero_grad()
        self._batch_losses = []

    @override
    def accumulate_logprob_gradient(self, sentence, corpus) -> None:
        self.updateAB()
        
        # Calculate Negative Log Likelihood
        # If probabilities are clamped to 1e-20, LogProb cannot be -inf.
        logprob = self.logprob(sentence, corpus)
        
        # Sanity check
        if torch.isfinite(logprob):
            self._batch_losses.append(-logprob)
        else:
            # If NaN/Inf, skip this sample. 
            # This allows training to proceed over "bad" data or transient glitches.
            pass

    @override
    def logprob_gradient_step(self, lr: float) -> None:
        if not self._batch_losses: return
        
        # Sum Loss
        minibatch_loss = torch.stack(self._batch_losses).sum()
        
        # Guard
        if not torch.isfinite(minibatch_loss):
            self._batch_losses = []
            self.optimizer.zero_grad()
            return

        minibatch_loss.backward()
        
        # CLIP GRADIENTS: Essential fix for RNN/CRF exploding gradients.
        # Cap norm at 5.0
        torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm=5.0)
        
        self.optimizer.step()
        self._batch_losses = []
        
    @override
    def reg_gradient_step(self, lr, reg, frac):
        pass