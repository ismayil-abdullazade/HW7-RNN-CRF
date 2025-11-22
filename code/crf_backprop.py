#!/usr/bin/env python3

# CS465 at Johns Hopkins University.
# Subclass ConditionalRandomField to use PyTorch facilities.

from __future__ import annotations
import logging
import torch.nn as nn
from pathlib import Path
from typing_extensions import override

import torch
from torch import Tensor, cuda
from crf import ConditionalRandomField

logger = logging.getLogger(Path(__file__).stem)  
torch.manual_seed(1337)
cuda.manual_seed(69_420)  

class ConditionalRandomFieldBackprop(ConditionalRandomField, nn.Module):
    @override
    def __init__(self, tagset, vocab, unigram=False):
        nn.Module.__init__(self)  
        super().__init__(tagset, vocab, unigram)
        self.count_params()
        self._batch_losses = []
        
    @override
    def init_params(self) -> None:
        # Init Parameters
        self.WB = nn.Parameter(torch.empty(self.k, self.V))
        nn.init.uniform_(self.WB, -0.01, 0.01)
        
        rows = 1 if self.unigram else self.k
        self.WA = nn.Parameter(torch.empty(rows, self.k))
        nn.init.uniform_(self.WA, -0.01, 0.01)

        # Structural Constraints (Init to valid logits, not -inf)
        # We use -100.0: large enough to be rare after softmax, small enough to not break float32 gradients.
        with torch.no_grad():
            large_neg = -100.0 
            self.WA[:, self.bos_t] = large_neg
            if not self.unigram:
                self.WA[self.eos_t, :] = large_neg
            self.WB[self.bos_t, :] = large_neg
            self.WB[self.eos_t, :] = large_neg

        self.updateAB() 

    @override
    def updateAB(self) -> None:
        # Use epsilon larger than float32 epsilon (approx 1e-38)
        # 1e-20 ensures log(epsilon) is roughly -46, which is a healthy gradient range.
        EPSILON = 1e-20 
        LARGE_NEG_LOGIT = -1e4 

        # 1. Transition A
        wa_safe = self.WA.clone()
        # Manual hard masking on logits
        wa_safe[:, self.bos_t] = LARGE_NEG_LOGIT
        if not self.unigram:
             wa_safe[self.eos_t, :] = LARGE_NEG_LOGIT 

        A_soft = wa_safe.softmax(dim=1)
        
        # Clamp helps in case Softmax produces strict 0 due to underflow on the -1e4
        self.A = torch.clamp(A_soft, min=EPSILON)
        
        if self.unigram:
            self.A = self.A.repeat(self.k, 1)

        # 2. Emission B
        wb_safe = self.WB.clone()
        wb_safe[self.bos_t, :] = LARGE_NEG_LOGIT
        wb_safe[self.eos_t, :] = LARGE_NEG_LOGIT
        
        B_soft = wb_safe.softmax(dim=1)
        self.B = torch.clamp(B_soft, min=EPSILON)

    def init_optimizer(self, lr: float, weight_decay: float) -> None:      
        self.optimizer = torch.optim.SGD(
            params=self.parameters(), 
            lr=lr, weight_decay=weight_decay
        )

    def count_params(self) -> None:
        paramcount = sum(p.numel() for p in self.parameters() if p.requires_grad)
        logger.info(f"Parameters: {paramcount}")

    @override
    def train(self, corpus, *args, minibatch_size=1, lr=1.0, reg=0.0, **kwargs) -> None:
        self.init_optimizer(lr=lr, weight_decay = 2 * reg * minibatch_size / len(corpus))
        super().train(corpus, *args, minibatch_size=minibatch_size, lr=lr, reg=reg, **kwargs)

    @override        
    def _zero_grad(self):
        self.optimizer.zero_grad()
        self._batch_losses = []

    @override
    def accumulate_logprob_gradient(self, sentence, corpus) -> None:
        # Ensure A/B are attached to graph
        self.updateAB()
        
        # Calculate Loss (Forward Pass)
        logprob = self.logprob(sentence, corpus)
        
        # Store negative logprob
        # NOTE: If forward pass hit extreme instabilities, logprob might be inf or nan.
        # We assume 1e-20 clamp fixed that. 
        self._batch_losses.append(-logprob)

    @override
    def logprob_gradient_step(self, lr: float) -> None:
        if not self._batch_losses: return
        
        # Total batch loss
        minibatch_loss = torch.stack(self._batch_losses).sum()
        
        # Safety Guard
        if torch.isnan(minibatch_loss) or torch.isinf(minibatch_loss):
            logger.warning("NaN/Inf loss detected. Resetting batch.")
            self._batch_losses = []
            self.optimizer.zero_grad()
            return

        # Backward
        minibatch_loss.backward()
        
        # Gradient Clipping (Vital for preventing future NaNs)
        torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm=5.0)
        
        self.optimizer.step()
        self._batch_losses = []
        
    @override
    def reg_gradient_step(self, lr, reg, frac):
        # Optimizer handles weight decay
        pass

    def learning_speed(self, lr: float, minibatch_size: int) -> float:
        try:
            total_grad_norm = sum(torch.sum(p.grad**2).item() for p in self.parameters() if p.grad is not None)
            return lr * total_grad_norm / minibatch_size
        except:
            return 0.0