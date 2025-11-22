#!/usr/bin/env python3

# CS465 at Johns Hopkins University.
# Subclass ConditionalRandomField to use PyTorch facilities.

from __future__ import annotations
import logging
import torch.nn as nn
from pathlib import Path
import time
from typing_extensions import override

import torch
from torch import Tensor, cuda
from jaxtyping import Float

from corpus import Sentence, Tag, TaggedCorpus, Word
from integerize import Integerizer
from crf import ConditionalRandomField

TorchScalar = Float[Tensor, ""] 

logger = logging.getLogger(Path(__file__).stem)  
torch.manual_seed(1337)
cuda.manual_seed(69_420)  

class ConditionalRandomFieldBackprop(ConditionalRandomField, nn.Module):
    @override
    def __init__(self, 
                 tagset: Integerizer[Tag],
                 vocab: Integerizer[Word],
                 unigram: bool = False):
        # Need to initialize nn.Module logic
        nn.Module.__init__(self)  
        super().__init__(tagset, vocab, unigram)
        
        self.count_params()
        self._batch_losses = [] # Store individual sentence losses here
        
    @override
    def init_params(self) -> None:
        """
        Initialize global stationary parameters WA and WB (used if not using Neural/RNN).
        """
        # Simple linear chain params
        self.WB = nn.Parameter(torch.empty(self.k, self.V))
        nn.init.uniform_(self.WB, -0.01, 0.01)
        
        rows = 1 if self.unigram else self.k
        self.WA = nn.Parameter(torch.empty(rows, self.k))
        nn.init.uniform_(self.WA, -0.01, 0.01)

        # Initialize Structural Zeroes to neg large number 
        # We use masking later, but this helps numerical stability
        with torch.no_grad():
            self.WA[:, self.bos_t] = -999
            if not self.unigram:
                self.WA[self.eos_t, :] = -999
            self.WB[self.bos_t, :] = -999
            self.WB[self.eos_t, :] = -999

        self.updateAB() 

    @override
    def updateAB(self) -> None:
        """
        Compute A and B using softmax.
        CRITICAL FIX: Use an epsilon (1e-45) instead of 0.0 for masking.
        log(0) -> -inf, which creates NaNs in gradients (grad = 1/x = 1/0).
        """
        EPSILON = 1e-45 

        # 1. Transition Matrix A
        # self.WA is unnormalized logits.
        A_soft = self.WA.softmax(dim=1)
        
        # Mask structural zeroes
        # Instead of A_soft * mask_A (which makes 0), we perform mixing
        mask_A = torch.ones_like(A_soft)
        mask_A[:, self.bos_t] = 0
        if not self.unigram:
             mask_A[self.eos_t, :] = 0 
        
        # Apply Mask: Keep soft value where mask is 1, put EPSILON where mask is 0
        self.A = A_soft * mask_A + EPSILON * (1 - mask_A)

        # 2. Emission Matrix B
        B_soft = self.WB.softmax(dim=1)
        
        mask_B = torch.ones_like(B_soft)
        mask_B[self.bos_t, :] = 0
        mask_B[self.eos_t, :] = 0
        
        # Apply Mask with Epsilon
        self.B = B_soft * mask_B + EPSILON * (1 - mask_B)

    def init_optimizer(self, lr: float, weight_decay: float) -> None:      
        # Standard SGD for stationary model
        self.optimizer = torch.optim.SGD(
            params=self.parameters(), 
            lr=lr, weight_decay=weight_decay
        )

    def count_params(self) -> None:
        paramcount = sum(p.numel() for p in self.parameters() if p.requires_grad)
        try:
            paramshapes = " + ".join("*".join(str(dim) for dim in p.size()) for p in self.parameters() if p.requires_grad)
            logger.info(f"Parameters: {paramcount} = {paramshapes}")
        except:
            pass

    @override
    def train(self, corpus: TaggedCorpus, *args, minibatch_size: int = 1, lr: float = 1.0, reg: float = 0.0, **kwargs) -> None:
        # Reset optimizer
        self.init_optimizer(lr=lr, weight_decay = 2 * reg * minibatch_size / len(corpus))
        self._save_time = time.time() 
        # Parent train calls the below override hooks
        super().train(corpus, *args, minibatch_size=minibatch_size, lr=lr, reg=reg, **kwargs)

    @override        
    def _zero_grad(self):
        self.optimizer.zero_grad()
        self._batch_losses = []

    @override
    def accumulate_logprob_gradient(self, sentence: Sentence, corpus: TaggedCorpus) -> None:
        """Compute logprob and append to batch list. """
        # Recalculate A/B if parameters changed (rebuilds graph)
        # Note: in BiRNN-CRF this method is 'pass', but needed for base model.
        self.updateAB()
        
        # Minimize negative log likelihood
        # Note: This returns a scalar Tensor if implemented with PyTorch
        logprob = self.logprob(sentence, corpus)
        
        # Store negative logprob for minimization
        self._batch_losses.append(-logprob)

    @override
    def logprob_gradient_step(self, lr: float) -> None:
        if not self._batch_losses: return
        
        # Accumulate gradient for the ENTIRE batch at once.
        # This solves the "Backward through graph twice" issue mentioned in Instructions.
        minibatch_loss = torch.stack(self._batch_losses).sum()
        minibatch_loss.backward()
        
        self.optimizer.step()
        self._batch_losses = []
        
    @override
    def reg_gradient_step(self, lr: float, reg: float, frac: float):
        # L2 Regularization is handled by optimizer weight_decay
        pass

    def learning_speed(self, lr: float, minibatch_size: int) -> float:
        # Helper to track gradient magnitude
        try:
            return lr * sum(torch.sum(p.grad * p.grad).item() 
                            for p in self.parameters() if p.grad is not None) / minibatch_size
        except:
            return 0.0