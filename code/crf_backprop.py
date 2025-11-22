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
        nn.Module.__init__(self)  
        super().__init__(tagset, vocab, unigram)
        self.count_params()
        self._batch_losses = []
        
    @override
    def init_params(self) -> None:
        # Define parameters WA and WB.
        # We assume dimensions based on vocab and tagset.
        
        self.WB = nn.Parameter(torch.empty(self.k, self.V))
        nn.init.uniform_(self.WB, -0.01, 0.01)
        
        rows = 1 if self.unigram else self.k
        self.WA = nn.Parameter(torch.empty(rows, self.k))
        nn.init.uniform_(self.WA, -0.01, 0.01)

        # Initialize Structural Zeroes to neg large number for softmasking later
        # (Gradient friendly alternative to -inf)
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
        CRITICAL: Avoids in-place operations (like A[x]=0) to satisfy PyTorch Autograd.
        Uses masking instead.
        """
        # 1. Transition Matrix A
        # self.WA is unnormalized logits.
        A_soft = self.WA.softmax(dim=1)
        
        # Mask structural zeroes
        mask_A = torch.ones_like(A_soft)
        mask_A[:, self.bos_t] = 0  # No transition to BOS
        if not self.unigram:
             mask_A[self.eos_t, :] = 0 # EOS has no outgoing transitions
        
        self.A = A_soft * mask_A

        # 2. Emission Matrix B
        B_soft = self.WB.softmax(dim=1)
        
        mask_B = torch.ones_like(B_soft)
        mask_B[self.bos_t, :] = 0
        mask_B[self.eos_t, :] = 0
        
        self.B = B_soft * mask_B

    def init_optimizer(self, lr: float, weight_decay: float) -> None:      
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
        self.init_optimizer(lr=lr, weight_decay = 2 * reg * minibatch_size / len(corpus))
        self._save_time = time.time() 
        super().train(corpus, *args, minibatch_size=minibatch_size, lr=lr, reg=reg, **kwargs)

    @override        
    def _zero_grad(self):
        self.optimizer.zero_grad()
        self._batch_losses = []

    @override
    def accumulate_logprob_gradient(self, sentence: Sentence, corpus: TaggedCorpus) -> None:
        """Compute logprob and append to batch list. """
        # Important: Update A/B from the current parameters WA/WB for this batch step.
        # Otherwise the computation graph is stale.
        self.updateAB()
        
        # We minimize negative log likelihood
        logprob = self.logprob(sentence, corpus)
        self._batch_losses.append(-logprob)

    @override
    def logprob_gradient_step(self, lr: float) -> None:
        if not self._batch_losses: return
        
        # Single backward pass for the whole batch
        minibatch_loss = torch.stack(self._batch_losses).sum()
        minibatch_loss.backward()
        
        self.optimizer.step()
        self._batch_losses = []
        
    @override
    def reg_gradient_step(self, lr: float, reg: float, frac: float):
        # L2 Regularization handled by optimizer
        pass

    def learning_speed(self, lr: float, minibatch_size: int) -> float:
        try:
            return lr * sum(torch.sum(p.grad * p.grad).item() 
                            for p in self.parameters() if p.grad is not None) / minibatch_size
        except:
            return 0.0