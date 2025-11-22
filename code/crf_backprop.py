#!/usr/bin/env python3
# CS465 at Johns Hopkins University.
# Subclass ConditionalRandomField to use PyTorch facilities.

from __future__ import annotations
import logging
import torch.nn as nn
from pathlib import Path
import time
from typing_extensions import override
import math

import torch
from torch import Tensor, cuda
from jaxtyping import Float

from corpus import Sentence, Tag, TaggedCorpus, Word
from integerize import Integerizer
from crf import ConditionalRandomField

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
        # Standard Initialization
        self.WB = nn.Parameter(torch.empty(self.k, self.V))
        nn.init.uniform_(self.WB, -0.01, 0.01)
        
        rows = 1 if self.unigram else self.k
        self.WA = nn.Parameter(torch.empty(rows, self.k))
        nn.init.uniform_(self.WA, -0.01, 0.01)

        # Robust Structural Zeroes Initialization
        # We use -1e4 (logits) rather than -inf for stability before Softmax
        with torch.no_grad():
            large_neg = -1e4
            self.WA[:, self.bos_t] = large_neg
            if not self.unigram:
                self.WA[self.eos_t, :] = large_neg
            self.WB[self.bos_t, :] = large_neg
            self.WB[self.eos_t, :] = large_neg

        self.updateAB() 

    @override
    def updateAB(self) -> None:
        """
        Compute A and B with extensive safety masking.
        """
        epsilon = 1e-45 # Prevents log(0) in HMM forward pass

        # 1. Transition A
        wa_logits = self.WA.clone()
        # Masking: Force "Impossible" transitions to be effectively 0 probability
        wa_logits[:, self.bos_t] = -1e9
        if not self.unigram:
             wa_logits[self.eos_t, :] = -1e9 
        
        A_soft = wa_masked = wa_logits.softmax(dim=1)
        if self.unigram:
            A_soft = A_soft.repeat(self.k, 1)
            
        # 2. Emission B
        wb_logits = self.WB.clone()
        wb_logits[self.bos_t, :] = -1e9
        wb_logits[self.eos_t, :] = -1e9
        B_soft = wb_logits.softmax(dim=1)

        # 3. Clamping
        # If p=0, log(p) = -inf. This causes NaNs during gradient calculation (1/0).
        # We clamp probability to epsilon.
        self.A = torch.clamp(A_soft, min=epsilon)
        self.B = torch.clamp(B_soft, min=epsilon)
        
        # DEBUG SAMPLER (1% of the time)
        if torch.rand(1).item() < 0.01:
            if torch.isnan(self.A).any(): logger.critical("NaN detected in A matrix inside updateAB")
            if torch.isnan(self.B).any(): logger.critical("NaN detected in B matrix inside updateAB")
            # logger.debug(f"A_stats: max={self.A.max().item():.2e}, min={self.A.min().item():.2e}")

    def init_optimizer(self, lr: float, weight_decay: float) -> None:      
        self.optimizer = torch.optim.SGD(
            params=self.parameters(), 
            lr=lr, weight_decay=weight_decay
        )

    def count_params(self) -> None:
        paramcount = sum(p.numel() for p in self.parameters() if p.requires_grad)
        logger.info(f"Parameters: {paramcount}")

    @override
    def train(self, corpus: TaggedCorpus, *args, minibatch_size: int = 1, lr: float = 1.0, reg: float = 0.0, **kwargs) -> None:
        self.init_optimizer(lr=lr, weight_decay = 2 * reg * minibatch_size / len(corpus))
        super().train(corpus, *args, minibatch_size=minibatch_size, lr=lr, reg=reg, **kwargs)

    @override        
    def _zero_grad(self):
        self.optimizer.zero_grad()
        self._batch_losses = []

    @override
    def accumulate_logprob_gradient(self, sentence: Sentence, corpus: TaggedCorpus) -> None:
        # Ensure A/B is fresh from parameters
        self.updateAB()
        
        # Calculate Loss
        logprob = self.logprob(sentence, corpus)
        
        # Debug Checks
        is_nan = torch.isnan(logprob)
        is_inf = torch.isinf(logprob)
        
        if is_nan or is_inf:
            if is_inf and logprob.item() < 0:
                # This is Log(0) case. Forward pass calculated probability 0 for the supervised sentence.
                # This means the parameters believe this sentence is Impossible.
                # We must cap the loss or gradient will explode.
                # logger.warning("LogProb is -inf (Prob 0). Clamping gradient source.")
                safe_loss = torch.tensor(-1e5, device=logprob.device, requires_grad=True)
                self._batch_losses.append(-safe_loss)
                return
            else:
                logger.error(f"LogProb NaN/Inf detected! Value: {logprob.item()}")
                # Fallback
                self._batch_losses.append(torch.tensor(0.0, device=logprob.device, requires_grad=True))
        else:
            self._batch_losses.append(-logprob)

    @override
    def logprob_gradient_step(self, lr: float) -> None:
        if not self._batch_losses: return
        
        # Sum minibatch loss
        minibatch_loss = torch.stack(self._batch_losses).sum()
        
        if torch.isnan(minibatch_loss):
            logger.critical("Minibatch Total Loss is NaN. Skipping update.")
            self._batch_losses = []
            self.optimizer.zero_grad()
            return

        # Backward Pass
        minibatch_loss.backward()
        
        # Debug Gradients
        has_nan = False
        norm = 0.0
        for n, p in self.named_parameters():
            if p.grad is not None:
                g_norm = p.grad.norm()
                norm += g_norm.item() ** 2
                if torch.isnan(p.grad).any():
                    logger.warning(f"NaN gradient in {n}")
                    has_nan = True
        
        if has_nan:
            logger.critical("Skipping optimizer step due to NaN gradients.")
            self.optimizer.zero_grad()
            self._batch_losses = []
            return

        # Gradient Clipping (Essential for RNN/CRF stability)
        torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm=5.0)
        
        self.optimizer.step()
        self._batch_losses = []
        
    @override
    def reg_gradient_step(self, lr: float, reg: float, frac: float):
        pass