#!/usr/bin/env python3

# CS465 at Johns Hopkins University.
# Subclass ConditionalRandomField to use PyTorch facilities.

from __future__ import annotations
import logging
import torch.nn as nn
import torch.nn.functional as F
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
        
        # Using uniform small init
        self.WB = nn.Parameter(torch.empty(self.k, self.V))
        nn.init.uniform_(self.WB, -0.01, 0.01)
        
        rows = 1 if self.unigram else self.k
        self.WA = nn.Parameter(torch.empty(rows, self.k))
        nn.init.uniform_(self.WA, -0.01, 0.01)

        # Initialize Structural Zeroes logic
        # We do NOT set them to -inf here because they are trainable parameters 
        # in this setup and will be handled via Softmax masking in updateAB.
        # But we can set them to a large negative value to start right.
        with torch.no_grad():
            self.WA[:, self.bos_t] = -1e4 
            if not self.unigram:
                self.WA[self.eos_t, :] = -1e4
            self.WB[self.bos_t, :] = -1e4
            self.WB[self.eos_t, :] = -1e4

        self.updateAB() 

    @override
    def updateAB(self) -> None:
        """
        Compute A and B using softmax.
        Applies -1e9 mask to logits BEFORE softmax to maintain probability summing to 1.
        Applies clamps to avoid strict zeros.
        """
        
        # --- 1. Transition Matrix A ---
        wa_logits = self.WA.clone()
        
        # Mask invalid logits with large negative value
        # (Transition TO BOS is impossible)
        wa_logits[:, self.bos_t] = -1e9
        
        if not self.unigram:
            # (Transition FROM EOS is impossible)
            wa_logits[self.eos_t, :] = -1e9 
            
        A_soft = torch.softmax(wa_logits, dim=1)
        
        if self.unigram:
            # Broadcast unigram logits to full transition matrix
            A_soft = A_soft.repeat(self.k, 1)

        # --- 2. Emission Matrix B ---
        wb_logits = self.WB.clone()
        
        # (BOS and EOS emit nothing)
        wb_logits[self.bos_t, :] = -1e9
        wb_logits[self.eos_t, :] = -1e9
        
        B_soft = torch.softmax(wb_logits, dim=1)

        # --- 3. Stabilization ---
        # Replace strict 0.0 with epsilon (1e-45) to prevent log(0) -> NaN 
        # in the forward/logprob pass calculation.
        # Using a value smaller than float precision epsilon ensures it acts like zero 
        # but doesn't crash log().
        epsilon = 1e-45
        self.A = torch.clamp(A_soft, min=epsilon)
        self.B = torch.clamp(B_soft, min=epsilon)

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
        # Reset optimizer with new hyperparameters
        self.init_optimizer(lr=lr, weight_decay = 2 * reg * minibatch_size / len(corpus))
        super().train(corpus, *args, minibatch_size=minibatch_size, lr=lr, reg=reg, **kwargs)

    @override        
    def _zero_grad(self):
        self.optimizer.zero_grad()
        self._batch_losses = []

    @override
    def accumulate_logprob_gradient(self, sentence: Sentence, corpus: TaggedCorpus) -> None:
        """Compute logprob and append to batch list. """
        # Ensure A/B reflect current parameters for graph construction
        self.updateAB()
        
        # We minimize negative log likelihood
        # Forward pass happens here
        logprob = self.logprob(sentence, corpus)
        
        self._batch_losses.append(-logprob)

    @override
    def logprob_gradient_step(self, lr: float) -> None:
        if not self._batch_losses: return
        
        # Sum loss for batch and backward ONCE
        minibatch_loss = torch.stack(self._batch_losses).sum()
        
        if torch.isnan(minibatch_loss):
            # Emergency Guard: If loss is already nan, skip update to save the model params
            # logger.warning("NaN loss detected, skipping step")
            self._batch_losses = []
            self.optimizer.zero_grad()
            return

        minibatch_loss.backward()
        
        # Gradient Clipping is crucial for RNNs/CRFs to prevent explosion
        torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm=5.0)
        
        self.optimizer.step()
        self._batch_losses = []
        
    @override
    def reg_gradient_step(self, lr: float, reg: float, frac: float):
        # L2 Regularization handled by optimizer's weight_decay
        pass

    def learning_speed(self, lr: float, minibatch_size: int) -> float:
        try:
            return lr * sum(torch.sum(p.grad * p.grad).item() 
                            for p in self.parameters() if p.grad is not None) / minibatch_size
        except:
            return 0.0