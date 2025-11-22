#!/usr/bin/env python3

# CS465 at Johns Hopkins University.

# Subclass ConditionalRandomField to use PyTorch facilities for gradient computation and gradient-based optimization.

from __future__ import annotations
import logging
import torch.nn as nn
from math import inf
from pathlib import Path
import time
from typing_extensions import override

import torch
from torch import tensor, Tensor, cuda
from jaxtyping import Float

from corpus import Sentence, Tag, TaggedCorpus, Word
from integerize import Integerizer
from crf import ConditionalRandomField

TorchScalar = Float[Tensor, ""] # a Tensor with no dimensions, i.e., a scalar

logger = logging.getLogger(Path(__file__).stem)  # For usage, see findsim.py in earlier assignment.
    # Note: We use the name "logger" this time rather than "log" since we
    # are already using "log" for the mathematical log!

# Set the seed for random numbers in torch, for replicability
torch.manual_seed(1337)
cuda.manual_seed(69_420)  # No-op if CUDA isn't available


class ConditionalRandomFieldBackprop(ConditionalRandomField, nn.Module):
    """An implementation of a CRF that has only transition and 
    emission features, just like an HMM."""
    
    # We inherit most of the functionality from the parent CRF (and HMM)
    # classes. The overridden methods will allow for backpropagation to be done
    # automatically by PyTorch rather than manually as in the parent class.
    # CRFBackprop also inherits from nn.Module so that nn.Parameter will
    # be able to register parameters to be found by self.parameters().
    
    @override
    def __init__(self, 
                 tagset: Integerizer[Tag],
                 vocab: Integerizer[Word],
                 unigram: bool = False):
        # [docstring will be inherited from parent method]
        
        # Call both parent classes' initializers
        nn.Module.__init__(self)  
        super().__init__(tagset, vocab, unigram)

        # Print number of parameters        
        self.count_params()
        
    @override
    def init_params(self) -> None:
        # [docstring will be inherited from parent method]
        
        # This version overrides the parent to use nn.Parameter.
        self.WB = nn.Parameter(torch.empty(self.k, self.V))
        nn.init.uniform_(self.WB, -0.01, 0.01)
        
        rows = 1 if self.unigram else self.k
        self.WA = nn.Parameter(torch.empty(rows, self.k))
        nn.init.uniform_(self.WA, -0.01, 0.01)

        # Structural Zeroes for HMM topology, using -999 to avoid nan during decay
        with torch.no_grad():
            # Transitions to BOS are impossible
            self.WA[:, self.bos_t] = -999
            # Transitions from EOS are impossible (technically unused rows, but good hygiene)
            # However, in HMM code: self.A[self.eos_t, :] = 0. 
            # WA corresponds to transition logits FROM s TO t.
            if not self.unigram:
                self.WA[self.eos_t, :] = -999
            
            # Emissions: BOS and EOS tags emit nothing
            self.WB[self.bos_t, :] = -999
            self.WB[self.eos_t, :] = -999

        self.updateAB()  # update A and B potential matrices from new params

    def init_optimizer(self, lr: float, weight_decay: float) -> None:      
        """Creates an optimizer for training.
        A subclass may override this method to select a different optimizer."""
        self.optimizer = torch.optim.SGD(
            params=self.parameters(),  # all nn.Parameter objects that are stored in attributes of self
            lr=lr, weight_decay=weight_decay
        )

    def count_params(self) -> None:
        paramcount = sum(p.numel() for p in self.parameters() if p.requires_grad)
        paramshapes = " + ".join("*".join(str(dim) for dim in p.size()) for p in self.parameters() if p.requires_grad)
        logger.info(f"Parameters: {paramcount} = {paramshapes}")

    @override
    def train(self,
              corpus: TaggedCorpus,
              *args,
              minibatch_size: int = 1,
              lr: float = 1.0,  # same defaults as in parent
              reg: float = 0.0,
              **kwargs) -> None:
        # [docstring will be inherited from parent method]
        
        # Configure an optimizer.
        # Weight decay in the optimizer augments the minimization
        # objective by adding an L2 regularizer.
        self.init_optimizer(lr=lr,
                            weight_decay = 2 * reg * minibatch_size / len(corpus))
        
        self._save_time = time.time() 
        super().train(corpus, *args, minibatch_size=minibatch_size, lr=lr, reg=reg, **kwargs)

    @override        
    def _zero_grad(self):
        """Resets gradient accumulation."""
        # Instead of just zeroing the gradients, we also prepare a container
        # for the batch losses.
        self.optimizer.zero_grad()
        self._batch_losses = []

    @override
    def accumulate_logprob_gradient(self, sentence: Sentence, corpus: TaggedCorpus) -> None:
        """Computes the log-probability of a sentence and adds it to the batch list."""
        
        # We assume this is called inside the minibatch loop in parent.train().
        # To be efficient, we delay the backward() call until the end of the batch.
        # We negate the logprob because optimizers minimize loss.
        logprob = self.logprob(sentence, corpus)
        self._batch_losses.append(-logprob)

    @override
    def logprob_gradient_step(self, lr: float) -> None:
        """Sum the accumulated batch losses and take a gradient step."""
        
        if not self._batch_losses:
            return

        # Summing log probabilities corresponds to log P(batch).
        # We sum the negated losses (since we minimize).
        minibatch_loss = torch.stack(self._batch_losses).sum()
        
        # Back-propagation to compute gradients
        minibatch_loss.backward()
        
        # Update parameters
        self.optimizer.step()
        
        # Clear the batch for the next step
        self._batch_losses = []
        
    @override
    def reg_gradient_step(self, lr: float, reg: float, frac: float):
        """Perform the regularization gradient step."""
        # L2 Regularization is handled by the optimizer's weight_decay.
        pass

    def learning_speed(self, lr: float, minibatch_size: int) -> float:
        """Estimates how fast we are trying to learn."""     
        return lr * sum(torch.sum(p.grad * p.grad).item()   # squared norm of p, as a float
                        for p in self.parameters() 
                        if p.grad is not None) / minibatch_size