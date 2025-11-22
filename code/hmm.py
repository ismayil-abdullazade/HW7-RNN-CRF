#!/usr/bin/env python3

# CS465 at Johns Hopkins University.
# Starter code for Hidden Markov Models.

from __future__ import annotations
import logging
from math import inf, log, exp
from pathlib import Path
import os, time
from typing import Callable, List, Optional, cast
from typeguard import typechecked

import torch
from torch import Tensor, cuda, nn
from jaxtyping import Float

from tqdm import tqdm # type: ignore
import pickle

from integerize import Integerizer
from corpus import BOS_TAG, BOS_WORD, EOS_TAG, EOS_WORD, Sentence, Tag, TaggedCorpus, IntegerizedSentence, Word

TorchScalar = Float[Tensor, ""] # a Tensor with no dimensions, i.e., a scalar

logger = logging.getLogger(Path(__file__).stem)

# Set the seed for random numbers in torch, for replicability
torch.manual_seed(1337)
cuda.manual_seed(69_420)  # No-op if CUDA isn't available

###
# HMM tagger
###
class HiddenMarkovModel:
    """An implementation of an HMM, whose emission probabilities are
    parameterized using the word embeddings in the lexicon.
    
    We'll refer to the HMM states as "tags" and the HMM observations 
    as "words."
    """
    
    def __init__(self, 
                 tagset: Integerizer[Tag],
                 vocab: Integerizer[Word],
                 unigram: bool = False):
        """Construct an HMM with initially random parameters."""

        if vocab[-2:] != [EOS_WORD, BOS_WORD]:
            raise ValueError("final two types of vocab should be EOS_WORD, BOS_WORD")

        self.k = len(tagset)       # number of tag types
        self.V = len(vocab) - 2    # number of word types (not counting EOS_WORD and BOS_WORD)
        self.unigram = unigram     # do we fall back to a unigram model?

        self.tagset = tagset
        self.vocab = vocab

        # Useful constants that are referenced by the methods
        self.bos_t: Optional[int] = tagset.index(BOS_TAG)
        self.eos_t: Optional[int] = tagset.index(EOS_TAG)
        if self.bos_t is None or self.eos_t is None:
            raise ValueError("tagset should contain both BOS_TAG and EOS_TAG")
        assert self.eos_t is not None    # we need this to exist
        self.eye: Tensor = torch.eye(self.k)  # identity matrix

        self.init_params()     # create and initialize model parameters
 
    def init_params(self) -> None:
        """Initialize params to small random values."""

        WB = 0.01*torch.rand(self.k, self.V)  # choose random logits
        self.B = WB.softmax(dim=1)            # construct emission distributions p(w | t)
        self.B[self.eos_t, :] = 0             # EOS_TAG can't emit any column's word
        self.B[self.bos_t, :] = 0             # BOS_TAG can't emit any column's word
        
        rows = 1 if self.unigram else self.k
        WA = 0.01*torch.rand(rows, self.k)
        WA[:, self.bos_t] = -inf    # correct the BOS_TAG column
        self.A = WA.softmax(dim=1)  # construct transition distributions p(t | s)
        if self.unigram:
            self.A = self.A.repeat(self.k, 1)   # copy the single row k times  

    @typechecked
    def setup_sentence(self, isent: IntegerizedSentence) -> None:
        """Precompute any quantities needed for forward/backward/Viterbi algorithms.
        This method may be overridden in subclasses (e.g., for RNN context)."""
        pass

    @typechecked
    def A_at(self, position: int, sentence: IntegerizedSentence) -> Tensor:
        """Return the transition matrix A at the given position."""
        return self.A

    @typechecked
    def B_at(self, position: int, sentence: IntegerizedSentence) -> Tensor:
        """Return the emission matrix B at the given position."""
        return self.B

    def printAB(self) -> None:
        """Print the A and B matrices."""
        print("Transition matrix A:")
        col_headers = [""] + [str(self.tagset[t]) for t in range(self.A.size(1))]
        print("\t".join(col_headers))
        for s in range(self.A.size(0)):   # rows
            row = [str(self.tagset[s])] + [f"{self.A[s,t]:.3f}" for t in range(self.A.size(1))]
            print("\t".join(row))
        print("\nEmission matrix B:")        
        col_headers = [""] + [str(self.vocab[w]) for w in range(self.B.size(1))]
        print("\t".join(col_headers))
        for t in range(self.A.size(0)):   # rows
            row = [str(self.tagset[t])] + [f"{self.B[t,w]:.3f}" for w in range(self.B.size(1))]
            print("\t".join(row))
        print("\n")

    def M_step(self, λ: float) -> None:
        """Set the transition and emission matrices (A, B), using expected counts."""

        # we should have seen no emissions from BOS or EOS tags
        assert self.B_counts[self.eos_t, :].sum() == 0, 'Incorrect accumulated emission counts from EOS'
        assert self.B_counts[self.bos_t, :].sum() == 0, 'Incorrect accumulated emission counts from BOS'

        smoothed_B_counts = self.B_counts + λ
        self.B = smoothed_B_counts / smoothed_B_counts.sum(dim=1, keepdim=True)
        self.B[self.eos_t, :] = 0 
        self.B[self.bos_t, :] = 0

        assert self.A_counts[:, self.bos_t].sum() == 0, 'Incorrect accumulated transition counts to BOS'
        assert self.A_counts[self.eos_t, :].sum() == 0, 'Incorrect accumulated transition counts from EOS'
                
        if self.unigram:
            tag_counts = self.A_counts.sum(dim=0, keepdim=True)
            smoothed_tag_counts = tag_counts + λ
            smoothed_tag_counts[:, self.bos_t] = 0
            smoothed_tag_counts[:, self.eos_t] = 0
            unigram_probs = smoothed_tag_counts / smoothed_tag_counts.sum(dim=1, keepdim=True)
            self.A = unigram_probs.repeat(self.k, 1)
            self.A[:, self.bos_t] = 0 
            self.A[self.bos_t, self.eos_t] = 0 
            self.A[self.eos_t, :] = 0
        else:
            smoothed_A_counts = self.A_counts + λ 
            smoothed_A_counts[:, self.bos_t] = 0 
            smoothed_A_counts[self.bos_t, self.eos_t] = 0 
            self.A = smoothed_A_counts / smoothed_A_counts.sum(dim=1, keepdim=True)
            self.A[self.eos_t, :] = 0 

    def _zero_counts(self):
        """Set the expected counts to 0."""
        self.A_counts = torch.zeros((self.k, self.k), requires_grad=False)
        self.B_counts = torch.zeros((self.k, self.V), requires_grad=False)

    def train(self,
              corpus: TaggedCorpus,
              loss: Callable[[HiddenMarkovModel], float],
              λ: float = 0,
              tolerance: float = 0.001,
              max_steps: int = 50000,
              save_path: Optional[Path|str] = "my_hmm.pkl") -> None:
        """Train the HMM."""
        
        if λ < 0: raise ValueError(f"{λ=} but should be >= 0")
        elif λ == 0: λ = 1e-20    

        self._save_time = time.time()           
        dev_loss = loss(self)             
        old_dev_loss: float = dev_loss    
        steps: int = 0       
        while steps < max_steps:
            self._zero_counts()
            for sentence in tqdm(corpus, total=len(corpus), leave=True):
                isent = self._integerize_sentence(sentence, corpus)
                self.E_step(isent)
                steps += 1

            self.M_step(λ)
            if save_path: self.save(save_path, checkpoint=steps) 
            
            dev_loss = loss(self)  
            if dev_loss >= old_dev_loss * (1-tolerance):
                break
            old_dev_loss = dev_loss          
        
        if save_path: self.save(save_path)
  
    def _integerize_sentence(self, sentence: Sentence, corpus: TaggedCorpus) -> IntegerizedSentence:
        if corpus.tagset != self.tagset or corpus.vocab != self.vocab:
            raise TypeError("The corpus that this sentence came from uses a different tagset or vocab")
        return corpus.integerize_sentence(sentence)

    @typechecked
    def logprob(self, sentence: Sentence, corpus: TaggedCorpus) -> TorchScalar:
        isent = self._integerize_sentence(sentence, corpus)
        return self.forward_pass(isent)

    def E_step(self, isent: IntegerizedSentence, mult: float = 1) -> None:
        if all(tag is not None for _, tag in isent):
            for j in range(1, len(isent)):
                w_j, t_j = isent[j]
                _, t_prev = isent[j-1]
                assert t_j is not None and t_prev is not None
                self.A_counts[t_prev, t_j] += mult
                if w_j < self.V:
                    self.B_counts[t_j, w_j] += mult
            return

        log_Z_forward = self.forward_pass(isent)
        log_Z_backward = self.backward_pass(isent, mult=mult)
        assert torch.isclose(log_Z_forward, log_Z_backward), f"Mismatch: back {log_Z_backward} != forw {log_Z_forward}"

    @typechecked
    def forward_pass(self, isent: IntegerizedSentence) -> TorchScalar:
        self.setup_sentence(isent)
        
        # log-space to avoid underflow
        log_alpha = [torch.empty(self.k) for _ in isent]
        log_alpha[0] = self.eye[self.bos_t].log().nan_to_num(-inf)
        # Wait, eye has 1.0 and 0.0. log(1)=0, log(0)=-inf.
        # safe version used in original: log(alpha + epsilon). 
        # But let's respect logic: log_alpha[0][bos_t] = 0.0, else -inf
        
        # Vectorized forward pass with non-stationary potentials
        for j in range(1, len(isent)):
            w_j, t_j = isent[j]
            
            # Context-dependent matrices
            A = self.A_at(j, isent)
            B = self.B_at(j, isent)
            
            log_A = torch.log(A + 1e-45)
            
            if w_j < self.V:
                log_B_col = torch.log(B[:, w_j] + 1e-45)
            else:
                log_B_col = torch.full((self.k,), float('-inf'))
                if w_j == self.vocab.index(EOS_WORD):
                    log_B_col[self.eos_t] = 0.0
                elif w_j == self.vocab.index(BOS_WORD):
                    log_B_col[self.bos_t] = 0.0
            
            # (k,1) + (k,k) + (1,k) broadcasting
            log_scores = log_alpha[j-1].unsqueeze(1) + log_A + log_B_col.unsqueeze(0)
            log_alpha[j] = torch.logsumexp(log_scores, dim=0) 
            
            if t_j is not None:
                mask = torch.full((self.k,), float('-inf'))
                mask[t_j] = 0.0
                log_alpha[j] = log_alpha[j] + mask
        
        self.log_alpha = log_alpha
        self.log_Z = log_alpha[-1][self.eos_t]
        return self.log_Z

    @typechecked
    def backward_pass(self, isent: IntegerizedSentence, mult: float = 1) -> TorchScalar:
        self.setup_sentence(isent)

        beta = [torch.empty(self.k) for _ in isent]
        beta[-1] = self.eye[self.eos_t]
        
        log_beta = [torch.empty(self.k) for _ in isent]
        log_beta[-1] = self.eye[self.eos_t].log().nan_to_num(-inf)
        
        # Backward pass
        for j in range(len(isent) - 2, -1, -1):
            w_next, t_next = isent[j+1]
            
            # Matrices for position j+1 (the transition happens *to* j+1)
            A = self.A_at(j+1, isent)
            B = self.B_at(j+1, isent)
            log_A = torch.log(A + 1e-45)
            
            if w_next < self.V:
                log_B_col = torch.log(B[:, w_next] + 1e-45)
            else:
                log_B_col = torch.full((self.k,), float('-inf'))
                if w_next == self.vocab.index(EOS_WORD):
                    log_B_col[self.eos_t] = 0.0
                elif w_next == self.vocab.index(BOS_WORD):
                    log_B_col[self.bos_t] = 0.0

            log_scores = log_beta[j+1].unsqueeze(0) + log_A + log_B_col.unsqueeze(0)
            
            if t_next is not None:
                mask = torch.full((self.k,), float('-inf'))
                mask[t_next] = 0.0
                log_scores = log_scores + mask.unsqueeze(0) 
            
            log_beta[j] = torch.logsumexp(log_scores, dim=1)
        
        log_Z_backward = log_beta[0][self.bos_t]
        
        # Accumulate counts with appropriate local A and B
        for j in range(1, len(isent)):
            w_j, t_j = isent[j]
            
            # Local Matrices for current step j
            A = self.A_at(j, isent)
            B = self.B_at(j, isent)
            log_A = torch.log(A + 1e-45)

            if w_j < self.V:
                log_B_col = torch.log(B[:, w_j] + 1e-45)
            else:
                log_B_col = torch.full((self.k,), float('-inf'))
                if w_j == self.vocab.index(EOS_WORD):
                    log_B_col[self.eos_t] = 0.0
                elif w_j == self.vocab.index(BOS_WORD):
                    log_B_col[self.bos_t] = 0.0
            
            log_posteriors = (self.log_alpha[j-1].unsqueeze(1) + log_A + 
                            log_B_col.unsqueeze(0) + log_beta[j].unsqueeze(0) - self.log_Z)
            
            log_posteriors[:, self.bos_t] = float('-inf')
            log_posteriors[self.eos_t, :] = float('-inf')
            
            if t_j is not None:
                mask = torch.full_like(log_posteriors, float('-inf'))
                mask[:, t_j] = 0.0
                log_posteriors = log_posteriors + mask
            
            posteriors = torch.where(log_posteriors > -100, 
                                    torch.exp(log_posteriors), 
                                    torch.tensor(0.0))
            self.A_counts += mult * posteriors
            
            if w_j < self.V:
                log_posteriors_emit = self.log_alpha[j] + log_beta[j] - self.log_Z
                log_posteriors_emit[self.eos_t] = float('-inf')
                log_posteriors_emit[self.bos_t] = float('-inf')
                
                if t_j is not None:
                    mask = torch.full_like(log_posteriors_emit, float('-inf'))
                    mask[t_j] = 0.0
                    log_posteriors_emit = log_posteriors_emit + mask
                
                posteriors_emit = torch.where(log_posteriors_emit > -100,
                                            torch.exp(log_posteriors_emit),
                                            torch.tensor(0.0))
                self.B_counts[:, w_j] += mult * posteriors_emit
        
        return log_Z_backward

    def viterbi_tagging(self, sentence: Sentence, corpus: TaggedCorpus) -> Sentence:
        isent = self._integerize_sentence(sentence, corpus)
        self.setup_sentence(isent) # Prepare contexts

        log_alpha    = [torch.empty(self.k)                  for _ in isent]   
        backpointers = [torch.empty(self.k, dtype=torch.int) for _ in isent]
        tags: List[int] 

        log_alpha[0] = torch.full((self.k,), float('-inf'))
        log_alpha[0][self.bos_t] = 0.0 
        
        for j in range(1, len(isent)):
            w_j, t_j = isent[j]
            
            # Local Matrices
            A = self.A_at(j, isent)
            B = self.B_at(j, isent)
            log_A = torch.log(A + 1e-45)
            
            for t in range(self.k):
                if t_j is not None and t != t_j:
                    log_alpha[j][t] = float('-inf')
                    backpointers[j][t] = -1
                    continue
                
                if w_j < self.V:
                    log_emissions = torch.log(B[t, w_j] + 1e-45) # Specific value from B
                else:
                    if (t == self.eos_t and w_j == self.vocab.index(EOS_WORD)) or \
                       (t == self.bos_t and w_j == self.vocab.index(BOS_WORD)):
                        log_emissions = 0.0 
                    else:
                        log_emissions = float('-inf')
                
                log_scores = log_alpha[j-1] + log_A[:, t] + log_emissions
                log_alpha[j][t], backpointers[j][t] = log_scores.max(dim=0)
        
        tags = [0] * len(isent)
        tags[-1] = self.eos_t
        for j in range(len(isent) - 1, 0, -1):
            tags[j-1] = backpointers[j][tags[j]].item()
        
        if len(isent) <= 10:
            logger.debug(f"Viterbi tags: {[self.tagset[t] for t in tags]}")

        return Sentence([(word, self.tagset[tags[j]]) for j, (word, tag) in enumerate(sentence)])

    def posterior_tagging(self, sentence: Sentence, corpus: TaggedCorpus) -> Sentence:
        isent = self._integerize_sentence(sentence, corpus)
        self.setup_sentence(isent)

        log_Z = self.forward_pass(isent)
        
        old_A_counts = getattr(self, 'A_counts', None)
        old_B_counts = getattr(self, 'B_counts', None)
        self._zero_counts() 
        
        beta = [torch.empty(self.k) for _ in isent]
        beta[-1] = self.eye[self.eos_t]
        log_beta = [torch.empty(self.k) for _ in isent]
        log_beta[-1] = self.eye[self.eos_t].log().nan_to_num(-inf)
        
        # Backward pass (probs only)
        for j in range(len(isent) - 2, -1, -1):
            w_next, t_next = isent[j+1]
            
            # Local A, B
            A = self.A_at(j+1, isent)
            B = self.B_at(j+1, isent)
            log_A = torch.log(A + 1e-45)

            if w_next < self.V:
                log_B_col = torch.log(B[:, w_next] + 1e-45)
            else:
                log_B_col = torch.full((self.k,), float('-inf'))
                if w_next == self.vocab.index(EOS_WORD):
                    log_B_col[self.eos_t] = 0.0
                elif w_next == self.vocab.index(BOS_WORD):
                    log_B_col[self.bos_t] = 0.0
            
            log_scores = log_beta[j+1].unsqueeze(0) + log_A + log_B_col.unsqueeze(0)
            
            if t_next is not None:
                mask = torch.full((self.k,), float('-inf'))
                mask[t_next] = 0.0
                log_scores = log_scores + mask.unsqueeze(0)
            
            log_beta[j] = torch.logsumexp(log_scores, dim=1)
        
        if old_A_counts is not None: self.A_counts = old_A_counts
        if old_B_counts is not None: self.B_counts = old_B_counts
        
        tags: List[int] = []
        for j in range(len(isent)):
            w_j, t_j = isent[j]
            log_posteriors = self.log_alpha[j] + log_beta[j] - log_Z
            if t_j is not None:
                tags.append(t_j)
            else:
                best_tag = log_posteriors.argmax().item()
                tags.append(best_tag)
        
        if len(isent) <= 10:
            logger.debug(f"Posterior tags: {[self.tagset[t] for t in tags]}")
        
        return Sentence([(word, self.tagset[tags[j]]) for j, (word, tag) in enumerate(sentence)])

    def save(self, path: Path|str, checkpoint=None, checkpoint_interval: int = 300) -> None:
        if isinstance(path, str): path = Path(path)
        now = time.time()
        old_save_time =           getattr(self, "_save_time", None)
        old_checkpoint_path =     getattr(self, "_checkpoint_path", None)
        old_total_training_time = getattr(self, "total_training_time", 0)

        if checkpoint is None:
            self._checkpoint_path = None 
        else:    
            if old_save_time is not None and now < old_save_time + checkpoint_interval: 
                return 
            path = path.with_name(f"{path.stem}-{checkpoint}{path.suffix}")  
            self._checkpoint_path = path

        if old_save_time is not None:
            self.total_training_time = old_total_training_time + (now - old_save_time)
        del self._save_time
        
        try:
            torch.save(self, path, pickle_protocol=pickle.HIGHEST_PROTOCOL)
            logger.info(f"Saved model to {path}")
        except Exception as e:   
            self._save_time          = old_save_time
            self._checkpoint_path    = old_checkpoint_path
            self.total_training_time = old_total_training_time
            raise e
        
        self._save_time = now
        if old_checkpoint_path: 
            try: os.remove(old_checkpoint_path)
            except FileNotFoundError: pass 

    @classmethod
    def load(cls, path: Path|str, device: str = 'cpu') -> HiddenMarkovModel:
        if isinstance(path, str): path = Path(path)   
        model = torch.load(path, map_location=device)

        if not isinstance(model, cls):
            raise ValueError(f"Type Error: expected object of type {cls.__name__} but got {model.__class__.__name__} " \
                             f"from saved file {path}.")

        logger.info(f"Loaded model from {path}")
        return model