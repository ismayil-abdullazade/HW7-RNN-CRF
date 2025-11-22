#!/usr/bin/env python3

# CS465 at Johns Hopkins University.
# Module for constructing a lexicon of word features.

import logging
from pathlib import Path
from typing import Optional, Set, List

import torch
from torch import Tensor

from corpus import TaggedCorpus, BOS_WORD, EOS_WORD, BOS_TAG, EOS_TAG, OOV_WORD, Word
from integerize import Integerizer

log = logging.getLogger(Path(__file__).stem)

def build_lexicon(corpus: TaggedCorpus,
                  one_hot: bool = False,
                  embeddings_file: Optional[Path] = None,
                  newvocab: Optional[Integerizer[Word]] = None,
                  problex: bool = False,
                  affixes: bool = False) -> torch.Tensor:
    
    """Returns a lexicon, implemented as a matrix Tensor."""
    
    matrices = [torch.empty(len(corpus.vocab), 0)]  

    if one_hot: 
        matrices.append(one_hot_lexicon(corpus))
    if problex:
        matrices.append(problex_lexicon(corpus))
    if embeddings_file is not None:
        matrices.append(embeddings_lexicon(corpus, embeddings_file,
                                           newvocab=newvocab))  
    # Affixes typically called last so it can feature-ize words added by embeddings
    if affixes:
        matrices.append(affixes_lexicon(corpus, newvocab=newvocab))
    
    # Handle padding for disparate vocabulary sizes
    # (e.g. if embeddings_lexicon or affixes_lexicon grew the vocab)
    padded_matrices = []
    vocab_size = len(corpus.vocab)
    
    for m in matrices:
        diff = vocab_size - m.size(0)
        if diff > 0:
            # Pad with zeros for the newly added words
            padding = torch.zeros(diff, m.size(1))
            m = torch.cat([m, padding], dim=0)
        padded_matrices.append(m)

    return torch.cat(padded_matrices, dim=1)   

def one_hot_lexicon(corpus: TaggedCorpus) -> torch.Tensor:
    """One-hot embedding of the corresponding word."""
    return torch.eye(len(corpus.vocab)) 

def embeddings_lexicon(corpus: TaggedCorpus, file: Path,
                       newvocab: Optional[Integerizer[Word]] = None) -> torch.Tensor:
    """Read pre-trained embeddings."""
    vocab = corpus.vocab

    with open(file) as f:
        filerows, cols = [int(i) for i in next(f).split()]   
        words = list(vocab)                      
        embeddings: List[Optional[Tensor]] = [None] * len(vocab)   
        specials = {'BOS': BOS_WORD, 'EOS': EOS_WORD, 'OOV': OOV_WORD}
        ool_vector = torch.zeros(cols)          

        for line in f:
            first, *rest = line.strip().split("\t")
            word = Word(first)
            vector = torch.tensor([float(v) for v in rest])
            
            if word == 'OOL':
                ool_vector = vector
            else:
                if word in specials: word = specials[word]    
                w = vocab.index(word)
                if w is None:
                    if newvocab is None or word in newvocab:
                        words.append(word)
                        embeddings.append(vector)     
                else:
                    embeddings[w] = vector        

    # Fill OOL
    ool_words = 0
    for w in range(len(vocab)):
        if embeddings[w] is None:
            embeddings[w] = ool_vector
            ool_words += 1

    # Maintain BOS/EOS at end
    slice = words[len(vocab)-2:len(vocab)]
    del words[len(vocab)-2:len(vocab)]
    words.extend(slice)
    
    slice = embeddings[len(vocab)-2:len(vocab)]
    del embeddings[len(vocab)-2:len(vocab)]
    embeddings.extend(slice)
    
    corpus.vocab = Integerizer(words)
    return torch.stack(embeddings)

def problex_lexicon(corpus: TaggedCorpus) -> torch.Tensor:
    """
    Return a V x (K+1) matrix. 
    Cols 0..K-1: log(p(t|w)). Col K: log(p(w)).
    Based on supervised training data counts + add-1 smoothing.
    """
    
    V = len(corpus.vocab)
    K = len(corpus.tagset)
    
    # 1. Count occurrences
    count_wt = torch.zeros(V, K)
    count_w = torch.zeros(V)
    
    for sent in corpus:
        for word, tag in sent:
            w_idx = corpus.vocab.index(word)
            # Words might appear in corpus but not be in corpus.vocab 
            # if corpus.vocab was restricted (unlikely here)
            if w_idx is not None:
                count_w[w_idx] += 1
                if tag is not None:
                    t_idx = corpus.tagset.index(tag)
                    if t_idx is not None:
                        count_wt[w_idx, t_idx] += 1
                        
    # 2. Compute probabilities
    # p(t|w)
    denom = count_w.unsqueeze(1) + K
    log_ptw = torch.log(count_wt + 1) - torch.log(denom)
    
    # p(w)
    total_tokens = count_w.sum()
    log_pw = torch.log(count_w + 1) - torch.log(total_tokens + V)
    
    # Concatenate
    return torch.cat([log_ptw, log_pw.unsqueeze(1)], dim=1)

def affixes_lexicon(corpus: TaggedCorpus,
                    newvocab: Optional[Integerizer[Word]] = None) -> torch.Tensor:
    """Return feature matrix for common suffixes/affixes.
    Also expands corpus.vocab to include words in newvocab."""
    
    # 1. Add newvocab words to corpus.vocab as requested in docstring
    if newvocab is not None:
        for word in newvocab:
            # Integerizer.add checks existence internally usually, or we check manually
            if word not in corpus.vocab:
                corpus.vocab.add(word)
    
    # We might need to move BOS/EOS to the end again if we added words,
    # but corpus.vocab usually appends. 
    # For safety with HMM index assumptions (BOS/EOS last):
    # We grab the strings, check if BOS/EOS are at end. If not, reorder.
    # However, typical Integers assume 0..N. If we append, we are safe IF HMM uses index()
    # BUT HMM `vocab[-2:]` check implies strict ordering.
    # If we add words, we disrupt the BOS/EOS at end property.
    
    # Let's reorder to ensure BOS/EOS remain at the end.
    # (Standard trick seen in embeddings_lexicon)
    if corpus.vocab[-2:] != [EOS_WORD, BOS_WORD] and len(corpus.vocab) > 2:
        # Extract list
        all_words = list(corpus.vocab)
        # Remove BOS/EOS where they are
        if BOS_WORD in all_words: all_words.remove(BOS_WORD)
        if EOS_WORD in all_words: all_words.remove(EOS_WORD)
        # Append them back
        all_words.extend([EOS_WORD, BOS_WORD])
        # Rebuild Index
        corpus.vocab = Integerizer(all_words)
        
    
    # 2. Define Features
    # Common English Suffixes (Morphological hints for POS)
    suffixes = ["ing", "ed", "ly", "s", "tion", "ity", "er", "est", "ous", "al"]
    # Prefixes
    prefixes = ["un", "in", "dis", "re", "pre"]
    
    num_affixes = len(suffixes) + len(prefixes)
    num_ortho = 3 # Cap, Digit, Hyphen
    total_dim = num_affixes + num_ortho
    
    # 3. Build Matrix
    V = len(corpus.vocab)
    features = torch.zeros(V, total_dim)
    
    for i, word_str in enumerate(corpus.vocab):
        col = 0
        
        # Orthography
        features[i, col] = 1.0 if word_str[0].isupper() else 0.0; col += 1
        features[i, col] = 1.0 if any(c.isdigit() for c in word_str) else 0.0; col += 1
        features[i, col] = 1.0 if '-' in word_str else 0.0; col += 1
        
        # Suffixes
        w_lower = word_str.lower()
        for s in suffixes:
            features[i, col] = 1.0 if w_lower.endswith(s) else 0.0
            col += 1
            
        # Prefixes
        for p in prefixes:
            features[i, col] = 1.0 if w_lower.startswith(p) else 0.0
            col += 1
            
    return features