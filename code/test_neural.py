#!/usr/bin/env python3

import torch
from corpus import TaggedCorpus
from pathlib import Path
from lexicon import build_lexicon
from crf_neural import ConditionalRandomFieldNeural

# Load corpus
train_corpus = TaggedCorpus(Path('../data/nextsup'))
print(f'Vocab size: {len(train_corpus.vocab)}')
print(f'Tagset size: {len(train_corpus.tagset)}')

# Build lexicon
lexicon = build_lexicon(train_corpus, one_hot=True)
print(f'Lexicon shape: {lexicon.shape}')

# Create model
model = ConditionalRandomFieldNeural(
    tagset=train_corpus.tagset,
    vocab=train_corpus.vocab,
    lexicon=lexicon,
    rnn_dim=2,
    unigram=False
)

print(f'\nModel parameters:')
print(f'M: {model.M.shape}')
print(f'M_prime: {model.M_prime.shape}')
print(f'U_a: {model.U_a.shape}')
print(f'theta_a: {model.theta_a.shape}')
print(f'U_b: {model.U_b.shape}')
print(f'theta_b: {model.theta_b.shape}')

# Test on first sentence
sentence = list(train_corpus)[0]
print(f'\nTest sentence: {sentence}')

# Compute logprob
try:
    logprob = model.logprob(sentence, train_corpus)
    print(f'Log probability: {logprob.item():.4f}')
    
    if torch.isnan(logprob):
        print('ERROR: NaN detected!')
    else:
        print('SUCCESS: No NaN')
        
except Exception as e:
    print(f'ERROR: {e}')
    import traceback
    traceback.print_exc()
