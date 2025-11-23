#!/usr/bin/env python3

import torch
from corpus import TaggedCorpus
from pathlib import Path
from lexicon import build_lexicon
from crf_neural import ConditionalRandomFieldNeural

# Load corpus
train_corpus = TaggedCorpus(Path('../data/nextsup'))
print(f'Corpus: {len(train_corpus)} sentences')

# Build lexicon
lexicon = build_lexicon(train_corpus, one_hot=True)

# Create model
model = ConditionalRandomFieldNeural(
    tagset=train_corpus.tagset,
    vocab=train_corpus.vocab,
    lexicon=lexicon,
    rnn_dim=2,
    unigram=False
)

# Initialize optimizer
model.init_optimizer(lr=0.01, weight_decay=0.0)

# Get first few sentences
sentences = list(train_corpus)[:5]

print("\n=== Testing forward pass on first 5 sentences ===")
for i, sent in enumerate(sentences):
    try:
        logprob = model.logprob(sent, train_corpus)
        print(f"Sentence {i}: logprob = {logprob.item():.4f}")
        if torch.isnan(logprob):
            print(f"  ERROR: NaN detected!")
    except Exception as e:
        print(f"Sentence {i}: ERROR - {e}")

print("\n=== Testing one gradient step ===")
model._zero_grad()

# Accumulate gradient from first sentence
sent = sentences[0]
logprob = model.logprob(sent, train_corpus)
print(f"Initial logprob: {logprob.item():.4f}")

loss = -logprob
loss.backward()

# Check gradients
print("\nGradient norms:")
for name, param in model.named_parameters():
    if param.grad is not None:
        grad_norm = param.grad.norm().item()
        has_nan = torch.isnan(param.grad).any().item()
        print(f"  {name}: norm={grad_norm:.4f}, has_nan={has_nan}")

# Take step
torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
model.optimizer.step()

# Check parameters after update
print("\nParameter values after update:")
for name, param in model.named_parameters():
    has_nan = torch.isnan(param).any().item()
    param_norm = param.norm().item()
    print(f"  {name}: norm={param_norm:.4f}, has_nan={has_nan}")

# Test forward pass again
print("\n=== Testing forward pass after one update ===")
try:
    logprob2 = model.logprob(sent, train_corpus)
    print(f"Logprob after update: {logprob2.item():.4f}")
    if torch.isnan(logprob2):
        print("ERROR: NaN after update!")
except Exception as e:
    print(f"ERROR: {e}")
