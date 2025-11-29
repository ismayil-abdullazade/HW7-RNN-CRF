#!/usr/bin/env python3
"""
Extra Credit Experiments: Testing Performance Improvements
Tests various improvements on ensup10k (fast) and reports results.
"""

import subprocess
import time
import re
from pathlib import Path

# Experiment configurations
experiments = [
    {
        'name': '1. Baseline CRF (no RNN)',
        'cmd': 'python3 tag.py ../data/endev --train ../data/ensup10k --crf --model baseline_crf.pkl --max_steps 10000 --eval_interval 1000',
        'description': 'Simple stationary CRF without neural features'
    },
    {
        'name': '2. Neural CRF (biRNN, fixed embeddings)',
        'cmd': 'python3 tag.py ../data/endev --train ../data/ensup10k --crf --model neural_crf.pkl --lexicon ../lexicons/words-100.txt --rnn_dim 50 --problex --max_steps 10000 --eval_interval 1000',
        'description': 'BiRNN-CRF with fixed pretrained word embeddings'
    },
    {
        'name': '3. Awesome CRF (trainable embeddings + LayerNorm)',
        'cmd': 'python3 tag.py ../data/endev --train ../data/ensup10k --crf --model awesome_crf.pkl --lexicon ../lexicons/words-100.txt --rnn_dim 50 --problex --awesome --max_steps 10000 --eval_interval 1000',
        'description': 'Improvements: tunable embeddings, layer norm, efficient architecture'
    },
    {
        'name': '4. Awesome CRF with higher LR',
        'cmd': 'python3 tag.py ../data/endev --train ../data/ensup10k --crf --model awesome_crf_highlr.pkl --lexicon ../lexicons/words-100.txt --rnn_dim 50 --problex --awesome --lr 0.15 --max_steps 10000 --eval_interval 1000',
        'description': 'LayerNorm allows stable training with higher learning rate'
    },
    {
        'name': '5. Awesome CRF with affixes',
        'cmd': 'python3 tag.py ../data/endev --train ../data/ensup10k --crf --model awesome_crf_affixes.pkl --lexicon ../lexicons/words-100.txt --rnn_dim 50 --problex --affixes --awesome --max_steps 10000 --eval_interval 1000',
        'description': 'Add prefix/suffix features for better morphological generalization'
    },
]

def parse_output(output):
    """Extract accuracy and cross-entropy from output"""
    accuracy_match = re.search(r'Error rate:\s+([\d.]+)%.*accuracy:\s+([\d.]+)%', output, re.DOTALL)
    ce_match = re.search(r'Cross-entropy:\s+([\d.]+)', output)
    time_match = re.search(r'Total training time:\s+(\d+:\d+:\d+)', output)
    
    results = {}
    if accuracy_match:
        results['error_rate'] = float(accuracy_match.group(1))
        results['accuracy'] = float(accuracy_match.group(2))
    if ce_match:
        results['cross_entropy'] = float(ce_match.group(1))
    if time_match:
        results['train_time'] = time_match.group(1)
    
    return results

def run_experiment(exp):
    """Run a single experiment and return results"""
    print(f"\n{'='*80}")
    print(f"Running: {exp['name']}")
    print(f"Description: {exp['description']}")
    print(f"Command: {exp['cmd']}")
    print('='*80)
    
    start_time = time.time()
    try:
        result = subprocess.run(
            exp['cmd'],
            shell=True,
            cwd='code',
            capture_output=True,
            text=True,
            timeout=600  # 10 minute timeout
        )
        elapsed = time.time() - start_time
        
        output = result.stdout + result.stderr
        results = parse_output(output)
        results['wall_time'] = f"{elapsed:.1f}s"
        results['success'] = result.returncode == 0
        
        if results['success']:
            print(f"✓ SUCCESS")
            print(f"  Accuracy: {results.get('accuracy', 'N/A')}%")
            print(f"  Cross-entropy: {results.get('cross_entropy', 'N/A')}")
            print(f"  Training time: {results.get('train_time', 'N/A')}")
            print(f"  Wall time: {results['wall_time']}")
        else:
            print(f"✗ FAILED (return code {result.returncode})")
            print(f"Error output:\n{result.stderr[:500]}")
        
        return results
        
    except subprocess.TimeoutExpired:
        print(f"✗ TIMEOUT (> 10 minutes)")
        return {'success': False, 'error': 'timeout'}
    except Exception as e:
        print(f"✗ ERROR: {e}")
        return {'success': False, 'error': str(e)}

def main():
    print("="*80)
    print("EXTRA CREDIT EXPERIMENTS: Performance Improvements")
    print("="*80)
    print(f"Testing on ensup10k (10k training sentences)")
    print(f"Evaluating on endev")
    print()
    
    all_results = []
    
    for i, exp in enumerate(experiments, 1):
        results = run_experiment(exp)
        all_results.append({
            'name': exp['name'],
            'description': exp['description'],
            **results
        })
        
        # Brief pause between experiments
        if i < len(experiments):
            time.sleep(2)
    
    # Print summary table
    print("\n" + "="*80)
    print("RESULTS SUMMARY")
    print("="*80)
    print(f"{'Experiment':<45} {'Accuracy':<12} {'CrossEnt':<10} {'Time'}")
    print("-"*80)
    
    for r in all_results:
        if r.get('success'):
            acc = f"{r.get('accuracy', 0):.2f}%"
            ce = f"{r.get('cross_entropy', 0):.3f}"
            time_str = r.get('wall_time', 'N/A')
            print(f"{r['name']:<45} {acc:<12} {ce:<10} {time_str}")
        else:
            print(f"{r['name']:<45} {'FAILED':<12} {'-':<10} {'-'}")
    
    print("="*80)
    
    # Save detailed results
    report_path = Path('experiments_report.txt')
    with open(report_path, 'w') as f:
        f.write("EXTRA CREDIT EXPERIMENTS REPORT\n")
        f.write("="*80 + "\n\n")
        
        for r in all_results:
            f.write(f"Experiment: {r['name']}\n")
            f.write(f"Description: {r['description']}\n")
            if r.get('success'):
                f.write(f"  Accuracy: {r.get('accuracy', 'N/A')}%\n")
                f.write(f"  Error rate: {r.get('error_rate', 'N/A')}%\n")
                f.write(f"  Cross-entropy: {r.get('cross_entropy', 'N/A')}\n")
                f.write(f"  Training time: {r.get('train_time', 'N/A')}\n")
                f.write(f"  Wall time: {r.get('wall_time', 'N/A')}\n")
            else:
                f.write(f"  Status: FAILED ({r.get('error', 'unknown error')})\n")
            f.write("\n" + "-"*80 + "\n\n")
    
    print(f"\nDetailed report saved to: {report_path.absolute()}")

if __name__ == '__main__':
    main()
