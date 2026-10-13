"""Batch benchmark across seeded random chambers. No external dependencies."""
from __future__ import annotations
import argparse
import csv
import json
from pathlib import Path
from run_demo import run


def evaluate(episodes: int = 20, first_seed: int = 0, out: str = 'batch_results.csv') -> dict:
    if episodes < 1:
        raise ValueError('episodes must be positive')
    results = [run('random', seed, verbose=False)[0]
               for seed in range(first_seed, first_seed + episodes)]
    with Path(out).open('w', newline='', encoding='utf8') as f:
        writer = csv.DictWriter(f, fieldnames=list(results[0]))
        writer.writeheader()
        writer.writerows(results)
    aggregate = {
        'episodes': episodes,
        'first_seed': first_seed,
        'successes': sum(r['success'] for r in results),
        'collisions': sum(r['collision'] for r in results),
        'success_rate': sum(r['success'] for r in results) / episodes,
        'mean_steps': round(sum(r['steps'] for r in results) / episodes, 3),
        'mean_expansions': round(sum(r['total_search_expansions'] for r in results) / episodes, 3),
        'csv': out,
        'planner': 'oracle_time_expanded_a_star_not_oops',
    }
    return aggregate


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Reproducible batch of randomized chambers')
    parser.add_argument('--episodes', type=int, default=20)
    parser.add_argument('--first-seed', type=int, default=0)
    parser.add_argument('--out', type=str, default='batch_results.csv')
    args = parser.parse_args()
    print(json.dumps(evaluate(args.episodes, args.first_seed, args.out), indent=2))
