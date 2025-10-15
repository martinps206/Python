#!/usr/bin/env python3
# shor_pipeline.py
import argparse, json, random
from pathlib import Path
from shor_core import simulate_shor_once, gcd

def main():
    parser = argparse.ArgumentParser(description='Educational Shor simulation pipeline')
    parser.add_argument('--N', type=int, required=True)
    parser.add_argument('--shots', type=int, default=1024)
    parser.add_argument('--tries', type=int, default=6)
    parser.add_argument('--t', type=int, default=None)
    parser.add_argument('--seed', type=int, default=None)
    parser.add_argument('--outdir', type=str, default='shor_outputs')
    args = parser.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    results = []
    for attempt in range(args.tries):
        if args.seed is not None:
            seed = args.seed + attempt
        else:
            seed = None
        # elegir 'a' coprimo con N
        while True:
            a = random.randrange(2, args.N)
            if gcd(a, args.N) == 1:
                break
        res = simulate_shor_once(args.N, a=a, t=args.t, shots=args.shots, seed=seed)
        results.append(res)
        with open(outdir / f'result_N{args.N}_a{a}.json','w') as f:
            json.dump(res,f,indent=2)
    with open(outdir / f'summary_N{args.N}.json','w') as f:
        json.dump(results,f,indent=2)
    print(f'Saved {len(results)} attempt results to {outdir.resolve()}')

if __name__ == "__main__":
    main()
