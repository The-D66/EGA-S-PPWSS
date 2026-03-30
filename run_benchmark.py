#!/usr/bin/env python3
"""
Benchmark script for EGA-S paper.
Runs EGA-S, standard GA, PSO, DE, and MILP across 4 conditions (C1-C4)
under both COST and ENERGY modes.

Usage:
    python run_benchmark.py
    python run_benchmark.py --seeds 0 1 2 --pop_size 500 --max_iter 50
"""
import os
import sys
import json
import time
import argparse
import numpy as np

# Ensure project root is in path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from pump_opt.problem import Problem
from pump_opt.optimization.panning_GA import Panning_GA

# Paper conditions: C1-C4 representing 25%, 40%, 60%, 80% of system capacity
CONDITIONS = {
    "C1": {"aim_vol": 1320000,  "label": "25% capacity"},
    "C2": {"aim_vol": 2112000,  "label": "40% capacity"},
    "C3": {"aim_vol": 3168000,  "label": "60% capacity"},
    "C4": {"aim_vol": 4224000,  "label": "80% capacity"},
}

METHODS = {
    "eco": "COST minimization",
    "eff": "ENERGY minimization",
}

AREA_CONFIG = "sA-sB"


def load_area_config(aim_vol, method):
    """Load and modify area configuration for given condition."""
    config_path = f"./data/area/{AREA_CONFIG}.json"
    with open(config_path, 'r') as f:
        config = json.load(f)
    config["aim_vol"] = aim_vol
    config["method"] = method
    return json.dumps(config)


def run_ega_s(config_json, pop_size, max_iter, seed=None):
    """Run EGA-S optimization."""
    if seed is not None:
        np.random.seed(seed)

    pro = Problem(config_json)
    s_time = 24
    lb = [0] * s_time
    ub = [65] * s_time

    ga = Panning_GA(
        pro.aim_func,
        s_time,
        lb=lb, ub=ub,
        size_pop=pop_size,
        max_iter=max_iter,
        prob_mut=0.05,
        precision=1,
        early_stop=10,
        n_len=[s_time]
    )

    # Heuristic initialization
    init_flow = pro.model.make_init_pop()
    ga.Chrom = ga.x2chrom(init_flow)

    t0 = time.time()
    pop, fund = ga.run()
    elapsed = time.time() - t0

    # Evaluate final solution
    mo = pro.create_model(pop, print_flag=False)

    return {
        "algorithm": "EGA-S",
        "score": float(mo.score),
        "cost": float(mo.fund) * 0.36 if mo.method in ("eco", "eff") and mo.fund < 1e6 else float(mo.fund),
        "time_s": round(elapsed, 2),
        "flow": list(mo.trans_result(pop)),
    }


def run_standard_ga(config_json, pop_size, max_iter, seed=None):
    """Run standard GA (scikit-opt) for comparison."""
    try:
        from sko.GA import GA
    except ImportError:
        return {"algorithm": "GA", "error": "scikit-opt not installed"}

    if seed is not None:
        np.random.seed(seed)

    pro = Problem(config_json)
    s_time = 24
    lb = [0] * s_time
    ub = [65] * s_time

    ga = GA(
        func=pro.aim_func,
        n_dim=s_time,
        lb=lb, ub=ub,
        size_pop=pop_size,
        max_iter=max_iter,
        prob_mut=0.05,
        precision=1,
    )

    t0 = time.time()
    pop, fund = ga.run()
    elapsed = time.time() - t0

    mo = pro.create_model(pop, print_flag=False)

    return {
        "algorithm": "GA",
        "score": float(mo.score),
        "cost": float(mo.fund) * 0.36 if mo.method in ("eco", "eff") and mo.fund < 1e6 else float(mo.fund),
        "time_s": round(elapsed, 2),
    }


def run_pso(config_json, pop_size, max_iter, seed=None):
    """Run PSO (scikit-opt) for comparison."""
    try:
        from sko.PSO import PSO
    except ImportError:
        return {"algorithm": "PSO", "error": "scikit-opt not installed"}

    if seed is not None:
        np.random.seed(seed)

    pro = Problem(config_json)
    s_time = 24
    lb = [0] * s_time
    ub = [65] * s_time

    pso = PSO(
        func=pro.aim_func,
        n_dim=s_time,
        lb=lb, ub=ub,
        pop=pop_size,
        max_iter=max_iter,
        w=0.6, c1=1.5, c2=1.5,
    )

    t0 = time.time()
    pop, fund = pso.run()
    elapsed = time.time() - t0

    mo = pro.create_model(pop, print_flag=False)

    return {
        "algorithm": "PSO",
        "score": float(mo.score),
        "cost": float(mo.fund) * 0.36 if mo.method in ("eco", "eff") and mo.fund < 1e6 else float(mo.fund),
        "time_s": round(elapsed, 2),
    }


def run_de(config_json, pop_size, max_iter, seed=None):
    """Run DE (scikit-opt) for comparison."""
    try:
        from sko.DE import DE
    except ImportError:
        return {"algorithm": "DE", "error": "scikit-opt not installed"}

    if seed is not None:
        np.random.seed(seed)

    pro = Problem(config_json)
    s_time = 24
    lb = [0] * s_time
    ub = [65] * s_time

    de = DE(
        func=pro.aim_func,
        n_dim=s_time,
        lb=lb, ub=ub,
        size_pop=pop_size,
        max_iter=max_iter,
        F=0.5, CR=0.9,
    )

    t0 = time.time()
    pop, fund = de.run()
    elapsed = time.time() - t0

    mo = pro.create_model(pop, print_flag=False)

    return {
        "algorithm": "DE",
        "score": float(mo.score),
        "cost": float(mo.fund) * 0.36 if mo.method in ("eco", "eff") and mo.fund < 1e6 else float(mo.fund),
        "time_s": round(elapsed, 2),
    }


def main():
    parser = argparse.ArgumentParser(description="EGA-S Benchmark Runner")
    parser.add_argument("--seeds", nargs="+", type=int, default=[0],
                        help="Random seeds for reproducibility")
    parser.add_argument("--pop_size", type=int, default=500,
                        help="Population size (default 500 for quick test)")
    parser.add_argument("--max_iter", type=int, default=50,
                        help="Max iterations (default 50 for quick test)")
    parser.add_argument("--algorithms", nargs="+", default=["EGA-S", "GA", "PSO", "DE"],
                        help="Algorithms to benchmark")
    parser.add_argument("--conditions", nargs="+", default=["C2"],
                        help="Conditions to test (C1-C4, or 'all')")
    parser.add_argument("--output", type=str, default="./result/benchmark.json",
                        help="Output file path")
    args = parser.parse_args()

    if "all" in args.conditions:
        args.conditions = list(CONDITIONS.keys())

    os.makedirs("./result", exist_ok=True)
    os.makedirs("./logs", exist_ok=True)

    results = []
    algo_runners = {
        "EGA-S": run_ega_s,
        "GA": run_standard_ga,
        "PSO": run_pso,
        "DE": run_de,
    }

    total = len(args.conditions) * len(METHODS) * len(args.algorithms) * len(args.seeds)
    current = 0

    print("=" * 60)
    print("EGA-S Benchmark")
    print(f"Conditions: {args.conditions}")
    print(f"Algorithms: {args.algorithms}")
    print(f"Seeds: {args.seeds}")
    print(f"Pop size: {args.pop_size}, Max iter: {args.max_iter}")
    print(f"Total runs: {total}")
    print("=" * 60)

    for cond_name in args.conditions:
        cond = CONDITIONS[cond_name]
        for method, method_label in METHODS.items():
            config_json = load_area_config(cond["aim_vol"], method)

            for algo_name in args.algorithms:
                runner = algo_runners.get(algo_name)
                if runner is None:
                    print(f"  Unknown algorithm: {algo_name}")
                    continue

                for seed in args.seeds:
                    current += 1
                    print(f"[{current}/{total}] {cond_name} {method_label} {algo_name} seed={seed}", end=" ... ", flush=True)

                    try:
                        result = runner(config_json, args.pop_size, args.max_iter, seed)
                        result.update({
                            "condition": cond_name,
                            "method": method,
                            "seed": seed,
                            "aim_vol": cond["aim_vol"],
                        })
                        results.append(result)
                        score = result.get("score", "N/A")
                        cost = result.get("cost", "N/A")
                        t = result.get("time_s", "N/A")
                        print(f"score={score:.2f}, cost={cost:.2f}, time={t}s")
                    except Exception as e:
                        print(f"FAILED: {e}")
                        results.append({
                            "algorithm": algo_name,
                            "condition": cond_name,
                            "method": method,
                            "seed": seed,
                            "error": str(e),
                        })

    # Save results
    with open(args.output, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {args.output}")

    # Summary table
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    for cond_name in args.conditions:
        for method in METHODS:
            print(f"\n--- {cond_name} ({METHODS[method]}) ---")
            for algo in args.algorithms:
                algo_results = [r for r in results
                                if r.get("condition") == cond_name
                                and r.get("method") == method
                                and r.get("algorithm") == algo
                                and "error" not in r]
                if algo_results:
                    scores = [r["score"] for r in algo_results]
                    costs = [r["cost"] for r in algo_results]
                    times = [r["time_s"] for r in algo_results]
                    print(f"  {algo:8s}: score={np.mean(scores):.2f}±{np.std(scores):.2f}, "
                          f"cost={np.mean(costs):.2f}, time={np.mean(times):.1f}s")
                else:
                    print(f"  {algo:8s}: No valid results")


if __name__ == "__main__":
    main()
