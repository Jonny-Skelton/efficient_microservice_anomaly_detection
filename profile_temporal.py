"""profile_temporal.py — complexity profiling for TAM vs Mamba."""
import argparse, json, subprocess, sys, time, traceback
from pathlib import Path


def measure_one(module_name, window, num_services, batch_size,
                node_dim, edge_dim, log_dim, warmup_iters, measure_iters):
    import torch
    device = torch.device("cuda")
    torch.backends.cudnn.benchmark = False

    num_edges = num_services * 2
    trace2pod = torch.ones(num_edges, num_services, device=device) / num_services

    if module_name == "tam":
        from src.model_util import Temporal_Attention
        module = Temporal_Attention(
            node_dim, edge_dim, log_dim, trace2pod,
            heads_node=4, heads_edge=4, heads_log=4,
            dropout=0.1, window_size=window, batch_size=batch_size).to(device)
    elif module_name == "mamba":
        from src.MESTGAD_util import MambaTemporalModule
        module = MambaTemporalModule(
            node_dim, edge_dim, log_dim, trace2pod,
            d_state=16, d_conv=4, expand=2,
            dropout=0.1, window_size=window, batch_size=batch_size).to(device)
    else:
        raise ValueError(f"unknown module: {module_name}")

    def make_inputs():
        xn = torch.randn(batch_size, window, num_services, node_dim,
                         device=device, requires_grad=True)
        xt = torch.randn(batch_size, window, num_edges, edge_dim,
                         device=device, requires_grad=True)
        xl = torch.randn(batch_size, window, num_services, log_dim,
                         device=device, requires_grad=True)
        return xn, xt, xl

    def run_step():
        xn, xt, xl = make_inputs()
        out = module(xn, xt, xl)
        loss = sum(o.sum() for o in out) if isinstance(out, (tuple, list)) else out.sum()
        loss.backward()

    for _ in range(warmup_iters):
        run_step()
    torch.cuda.synchronize()

    torch.cuda.reset_peak_memory_stats()
    times = []
    for _ in range(measure_iters):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        run_step()
        torch.cuda.synchronize()
        times.append(time.perf_counter() - t0)
    times.sort()
    return {"module": module_name, "window": window, "status": "ok",
            "num_services": num_services, "num_edges": num_edges,
            "batch_size": batch_size, "measure_iters": measure_iters,
            "time_mean_s": sum(times) / len(times),
            "time_median_s": times[len(times) // 2],
            "time_min_s": times[0], "time_max_s": times[-1],
            "peak_mem_mb": torch.cuda.max_memory_allocated() / (1024**2)}


def run_worker(a):
    try:
        r = measure_one(a.module, a.window, a.num_services, a.batch_size,
                        a.node_dim, a.edge_dim, a.log_dim,
                        a.warmup_iters, a.measure_iters)
    except Exception as e:
        msg = str(e).lower()
        status = "oom" if "out of memory" in msg else "error"
        r = {"module": a.module, "window": a.window, "status": status,
             "error": str(e), "traceback": traceback.format_exc()}
    print("RESULT_JSON:" + json.dumps(r))


def run_driver(a):
    Path(a.output).parent.mkdir(parents=True, exist_ok=True)
    results = []
    for m in a.modules:
        for w in a.windows:
            print(f"\n=== {m} W={w} ===", flush=True)
            cmd = [sys.executable, __file__, "--worker",
                   "--module", m, "--window", str(w),
                   "--num-services", str(a.num_services),
                   "--batch-size", str(a.batch_size),
                   "--node-dim", str(a.node_dim),
                   "--edge-dim", str(a.edge_dim),
                   "--log-dim", str(a.log_dim),
                   "--warmup-iters", str(a.warmup_iters),
                   "--measure-iters", str(a.measure_iters)]
            p = subprocess.run(cmd, capture_output=True, text=True)
            r = None
            for line in p.stdout.splitlines():
                if line.startswith("RESULT_JSON:"):
                    r = json.loads(line[len("RESULT_JSON:"):])
                    break
            if r is None:
                r = {"module": m, "window": w, "status": "error",
                     "error": "no RESULT_JSON",
                     "stdout_tail": p.stdout[-1500:],
                     "stderr_tail": p.stderr[-1500:]}
            print(json.dumps(r, indent=2), flush=True)
            results.append(r)
            with open(a.output, "w") as f:
                json.dump(results, f, indent=2)
    print(f"\nWrote {len(results)} results to {a.output}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--worker", action="store_true")
    p.add_argument("--modules", nargs="+", default=["tam", "mamba"])
    p.add_argument("--module", type=str)
    p.add_argument("--windows", nargs="+", type=int,
                   default=[10, 20, 40, 80, 160, 320, 640, 1280])
    p.add_argument("--window", type=int)
    p.add_argument("--num-services", type=int, default=5)
    p.add_argument("--batch-size", type=int, default=50)
    p.add_argument("--node-dim", type=int, default=64)
    p.add_argument("--edge-dim", type=int, default=64)
    p.add_argument("--log-dim", type=int, default=64)
    p.add_argument("--warmup-iters", type=int, default=10)
    p.add_argument("--measure-iters", type=int, default=50)
    p.add_argument("--fresh-subprocess", action="store_true")
    p.add_argument("--output", type=str,
                   default="results/profile/temporal_complexity.json")
    a = p.parse_args()
    (run_worker if a.worker else run_driver)(a)


if __name__ == "__main__":
    main()