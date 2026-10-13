import sys, time
sys.path.insert(0, '.')
from oops_prefix import PrefixOOPS, standard_tasks
for depth in (4, 5, 6):
    m = PrefixOOPS(reuse=True, max_depth=depth, starting_budget=256,
                   max_phase_budget=1048576, max_prefixes_per_phase=30000)
    t0 = time.perf_counter()
    ok = True
    for task in standard_tasks():
        r = m.train_task(task)
        ok = ok and r.success
    dt = time.perf_counter() - t0
    print(f"depth={depth} {dt:.3f}s stages={sum(x.success for x in m.history)} "
          f"ok={ok} replay_calls={m.real_replay_calls} steps={m.simulated_steps} "
          f"tested={sum(x.prefixes_tested for x in m.history)}", flush=True)
