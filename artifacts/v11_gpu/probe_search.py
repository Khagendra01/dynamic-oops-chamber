"""Instrument the reference prefix search: query trace, costs, phase structure."""
import json, sys
from collections import Counter
sys.path.insert(0, '/tmp/opencode/oops_v10/stage/dynamic_oops_chamber_v1_0')
from oops_prefix import PrefixOOPS, standard_tasks

queries = []          # (rules, task_name) in evaluation order
costs = []
phase_rows = []


def speculative(base, budget, vocab, max_depth):
    out = []

    def visit(prefix, allocation, depth, state):
        if allocation < 1:
            return 0.0
        state[0] += 1
        consumed = 0.0
        if prefix:
            out.append(prefix)
            consumed = min(1.0, allocation)
        else:
            consumed = 1.0
        if depth >= max_depth - len(base):
            return consumed
        remaining = max(0.0, allocation - consumed)
        for token, weight in vocab:
            if token in prefix:
                continue
            child = remaining * weight
            if child < 1:
                continue
            consumed += visit(prefix + (token,), child, depth + 1, state)
        return min(allocation, consumed)

    visit(base, float(budget), 0, [0])
    return out


def run():
    model = PrefixOOPS(reuse=True, max_depth=4, starting_budget=256,
                        max_phase_budget=1048576, max_prefixes_per_phase=30000)
    orig_lane = PrefixOOPS._run_lane
    orig_eval = PrefixOOPS._evaluate

    def run_lane(self, lane, budget, stage_tasks, prev):
        base = prev.rules if lane == 'extend' and prev is not None else ()
        vocab = self._vocabulary(lane)
        spec = speculative(base, budget, vocab, self.max_depth)
        res = orig_lane(self, lane, budget, stage_tasks, prev)
        phase_rows.append(dict(lane=lane, budget=budget, speculative=len(spec),
                               visited=res['visits'], tested=res['tested'],
                               spent=res['spent'], found=res['found'] is not None,
                               tasks=len(stage_tasks)))
        return res

    def evaluate(self, rules, task):
        result = orig_eval(self, rules, task)
        queries.append((list(rules), task.name))
        costs.append(result[1])
        return result

    PrefixOOPS._run_lane = run_lane
    PrefixOOPS._evaluate = evaluate

    tasks = standard_tasks()
    for task in tasks:
        r = model.train_task(task)
        print(f"{task.name:16} success={r.success} lane={r.selected_lane:8} "
              f"phases={len(r.phases)} visits={r.total_prefix_visits} "
              f"tested={r.prefixes_tested} replay_calls={r.replay_calls} "
              f"work={r.search_work_units}")

    print('\nphase rows:')
    for row in phase_rows:
        print('  ', row)
    print('\nTOTAL real (candidate,task) queries:', len(queries))
    print('unique candidates:', len({tuple(q[0]) for q in queries}))
    print('queries per task:', Counter(q[1] for q in queries))
    print('cost stats: min', min(costs), 'max', max(costs),
          'mean', round(sum(costs) / len(costs), 1))
    print('frozen:', {k: list(v.rules) for k, v in model.frozen.items()})
    print('speculative total:', sum(r['speculative'] for r in phase_rows))
    json.dump(dict(queries=queries, costs=costs, phases=phase_rows,
                   frozen={k: list(v.rules) for k, v in model.frozen.items()}),
              open('/tmp/opencode/oops_v10/reference_queries.json', 'w'), indent=1)


if __name__ == '__main__':
    run()
