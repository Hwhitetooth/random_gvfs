import numpy as onp


def random_td_network(seed, depth, repeat, discount_factors, num_targets, num_actions):
    total_nodes = num_targets * (1 + len(discount_factors)) + depth * repeat * num_actions
    mat = onp.zeros((total_nodes, total_nodes))
    masks = onp.zeros((num_actions, total_nodes))
    dep = onp.zeros((total_nodes,), dtype=onp.int32)

    rng = onp.random.RandomState(seed=seed)
    idx = num_targets
    nodes_d = []
    for f in range(num_targets):
        nodes_d.append(f)
        for gamma in discount_factors:
            mat[idx, idx] = gamma
            mat[idx, f] = 1.
            masks[:, idx] = 1
            dep[idx] = 0
            nodes_d.append(idx)
            idx += 1
    for d in range(depth):
        nodes_dp1 = []
        for a in range(num_actions):
            # assert repeat <= len(nodes_d)
            _repeat = min(repeat, len(nodes_d))
            parents = rng.choice(nodes_d, size=_repeat, replace=False)
            for p in parents:
                mat[idx, p] = 1
                if p >= num_targets:
                    f = rng.choice(num_targets)
                    mat[idx, f] = 1
                masks[a, idx] = 1
                dep[idx] = d + 1
                nodes_dp1.append(idx)
                idx += 1
        nodes_d = nodes_dp1
    num_preds = total_nodes - num_targets
    masks = masks[:, num_targets:]
    dep = dep[num_targets:]
    return num_preds, mat, masks, dep
