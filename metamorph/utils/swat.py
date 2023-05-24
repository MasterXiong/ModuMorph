import numpy as np
import torch

from metamorph.config import cfg


def getChildrens(parents):
    childrens = []
    for cur_node_idx in range(len(parents)):
        childrens.append([])
        for node_idx in range(cur_node_idx, len(parents)):
            if cur_node_idx == parents[node_idx]:
                childrens[cur_node_idx].append(node_idx)
    return childrens


def lcrs(graph):
    new_graph = [[] for _ in graph]
    for node, children in enumerate(graph):
        if len(children) > 0:
            temp = children[0]
            new_graph[node].insert(0, temp)
            for sibling in children[1:]:
                new_graph[temp].append(sibling)
                temp = sibling
    return new_graph


def getTraversal(parents, traversal_types=cfg.MODEL.TRANSFORMER.TRAVERSALS):
    """Reconstruct tree and return a lists of node position in multiple traversals"""
    
    def postorder(children):
        trav = []
        def visit(node):
            for i in children[node]:
                visit(i)
            trav.append(node)
        visit(0)
        return trav

    def inorder(children):
        # assert binary tree
        trav = []
        def visit(node):
            if children[node]:
                visit(children[node][0])
            trav.append(node)
            if len(children[node]) == 2:
                visit(children[node][1])
        visit(0)
        return trav

    children = getChildrens(parents)
    traversals = []
    for ttype in traversal_types:
        if ttype == 'pre':
            indices = list(range(len(children)))
        else:
            if ttype == 'inlcrs':
                traversal = inorder(lcrs(children))
            elif ttype == 'postlcrs':
                traversal = postorder(lcrs(children))
            # indices = traversal
            indices = []
            for i in list(range(len(children))):
                indices.append(traversal.index(i))
        indices.extend([0 for _ in range(cfg.MODEL.MAX_LIMBS - len(indices))])
        traversals.append(indices)
    traversals = np.stack(traversals, axis=1)
    return traversals


def getAdjacency(parents):
    """Compute adjacency matrix of given graph"""
    N = len(parents)
    childrens = getChildrens(parents)
    adj = torch.zeros(N, N) # no self-loop
    for i, children in enumerate(childrens):
        for child in children:
            adj[i][child] = 1
            adj[child][i] = 1
    return adj  # (N, N)


def getGraphTransition(adjacency, self_loop=True):
    """Compute random walker transition in the given graph"""
    N = len(adjacency)
    if self_loop:
        adjacency = adjacency + torch.eye(N)
    degree = 1 / adjacency.sum(1).reshape(-1, 1) # for normalization
    transition = (adjacency * degree).T # (N, N)
    return transition


def PPR(transition, start=None, damping=0.9, max_iter=1000):
    """Compute Personalized PageRank vector"""
    N = transition.size(0)
    start = torch.ones(N, 1) / N \
            if start is None \
            else torch.eye(N)[start].reshape(N, 1)
    if damping == 1:
        prev_ppr = torch.ones(N, 1) / N
        for i in range(max_iter):
            ppr = damping * transition @ prev_ppr + (1 - damping) * start
            if ((ppr - prev_ppr).abs() < 1e-8).all():
                break
            prev_ppr = ppr
    else:
        inv = torch.inverse(torch.eye(N) - damping * transition)
        ppr = (1 - damping) * inv @ start
    return ppr  # (N, 1)


def getDistance(adjacency):
    def bfs(adjacency, root):
        dist = [-1] * adjacency.shape[0]
        dist[root] = 0
        Q = [(root, 0)]
        while len(Q):
            v, d = Q[0]; Q = Q[1:]
            for u, is_adj in enumerate(adjacency[v]):
                if is_adj and dist[u] == -1:
                    dist[u] = d + 1
                    Q.append((u, d+1))
        return dist

    return np.array([bfs(adjacency, i) for i in range(len(adjacency))]) / len(adjacency)


def getGraphDict(parents, trav_types=[], rel_types=[], self_loop=True, ppr_damping=0.9, device=None):

    if len(parents) == 1:
        return {'parents': parents}

    adjacency = getAdjacency(parents)
    transition = getGraphTransition(adjacency, self_loop)
    degree = adjacency.sum(1)
    laplacian = torch.diag(degree) - adjacency
    sym_lap = torch.diag(degree**-0.5) @ laplacian @ torch.diag(degree**-0.5)
    distance = torch.from_numpy(getDistance(adjacency)).float()
    graph_dict = {
        'parents': parents,
        'ppr': torch.cat([
            PPR(transition, i, ppr_damping)
            for i in range(len(parents))], dim=1).T,
        'transition': transition,
        'adjacency': adjacency,
        'distance': distance,
        'sym_lap': sym_lap,
    }
    # TODO: define relation (PPR, Laplacian, ...)
    graph_dict['relation'] = torch.stack(
                        [
                            graph_dict['ppr'],
                            graph_dict['sym_lap'],
                            graph_dict['distance']
                        ], dim=2)
    # graph_dict['relation'] = graph_dict['ppr'].unsqueeze(-1)
    # graph_dict['ppr'][i] represents PPR(i, j) for all j (N,)
    relational_encoding = np.zeros([cfg.MODEL.MAX_LIMBS, cfg.MODEL.MAX_LIMBS, graph_dict['relation'].size(-1)])
    limb_num = graph_dict['relation'].size(0)
    relational_encoding[:limb_num, :limb_num, :] = graph_dict['relation'].numpy()
    return relational_encoding