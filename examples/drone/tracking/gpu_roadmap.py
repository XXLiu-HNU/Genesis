import math
from typing import List, Tuple, Optional

import torch


def sample_free_goals_gpu(
    world_min: Tuple[float, float],
    world_max: Tuple[float, float],
    obs_xy: torch.Tensor,   # (M,2) on device
    obs_r: torch.Tensor,    # (M,) on device
    clearance: float,
    n_goals: int,
    device: torch.device,
    max_tries: int = 10000,
) -> torch.Tensor:
    """Sample n_goals collision-free points on GPU. Returns (n_goals, 2)."""
    xmin, ymin = world_min
    xmax, ymax = world_max
    out = torch.empty((n_goals, 2), device=device, dtype=torch.float32)
    ok = torch.zeros((n_goals,), device=device, dtype=torch.bool)
    if obs_xy.numel() == 0:
        out[:, 0].uniform_(xmin, xmax)
        out[:, 1].uniform_(ymin, ymax)
        return out
    inflated = (obs_r + clearance).to(device=device, dtype=torch.float32)
    tries = 0
    batch_size = 512
    while (~ok).any() and tries < max_tries:
        tries += 1
        need_idx = (~ok).nonzero(as_tuple=False).squeeze(-1)
        M = need_idx.numel()
        if M == 0:
            break
        cand = torch.empty((min(M, batch_size), 2), device=device, dtype=torch.float32)
        cand[:, 0].uniform_(xmin, xmax)
        cand[:, 1].uniform_(ymin, ymax)
        # check collision (cand: (B,2), obs: (M,2))
        d = torch.linalg.norm(cand.unsqueeze(1) - obs_xy.unsqueeze(0), dim=-1)  # (B,M)
        free = (d >= inflated.unsqueeze(0)).all(dim=1)
        # assign
        take = min(free.sum().item(), M)
        if take > 0:
            out[need_idx[:take]] = cand[free][:take]
            ok[need_idx[:take]] = True
    if (~ok).any():
        # fallback: fill remaining with any candidate (will likely fail LOS but avoids crash)
        rem = (~ok).nonzero(as_tuple=False).squeeze(-1)
        out[rem, 0].uniform_(xmin, xmax)
        out[rem, 1].uniform_(ymin, ymax)
    return out


def _pair_los_free_edges(
    p_u: torch.Tensor,   # (E,2)
    p_v: torch.Tensor,   # (E,2)
    obs_xy: torch.Tensor,  # (M,2)
    obs_r: torch.Tensor,   # (M,)
    clearance: float,
    chunk_edges: int = 65536,
) -> torch.Tensor:
    """Check line-of-sight for many edges against circular obstacles, on device.
    Returns: (E,) bool tensor, True if LOS is free.
    """
    device = p_u.device
    M = obs_xy.shape[0]
    if M == 0:
        return torch.ones(p_u.shape[0], dtype=torch.bool, device=device)
    inflated = (obs_r + float(clearance)).to(device=device, dtype=p_u.dtype)  # (M,)
    out = torch.empty(p_u.shape[0], dtype=torch.bool, device=device)
    # chunk to control memory
    for s in range(0, p_u.shape[0], chunk_edges):
        e_u = p_u[s:s+chunk_edges]      # (e,2)
        e_v = p_v[s:s+chunk_edges]      # (e,2)
        v = e_v - e_u                   # (e,2)
        vv = torch.clamp((v * v).sum(-1), min=1e-9)  # (e,)
        # broadcast to (e,M)
        w = obs_xy.unsqueeze(0) - e_u.unsqueeze(1)    # (e,M,2)
        t = ((w * v.unsqueeze(1)).sum(-1) / vv.unsqueeze(1)).clamp(0.0, 1.0)  # (e,M)
        proj = e_u.unsqueeze(1) + t.unsqueeze(-1) * v.unsqueeze(1)            # (e,M,2)
        d = torch.linalg.norm(proj - obs_xy.unsqueeze(0), dim=-1)             # (e,M)
        blocked = d <= inflated.unsqueeze(0)
        out[s:s+chunk_edges] = ~blocked.any(dim=1)
    return out


def _knn_chunked(points: torch.Tensor, k: int, chunk_rows: int = 4096) -> Tuple[torch.Tensor, torch.Tensor]:
    """Compute kNN using chunked pairwise distances.
    points: (N,2) on device
    Returns (idx: (N,k) long, dist: (N,k) float)
    """
    device = points.device
    dtype = points.dtype
    N = points.shape[0]
    k = min(k, N)
    # initialize with +inf distances
    topk_dist = torch.full((N, k), float('inf'), device=device, dtype=dtype)
    topk_idx = torch.full((N, k), -1, device=device, dtype=torch.long)

    # precompute squared norms
    pts = points
    for s in range(0, N, chunk_rows):
        a = pts[s:s+chunk_rows]                    # (a,2)
        # compute distances to all points in sub-chunks to control mem further if needed
        # here we compute against all (N,2) directly (fits for 2D up to large N)
        diff = a.unsqueeze(1) - pts.unsqueeze(0)   # (a,N,2)
        d = torch.linalg.norm(diff, dim=-1)        # (a,N)
        # exclude self
        row_idx = torch.arange(s, min(s+chunk_rows, N), device=device)
        d[torch.arange(d.shape[0], device=device), row_idx] = float('inf')
        dist_k, idx_k = torch.topk(d, k=k, dim=1, largest=False)
        topk_dist[s:s+chunk_rows] = dist_k
        topk_idx[s:s+chunk_rows] = idx_k
    return topk_idx, topk_dist


class GPURoadmap:
    def __init__(self,
                 nodes_xy: torch.Tensor,   # (N,2) float32 on device
                 adj_list: List[List[Tuple[int, float]]],
                 clearance: float,
                 obs_xy: torch.Tensor,     # (M,2) on device
                 obs_r: torch.Tensor,      # (M,) on device
                 world_min: Tuple[float, float],
                 world_max: Tuple[float, float]):
        self.nodes = nodes_xy
        self.adj = adj_list
        self.clearance = float(clearance)
        self.obs_xy = obs_xy
        self.obs_r = obs_r
        self.world_min = (float(world_min[0]), float(world_min[1]))
        self.world_max = (float(world_max[0]), float(world_max[1]))
        # Build edge tensors for GPU SSSP (CSR-like simple edge list)
        self._build_edge_tensors()

    @classmethod
    def build(cls,
              world_min: Tuple[float, float],
              world_max: Tuple[float, float],
              obs_xy: torch.Tensor,     # (M,2), device
              obs_r: torch.Tensor,      # (M,), device
              n_nodes: int,
              k: int,
              max_edge_len: float,
              clearance: float,
              device: torch.device,
              sample_max_tries: int = 20000,
              ) -> "GPURoadmap":
        # Rejection sample nodes uniformly in bounds avoiding inflated obstacles
        xmin, ymin = world_min
        xmax, ymax = world_max
        nodes: List[Tuple[float, float]] = []
        tries = 0
        while len(nodes) < n_nodes and tries < sample_max_tries:
            tries += 1
            cand = torch.empty((1024, 2), device=device, dtype=torch.float32)
            cand[:, 0].uniform_(xmin, xmax)
            cand[:, 1].uniform_(ymin, ymax)
            if obs_xy.numel() > 0:
                # distances to all obstacles
                d = torch.linalg.norm(cand.unsqueeze(1) - obs_xy.unsqueeze(0), dim=-1)  # (B,M)
                infl = (obs_r + clearance).unsqueeze(0)
                free = (d >= infl).all(dim=1)
            else:
                free = torch.ones(cand.shape[0], dtype=torch.bool, device=device)
            picked = cand[free]
            # append up to needed
            need = n_nodes - len(nodes)
            if picked.shape[0] > 0:
                take = picked[:need].detach().to('cpu')
                nodes.extend([(float(x), float(y)) for x, y in take])
        if len(nodes) == 0:
            return cls(nodes_xy=torch.zeros((0, 2), device=device), adj_list=[], clearance=clearance,
                       obs_xy=obs_xy, obs_r=obs_r, world_min=world_min, world_max=world_max)
        nodes_xy = torch.tensor(nodes, device=device, dtype=torch.float32)

        # kNN on device
        idx_k, dist_k = _knn_chunked(nodes_xy, k=min(k+1, nodes_xy.shape[0]))  # include self

        # Build candidate edges (u->v) excluding self and long edges
        E_u: List[int] = []
        E_v: List[int] = []
        E_w: List[float] = []
        N = nodes_xy.shape[0]
        for u in range(N):
            for j in range(idx_k.shape[1]):
                v = int(idx_k[u, j].item())
                if v < 0 or v == u:
                    continue
                w = float(dist_k[u, j].item())
                if w <= 1e-6 or w > float(max_edge_len):
                    continue
                E_u.append(u); E_v.append(v); E_w.append(w)
        if len(E_u) == 0:
            return cls(nodes_xy=nodes_xy, adj_list=[[] for _ in range(N)], clearance=clearance,
                       obs_xy=obs_xy, obs_r=obs_r, world_min=world_min, world_max=world_max)

        e_u_xy = nodes_xy[torch.tensor(E_u, device=device, dtype=torch.long)]
        e_v_xy = nodes_xy[torch.tensor(E_v, device=device, dtype=torch.long)]
        los = _pair_los_free_edges(e_u_xy, e_v_xy, obs_xy, obs_r, clearance)

        adj: List[List[Tuple[int, float]]] = [[] for _ in range(N)]
        for i, ok in enumerate(los.tolist()):
            if ok:
                u = E_u[i]; v = E_v[i]; w = E_w[i]
                adj[u].append((v, w))
                adj[v].append((u, w))

        return cls(nodes_xy=nodes_xy, adj_list=adj, clearance=clearance,
                   obs_xy=obs_xy, obs_r=obs_r, world_min=world_min, world_max=world_max)

    def _build_edge_tensors(self) -> None:
        """Construct edge list tensors (src, dst, weight) on device for vectorized relaxations."""
        device = self.nodes.device
        src: List[int] = []
        dst: List[int] = []
        wts: List[float] = []
        for u, nbrs in enumerate(self.adj):
            for v, w in nbrs:
                src.append(u); dst.append(v); wts.append(w)
        if len(src) == 0:
            self.edge_src = torch.zeros((0,), device=device, dtype=torch.long)
            self.edge_dst = torch.zeros((0,), device=device, dtype=torch.long)
            self.edge_w = torch.zeros((0,), device=device, dtype=self.nodes.dtype)
        else:
            self.edge_src = torch.tensor(src, device=device, dtype=torch.long)
            self.edge_dst = torch.tensor(dst, device=device, dtype=torch.long)
            self.edge_w = torch.tensor(wts, device=device, dtype=self.nodes.dtype)

    def _attach_point(self, p_xy: Tuple[float, float], k_attach: int) -> List[Tuple[int, float]]:
        if self.nodes.shape[0] == 0:
            return []
        p = torch.tensor(p_xy, device=self.nodes.device, dtype=self.nodes.dtype)
        diff = self.nodes - p.unsqueeze(0)
        d = torch.linalg.norm(diff, dim=-1)
        val, idx = torch.topk(d, k=min(max(8, k_attach*2), d.shape[0]), largest=False)
        picks: List[Tuple[int, float]] = []
        # LOS check to nodes
        cand_nodes = idx.tolist()
        if len(cand_nodes) == 0:
            return []
        pu = p.unsqueeze(0).expand(len(cand_nodes), -1)
        pv = self.nodes[idx]
        los = _pair_los_free_edges(pu, pv, self.obs_xy, self.obs_r, self.clearance)
        for j, ok in enumerate(los.tolist()):
            if ok:
                v = int(idx[j].item())
                picks.append((v, float(val[j].item())))
                if len(picks) >= k_attach:
                    break
        return picks

    def _sssp_gpu(self,
                  start_links: List[Tuple[int, float]],
                  max_iters: Optional[int] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """Single-source shortest paths on GPU using frontier Bellman-Ford over edge list.
        start_links: list of (node, initial_cost), we seed distances with given costs.
        Returns (dist[N], parent[N]) tensors on device.
        """
        device = self.nodes.device
        N = self.nodes.shape[0]
        if max_iters is None:
            # upper bound; typically converges faster
            max_iters = max(1, len(start_links) + N)
        inf = float('inf')
        dist = torch.full((N,), inf, device=device, dtype=self.nodes.dtype)
        parent = torch.full((N,), -1, device=device, dtype=torch.long)
        frontier = torch.zeros((N,), device=device, dtype=torch.bool)
        for v, w in start_links:
            dist[v] = min(float(dist[v].item()), float(w))
            frontier[v] = True
            parent[v] = -1
        if self.edge_src.numel() == 0:
            return dist, parent

        for _ in range(max_iters):
            if not torch.any(frontier):
                break
            # Relax edges whose src is in frontier
            mask = frontier[self.edge_src]
            if not torch.any(mask):
                frontier[:] = False
                break
            cand = dist[self.edge_src] + self.edge_w
            cand = torch.where(mask, cand, torch.full_like(cand, inf))
            # reduce min per dst
            new_dist = dist.clone()
            # Using scatter_reduce_ amin to compute minimal candidate per dst
            tmp = torch.full((N,), inf, device=device, dtype=self.nodes.dtype)
            tmp.scatter_reduce_(0, self.edge_dst, cand, reduce="amin", include_self=True)
            improved = tmp < new_dist
            new_dist = torch.minimum(new_dist, tmp)

            # Update parents for improved nodes: find contributing src
            if torch.any(improved):
                # For improved destinations, find edges where cand equals tmp[dst]
                dst_vals = tmp[self.edge_dst]
                is_best = (cand == dst_vals) & mask
                # Pick one arbitrary contributing src per dst by first-true scatter
                # Initialize with existing parent
                new_parent = parent.clone()
                sel_src = torch.where(is_best, self.edge_src, torch.full_like(self.edge_src, -1))
                # For each edge where is_best True, write parent[dst]=src (last wins)
                write_mask = sel_src >= 0
                if torch.any(write_mask):
                    new_parent[self.edge_dst[write_mask]] = sel_src[write_mask]
                parent = new_parent

            # next frontier are nodes that improved
            frontier = improved
            dist = new_dist

        return dist, parent

    def query(self,
              start_xy: Tuple[float, float],
              goals_xy: List[Tuple[float, float]],
              k_attach: int = 8) -> Optional[List[Tuple[float, float]]]:
        if len(goals_xy) == 0 or self.nodes.shape[0] == 0:
            return None
        # Fast LOS to any goal on device
        s = torch.tensor(start_xy, device=self.nodes.device, dtype=self.nodes.dtype).unsqueeze(0)
        g = torch.tensor(goals_xy, device=self.nodes.device, dtype=self.nodes.dtype)
        los = _pair_los_free_edges(s.expand(g.shape[0], -1), g, self.obs_xy, self.obs_r, self.clearance)
        if torch.any(los):
            idx = int(torch.nonzero(los, as_tuple=False)[0].item())
            goal = goals_xy[idx]
            return [start_xy, (float(goal[0]), float(goal[1]))]

        # Attach start and all goals
        start_links = self._attach_point(start_xy, k_attach=k_attach)
        if len(start_links) == 0:
            return None
        goals_attached: List[Tuple[Tuple[float, float], List[Tuple[int, float]]]] = []
        for g_xy in goals_xy:
            links = self._attach_point(g_xy, k_attach=k_attach)
            if len(links) > 0:
                goals_attached.append((g_xy, links))
        if len(goals_attached) == 0:
            return None

        # GPU SSSP from start_links
        dist, parent = self._sssp_gpu(start_links)

        # Evaluate best goal reachable by LOS from any node
        best_total = math.inf
        best_goal_xy: Optional[Tuple[float, float]] = None
        best_last_node: Optional[int] = None
        # batch LOS from all nodes to goals (chunk to save memory)
        G = len(goals_attached)
        if G == 0:
            return None
        device = self.nodes.device
        goals_t = torch.tensor([g for (g, _) in goals_attached], device=device, dtype=self.nodes.dtype)  # (G,2)
        N = self.nodes.shape[0]
        chunk = 8192
        for s in range(0, N, chunk):
            u_nodes = self.nodes[s:s+chunk]  # (b,2)
            # LOS for pairs (b,G)
            ok = _pair_los_free_edges(u_nodes.repeat_interleave(G, 0), goals_t.repeat(u_nodes.shape[0], 1), self.obs_xy, self.obs_r, self.clearance)
            ok = ok.view(-1, G)
            if not torch.any(ok):
                continue
            # compute total cost
            d_to_u = dist[s:s+u_nodes.shape[0]].unsqueeze(1).expand(-1, G)
            cost_u_g = torch.linalg.norm(u_nodes.unsqueeze(1) - goals_t.unsqueeze(0), dim=-1)
            total = d_to_u + cost_u_g
            # invalidate non-LOS and inf distances
            total = torch.where(ok & torch.isfinite(d_to_u), total, torch.full_like(total, float('inf')))
            # find best
            min_val, min_idx = torch.min(total.view(-1), dim=0)
            if float(min_val.item()) < best_total:
                best_total = float(min_val.item())
                flat = int(min_idx.item())
                u_rel = flat // G
                g_rel = flat % G
                best_last_node = s + u_rel
                best_goal_xy = tuple(float(x) for x in goals_t[g_rel].tolist())

        if not math.isfinite(best_total) or best_goal_xy is None or best_last_node is None:
            return None

        # Reconstruct path from parent
        chain: List[Tuple[float, float]] = []
        u = best_last_node
        while u != -1 and u is not None:
            node = self.nodes[u]
            chain.append((float(node[0].item()), float(node[1].item())))
            pu = int(parent[u].item())
            if pu == -1:
                break
            u = pu
        chain = list(reversed(chain))
        path = [start_xy] + chain + [best_goal_xy]
        return path


