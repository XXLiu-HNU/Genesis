# -*- coding: utf-8 -*-
"""
PRM/可见图 + Dijkstra 的轻量实现
- Roadmap.build(...)：用静态圆柱障碍构建路网（CPU）
- Roadmap.query(start_xy, goals_xy)：一次性规划到任一目标（直线快路径 + Dijkstra）
"""

from __future__ import annotations
import math
import heapq
from typing import List, Tuple, Optional, Dict

import numpy as np
from numba import njit

@njit(cache=True, fastmath=True)
def _los_hits_any_numba(p0, p1, centers, radii):
    # p0, p1: (2,), centers: (M,2), radii: (M,)
    v0 = p1[0] - p0[0]
    v1 = p1[1] - p0[1]
    vv = v0*v0 + v1*v1 + 1e-9
    M = centers.shape[0]
    for i in range(M):
        w0 = centers[i,0] - p0[0]
        w1 = centers[i,1] - p0[1]
        t = (w0*v0 + w1*v1) / vv
        if t < 0.0: t = 0.0
        elif t > 1.0: t = 1.0
        proj0 = p0[0] + t * v0
        proj1 = p0[1] + t * v1
        d0 = proj0 - centers[i,0]
        d1 = proj1 - centers[i,1]
        if d0*d0 + d1*d1 <= radii[i]*radii[i]:
            return True
    return False

# 可选加速：scikit-learn NearestNeighbors 或 scipy KDTree
_HAS_SKLEARN = False
_HAS_SCIPY = False
try:
    from sklearn.neighbors import NearestNeighbors  # type: ignore
    _HAS_SKLEARN = True
except Exception:
    pass

if not _HAS_SKLEARN:
    try:
        from scipy.spatial import cKDTree  # type: ignore
        _HAS_SCIPY = True
    except Exception:
        pass


# ----------------------------
# 几何与基础工具
# ----------------------------

def _segments_intersect_circles(p0: np.ndarray,
                                p1: np.ndarray,
                                centers: np.ndarray,
                                radii: np.ndarray) -> bool:
    """
    线段 p0-p1 是否与任一圆 (center, radius) 相交（含切）
    p0, p1: (2,)
    centers: (M,2)
    radii: (M,)
    """
    if centers.shape[0] == 0:
        return False
    v = p1 - p0                     # (2,)
    w = centers - p0                # (M,2)
    vv = float(v[0] * v[0] + v[1] * v[1]) + 1e-9
    t = np.clip((w @ v) / vv, 0.0, 1.0)   # (M,)
    proj = p0 + t[:, None] * v            # (M,2)
    d2 = np.sum((proj - centers) ** 2, axis=1)
    return bool(np.any(d2 <= (radii ** 2)))


def _line_of_sight_free(a_xy: np.ndarray,
                        b_xy: np.ndarray,
                        obs_xy: np.ndarray,
                        obs_r: np.ndarray,
                        clearance: float) -> bool:
    """两点之间是否无遮挡（把障碍半径加上 clearance），使用 numba 加速版本。"""
    if obs_xy.shape[0] == 0:
        return True
    p0 = a_xy.astype(np.float32)
    p1 = b_xy.astype(np.float32)
    inflated = (obs_r.astype(np.float32) + float(clearance)).astype(np.float32)
    return not _los_hits_any_numba(p0, p1, obs_xy.astype(np.float32), inflated)

# # 在 Roadmap 类里，新增一个更快的 LOS：
# def _line_of_sight_free(a_xy: np.ndarray,
#                         b_xy: np.ndarray,
#                         obs_xy: np.ndarray,
#                         obs_r: np.ndarray,
#                         clearance: float):
#     if obs_xy.shape[0] == 0:
#         return True
#     p0 = a_xy.astype(np.float32)
#     p1 = b_xy.astype(np.float32)
#     inflated = (obs_r.astype(np.float32) + clearance).astype(np.float32)
#     return not _los_hits_any_numba(p0, p1, obs_xy, inflated)



def _sample_free_xy_cpu(n: int,
                        world_min: Tuple[float, float],
                        world_max: Tuple[float, float],
                        obs_xy: np.ndarray,
                        obs_r: np.ndarray,
                        clearance: float,
                        max_tries_factor: int = 60) -> np.ndarray:
    """
    拒绝采样：在 [world_min, world_max] 内采 n 个不碰障碍(加 clearance)的点
    返回 (K,2)，K<=n（若空间很挤可能略少）
    """
    xmin, ymin = float(world_min[0]), float(world_min[1])
    xmax, ymax = float(world_max[0]), float(world_max[1])

    obs_c = obs_xy.astype(np.float32)
    obs_rr = (obs_r.astype(np.float32) + float(clearance)).astype(np.float32)

    rng = np.random.default_rng()
    out = []
    tries = 0
    max_tries = max(n * max_tries_factor, n * 10)
    while len(out) < n and tries < max_tries:
        tries += 1
        cand = rng.uniform([xmin, ymin], [xmax, ymax]).astype(np.float32)
        if obs_c.shape[0] == 0:
            out.append(cand)
            continue
        d2 = np.sum((obs_c - cand) ** 2, axis=1)
        if np.all(d2 >= (obs_rr ** 2)):
            out.append(cand)
    return np.asarray(out, dtype=np.float32)


def _knn_indices(points: np.ndarray, k: int) -> Tuple[np.ndarray, np.ndarray]:
    """
    返回 (dists, idxs)
    - 若可用 sklearn/scipy，使用其加速
    - 否则退化为 O(N^2) 暴力
    idxs[i,0] 会是 i 自身（若使用 sklearn NearestNeighbors）
    """
    n = points.shape[0]
    if n == 0:
        return (np.zeros((0, 0), dtype=np.float32), np.zeros((0, 0), dtype=np.int32))

    if _HAS_SKLEARN:
        nn = NearestNeighbors(n_neighbors=min(k, n), algorithm="auto")
        nn.fit(points)
        d, idx = nn.kneighbors(points)  # d, idx: (n, k)
        return d.astype(np.float32), idx.astype(np.int32)

    if _HAS_SCIPY:
        tree = cKDTree(points)
        d, idx = tree.query(points, k=min(k, n))
        # scipy 当 k=1 时返回 shape (n,)；兼容成 (n, k)
        if d.ndim == 1:
            d = d[:, None]
            idx = idx[:, None]
        return d.astype(np.float32), idx.astype(np.int32)

    # 暴力版（无依赖）
    # 距离矩阵
    diff = points[:, None, :] - points[None, :, :]  # (n,n,2)
    D = np.linalg.norm(diff, axis=2)                # (n,n)
    idx = np.argsort(D, axis=1)[:, :min(k, n)]
    rows = np.arange(n)[:, None]
    d = D[rows, idx]
    return d.astype(np.float32), idx.astype(np.int32)


def _visible_simplify(path_xy: np.ndarray,
                      obs_xy: np.ndarray,
                      obs_r: np.ndarray,
                      clearance: float) -> np.ndarray:
    """可见性线性化简：尽可能合并折线段"""
    if path_xy.shape[0] <= 2:
        return path_xy
    out = [path_xy[0]]
    i = 0
    while i < path_xy.shape[0] - 1:
        j = i + 1
        # 把 j 尽量往后拉
        while (j + 1 < path_xy.shape[0] and
               _line_of_sight_free(path_xy[i], path_xy[j + 1], obs_xy, obs_r, clearance)):
            j += 1
        out.append(path_xy[j])
        i = j
    return np.asarray(out, dtype=np.float32)


# ----------------------------
# 路网主体
# ----------------------------

class Roadmap:
    """
    轻量 PRM/可见图路网（静态圆障碍）
    - nodes: (N,2) float32
    - adj:   邻接表 List[List[Tuple(int node, float weight)]]
    """
    def __init__(self,
                 nodes: np.ndarray,
                 adj: List[List[Tuple[int, float]]],
                 clearance: float,
                 obs_xy: np.ndarray,
                 obs_r: np.ndarray,
                 world_min: Tuple[float, float],
                 world_max: Tuple[float, float]):
        self.nodes = nodes.astype(np.float32)
        self.adj = adj
        self.clearance = float(clearance)
        self.obs_xy = obs_xy.astype(np.float32)
        self.obs_r = obs_r.astype(np.float32)
        self.world_min = (float(world_min[0]), float(world_min[1]))
        self.world_max = (float(world_max[0]), float(world_max[1]))

        # 预建一个 KNN 索引，便于 query 时“接入路网”
        # 若没有依赖，我们在 query 里会临时走暴力
        self._knn_index = None
        try:
            if _HAS_SKLEARN:
                self._knn_index = NearestNeighbors(n_neighbors=min(32, len(self.nodes)))
                self._knn_index.fit(self.nodes)
            elif _HAS_SCIPY and len(self.nodes) > 0:
                self._knn_index = cKDTree(self.nodes)
        except Exception:
            self._knn_index = None

    # ---------- pickling (exclude indices; rebuild on load) ----------
    def __getstate__(self):
        state = self.__dict__.copy()
        # Exclude non-picklable/search index; rebuild after loading
        if "_knn_index" in state:
            state["_knn_index"] = None
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)
        # Rebuild neighbor index if possible
        self._knn_index = None
        try:
            if _HAS_SKLEARN:
                self._knn_index = NearestNeighbors(n_neighbors=min(32, len(self.nodes)))
                self._knn_index.fit(self.nodes)
            elif _HAS_SCIPY and len(self.nodes) > 0:
                self._knn_index = cKDTree(self.nodes)
        except Exception:
            self._knn_index = None

    # ---------- 构建 ----------
    @classmethod
    def build(cls,
              world_min: Tuple[float, float],
              world_max: Tuple[float, float],
              obs_xy: np.ndarray,
              obs_r: np.ndarray,
              n_nodes: int,
              k: int,
              max_edge_len: float,
              clearance: float) -> "Roadmap":
        """
        构建 PRM：采样路标点 → kNN → 过滤过长/遮挡 → 邻接表
        """
        nodes = _sample_free_xy_cpu(
            n=n_nodes,
            world_min=world_min,
            world_max=world_max,
            obs_xy=obs_xy,
            obs_r=obs_r,
            clearance=clearance,
        )
        if nodes.shape[0] == 0:
            return cls(nodes=np.zeros((0, 2), np.float32),
                       adj=[],
                       clearance=clearance,
                       obs_xy=obs_xy, obs_r=obs_r,
                       world_min=world_min, world_max=world_max)

        # kNN（含自身），后续会去掉自身边
        dists, idxs = _knn_indices(nodes, k=min(k + 1, len(nodes)))

        # 建无向图
        adj: List[List[Tuple[int, float]]] = [[] for _ in range(nodes.shape[0])]
        for u in range(nodes.shape[0]):
            for j in range(idxs.shape[1]):
                v = int(idxs[u, j])
                if v == u:
                    continue
                w = float(dists[u, j])
                if w <= 1e-6 or w > float(max_edge_len):
                    continue
                # 可见性检测
                if _line_of_sight_free(nodes[u], nodes[v], obs_xy, obs_r, clearance):
                    adj[u].append((v, w))
                    adj[v].append((u, w))

        return cls(nodes=nodes, adj=adj, clearance=clearance,
                   obs_xy=obs_xy, obs_r=obs_r,
                   world_min=world_min, world_max=world_max)

    # ---------- 查询 ----------
    def _attach_point(self, p_xy: np.ndarray, k_attach: int) -> List[Tuple[int, float]]:
        """
        将点 p_xy 接入路网：找若干最近的可见节点，返回 (node_idx, cost) 列表
        """
        if self.nodes.shape[0] == 0:
            return []

        # 找候选近邻
        max_k = min(max(8, k_attach * 2), len(self.nodes))
        if _HAS_SKLEARN and isinstance(self._knn_index, NearestNeighbors):
            d, idx = self._knn_index.kneighbors(p_xy[None, :], n_neighbors=max_k)
            d = d[0]; idx = idx[0]
        elif _HAS_SCIPY and self._knn_index is not None:
            d, idx = self._knn_index.query(p_xy[None, :], k=max_k)
            d = np.atleast_1d(d[0]); idx = np.atleast_1d(idx[0])
        else:
            # 暴力
            diff = self.nodes - p_xy[None, :]
            D = np.linalg.norm(diff, axis=1)
            idx = np.argsort(D)[:max_k]
            d = D[idx]

        picks: List[Tuple[int, float]] = []
        for j in range(idx.shape[0]):
            v = int(idx[j]); w = float(d[j])
            if _line_of_sight_free(p_xy, self.nodes[v], self.obs_xy, self.obs_r, self.clearance):
                picks.append((v, w))
                if len(picks) >= k_attach:
                    break
        return picks

    def _dijkstra_until_any(self,
                            start_xy: np.ndarray,
                            goals_xy: List[Tuple[float, float]],
                            k_attach: int) -> Optional[np.ndarray]:
        """
        Dijkstra：从 start 接入路网，扩展时一旦能直达任一目标即停止。
        返回路径 (K,2) 或 None
        """
        # 直线快路径
        for g in goals_xy:
            if _line_of_sight_free(start_xy, np.asarray(g, np.float32),
                                   self.obs_xy, self.obs_r, self.clearance):
                return np.asarray([start_xy, np.asarray(g, np.float32)], dtype=np.float32)

        # 接入
        start_links = self._attach_point(start_xy, k_attach=k_attach)
        if len(start_links) == 0:
            return None

        # 预先过滤“可接入”的目标（至少能连到一个路网点）
        goals_attached: List[Tuple[np.ndarray, List[Tuple[int, float]]]] = []
        for g in goals_xy:
            g_xy = np.asarray(g, np.float32)
            links = self._attach_point(g_xy, k_attach=k_attach)
            if len(links) > 0:
                goals_attached.append((g_xy, links))
        if len(goals_attached) == 0:
            return None

        N = self.nodes.shape[0]
        dist = np.full(N, np.inf, dtype=np.float32)
        parent = np.full(N, -1, dtype=np.int32)
        pq: List[Tuple[float, int]] = []

        for v, w in start_links:
            if w < dist[v]:
                dist[v] = w
                parent[v] = -1
                heapq.heappush(pq, (w, v))

        best_total = np.inf
        best_goal_xy = None
        best_last_node = None

        while pq:
            d_u, u = heapq.heappop(pq)
            if d_u > dist[u] + 1e-9:
                continue

            # 能否直接从 u 到某个目标（更快的终止）
            for g_xy, _ in goals_attached:
                if d_u >= best_total:
                    break
                if _line_of_sight_free(self.nodes[u], g_xy, self.obs_xy, self.obs_r, self.clearance):
                    total = d_u + float(np.linalg.norm(self.nodes[u] - g_xy))
                    if total < best_total:
                        best_total = total
                        best_goal_xy = g_xy
                        best_last_node = u

            # 常规扩展
            for v, w_uv in self.adj[u]:
                nd = d_u + w_uv
                if nd + 1e-6 < dist[v]:
                    dist[v] = nd
                    parent[v] = u
                    heapq.heappush(pq, (nd, v))

        if not np.isfinite(best_total) or best_goal_xy is None or best_last_node is None:
            return None

        # 回溯
        chain = []
        u = best_last_node
        while u != -1 and u >= 0:
            chain.append(self.nodes[u])
            u = parent[u]
        chain = chain[::-1]

        path = [start_xy] + chain + [best_goal_xy]
        path = np.asarray(path, dtype=np.float32)
        # 可见化简（极快）
        path = _visible_simplify(path, self.obs_xy, self.obs_r, self.clearance)
        return path

    # ---------- 外部接口 ----------
    def query(self,
              start_xy: Tuple[float, float],
              goals_xy: List[Tuple[float, float]],
              k_attach: int = 8) -> Optional[np.ndarray]:
        """
        输入：
          - start_xy: (x,y)
          - goals_xy: [(x,y), ...] 可多个，命中任一即可
        输出：
          - None 或 路径 ndarray [K,2]
        """
        if len(goals_xy) == 0:
            return None
        s = np.asarray(start_xy, dtype=np.float32)
        return self._dijkstra_until_any(s, goals_xy, k_attach=k_attach)
