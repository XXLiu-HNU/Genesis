"""

A* 规划器
"""
import os
import math
import yaml
import random
import torch
import genesis as gs
from pid import PIDcontroller
from odom import Odom
from utils import setup_random_cylindrical_obstacles

# -------------------- 工具函数 --------------------
def torch_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")

def sample_free_point(xy_min, xy_max, obs_xy, obs_r, safety_radius, max_tries=1000, device="cpu"):
    """
    从矩形边界内随机采样一个与任意圆柱（投影为圆）不相交的点
    obs_xy: (n,2) tensor on device
    obs_r:  (n,)  tensor on device
    """
    for _ in range(max_tries):
        x = random.uniform(xy_min[0], xy_max[0])
        y = random.uniform(xy_min[1], xy_max[1])
        p = torch.tensor([x, y], device=device, dtype=obs_xy.dtype)
        if obs_xy.numel() == 0:
            return x, y
        d2 = torch.sum((obs_xy - p)**2, dim=1)
        if torch.all(torch.sqrt(d2) >= (obs_r + safety_radius)):
            return x, y
    raise RuntimeError("Failed to sample a collision-free point in given tries.")

def build_occupancy_grid(xy_min, xy_max, cell, obs_xy, obs_r, inflation):
    """
    构建占据栅格（更保守）：对每个方格，以“圆心到方格的最短距离<=r”作为占据条件。
    obs膨胀半径 = 障碍半径 + inflation（inflation = drone_radius + safety_margin）
    返回:
        grid: HxW 的 bool 张量（True=占据）
        xs, ys: 每个栅格中心坐标的一维张量
    """
    w = int(math.ceil((xy_max[0] - xy_min[0]) / cell))
    h = int(math.ceil((xy_max[1] - xy_min[1]) / cell))
    grid = torch.zeros((h, w), dtype=torch.bool)

    # 每个栅格的中心 & 半边长
    xs = xy_min[0] + (torch.arange(w) + 0.5) * cell   # (W,)
    ys = xy_min[1] + (torch.arange(h) + 0.5) * cell   # (H,)
    half = cell * 0.5

    if obs_xy.numel() == 0:
        return grid, xs, ys

    # 网格中心坐标场
    X, Y = torch.meshgrid(ys, xs, indexing='ij')      # (H,W)

    # 对每个障碍圆做覆盖判定（累积 or）
    for i in range(obs_xy.shape[0]):
        cx, cy = obs_xy[i, 0].item(), obs_xy[i, 1].item()
        r = (obs_r[i].item() + inflation)

        # 计算“圆心到方格”的最小距离（矩形最近点距离）
        dx = torch.clamp(torch.abs(X - cy) - half, min=0.0)  # 注意：X是y方向
        dy = torch.clamp(torch.abs(Y - cx) - half, min=0.0)  # 注意：Y是x方向
        # 纠正一下变量名别混：X是行方向坐标(对应y)，Y是列方向坐标(对应x)

        # 以上写法有点易混淆，换成更直观的：
        dx = torch.clamp(torch.abs(Y - cx) - half, min=0.0)  # 与x轴的剩余距离
        dy = torch.clamp(torch.abs(X - cy) - half, min=0.0)  # 与y轴的剩余距离

        dist = torch.sqrt(dx * dx + dy * dy)                 # 到方格的最短距离
        grid |= (dist <= r)                                  # 相交则置占据

    return grid, xs, ys


def world_to_grid(x, y, xy_min, cell, w, h):
    j = int((x - xy_min[0]) / cell)
    i = int((y - xy_min[1]) / cell)
    # clamp 到网格内
    j = max(0, min(w - 1, j))
    i = max(0, min(h - 1, i))
    return i, j

def grid_to_world(i, j, xy_min, cell):
    x = xy_min[0] + (j + 0.5) * cell
    y = xy_min[1] + (i + 0.5) * cell
    return x, y

def astar(grid, start_ij, goal_ij):
    """
    grid: HxW bool, True=障碍
    start_ij / goal_ij: (i, j)
    返回路径：[(i,j), ...] 或 None
    """
    import heapq
    H, W = grid.shape
    si, sj = start_ij
    gi, gj = goal_ij
    if grid[si, sj] or grid[gi, gj]:
        return None

    def h(i, j):  # 启发式：欧氏距离
        return math.hypot(i - gi, j - gj)

    open_heap = []
    heapq.heappush(open_heap, (h(si, sj), 0.0, (si, sj)))
    came = { (si, sj): None }
    gscore = { (si, sj): 0.0 }

    # 8邻域
    nbrs = [(-1,0),(1,0),(0,-1),(0,1),(-1,-1),(-1,1),(1,-1),(1,1)]

    while open_heap:
        f, gs, (ci, cj) = heapq.heappop(open_heap)
        if (ci, cj) == (gi, gj):
            # 回溯路径
            path = []
            cur = (ci, cj)
            while cur is not None:
                path.append(cur)
                cur = came[cur]
            path.reverse()
            return path

        for di, dj in nbrs:
            ni, nj = ci + di, cj + dj
            if not (0 <= ni < H and 0 <= nj < W):
                continue
            if grid[ni, nj]:
                continue
            step_cost = math.hypot(di, dj)
            tentative = gs + step_cost
            if (ni, nj) not in gscore or tentative < gscore[(ni, nj)]:
                gscore[(ni, nj)] = tentative
                came[(ni, nj)] = (ci, cj)
                heapq.heappush(open_heap, (tentative + h(ni, nj), tentative, (ni, nj)))
    return None

def line_collision_free(p, q, obs_xy, obs_r, inflation):
    """
    检查线段 p->q 是否与任一膨胀圆相交。
    p, q: (2,)
    """
    if obs_xy.numel() == 0:
        return True
    # 线段到圆心距离小于等于半径 => 碰撞
    pq = q - p
    pq2 = torch.dot(pq, pq).item()
    if pq2 == 0:
        d = torch.norm(obs_xy - p, dim=1)
        return torch.all(d > (obs_r + inflation))
    t = torch.clamp(torch.sum((obs_xy - p) * pq, dim=1) / pq2, 0.0, 1.0)  # (n,)
    proj = p + t.unsqueeze(1) * pq  # (n,2)
    d = torch.norm(obs_xy - proj, dim=1)  # (n,)
    return torch.all(d > (obs_r + inflation))

def shortcut_smooth(path_xy, obs_xy, obs_r, inflation, max_trials=200):
    """
    对路径做捷径平滑：随机挑两点，若直连不碰撞则删除中间段。
    path_xy: List[ (x,y) ]
    """
    if len(path_xy) <= 2:
        return path_xy
    pts = [torch.tensor(p) for p in path_xy]
    for _ in range(max_trials):
        if len(pts) <= 2:
            break
        i = random.randint(0, len(pts) - 2)
        j = random.randint(i + 1, len(pts) - 1)
        if j == i + 1:
            continue
        if line_collision_free(pts[i], pts[j], obs_xy, obs_r, inflation + 0.05):
            # 删除中间
            pts = pts[:i+1] + pts[j:]
    return [(p[0].item(), p[1].item()) for p in pts]



def polyline_arclen(points_xy):
    """返回每个点对应的累计弧长数组 L[i]（单位 m），L[0]=0"""
    if len(points_xy) == 0:
        return [0.0]
    L = [0.0]
    for i in range(1, len(points_xy)):
        dx = points_xy[i][0] - points_xy[i-1][0]
        dy = points_xy[i][1] - points_xy[i-1][1]
        L.append(L[-1] + math.hypot(dx, dy))
    return L

def interp_along_polyline(points_xy, L, s):
    """
    在折线（points_xy）上按弧长 s 取插值点坐标 (x,y)。
    L 为 polyline_arclen 结果。
    """
    if len(points_xy) == 0:
        return (0.0, 0.0)
    s = max(0.0, min(s, L[-1]))
    # 找到区间
    lo, hi = 0, len(L) - 1
    while lo + 1 < hi:
        mid = (lo + hi) // 2
        if L[mid] <= s:
            lo = mid
        else:
            hi = mid
    if L[hi] - L[lo] < 1e-9:
        return points_xy[hi]
    r = (s - L[lo]) / (L[hi] - L[lo])
    x = points_xy[lo][0] + r * (points_xy[hi][0] - points_xy[lo][0])
    y = points_xy[lo][1] + r * (points_xy[hi][1] - points_xy[lo][1])
    return (x, y)



class PathFollower:
    """
    沿规划好的 2D 折线路径以限速/限加速度的方式前进，
    每一步给出一个“近一点”的参考点，避免对控制器造成过大跳变。
    """
    def __init__(self,
                 path_xy,
                 dt,
                 v_max=0.8,
                 a_max=1.5,
                 warmup_time=2.0,
                 v_init=0.1,
                 lookahead_time=0.6,
                 min_lookahead=0.15,
                 slow_down_k=1.0):
        """
        - v_max: 轨迹最大速度（m/s），可调
        - a_max: 最大加速度（m/s^2），用于速度限幅
        - warmup_time: 从 v_init 升到 v_max 的时间（线性），用于起步平滑
        - v_init: 初始速度上限
        - lookahead_time: 给 PID 的前瞻时间（用于减少横向抖动）
        - min_lookahead: 最小前瞻距离（m）
        - slow_down_k: 终点减速系数，越大离终点越保守
        """
        self.dt = dt
        self.v_max_nominal = v_max
        self.a_max = a_max
        self.v_init = max(0.05, v_init)
        self.warmup_time = max(1e-6, warmup_time)
        self.lookahead_time = max(0.0, lookahead_time)
        self.min_lookahead = min_lookahead
        self.slow_down_k = max(0.5, slow_down_k)

        self.reset_with_path(path_xy)

    def reset_with_path(self, path_xy):
        self.path_xy = list(path_xy) if path_xy is not None else []
        self.L = polyline_arclen(self.path_xy)
        self.s_total = self.L[-1] if len(self.L) > 0 else 0.0
        self.s_ref = 0.0          # 当前参考点的弧长位置
        self.v = 0.0              # 当前跟随速度标量
        self.t = 0.0              # 累计时间（用于 warmup）
        # 若路径为空，保持静止
        if self.s_total <= 1e-6:
            self.path_xy = [(0.0, 0.0)]
            self.L = [0.0]
            self.s_total = 0.0
            self.s_ref = 0.0

    def _warmup_vmax(self):
        # 线性从 v_init -> v_max_nominal
        alpha = min(1.0, self.t / self.warmup_time)
        return self.v_init + alpha * (self.v_max_nominal - self.v_init)

    def step(self, cur_xy):
        """
        输入当前实际位置 cur_xy=(x,y)（用于微弱的自适应），
        输出下一个“期望位置” ref_xy=(x,y)，保证每步推进不大于 v*dt，
        并提前少量前瞻。
        """
        self.t += self.dt
        vmax_now = self._warmup_vmax()

        # 距离终点剩余距离
        s_remain = max(0.0, self.s_total - self.s_ref)
        # 终点减速：v_des 不能超过“刹车可停下来的速度”
        # v^2 <= 2 * a_max * s_remain / slow_down_k
        v_brake = math.sqrt(max(0.0, 2.0 * self.a_max * s_remain / self.slow_down_k)) if self.a_max > 1e-9 else vmax_now
        v_des = min(vmax_now, v_brake)

        # 对速度进行加速度限幅
        dv = v_des - self.v
        dv = max(-self.a_max * self.dt, min(self.a_max * self.dt, dv))
        self.v += dv

        # 本步弧长推进
        ds = self.v * self.dt
        self.s_ref = min(self.s_total, self.s_ref + ds)

        # 取前瞻：避免前瞻太远（仍受速度约束）
        lookahead = max(self.min_lookahead, self.v * self.lookahead_time)
        s_query = min(self.s_total, self.s_ref + lookahead)

        ref_xy = interp_along_polyline(self.path_xy, self.L, s_query)
        return ref_xy

    def reached_goal(self, thresh=0.12):
        return (self.s_total - self.s_ref) <= thresh
# -------------------- 环境 --------------------

class TrackerEnv:
    def __init__(self, num_envs, show_viewer=False):
        self.num_envs = num_envs
        self.rendered_env_num = min(10, self.num_envs)
        self.device = torch_device()

        # 创建仿真场景
        self.scene = gs.Scene(
            sim_options=gs.options.SimOptions(dt=self.dt, substeps=2),
            viewer_options=gs.options.ViewerOptions(
                max_FPS=100,
                camera_pos=(3.0, 0.0, 3.0),
                camera_lookat=(0.0, 0.0, 1.0),
                camera_fov=40,
            ),
            vis_options=gs.options.VisOptions(rendered_envs_idx=list(range(self.rendered_env_num))),
            rigid_options=gs.options.RigidOptions(
                dt=self.dt,
                constraint_solver=gs.constraint_solver.Newton,
                enable_collision=True,
                enable_joint_limit=True,
            ),
            show_viewer=show_viewer,
            profiling_options=gs.options.ProfilingOptions(show_FPS=False)
        )

        # 地面
        self.scene.add_entity(gs.morphs.Plane())

        # 无人机
        self.drone = self.scene.add_entity(gs.morphs.Drone(file="urdf/drones/drone_urdf/drone.urdf"))

        # 控制参数
        script_dir = os.path.dirname(os.path.abspath(__file__))
        with open(os.path.join(script_dir, "config/pos.yaml"), "r") as file:
            self.pos_ctrl_config = yaml.load(file, Loader=yaml.FullLoader)

        self.set_drone_imu()
        self.set_drone_controller()

        # 构建障碍物（你的函数返回：obs实体, obs_xy(list/np), obs_r(list/np)）
        obs, obs_xy, obs_r = setup_random_cylindrical_obstacles(self.scene, n_obstacles=100)

        # 转张量保存
        self.obs_xy = torch.tensor(obs_xy, dtype=torch.float32, device=self.device) if len(obs_xy) > 0 else torch.zeros((0,2), dtype=torch.float32, device=self.device)
        self.obs_r  = torch.tensor(obs_r , dtype=torch.float32, device=self.device) if len(obs_r ) > 0 else torch.zeros((0,),  dtype=torch.float32, device=self.device)


        # 可视化目标
        self.target = self.scene.add_entity(
            morph=gs.morphs.Mesh(
                file="meshes/sphere.obj",
                scale=0.05,
                fixed=True,
                collision=False,
            ),
            surface=gs.surfaces.Rough(
                diffuse_texture=gs.textures.ColorTexture(
                    color=(1.0, 0.5, 0.5),
                ),
            ),
        )
        # 构建场景
        self.scene.build(n_envs=self.num_envs)



        # 状态缓冲
        self.step_count = 0
        self.nav_step_in_task = 0
        self.current_wp_idx = 0
        self.path_wps = []   # [(x,y), ...]
        self.goal_xy = None
        self.start_xy = None

        # 初始化无人机位置（从可行点采样）
        start_xy = self.sample_free_xy()
        self.start_xy = start_xy
        self.drone_init_pos = torch.zeros((self.num_envs, 3), device=self.device)
        self.drone_init_pos[:, 0] = start_xy[0]
        self.drone_init_pos[:, 1] = start_xy[1]
        self.drone_init_pos[:, 2] = self.drone_height
        self.drone_init_quat = torch.tensor([1,0,0,0], device=self.device).repeat(self.num_envs, 1)

        init_target = torch.zeros((self.num_envs, 3), device=self.device)
        init_target[:, 0] = self.drone_init_pos[:, 0]     # 初始就放在无人机正上方/当前位置
        init_target[:, 1] = self.drone_init_pos[:, 1]
        init_target[:, 2] = self.drone_height
        self.target.set_pos(init_target)

        self.drone.set_pos(self.drone_init_pos)
        self.drone.set_quat(self.drone_init_quat)

        # 生成目标与路径
        self.plan_new_mission()

                # 基本参数
        self.dt = 0.01
        self.drone_height = 1.0
        self.world_xy_min = (-10.0, -10.0)   # 规划/采样边界 (可根据你的场景尺寸调整)
        self.world_xy_max = ( 10.0,  10.0)

        self.grid_cell = 0.05              # 栅格分辨率（越小越细，计算越慢）
        self.drone_radius = 0.12           # 无人机水平投影近似半径
        self.safety_margin = 0.12         # 安全余量（与障碍保持）
        self.wp_reach_thresh = 0.10        # 认为到达路点的距离阈值
        self.goal_reach_thresh = 0.12      # 认为到达终点的距离阈值
        self.max_nav_steps = 5000          # 一次任务的最大步数，避免卡死

        # 轨迹跟随器参数（你可随时改）
        self.v_max = 0.6        # 最大巡航速度（建议先小点）
        self.a_max = 1.2        # 最大加速度
        self.warmup_time = 2.5  # 速度上限从 v_init 线性升到 v_max 的时间
        self.v_init = 0.08      # 初始速度上限
        self.lookahead_time = 0.5
        self.min_lookahead = 0.12
        self.follower = PathFollower(
            self.path_wps, self.dt,
            v_max=self.v_max, a_max=self.a_max,
            warmup_time=self.warmup_time, v_init=self.v_init,
            lookahead_time=self.lookahead_time, min_lookahead=self.min_lookahead,
            slow_down_k=1.2
        )

    # ---------- 初始化 / 规划 ----------

    def sample_free_xy(self):
        return sample_free_point(
            self.world_xy_min, self.world_xy_max,
            self.obs_xy, self.obs_r,
            safety_radius=self.drone_radius + self.safety_margin,
            device=self.device
        )

    def plan_new_mission(self):
        """ 重新采样终点并基于当前无人机位置规划路径 """
        # 当前无人机 xy
        cur_pos = self.drone.get_pos()  # (N,3)
        ux = cur_pos[0,0].item()
        uy = cur_pos[0,1].item()
        self.start_xy = (ux, uy)

        # 采样终点
        self.goal_xy = self.sample_free_xy()

        # 可视化目标
        goal_pos = torch.zeros((self.num_envs, 3), device=self.device)
        goal_pos[:, 0] = self.goal_xy[0]
        goal_pos[:, 1] = self.goal_xy[1]
        goal_pos[:, 2] = self.drone_height     # 你希望显示在什么高度就放什么值
        self.target.set_pos(goal_pos)

        # 构建栅格
        inflation = self.drone_radius + self.safety_margin
        grid, xs, ys = build_occupancy_grid(
            self.world_xy_min, self.world_xy_max,
            self.grid_cell, self.obs_xy, self.obs_r, inflation
        )
        H, W = grid.shape
        si, sj = world_to_grid(ux, uy, self.world_xy_min, self.grid_cell, W, H)
        gi, gj = world_to_grid(self.goal_xy[0], self.goal_xy[1], self.world_xy_min, self.grid_cell, W, H)

        # A* 搜索
        path_ij = astar(grid, (si, sj), (gi, gj))
        if path_ij is None or len(path_ij) < 2:
            # 如果规划失败，扩大安全参数或重采样终点再试一次
            # 为简单起见，这里直接重采 10 次
            for _ in range(10):
                self.goal_xy = self.sample_free_xy()
                gi, gj = world_to_grid(self.goal_xy[0], self.goal_xy[1], self.world_xy_min, self.grid_cell, W, H)
                path_ij = astar(grid, (si, sj), (gi, gj))
                if path_ij is not None and len(path_ij) >= 2:
                    break
            if path_ij is None or len(path_ij) < 2:
                raise RuntimeError("Path planning failed: A* could not find a path.")

        # 网格 -> 世界坐标路径
        raw_path_xy = [grid_to_world(i, j, self.world_xy_min, self.grid_cell) for (i, j) in path_ij]

        # 平滑
        smoothed_xy = shortcut_smooth(raw_path_xy, self.obs_xy, self.obs_r, inflation)

        # 保存路点
        self.path_wps = smoothed_xy
        self.current_wp_idx = 0
        self.nav_step_in_task = 0

    # ---------- 控制/观测配置 ----------

    def set_drone_imu(self):
        odom = Odom(num_envs=self.num_envs, device=self.device)
        odom.set_drone(self.drone)
        setattr(self.drone, 'odom', odom)

    def set_drone_controller(self):
        pid = PIDcontroller(
            num_envs=self.num_envs,
            odom=self.drone.odom,
            config=self.pos_ctrl_config,
            device=self.device,
            controller="position",
        )
        pid.set_drone(self.drone)
        setattr(self.drone, 'controller', pid)

    # ---------- 主循环 ----------

    def step(self):
        self.step_count += 1

        # 当前无人机 xy（用于轻微自适应；即使不传也行）
        cur = self.drone.get_pos()[0]
        cur_xy = (cur[0].item(), cur[1].item())

        # 时间参数化推进
        ref_xy = self.follower.step(cur_xy)


        # 形成期望 3D 位置
        target = torch.zeros((self.num_envs, 4), device=self.device)
        target[:, 0] = ref_xy[0]
        target[:, 1] = ref_xy[1]
        target[:, 2] = self.drone_height
        target[:, 3] = 0.0

        # 发送给已有控制器
        rpms = self.drone.controller.step(target)
        self.drone.set_propellels_rpm(rpms)

        # 仿真步进
        self.scene.step()

        # 判断是否到达终点（按弧长）
        if self.follower.reached_goal(thresh=self.goal_reach_thresh):
            print(f"[Mission] Reached goal. Replan.")
            self.plan_new_mission()
            # 用新路径重置 follower，并且把起步速度上限重新 warmup
            self.follower.reset_with_path(self.path_wps)

        # 简单日志
        if self.step_count % 50 == 0:
            print(f"Step {self.step_count:05d} | Pos({cur[0]:.2f},{cur[1]:.2f},{cur[2]:.2f}) "
                  f"| s={self.follower.s_ref:.2f}/{self.follower.s_total:.2f} v={self.follower.v:.2f}")
            
# -------------------- 入口 --------------------

if __name__ == "__main__":
    gs.init()
    env = TrackerEnv(num_envs=1, show_viewer=True)
    while True:
        env.step()
