# Proximal Policy Optimization (PPO)

基于 PyTorch 的 PPO 算法实现，支持**离散动作**和**连续动作**空间。使用截断（Clipping）方式约束策略更新幅度。

## 算法简介

PPO 是 OpenAI 提出的一种 policy gradient 算法，核心思想是通过截断重要性采样比率来限制策略更新幅度，避免策略崩溃。损失函数：

$$L^{CLIP}(\theta) = \mathbb{E}\left[\min\left(r_t(\theta)\hat{A}_t,\ \text{clip}(r_t(\theta), 1-\epsilon, 1+\epsilon)\hat{A}_t\right)\right]$$

其中 $r_t(\theta)$ 为新旧策略概率比，$\hat{A}_t$ 为 GAE 优势估计。

## 项目结构

```
.
├── PPO.py          # 主文件：网络定义 + PPO 算法 + 训练入口
├── rl_utils.py     # 工具函数：GAE、训练循环、移动平均、回放缓冲区
└── README.md
```

### PPO.py

| 类 | 说明 |
|---|---|
| `PolicyNet` | 离散策略网络，输出 softmax 概率分布 |
| `ValueNet` | 价值网络，输出状态价值 V(s) |
| `PPO` | 离散动作 PPO（Categorical 分布 + 截断） |
| `PolicyNetContinuous` | 连续策略网络，输出 Normal 分布的 μ 和 σ |
| `PPOContinuous` | 连续动作 PPO（Normal 分布 + 截断） |

### rl_utils.py

| 函数 | 说明 |
|---|---|
| `compute_advantage` | GAE（Generalized Advantage Estimation） |
| `train_on_policy_agent` | on-policy 训练循环（带 tqdm 进度条） |
| `train_off_policy_agent` | off-policy 训练循环 |
| `moving_average` | 滑动平均（保持原长度） |
| `ReplayBuffer` | 经验回放缓冲区 |

## 安装与运行

```bash
# 创建虚拟环境
python -m venv .venv
.venv\Scripts\activate     # Windows
# source .venv/bin/activate  # Linux/Mac

# 安装依赖
pip install torch gymnasium matplotlib tqdm

# 运行
python PPO.py
```

## 实验

### 离散动作 — CartPole-v0

保持平衡杆不倒，每步 reward=+1，最大 200 步。

| 超参数 | 值 |
|---|---|
| actor_lr | 1e-3 |
| critic_lr | 1e-2 |
| γ (gamma) | 0.98 |
| λ (GAE) | 0.95 |
| ε (clip) | 0.2 |
| epochs | 10 |
| episodes | 500 |

### 连续动作 — Pendulum-v1

控制摆杆到竖直位置，reward ∈ [-16.27, 0]。训练前对奖励做了缩放：`(reward + 8) / 8`。

| 超参数 | 值 |
|---|---|
| actor_lr | 1e-4 |
| critic_lr | 5e-3 |
| γ (gamma) | 0.9 |
| λ (GAE) | 0.9 |
| ε (clip) | 0.2 |
| epochs | 10 |
| episodes | 2000 |

## 依赖

- Python ≥ 3.9
- PyTorch ≥ 2.0
- Gymnasium ≥ 1.0
- NumPy
- Matplotlib
- tqdm

## 参考资料

- [Proximal Policy Optimization Algorithms](https://arxiv.org/abs/1707.06347) — Schulman et al., 2017
- [High-Dimensional Continuous Control Using Generalized Advantage Estimation](https://arxiv.org/abs/1506.02438) — Schulman et al., 2016
- https://hrl.boyuai.com/chapter/2/ppo%E7%AE%97%E6%B3%95
