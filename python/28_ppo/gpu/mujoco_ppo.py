"""PPO on a continuous-control benchmark, which is what the paper actually tested.

CartPole has discrete actions and a handful of states; the paper's results are on MuJoCo
locomotion with continuous actions. This runs PPO on HalfCheetah-v4 with a Gaussian policy,
generalised advantage estimation and the same clipped objective.

Run: ~/.venvs/gpu/bin/python mujoco_ppo.py --steps 200000
"""

import argparse
import json
import os
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

HERE = os.path.dirname(os.path.abspath(__file__))


class ActorCritic(nn.Module):
    def __init__(self, obs_dim, act_dim, hidden=64):
        super().__init__()
        self.pi = nn.Sequential(nn.Linear(obs_dim, hidden), nn.Tanh(),
                                nn.Linear(hidden, hidden), nn.Tanh(), nn.Linear(hidden, act_dim))
        self.v = nn.Sequential(nn.Linear(obs_dim, hidden), nn.Tanh(),
                               nn.Linear(hidden, hidden), nn.Tanh(), nn.Linear(hidden, 1))
        self.log_std = nn.Parameter(torch.zeros(act_dim))

    def dist(self, obs):
        mean = self.pi(obs)
        std = self.log_std.exp().expand_as(mean)
        return torch.distributions.Normal(mean, std)

    def value(self, obs):
        return self.v(obs).squeeze(-1)


def collect(env, model, steps, device, obs):
    obs_buf, act_buf, logp_buf, rew_buf, done_buf, val_buf = [], [], [], [], [], []
    ep_returns, ep_ret, ep_len = [], 0.0, 0
    for _ in range(steps):
        t = torch.as_tensor(obs, dtype=torch.float32, device=device)
        with torch.no_grad():
            dist = model.dist(t)
            a = dist.sample()
            logp = dist.log_prob(a).sum(-1)
            v = model.value(t)
        nobs, r, term, trunc, _ = env.step(a.cpu().numpy())
        done = term or trunc
        obs_buf.append(obs); act_buf.append(a.cpu().numpy())
        logp_buf.append(logp.item()); rew_buf.append(r)
        done_buf.append(float(done)); val_buf.append(v.item())
        ep_ret += r; ep_len += 1
        obs = nobs
        if done:
            ep_returns.append(ep_ret)
            obs, _ = env.reset()
            ep_ret, ep_len = 0.0, 0
    return (np.array(obs_buf), np.array(act_buf), np.array(logp_buf),
            np.array(rew_buf), np.array(done_buf), np.array(val_buf), ep_returns, obs)


def gae(rewards, values, dones, gamma=0.99, lam=0.95):
    n = len(rewards)
    adv = np.zeros(n)
    for t in reversed(range(n)):
        next_v = 0.0 if (t == n - 1 or dones[t]) else values[t + 1]
        next_a = 0.0 if (t == n - 1 or dones[t]) else adv[t + 1]
        adv[t] = rewards[t] + gamma * next_v - values[t] + gamma * lam * next_a
    return adv


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--env", default="HalfCheetah-v4")
    ap.add_argument("--steps", type=int, default=200_000)
    ap.add_argument("--rollout", type=int, default=2048)
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--out", default=os.path.join(HERE, "results_mujoco.json"))
    args = ap.parse_args()

    import gymnasium as gym
    device = "cuda" if torch.cuda.is_available() else "cpu"
    env = gym.make(args.env)
    obs_dim, act_dim = env.observation_space.shape[0], env.action_space.shape[0]
    torch.manual_seed(0)
    model = ActorCritic(obs_dim, act_dim).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    obs, _ = env.reset(seed=0)
    history = []
    iters = args.steps // args.rollout
    t0 = time.time()
    for it in range(iters):
        (o, a, lp, r, d, v, ep_ret, obs) = collect(env, model, args.rollout, device, obs)
        adv = gae(r, v, d)
        ret = adv + v
        adv_t = torch.as_tensor((adv - adv.mean()) / (adv.std() + 1e-8), dtype=torch.float32, device=device)
        ret_t = torch.as_tensor(ret, dtype=torch.float32, device=device)
        ot = torch.as_tensor(o, dtype=torch.float32, device=device)
        at = torch.as_tensor(a, dtype=torch.float32, device=device)
        lpt = torch.as_tensor(lp, dtype=torch.float32, device=device)
        n = len(o)
        for _ in range(args.epochs):
            idx = np.random.permutation(n)
            for s in range(0, n, args.batch):
                b = idx[s:s + args.batch]
                dist = model.dist(ot[b])
                logp = dist.log_prob(at[b]).sum(-1)
                ratio = (logp - lpt[b]).exp()
                surr = torch.min(ratio * adv_t[b],
                                 ratio.clamp(1 - 0.2, 1 + 0.2) * adv_t[b]).mean()
                vloss = F.mse_loss(model.value(ot[b]), ret_t[b])
                loss = -surr + 0.5 * vloss
                opt.zero_grad(set_to_none=True)
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 0.5)
                opt.step()
        if ep_ret:
            history.append(float(np.mean(ep_ret)))
        if (it + 1) % max(1, iters // 10) == 0:
            print(f"  iter {it+1}/{iters}: mean return {np.mean(ep_ret) if ep_ret else 0:.1f} "
                  f"({time.time() - t0:.0f}s)", flush=True)
    final = history[-1] if history else float("nan")
    out = {"env": args.env, "steps": args.steps, "iterations": iters,
           "final_mean_return": round(float(final), 1),
           "history": [round(h, 1) for h in history],
           "seconds": round(time.time() - t0, 1)}
    print(json.dumps({k: v for k, v in out.items() if k != "history"}))
    with open(args.out, "w") as fh:
        json.dump(out, fh, indent=2)


if __name__ == "__main__":
    main()
