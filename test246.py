# -*- coding: utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from scipy.integrate import solve_ivp
from scipy.interpolate import interp1d
from scipy.signal import find_peaks
import os
from matplotlib.backends.backend_pdf import PdfPages

# ============================================================
# 設定（共通）
# ============================================================
mu = 3.0
DT = 0.01
output_dir = "output246"
os.makedirs(output_dir, exist_ok=True)

# 再現性（乱数シード）
GLOBAL_SEED = 42          # train/val 用
TEST_SEED   = 424242      # test 用（独立生成）

np.random.seed(GLOBAL_SEED)
torch.manual_seed(GLOBAL_SEED)

# ============================================================
# 0) ユーティリティ（検証）
# ============================================================
def validate_profiles(profiles, name="profiles"):
    """
    各プロファイルが必須キー {'u_func','t','y'} を持つか検証。欠損があれば例外。
    """
    required = {"u_func", "t", "y"}
    bad = []
    for i, p in enumerate(profiles):
        missing = required - set(p.keys())
        if missing:
            bad.append((i, missing, list(p.keys())))
    if bad:
        print(f"[VALIDATE] {name}: {len(bad)} item(s) missing required keys")
        for i, missing, keys in bad[:10]:
            print(f"  - idx {i}: missing {missing}, keys={keys}")
        raise KeyError(f"{name} contains items without required keys.")
    else:
        print(f"[VALIDATE] {name}: OK ({len(profiles)} items)")

# ============================================================
# 1) フルモデル（Van der Pol）
# ============================================================
def full_model_ode_vdp(t, X, u_func):
    x1, x2 = X
    u_val = u_func(t)
    dx1 = x2
    dx2 = mu*(1 - x1**2)*x2 - x1 + u_val
    return [dx1, dx2]

def simulate_full_model(x1_init, x2_init, t_span, u_func):
    sol = solve_ivp(lambda tt, X: full_model_ode_vdp(tt, X, u_func),
                    t_span,
                    [x1_init, x2_init],
                    max_step=DT*0.5,
                    dense_output=True)
    t_eval = np.arange(t_span[0], t_span[1] + DT, DT)
    X_sol = sol.sol(t_eval)
    x1_sol = X_sol[0, :]
    return t_eval, x1_sol

# ============================================================
# 2) リミットサイクル + 周期推定
# ============================================================
def precompute_vdp_limit_cycle(dt=0.001, transient=50.0, period_guess=6.5):
    def zero_input(t): return 0.0

    # 過渡を流す
    t_span_trans = (0.0, transient)
    sol_trans = solve_ivp(lambda tt, X: full_model_ode_vdp(tt, X, zero_input),
                          t_span_trans, [2.0, 0.0], max_step=dt*0.5, dense_output=True)

    # 数周期分を追加で走らせる
    T_extra = period_guess * 5
    t_span_extra = (transient, transient + T_extra)
    X_end = sol_trans.sol([transient])[:, 0]
    sol_extra = solve_ivp(lambda tt, X: full_model_ode_vdp(tt, X, zero_input),
                          t_span_extra, X_end, max_step=dt*0.5, dense_output=True)
    t_extra = np.arange(t_span_extra[0], t_span_extra[1] + dt, dt)
    X_extra = sol_extra.sol(t_extra)
    x1_extra = X_extra[0, :]
    x2_extra = X_extra[1, :]

    # 周期推定（ピーク間隔平均）
    peaks, _ = find_peaks(x1_extra)
    if len(peaks) < 2:
        T_est = period_guess
        print("ピーク検出不足 → period_guess を使用")
        t0 = t_extra[0]
    else:
        peak_times = t_extra[peaks]
        T_est = np.mean(np.diff(peak_times))
        print(f"推定された周期 T_est = {T_est:.4f}")
        t0 = peak_times[0]

    # 1 周期切り出し
    mask = (t_extra >= t0) & (t_extra <= t0 + T_est)
    t_cut = t_extra[mask] - t0
    x1_cut = x1_extra[mask]
    x2_cut = x2_extra[mask]
    theta_vals = 2.0 * np.pi * (t_cut / T_est)
    return theta_vals, x1_cut, x2_cut, t_cut, T_est

# ============================================================
# 3) x1*(θ) 補間（NumPy版 + Torch用グリッド生成）
# ============================================================
class VdpLimitCycleTable:
    def __init__(self, theta_table, x1_table, kind='cubic'):
        self.theta_max = 2.0 * np.pi
        self.f = interp1d(theta_table, x1_table, kind=kind, fill_value="extrapolate")
    def __call__(self, theta):
        theta_mod = np.mod(theta, self.theta_max)
        return self.f(theta_mod)

def _prepare_monotonic_2pi_grid(theta_np, x1_np):
    th = np.asarray(theta_np).copy()
    xx = np.asarray(x1_np).copy()
    if np.isclose(th[-1], 2*np.pi):
        th = th[:-1]; xx = xx[:-1]
    idx = np.argsort(th)
    return th[idx], xx[idx]

def torch_periodic_linear_interp(theta, theta_grid, x_grid):
    two_pi = 2.0 * np.pi
    th = torch.remainder(theta, two_pi)
    idx_right = torch.bucketize(th, theta_grid)
    idx_right = torch.remainder(idx_right, theta_grid.numel())
    idx_left = (idx_right - 1) % theta_grid.numel()
    th_left = theta_grid[idx_left]; th_right = theta_grid[idx_right]
    x_left = x_grid[idx_left];     x_right = x_grid[idx_right]
    seg = th_right - th_left
    seg = torch.where(seg <= 0, seg + two_pi, seg)
    th_adj = torch.where(th_right <= th_left, th + two_pi, th)
    w = (th_adj - th_left) / seg
    return (1.0 - w) * x_left + w * x_right

# ============================================================
# 4) フロケ由来の真値 Z(θ), I(θ)
# ============================================================
def make_periodic_time_interp(t_grid, v_grid, T):
    if np.isclose(t_grid[-1], T):
        t_use = t_grid[:-1]; v_use = v_grid[:-1]
    else:
        t_use = t_grid; v_use = v_grid
    def f(t):
        tau = np.mod(t, T)
        return np.interp(tau, t_use, v_use)
    return f

def jacobian_vdp(x1, x2):
    a11 = 0.0; a12 = 1.0
    a21 = -1.0 - 2.0*mu*x1*x2
    a22 = mu*(1.0 - x1**2)
    return np.array([[a11, a12],[a21, a22]], dtype=float)

def monodromy_matrix_RK4(t0, T, x1_of_t, x2_of_t, dt=1e-3):
    I = np.eye(2)
    Phi = I.copy()
    tau = 0.0
    while tau < T - 1e-15:
        h = min(dt, T - tau)
        t = t0 + tau
        A1 = jacobian_vdp(x1_of_t(t),           x2_of_t(t))
        A2 = jacobian_vdp(x1_of_t(t + 0.5*h),   x2_of_t(t + 0.5*h))
        A3 = A2
        A4 = jacobian_vdp(x1_of_t(t + h),       x2_of_t(t + h))
        k1 = A1 @ Phi
        k2 = A2 @ (Phi + 0.5*h*k1)
        k3 = A3 @ (Phi + 0.5*h*k2)
        k4 = A4 @ (Phi + h*k3)
        Phi = Phi + (h/6.0)*(k1 + 2*k2 + 2*k3 + k4)
        tau += h
    return Phi

def compute_ground_truth_ZI(theta_grid, T, t_cut, x1_cut, x2_cut, dt_floq=1e-3):
    x1_of_t = make_periodic_time_interp(t_cut, x1_cut, T)
    x2_of_t = make_periodic_time_interp(t_cut, x2_cut, T)
    B = np.array([0.0, 1.0])
    omega = 2.0*np.pi / T

    Z_true, I_true = [], []
    for theta in theta_grid:
        t0 = (theta/(2.0*np.pi))*T
        M = monodromy_matrix_RK4(t0, T, x1_of_t, x2_of_t, dt=dt_floq)
        evals, R = np.linalg.eig(M)
        evalsT, L = np.linalg.eig(M.T)

        # 位相方向（固有値=1）
        idx1 = np.argmin(np.abs(evalsT - 1.0))
        V = np.real(L[:, idx1])
        x1 = x1_of_t(t0); x2 = x2_of_t(t0)
        fvec = np.array([x2, mu*(1.0 - x1**2)*x2 - x1], dtype=float)
        scale = omega / (V @ fvec)
        V = V * scale
        Z_scalar = float(V @ B)

        # 振幅方向（もう一方）
        idx2 = 1 - idx1
        w = np.real(R[:, idx2])
        xi = np.real(L[:, idx2])
        xi = xi / (xi @ w)
        I_scalar = float(xi @ B)

        Z_true.append(Z_scalar); I_true.append(I_scalar)
    return np.array(Z_true), np.array(I_true)

# ============================================================
# 5) ネットワーク：Fourier / SIREN（論文準拠版）
# ============================================================
class FourierSineNN_ZIG(nn.Module):
    def __init__(self, order=7):
        super().__init__()
        self.order = order
        self.num_basis = 1 + 2*order
        self.Z_params = nn.Parameter(torch.zeros(self.num_basis))
        self.I_params = nn.Parameter(torch.zeros(self.num_basis))
        self.g_params = nn.Parameter(torch.zeros(self.num_basis))
        nn.init.uniform_(self.Z_params, -0.1, 0.1)
        nn.init.uniform_(self.I_params, -0.1, 0.1)
        nn.init.uniform_(self.g_params, -0.1, 0.1)

    def _basis(self, theta):
        if theta.ndim == 2:
            theta = theta.squeeze(1)
        b = [torch.ones_like(theta)]
        for k in range(1, self.order+1):
            b.append(torch.sin(k*theta))
            b.append(torch.cos(k*theta))
        return torch.stack(b, dim=1)

    def forward(self, theta_psi):
        theta = theta_psi[:, 0]
        psi   = theta_psi[:, 1]
        B = self._basis(theta)
        Z = torch.sum(B * self.Z_params, dim=1)
        I = torch.sum(B * self.I_params, dim=1)
        g = torch.sum(B * self.g_params, dim=1)
        G = psi * g
        return torch.stack([Z, I, G], dim=1)

    @torch.no_grad()
    def components(self, theta_1d):
        theta = torch.tensor(theta_1d, dtype=torch.float32)
        B = self._basis(theta)
        Z = (B * self.Z_params).sum(dim=1).cpu().numpy()
        I = (B * self.I_params).sum(dim=1).cpu().numpy()
        g = (B * self.g_params).sum(dim=1).cpu().numpy()
        return Z, I, g

class SIRENLayer(nn.Module):
    """
    論文準拠 SIREN 層：
      - 第1層のみ omega_0 を重み項にだけ掛ける（bias には掛けない）
      - 初期化：第1層 U(-1/in,1/in)、以降 U(-√(6/in)/ω0, √(6/in)/ω0)
    """
    def __init__(self, in_features, out_features, omega_0=30.0, is_first=False, bias=True):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.omega_0 = float(omega_0)
        self.is_first = bool(is_first)
        self.weight = nn.Parameter(torch.empty(out_features, in_features))
        self.bias   = nn.Parameter(torch.empty(out_features)) if bias else None
        self.reset_parameters()

    @torch.no_grad()
    def reset_parameters(self):
        if self.is_first:
            bound = 1.0 / self.in_features
            self.weight.uniform_(-bound, bound)
            if self.bias is not None:
                self.bias.uniform_(-bound, bound)
        else:
            bound = np.sqrt(6.0 / self.in_features) / self.omega_0
            self.weight.uniform_(-bound, bound)
            if self.bias is not None:
                self.bias.uniform_(-bound, bound)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        Wx = F.linear(x, self.weight, bias=None)  # [B, out]
        if self.is_first:
            if self.bias is not None:
                Wx = self.omega_0 * Wx + self.bias
            else:
                Wx = self.omega_0 * Wx
            return torch.sin(Wx)
        else:
            if self.bias is not None:
                Wx = Wx + self.bias
            return torch.sin(Wx)

class SIREN_SineNN_ZIG(nn.Module):
    """
    Z(θ), I(θ), g(θ) を同時出力する SIREN。
    """
    def __init__(self, hidden_dim=3, omega_0=30.0, num_hidden_layers=1):
        super().__init__()
        assert num_hidden_layers >= 1
        layers = [SIRENLayer(1, hidden_dim, omega_0=omega_0, is_first=True)]
        for _ in range(num_hidden_layers - 1):
            layers.append(SIRENLayer(hidden_dim, hidden_dim, omega_0=1.0, is_first=False))
        self.net = nn.Sequential(*layers)
        self.final_linear = nn.Linear(hidden_dim, 3, bias=True)
        with torch.no_grad():
            bound = np.sqrt(6.0 / self.final_linear.in_features) / float(omega_0)
            self.final_linear.weight.uniform_(-bound, bound)
            self.final_linear.bias.zero_()

    def forward(self, theta_psi: torch.Tensor) -> torch.Tensor:
        theta = theta_psi[:, 0:1]
        psi   = theta_psi[:, 1:2]
        h = self.net(theta)
        out = self.final_linear(h)
        Z = out[:, 0:1]; I = out[:, 1:2]; g_theta = out[:, 2:3]
        G_val = psi * g_theta
        return torch.cat([Z, I, G_val], dim=1)

    @torch.no_grad()
    def components(self, theta_1d):
        theta = torch.as_tensor(theta_1d, dtype=torch.float32).reshape(-1, 1)
        h = self.net(theta)
        out = self.final_linear(h)
        return out[:, 0].cpu().numpy(), out[:, 1].cpu().numpy(), out[:, 2].cpu().numpy()

# ============================================================
# 6) 縮約モデル：dθ/dt, dψ/dt, y = x1*(θ)+ψ g(θ)
# ============================================================
class ReducedModelPT:
    def __init__(self, net, x1_orbit_func, dt=DT, omega0=1.0, theta_init=0.0, psi_init=0.0):
        self.net = net
        self.x1_orbit_func = x1_orbit_func
        self.dt = dt
        self.omega0 = omega0
        self.theta0 = theta_init
        self.psi0 = psi_init
        self.theta_grid_torch = None
        self.x1_grid_torch = None

    def simulate(self, u_func, t_span, *, return_states=False, track_grad=True, truncate_k=200):
        t_start, t_end = t_span
        t_eval = np.arange(t_start, t_end + self.dt, self.dt)
        req = bool(track_grad)
        theta = torch.tensor(self.theta0, dtype=torch.float32, requires_grad=req)
        psi   = torch.tensor(self.psi0,   dtype=torch.float32, requires_grad=req)

        outputs, thetas, psis, Gs = [], [], [], []
        for i, t in enumerate(t_eval):
            if (truncate_k is not None) and (i > 0) and (i % truncate_k == 0):
                theta = theta.detach().requires_grad_(req)
                psi   = psi.detach().requires_grad_(req)

            u_val = torch.tensor(u_func(t), dtype=torch.float32)

            inp = torch.stack([theta, psi]).unsqueeze(0)
            out = self.net(inp)
            out = torch.nan_to_num(out, nan=0.0, posinf=1e6, neginf=-1e6)

            Z_val, I_val, G_val = out[0,0], out[0,1], out[0,2]

            theta = theta + self.dt * (self.omega0 + Z_val * u_val)
            psi   = psi   + self.dt * (-3.5578 * psi + I_val * u_val)  # κ固定（必要なら推定へ拡張可）

            if (self.theta_grid_torch is None) or (self.x1_grid_torch is None):
                x1_star_np = self.x1_orbit_func(float(torch.remainder(theta, 2*np.pi).detach().cpu().numpy()))
                x1_star = torch.tensor(x1_star_np, dtype=torch.float32)
            else:
                x1_star = torch_periodic_linear_interp(theta, self.theta_grid_torch, self.x1_grid_torch)

            y = x1_star + G_val
            y = torch.nan_to_num(y, nan=0.0, posinf=1e6, neginf=-1e6)

            outputs.append(y)
            if return_states:
                thetas.append(theta.detach().item())
                psis.append(psi.detach().item())
                Gs.append(G_val.detach().item())

        y_reduced = torch.stack(outputs)
        if return_states:
            return t_eval, y_reduced, np.array(thetas), np.array(psis), np.array(Gs)
        else:
            return t_eval, y_reduced

# ============================================================
# 7) 入力生成：矩形パルス（0→α→0）
# ============================================================
def make_rect_pulse(alpha: float, t_on: float, t_off: float):
    """
    u(t) = alpha (t_on <= t < t_off), それ以外は 0
    """
    assert alpha >= 0.0
    assert t_off > t_on
    return lambda t, a=alpha, on=t_on, off=t_off: (a if (t >= on and t < off) else 0.0)

def generate_random_pulse_profiles(num_profiles, t_span, alpha_max=0.1, seed=None,
                                   min_width=0.1, max_width=None):
    """
    ランダムな ON/OFF 時刻を持つ矩形パルス（0→α→0）を num_profiles 本生成。
    - alpha ~ U(0, alpha_max)
    - t_on  ~ U(T0, T1 - min_width)
    - width ~ U(min_width, min(max_width, T1 - t_on))
    - t_off = t_on + width
    """
    rng = np.random.default_rng(seed) if seed is not None else np.random.default_rng()
    T0, T1 = t_span
    if max_width is None:
        max_width = (T1 - T0)

    profiles = []
    for _ in range(num_profiles):
        a = float(rng.uniform(0.0, alpha_max))
        t_on = float(rng.uniform(T0, T1 - min_width))
        max_w = min(max_width, T1 - t_on)
        width = float(rng.uniform(min_width, max_w))
        t_off = t_on + width

        u_func = make_rect_pulse(a, t_on, t_off)
        t_arr, y_arr = simulate_full_model(2.0, 0.0, t_span, u_func)

        profiles.append({
            "alpha": a,
            "t_on": t_on,
            "t_off": t_off,
            "u_func": u_func,     # ← 必須：学習で使うコールバック
            "t": t_arr,
            "y": y_arr,
            # 互換表示用（旧コードが t_switch を参照していても落ちないように）
            "t_switch": t_on
        })
    return profiles

# ============================================================
# 8) 学習（ミニバッチ + 勾配クリップ + Truncated BPTT）
#    ※ フォールバック：u_func が無い場合は再構成を試みる（保険）
# ============================================================
def train_model(model_pt, profiles, epochs, learning_rate, batch_size=16, truncate_k=200, max_grad_norm=1.0):
    optimizer = optim.Adam(model_pt.net.parameters(), lr=learning_rate)
    loss_history = []
    rng = np.random.default_rng(42)

    for ep in range(1, epochs + 1):
        order = rng.permutation(len(profiles))
        total_loss_val = 0.0
        num_batches = 0

        for start in range(0, len(profiles), batch_size):
            idxs = order[start:start+batch_size]
            batch = [profiles[i] for i in idxs]

            batch_loss = 0.0
            valid_profiles = 0
            for profile in batch:
                # --- フォールバック：u_func が無ければ再構成 ---
                if "u_func" not in profile:
                    if ("t_on" in profile) and ("t_off" in profile) and ("alpha" in profile):
                        profile["u_func"] = make_rect_pulse(profile["alpha"], profile["t_on"], profile["t_off"])
                    elif ("t_switch" in profile) and ("alpha" in profile):
                        sw = profile["t_switch"]; a = profile["alpha"]
                        profile["u_func"] = (lambda t, s=sw, aa=a: (aa if t < s else 0.0))
                    else:
                        raise KeyError(f"Profile missing 'u_func' and insufficient info to rebuild: keys={list(profile.keys())}")

                u_func = profile["u_func"]; t_arr = profile["t"]; y_true = profile["y"]
                _, y_red = model_pt.simulate(u_func, (t_arr[0], t_arr[-1]), track_grad=True, truncate_k=truncate_k)

                m = min(len(y_true), len(y_red))
                y_true_clip = torch.tensor(y_true[:m], dtype=torch.float32)
                y_red_clip  = torch.nan_to_num(y_red[:m], nan=0.0, posinf=1e6, neginf=-1e6)

                loss = torch.mean((y_true_clip - y_red_clip)**2)
                if torch.isnan(loss).item() or torch.isinf(loss).item():
                    print("[WARN] NaN/Inf loss detected. Skipped one profile.")
                    continue

                batch_loss = batch_loss + loss
                valid_profiles += 1

            if valid_profiles == 0:
                continue

            batch_loss = batch_loss / valid_profiles

            optimizer.zero_grad()
            batch_loss.backward()
            torch.nn.utils.clip_grad_norm_(model_pt.net.parameters(), max_grad_norm)
            optimizer.step()

            total_loss_val += batch_loss.item()
            num_batches += 1

        mean_loss = (total_loss_val / max(1, num_batches))
        loss_history.append(mean_loss)
        print(f"Epoch {ep}/{epochs}  Loss={mean_loss:.6e}")

    return loss_history

# ============================================================
# 9) 可視化ユーティリティ
# ============================================================
def plot_ZI_compare(theta_grid, Z_true, I_true, fourier_net, siren_net, path_png):
    Z_F, I_F, _ = fourier_net.components(theta_grid)
    Z_S, I_S, _ = siren_net.components(theta_grid)

    fig, axes = plt.subplots(2,1, figsize=(10,8), sharex=True)
    ax = axes[0]
    ax.plot(theta_grid, Z_true, label="Z true")
    ax.plot(theta_grid, Z_F, '--', label="Z Fourier")
    ax.plot(theta_grid, Z_S, ':', label="Z SIREN")
    ax.set_ylabel("Z(θ)"); ax.grid(True); ax.legend()

    ax = axes[1]
    ax.plot(theta_grid, I_true, label="I true")
    ax.plot(theta_grid, I_F, '--', label="I Fourier")
    ax.plot(theta_grid, I_S, ':', label="I SIREN")
    ax.set_xlabel("θ"); ax.set_ylabel("I(θ)"); ax.grid(True); ax.legend()

    fig.tight_layout()
    fig.savefig(path_png, dpi=150)
    plt.close(fig)

def plot_loss_histories(loss_F, loss_S, path_png, title="Training Loss History"):
    fig = plt.figure(figsize=(8,5))
    plt.plot(loss_F, label="Fourier loss")
    plt.plot(loss_S, label="SIREN loss")
    plt.xlabel("Epoch"); plt.ylabel("MSE (y=x1)"); plt.title(title)
    plt.grid(True); plt.legend()
    fig.tight_layout()
    fig.savefig(path_png, dpi=150)
    plt.close(fig)

def safe_mae(a, b):
    m = min(len(a), len(b))
    a = np.asarray(a[:m]); b = np.asarray(b[:m])
    mask = np.isfinite(a) & np.isfinite(b)
    if not np.any(mask):
        return np.nan
    return float(np.mean(np.abs(a[mask] - b[mask])))

def evaluate_on_dataset(profiles, fourier_model, siren_model,
                        theta_init, psi_init, t_span, out_pdf_path, label="VAL"):
    maes_F, maes_S = [], []
    with PdfPages(out_pdf_path) as pdf:
        for i, profile in enumerate(profiles):
            u_func = profile['u_func']; t_full = profile['t']; y_full = profile['y']

            # Fourier
            fourier_model.theta0 = theta_init; fourier_model.psi0 = psi_init
            _, yF = fourier_model.simulate(u_func, t_span, return_states=False, track_grad=False)
            yF = yF.detach().numpy()

            # SIREN
            siren_model.theta0 = theta_init; siren_model.psi0 = psi_init
            _, yS = siren_model.simulate(u_func, t_span, return_states=False, track_grad=False)
            yS = yS.detach().numpy()

            maeF = safe_mae(y_full, yF)
            maeS = safe_mae(y_full, yS)
            maes_F.append(maeF); maes_S.append(maeS)

            m = min(len(y_full), len(yF), len(yS))
            tt = t_full[:m]
            absF = np.abs(y_full[:m] - yF[:m])
            absS = np.abs(y_full[:m] - yS[:m])

            # タイトル：パルス情報も表示
            title_str = f"[{label} #{i+1}] alpha={profile['alpha']:.4f}"
            if 't_on' in profile and 't_off' in profile:
                title_str += f", t_on={profile['t_on']:.3f}, t_off={profile['t_off']:.3f}"
            elif 't_switch' in profile:
                title_str += f", t_switch={profile['t_switch']:.3f}"

            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(11,8), sharex=True)
            ax1.plot(tt, y_full[:m], 'k', label="Full (y=x1)")
            ax1.plot(tt, yF[:m],   'r--', label="Reduced (Fourier)")
            ax1.plot(tt, yS[:m],   'b-.', label="Reduced (SIREN)")
            ax1.set_title(title_str)
            ax1.set_ylabel("y(t)"); ax1.grid(True); ax1.legend()

            ax2.plot(tt, absF, 'r--', label=f"|err| Fourier (MAE={maeF:.4f})")
            ax2.plot(tt, absS, 'b-.', label=f"|err| SIREN (MAE={maeS:.4f})")
            ax2.set_xlabel("Time"); ax2.set_ylabel("|err|"); ax2.grid(True); ax2.legend()

            fig.tight_layout()
            pdf.savefig(fig)
            plt.close(fig)

    # まとめファイル
    summary_txt = os.path.join(os.path.dirname(out_pdf_path), f"{label.lower()}_summary.txt")
    with open(summary_txt, "w", encoding="utf-8") as f:
        f.write(f"[{label} {len(profiles)} profiles]\n")
        f.write(f"Fourier  MAE: mean={np.nanmean(maes_F):.6f}, median={np.nanmedian(maes_F):.6f}, std={np.nanstd(maes_F):.6f}\n")
        f.write(f"SIREN    MAE: mean={np.nanmean(maes_S):.6f}, median={np.nanmedian(maes_S):.6f}, std={np.nanstd(maes_S):.6f}\n")
    print(f"[{label}] Saved {label.lower()} PDF → {out_pdf_path}")
    print(f"[{label}] Summary → {summary_txt}")

# ============================================================
# 10) メイン：train/val/test（矩形パルス統一）
# ============================================================
def main():
    # (A) リミットサイクル & 周期
    theta_vals, x1_vals, x2_vals, t_cut, T_est = precompute_vdp_limit_cycle(
        dt=0.001, transient=50.0, period_guess=6.5)
    print(f"採用周期 T_est = {T_est:.4f}")
    x1_orbit = VdpLimitCycleTable(theta_vals, x1_vals, kind='cubic')

    # Torch補間グリッド
    th_grid_np, x1_grid_np = _prepare_monotonic_2pi_grid(theta_vals, x1_vals)
    theta_grid_torch = torch.tensor(th_grid_np, dtype=torch.float32)
    x1_grid_torch = torch.tensor(x1_grid_np, dtype=torch.float32)

    # (B) 基準位相（x1≈2に最も近い点）
    idx = np.argmin(np.abs(x1_vals - 2.0))
    theta_init = theta_vals[idx]; psi_init = 0.0

    # (C) 真値 Z,I（可視化用）
    theta_grid_gt = np.linspace(0, 2*np.pi, 400)
    Z_true, I_true = compute_ground_truth_ZI(theta_grid_gt, T_est, t_cut, x1_vals, x2_vals, dt_floq=1e-3)

    # (D) 入力データ生成（すべて矩形パルス）
    t_span = (0.0, 30.0)

    # train/val：同一母集団から分割（seed=GLOBAL_SEED）
    profiles_all = generate_random_pulse_profiles(
        num_profiles=310, t_span=t_span, alpha_max=0.1, seed=GLOBAL_SEED,
        min_width=0.2, max_width=5.0
    )
    rng = np.random.default_rng(GLOBAL_SEED)
    perm = rng.permutation(len(profiles_all))
    idx_train = perm[:250]
    idx_val   = perm[250:280]     # 30 本
    profiles_train = [profiles_all[i] for i in idx_train]
    profiles_val   = [profiles_all[i] for i in idx_val]

    # test：独立生成（seed=TEST_SEED）
    profiles_test = generate_random_pulse_profiles(
        num_profiles=30, t_span=t_span, alpha_max=0.1, seed=TEST_SEED,
        min_width=0.2, max_width=5.0
    )

    # バリデーション（必須キーの有無チェック）
    validate_profiles(profiles_train, "profiles_train")
    validate_profiles(profiles_val,   "profiles_val")
    validate_profiles(profiles_test,  "profiles_test")

    # 分割情報を保存
    np.savez(os.path.join(output_dir, "split_indices_3split_pulse.npz"),
             idx_train=idx_train, idx_val=idx_val,
             seed_trainval=GLOBAL_SEED, seed_test=TEST_SEED)

    # (E) 2つの縮約モデルを用意（同条件で学習）
    omega0_phase = 2.0 * np.pi / T_est
    print(f"omega0 = {omega0_phase:.4f}")

    # Fourier（order=7）
    net_F = FourierSineNN_ZIG(order=7)
    model_F = ReducedModelPT(net=net_F, x1_orbit_func=x1_orbit, dt=DT,
                             omega0=omega0_phase, theta_init=theta_init, psi_init=psi_init)
    model_F.theta_grid_torch = theta_grid_torch
    model_F.x1_grid_torch = x1_grid_torch

    # SIREN（論文準拠）
    omega_siren = 30.0
    net_S = SIREN_SineNN_ZIG(hidden_dim=3, omega_0=omega_siren, num_hidden_layers=1)
    model_S = ReducedModelPT(net=net_S, x1_orbit_func=x1_orbit, dt=DT,
                             omega0=omega0_phase, theta_init=theta_init, psi_init=psi_init)
    model_S.theta_grid_torch = theta_grid_torch
    model_S.x1_grid_torch = x1_grid_torch

    # (F) 学習（train=250 本）
    epochs = 15
    learning_rate = 1e-2
    print("=== Train: Fourier (train=250, pulse) ===")
    loss_F = train_model(model_F, profiles_train, epochs, learning_rate,
                         batch_size=16, truncate_k=200, max_grad_norm=1.0)
    print("=== Train: SIREN (train=250, pulse) ===")
    loss_S = train_model(model_S, profiles_train, epochs, learning_rate,
                         batch_size=16, truncate_k=200, max_grad_norm=1.0)

    # 損失履歴
    plot_loss_histories(loss_F, loss_S, os.path.join(output_dir, "loss_history_F_vs_S.png"))

    # Z,I 比較（両者+真値）
    plot_ZI_compare(theta_grid_gt, Z_true, I_true, net_F, net_S,
                    os.path.join(output_dir, "ZI_compare_true_Fourier_SIREN.png"))

    # (G) 検証（val=30 本）
    val_pdf_path = os.path.join(output_dir, "validation_30_profiles.pdf")
    evaluate_on_dataset(
        profiles_val, model_F, model_S,
        theta_init=theta_init, psi_init=psi_init, t_span=t_span,
        out_pdf_path=val_pdf_path, label="VAL"
    )

    # (H) テスト（test=30 本；独立生成）
    test_pdf_path = os.path.join(output_dir, "test_30_profiles.pdf")
    evaluate_on_dataset(
        profiles_test, model_F, model_S,
        theta_init=theta_init, psi_init=psi_init, t_span=t_span,
        out_pdf_path=test_pdf_path, label="TEST"
    )

if __name__ == "__main__":
    main()
