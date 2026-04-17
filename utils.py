"""Utility functions for simulation, metrics, and visualization."""

import numpy as np
import torch
from copy import deepcopy
from tqdm import tqdm
import math
import mlflow
from io import BytesIO
from matplotlib import pyplot as plt
import tempfile
import os


def sample_initial_states(num_trajectories, dim, strategy):
    """Sample initial states using a configurable strategy."""
    np.random.seed(strategy["seed"])
    if strategy["identifies"] == "uniform":
        X0 = np.random.uniform(0, 1, (num_trajectories, dim))
    elif strategy["identifies"] == "normal":
        X0 = np.random.normal(0, 1, (num_trajectories, dim))
    else:
        raise NotImplementedError("Sampling strategy not implemented")
    if "scale" in strategy:
        X0 = X0 * np.reshape(strategy["scale"], (1, dim))
    if "bias" in strategy:
        X0 = X0 + np.reshape(strategy["bias"], (1, dim))
    return X0

def multi_sin_signal(n_signals=1, f_0=.1, amplitude=.2, seed=None):
    """
    Create a multi-sine input signal function u(t).

    The returned signal uses 40 harmonics with random phases:
    u_k(t) = A * sum_{i=0}^{39} sin(2π i f_0 t + φ_{k,i})

    Parameters
    ----------
    n_signals : int, optional
        Number of independent signals (output dimension).
    f_0 : float, optional
        Base frequency used for all harmonics.
    amplitude : float, optional
        Global amplitude scaling A applied to the summed harmonics.
    seed : int or None, optional
        Seed for phase generation. If None, uses the current RNG state.

    Returns
    -------
    callable
        Function u(t) that accepts a scalar t and returns an array of shape
        (n_signals,) with the multi-sine values at time t.
    """
    if seed is not None:
        np.random.seed(seed)
    i = np.arange(40)
    i, phi = np.stack([i] * n_signals), np.random.uniform(0, 2 * np.pi, (n_signals, 40))
    def u(t):
        sins = np.sin(2 * np.pi * i * f_0 * t + phi)
        out = np.sum(sins, axis=-1) * amplitude
        return out
    return u


@torch.no_grad()
def forecast(model, X0, u, dt, signal, steps, clamp=None, a=None, integrator="RK45"):
    """
    Roll out model dynamics from initial states using a numerical integrator.

    Parameters
    ----------
    model : torch.nn.Module
        Model that returns at least the state derivative as the first output.
    X0 : np.ndarray or torch.Tensor
        Initial states with shape (batch, state_dim).
    u : np.ndarray or torch.Tensor
        Input sequence with shape (batch, steps, input_dim). Used when noise is
        enabled (a is not None) and for logging, otherwise signal is used.
    dt : float
        Time step for integration.
    signal : list[callable]
        List of input functions; each sig(t) returns input for one trajectory.
    steps : int
        Number of time steps to simulate (including the initial state).
    clamp : float or None, optional
        If provided, clamps states to [-clamp, clamp] after each step to prevent blow-up if deviation is high.
    a : float or np.ndarray or None, optional
        If provided, adds uniform white noise with bound a to midpoint inputs.
    integrator : str, optional
        Integration method, e.g., "RK4", "DOPRI5", or "IMPLICIT_MIDPOINT".

    Returns
    -------
    torch.Tensor
        Simulated state trajectory of shape (batch, steps, state_dim).
    """
    model.eval()
    X = torch.zeros(X0.shape[0], steps, X0.shape[-1])
    u = torch.tensor(u)
    X[:, 0] = torch.tensor(X0)
    integrator = integrator.upper()
    t = 0
    for i in tqdm(range(0, steps-1)):
        X = X.clone().detach()
        """# first order approximation
        with torch.autograd.enable_grad():
            dxdt = model(X[:, i].float(), u[:, i].float())[0]
        X[:, i+1] = dxdt * dt + X[:, i]"""
        # runge kutta
        if integrator == "RK4":
            u0 = torch.stack([torch.tensor(sig(t)) for sig in signal]).float()
            u_mid = torch.stack([torch.tensor(sig(t + dt/2)) for sig in signal]).float()
            u_end = torch.stack([torch.tensor(sig(t + dt)) for sig in signal]).float()
            if a is not None:
                u0 = torch.tensor(u[:, i]).float()
                u_mid += torch.tensor(get_uniform_white_noise(u_mid, a)).float()
                u_end += torch.tensor(u[:, i+1]).float()
            k1 = model(X[:, i], u0)[0]
            k2 = model(X[:, i] + k1 * dt / 2, u_mid)[0]
            k3 = model(X[:, i] + k2 * dt / 2, u_mid)[0]
            k4 = model(X[:, i] + k3 * dt, u_end)[0]
            X[:, i+1] = X[:, i] + (k1 + 2 * k2 + 2 * k3 + k4) * dt / 6
        elif integrator in ("DOPRI5", "RK5", "RK45"):
            # Dormand-Prince 5th-order explicit Runge-Kutta
            u0 = torch.stack([torch.tensor(sig(t)) for sig in signal]).float()
            u_1_5 = torch.stack([torch.tensor(sig(t + dt / 5)) for sig in signal]).float()
            u_3_10 = torch.stack([torch.tensor(sig(t + 3 * dt / 10)) for sig in signal]).float()
            u_4_5 = torch.stack([torch.tensor(sig(t + 4 * dt / 5)) for sig in signal]).float()
            u_8_9 = torch.stack([torch.tensor(sig(t + 8 * dt / 9)) for sig in signal]).float()
            u_end = torch.stack([torch.tensor(sig(t + dt)) for sig in signal]).float()
            if a is not None:
                u0 = torch.tensor(u[:, i]).float()
                u_1_5 += torch.tensor(get_uniform_white_noise(u_1_5, a)).float()
                u_3_10 += torch.tensor(get_uniform_white_noise(u_3_10, a)).float()
                u_4_5 += torch.tensor(get_uniform_white_noise(u_4_5, a)).float()
                u_8_9 += torch.tensor(get_uniform_white_noise(u_8_9, a)).float()
                u_end = torch.tensor(u[:, i + 1]).float()

            k1 = model(X[:, i], u0)[0]
            k2 = model(X[:, i] + dt * (1 / 5) * k1, u_1_5)[0]
            k3 = model(X[:, i] + dt * (3 / 40 * k1 + 9 / 40 * k2), u_3_10)[0]
            k4 = model(X[:, i] + dt * (44 / 45 * k1 - 56 / 15 * k2 + 32 / 9 * k3), u_4_5)[0]
            k5 = model(X[:, i] + dt * (19372 / 6561 * k1 - 25360 / 2187 * k2 + 64448 / 6561 * k3 - 212 / 729 * k4), u_8_9)[0]
            k6 = model(X[:, i] + dt * (9017 / 3168 * k1 - 355 / 33 * k2 + 46732 / 5247 * k3 + 49 / 176 * k4 - 5103 / 18656 * k5), u_end)[0]
            k7 = model(X[:, i] + dt * (35 / 384 * k1 + 500 / 1113 * k3 + 125 / 192 * k4 - 2187 / 6784 * k5 + 11 / 84 * k6), u_end)[0]
            X[:, i + 1] = X[:, i] + dt * (35 / 384 * k1 + 500 / 1113 * k3 + 125 / 192 * k4 - 2187 / 6784 * k5 + 11 / 84 * k6)
        elif integrator in ("IMPLICIT_MIDPOINT", "IMPLICIT-MIDPOINT", "MIDPOINT"):
            u_mid = torch.stack([torch.tensor(sig(t + dt / 2)) for sig in signal]).float()
            if a is not None:
                u_mid += torch.tensor(get_uniform_white_noise(u_mid, a)).float()
            x_i = X[:, i]
            x_next = x_i + dt * model(x_i, u_mid)[0]
            max_iters = 8
            tol = 1e-6
            for _ in range(max_iters):
                x_mid = 0.5 * (x_i + x_next)
                x_next_new = x_i + dt * model(x_mid, u_mid)[0]
                if torch.max(torch.abs(x_next_new - x_next)) < tol:
                    x_next = x_next_new
                    break
                x_next = x_next_new
            X[:, i+1] = x_next

        if clamp is not None:
            X[:, i+1] = torch.clamp(X[:, i+1], -clamp, clamp)
        t += dt
    return X


class Dataset(torch.utils.data.Dataset):

    def __init__(self, X, u, xdot, y):
        """Store dataset tensors."""
        self.X = X
        self.u = u
        self.xdot = xdot
        self.y = y


    def __len__(self):
        """Return number of samples."""
        return len(self.X)

    def __getitem__(self, idx):
        """Return a single sample tuple."""
        return self.X[idx], self.u[idx], self.xdot[idx], self.y[idx]


def normalized_mse(x, target):
    """Compute normalized mean squared error."""
    std = target.std()
    return ((x - target) / (std + 1e-6)).pow(2).mean()

def normalized_mae(x, target):
    """Compute normalized mean absolute error."""
    std = target.std()
    return ((x - target) / (std + 1e-6)).abs().mean()


def _ensure_model_artifacts(model):
    """
    Ensure MLflow stores both:
    1) a PyTorch model at artifact path `model` (for mlflow.pytorch.load_model),
    2) a root-level `model.pth` file artifact.
    """
    active_run = mlflow.active_run()
    if active_run is None:
        return

    run_id = active_run.info.run_id
    client = mlflow.tracking.MlflowClient()

    # Ensure `runs:/<run_id>/model` exists for mlflow.pytorch.load_model.
    has_model_dir = False
    try:
        has_model_dir = len(client.list_artifacts(run_id, "model")) > 0
    except Exception:
        has_model_dir = False
    if not has_model_dir:
        mlflow.pytorch.log_model(model, "model")

    # Ensure a top-level `model.pth` artifact also exists.
    has_model_pth = False
    try:
        has_model_pth = any(a.path == "model.pth" for a in client.list_artifacts(run_id))
    except Exception:
        has_model_pth = False
    if not has_model_pth:
        model_pth_path = mlflow.artifacts.download_artifacts(
            run_id=run_id,
            artifact_path="model/data/model.pth",
        )
        with tempfile.TemporaryDirectory() as tmpdir:
            dst = os.path.join(tmpdir, "model.pth")
            with open(model_pth_path, "rb") as src_f, open(dst, "wb") as dst_f:
                dst_f.write(src_f.read())
            mlflow.log_artifact(dst)


@torch.no_grad()
def compute_metrics(model, dt, X, u, xdot, y, bounds=np.reshape(np.array([.001, .005, .01, .025, .05]), (1, 1, 5)), integrator="RK45"):
    """
    Compute prediction and forecast metrics, logging to MLflow when active.

    Parameters
    ----------
    model : torch.nn.Module or list[torch.nn.Module]
        Model(s) that return (xdot_pred, y_pred).
    trajectories : list[tuple]
        List of trajectories where each tuple is (X, u, y, signal).
        X shape: (steps, state_dim), u shape: (steps, input_dim),
        y shape: (steps, output_dim), signal: list of callables.
    dt : float
        Time step used for forecast rollouts.
    X : torch.Tensor
        Batched state inputs used for one-step prediction.
    u : torch.Tensor
        Batched inputs aligned with X for one-step prediction.
    xdot : torch.Tensor
        Ground-truth state derivatives aligned with X.
    y : torch.Tensor
        Ground-truth outputs aligned with X.
    bounds : np.ndarray, optional
        Thresholds for accurate time calculation with shape (1, 1, K).
    integrator : str, optional
        Integration method for multi-step forecast.

    Returns
    -------
    dict
        Dictionary of scalar metrics (MAE/MSE for xdot, outputs, and forecast).
    """

    if isinstance(model, list):
        out_dict = {}
        all_metrics = {}
        for m in model:
            metrics = compute_metrics(m, dt, X, u, xdot, y)
            for k, v in metrics.items():
                if k not in all_metrics:
                    all_metrics[k] = []
                all_metrics[k].append(v)
        for k, v in all_metrics.items():
            out_dict[k] = np.mean(v)
        return out_dict
    else:
        _ensure_model_artifacts(model)
        N = len(X)
        x_dot_pred = []
        y_pred = []
        for i in tqdm(range(N // 512 + 1)):
            with torch.enable_grad():
                out = model(X[i*512:(i+1)*512], u[i*512:(i+1)*512])
            x_dot_pred.append(out[0].detach())
            y_pred.append(out[1].detach())
        xdot_pred = torch.cat(x_dot_pred, dim=0)
        y_pred = torch.cat(y_pred, dim=0)
        mae = torch.abs(xdot_pred - xdot).mean().item()
        mse = ((xdot_pred - xdot)**2).mean().item()
        mae_rel = normalized_mae(xdot_pred, xdot).item()
        mse_rel = normalized_mse(xdot_pred, xdot).item()

        output_mae = torch.abs(y_pred - y).mean().item()
        output_mse = ((y_pred - y)**2).mean().item()
        output_mae_rel = normalized_mae(y_pred, y).item()
        output_mse_rel = normalized_mse(y_pred, y).item()

        out_dict = {
            "mae": mae,
            "mse": mse,
            "mae_rel": mae_rel,
            "mse_rel": mse_rel,
            "output_mae": output_mae,
            "output_mse": output_mse,
            "output_mae_rel": output_mae_rel,
            "output_mse_rel": output_mse_rel
        }


        if mlflow.active_run() is not None:
            mlflow.log_metrics(out_dict)

        return out_dict


def visualize_trajectory(model, forecast_examples, steps, dt, trajectories, a=None, integrator="RK45"):
    """
    Plot and log trajectory forecasts and inputs.

    Parameters
    ----------
    model : torch.nn.Module
        Model used for forecasting.
    forecast_examples : int
        Number of trajectories to visualize.
    steps : int
        Number of time steps to plot.
    dt : float
        Time step between points.
    trajectories : list[tuple]
        List of trajectories where each tuple is (X, u, y, signal).
    a : float or np.ndarray or None, optional
        If provided, enables input noise during forecasting (passed to forecast).
    b : Any, optional
        Unused placeholder for compatibility with callers.
    integrator : str, optional
        Integration method for forecast rollouts.
    """

    t = np.array([dt * s for s in range(steps)])

    X, u, y, signal = np.stack([t[0] for t in trajectories]), np.stack([t[1] for t in trajectories]), np.stack([t[2] for t in trajectories]), [t[3] for t in trajectories]
    X0 = X[:, 0]
    X_pred = forecast(model, X0, u, dt, signal, steps, a=a, integrator=integrator).detach().numpy()
    # log predictions as numpy
    tmpdir = tempfile.TemporaryDirectory()
    np.save(os.path.join(tmpdir.name, "X_pred.npy"), X_pred)
    np.save(os.path.join(tmpdir.name, "X.npy"), X)
    m, M = np.min(X, axis=(0, 1)), np.max(X, axis=(0, 1))
    X_pred = np.clip(X_pred, m, M)
    if mlflow.active_run() is not None:
        mlflow.log_artifact(os.path.join(tmpdir.name, "X_pred.npy"))
        mlflow.log_artifact(os.path.join(tmpdir.name, "X.npy"))
    for j in range(forecast_examples):
        torch.cuda.empty_cache()

        fig, axs = plt.subplots(X.shape[-1] + u.shape[-1], 1, figsize=(30, 10))
        fig.suptitle(f'Trajectory example', fontsize=16)
        print(X.shape)
        for i in range(X.shape[-1]):
            axs[i].plot(t, X[j, :, i], linestyle="dashed", color="red", label=f"X_{i}", linewidth=4)
            axs[i].plot(t, X_pred[j, :, i], linestyle="dotted", color="blue", label=f"X_pred_{i}", linewidth=4)
            axs[i].set_xlabel("Time")
            axs[i].set_ylabel(f"X_{i}")
            axs[i].set_title(f"Trajectory example {j + 1}")
            axs[i].grid(True)
        for k in range(u.shape[-1]):
            col = ["green", "orange", "purple", "black"]
            axs[X.shape[-1] + k].plot(t, u[j, :, k], linestyle="solid", color=col[k], label=f"u_{k}", linewidth=2)
            axs[X.shape[-1] + k].set_xlabel("Time")
            axs[X.shape[-1] + k].set_ylabel(f"u_{k}")
            axs[X.shape[-1] + k].set_title(f"Trajectory example {j + 1}")
            axs[X.shape[-1] + k].grid(True)
        buf = BytesIO()
        fig.savefig(buf, format='png')
        buf.seek(0)  # Rewind the buffer to the beginning

        # Log the figure to MLflow
        if mlflow.active_run() is not None:
            mlflow.log_figure(fig, f"trajectory_example_{j + 1}.png")

        # Close the figure to free up memory
        plt.close(fig)


def scatter(cp, c, name, samples=1000):
    """
    Create and log scatter plots comparing predictions to targets.

    Parameters
    ----------
    cp : np.ndarray
        Predicted values with shape (N, ...).
    c : np.ndarray
        Ground-truth values with shape (N, ...), matching cp.
    name : str
        Base name used in plot titles and MLflow artifact names.
    samples : int, optional
        Number of random samples to plot.
    """

    cp = np.reshape(cp, (cp.shape[0], -1))
    c = np.reshape(c, (c.shape[0], -1))


    assert c.shape == cp.shape

    idx = list(range(len(c)))
    np.random.shuffle(idx)

    c = c[idx[:samples]]
    cp = cp[idx[:samples]]
    for j in range(c.shape[-1]):

        fig, axs = plt.subplots(1, 1, figsize=(10, 10))
        fig.suptitle(f'{name}', fontsize=16)
        axs.scatter(c[:, j], cp[:, j])
        axs.set_xlabel(f"True {name}_{j}")
        axs.set_ylabel(f"Predicted {name}_{j}")
        axs.set_title(f"Scatter plot of {name}_{j}")
        axs.grid(True)

        buf = BytesIO()
        fig.savefig(buf, format='png')
        buf.seek(0)  # Rewind the buffer to the beginning

        # Log the figure to MLflow
        if mlflow.active_run() is not None:
            mlflow.log_figure(fig, f"scatter_{name}_{j}.png")

        # Close the figure to free up memory
        plt.close(fig)

def get_noise_bound(X, sig_noise_ratio):
    """Compute uniform noise bound for a target signal-to-noise ratio."""
    power_signal = np.mean(X ** 2, axis=0)  # N x S
    # SNR (db): 10 * log10(P_signal / P_noise)
    P_noise = 10 ** (-sig_noise_ratio / 10) * power_signal
    # P_noise**2 = (2 a)**2 / 12
    a = np.sqrt(3 * P_noise)  # S
    #noise = np.random.uniform(-a, a, X.shape)
    #X += noise
    return a

def get_uniform_white_noise(X, a):
    """Generate uniform white noise with bound a."""
    noise = np.random.uniform(-a, a, X.shape)
    return noise



if __name__ == "__main__":
    from matplotlib import pyplot as plt
    u = multi_sin_signal(n_signals=2, amplitude=.2)
    t = np.linspace(0, 10, 1000)
    print(u(0))
    plt.plot(t, [u(tt)[0] for tt in t])
    plt.plot(t, [u(tt)[1] for tt in t])
    plt.show()
    print(np.mean(u, axis=0))
