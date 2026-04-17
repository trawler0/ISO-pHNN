from train import train
from model import *
from utils import Dataset, sample_initial_states, compute_metrics, visualize_trajectory, scatter, \
    get_uniform_white_noise, get_noise_bound
from data import simple_experiment
import torch
import numpy as np
from data import dim_bias_scale_sigs
import mlflow
import argparse
import random
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import os

parser = argparse.ArgumentParser()
parser.add_argument("--name", type=str, default="spring", help="Problem name: spring, ball, motor.")
parser.add_argument("--num_trajectories", type=int, default=16, help="Number of training trajectories.")
parser.add_argument("--num_val_trajectories", type=int, default=100, help="Number of validation trajectories.")
parser.add_argument("--hidden_dim", type=int, default=32, help="Hidden layer width.")
parser.add_argument("--depth", type=int, default=3, help="Number of hidden layers.")
parser.add_argument("--J", type=str, default="default", help="J parametrization: linear, default, default_kan, sigmoid, spring.")
parser.add_argument("--R", type=str, default="default", help="R parametrization: linear, quadratic, default, sigmoid, default_kan.")
parser.add_argument("--G", type=str, default="mlp", help="G parametrization: mlp, kan, linear.")
parser.add_argument("--grad_H", type=str, default="gradient", help="Hamiltonian gradient: gradient, gradient_positive, gradient_kan, linear.")
parser.add_argument("--output-weight", type=float, default=.25, help="Weight for output loss term.")
parser.add_argument("--excitation", type=str, default="mlp", help="Excitation for sigmoid-based J/R: linear or mlp.")
parser.add_argument("--time", type=float, default=10, help="Training simulation time horizon.")
parser.add_argument("--val_time", type=float, default=100, help="Validation simulation time horizon.")
parser.add_argument("--steps", type=int, default=None, help="Training data steps per trajectory; defaults to 100 * time.")
parser.add_argument("--val_steps", type=int, default=None, help="Validation data steps per trajectory; defaults to 100 * val_time.")
parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate.")
parser.add_argument("--epochs", type=int, default=500, help="Number of training epochs.")
parser.add_argument("--criterion", type=str, default="normalized_mse", help="Loss criterion: mse or normalized_mse.")
parser.add_argument("--batch_size", type=int, default=256, help="Batch size.")
parser.add_argument("--val_batch_size", type=int, default=16384, help="validation Batch size.")
parser.add_argument("--seed", type=int, default=1000, help="Random seed.")
parser.add_argument("--repeat", type=int, default=1, help="Concatenate the dataset N times and train for N times fewer epochs.")
parser.add_argument("--weight_decay", type=float, default=1e-2, help="Weight decay for optimizer.")
parser.add_argument("--forecast_examples", type=int, default=20, help="Number of trajectories to visualize.")
parser.add_argument("--forecast_time", type=float, default=100., help="Forecast time horizon.")
parser.add_argument("--forecast_steps", type=int, default=10000, help="Forecast steps.")
parser.add_argument("--run_name", type=str, default=None, help="MLflow run name.")
parser.add_argument("--example", type=str, default=None, help="Optional example tag or id.")
parser.add_argument("--tag", type=str, default=None, help="Optional MLflow tag value.")
parser.add_argument("--dB", type=float, default=None, help="SNR in dB for adding uniform noise; disabled if None.")
parser.add_argument("--baseline", action="store_true", default=False, help="Use baseline MLP model.")
parser.add_argument("--experiment", type=str, default="1", help="MLflow experiment name or id.")
parser.add_argument("--rescale_epochs", type=int, default=1, help="Epoch interval for rescaling (if used).")
parser.add_argument("--no-forecast", action="store_true", default=False, help="Disable forecasting and visualization.")
parser.add_argument("--normalize-u", action="store_true", default=False, help="Enable standardization of u.")
parser.add_argument("--no-normalize-x", action="store_true", default=False, help="Disable standardization of x.")
parser.add_argument("--normalize-y", action="store_true", default=False, help="Enable standardization of y.")
parser.add_argument("--normalize-xdot", action="store_true", default=False, help="Enable standardization of xdot.")
parser.add_argument("--integrator", default="RK45", type=str, help="Forecast integrator: RK45, RK4 or IMPLICIT_MIDPOINT.")
parser.add_argument("--device", type=str, default="cpu", help="Device used for training.")
parser.add_argument("--validation-frequency", type=int, default=None, help="How often to validate the model.")
parser.add_argument("--dataset", type=str, default=None)

args = parser.parse_args()

try:
    experiment = mlflow.get_experiment_by_name(args.experiment)
    if experiment is not None:
        experiment_id = experiment.experiment_id
    else:
        experiment_id = mlflow.create_experiment(args.experiment)
except Exception as e:
    print(f"Error fetching or creating experiment: {e}")


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


set_seed(args.seed)
if args.device == "cpu":
    torch.use_deterministic_algorithms(True)

time = args.time
steps = args.steps if args.steps is not None else int(100 * args.time)
val_steps = args.val_steps if args.val_steps is not None else int(100 * args.val_time)
dt = time / steps

name = args.name
DIM, scale, bias, sigs, amplitude_train, f0_train, amplitude_val, f0_val = dim_bias_scale_sigs(name)

if args.dataset is None:
    generator = simple_experiment(name, time, steps, amplitude_train, f0_train, start_seed=args.seed)
    generator_val = simple_experiment(name, args.val_time, val_steps, amplitude_val, f0_val, start_seed=args.seed+10**6)

    X0_train = sample_initial_states(args.num_trajectories, DIM,
                                        {"identifies": "uniform", "seed": args.seed, "scale": scale, "bias": bias})
    X0_val = sample_initial_states(args.num_val_trajectories, DIM,
                                {"identifies": "uniform", "seed": args.seed + 10**6, "scale": scale, "bias": bias})
    

    X, u, xdot, y, _ = generator.get_data(X0_train)
    X_val, u_val, xdot_val, y_val, _ = generator_val.get_data(X0_val)
else:
    exists = os.path.exists(args.dataset)
    print(args.dataset)
    if exists:
        data = np.load(args.dataset, allow_pickle=True).item()
        X, u, xdot, y = data["X"], data["u"], data["xdot"], data["y"]
        X_val, u_val, xdot_val, y_val = data["X_val"], data["u_val"], data["xdot_val"], data["y_val"]
    else:
        generator = simple_experiment(name, time, steps, amplitude_train, f0_train, start_seed=args.seed)
        generator_val = simple_experiment(name, args.val_time, val_steps, amplitude_val, f0_val, start_seed=args.seed+10**6)

        X0_train = sample_initial_states(args.num_trajectories, DIM,
                                            {"identifies": "uniform", "seed": args.seed, "scale": scale, "bias": bias})
        X0_val = sample_initial_states(args.num_val_trajectories, DIM,
                                        {"identifies": "uniform", "seed": args.seed + 10**6, "scale": scale, "bias": bias})
        

        X, u, xdot, y, _ = generator.get_data(X0_train)
        X_val, u_val, xdot_val, y_val, _ = generator_val.get_data(X0_val)
        os.makedirs(os.path.dirname(args.dataset), exist_ok=True) if os.path.dirname(args.dataset) else None
        np.save(
            args.dataset,
            {
                "X": X,
                "u": u,
                "xdot": xdot,
                "y": y,
                "X_val": X_val,
                "u_val": u_val,
                "xdot_val": xdot_val,
                "y_val": y_val,
            },
            allow_pickle=True,
        )
scaler_X = MinMaxScaler(feature_range=(-1, 1))
scaler_u = StandardScaler()
scaler_xdot = StandardScaler()
scaler_y = StandardScaler()

if not args.no_normalize_x:
    X = scaler_X.fit_transform(X)
if args.normalize_u:
    u = scaler_u.fit_transform(u)
if args.normalize_xdot:
    xdot = scaler_xdot.fit_transform(xdot)
if args.normalize_y:
    y = scaler_y.fit_transform(y)

class Predictor(nn.Module):
    """
    Wrapper that applies input/output standardization around a trained model.

    Parameters
    ----------
    model : torch.nn.Module
        Trained model that returns (xdot, y).
    scaler_X : sklearn.preprocessing.StandardScaler
        Scaler fit on state inputs.
    scaler_u : sklearn.preprocessing.StandardScaler
        Scaler fit on input signals.
    scaler_xdot : sklearn.preprocessing.StandardScaler
        Scaler fit on state derivatives.
    scaler_y : sklearn.preprocessing.StandardScaler
        Scaler fit on outputs.
    """

    def __init__(self, model, scaler_X, scaler_u, scaler_xdot, scaler_y):
        super().__init__()
        self.model = model
        self.scaler_X = scaler_X
        self.scaler_u = scaler_u
        self.scaler_xdot = scaler_xdot
        self.scaler_y = scaler_y

    def forward(self, x, u):
        """
        Run the model with optional normalization and inverse scaling.

        Parameters
        ----------
        x : torch.Tensor
            State tensor of shape (batch, state_dim).
        u : torch.Tensor
            Input tensor of shape (batch, input_dim).

        Returns
        -------
        tuple[torch.Tensor, torch.Tensor]
            (xdot, y) with the same shapes as model outputs, in original scale.
        """
        if not args.no_normalize_x:
            x = x.detach().numpy()
            x = self.scaler_X.transform(x)
        if args.normalize_u:
            u = u.detach().numpy()
            u = self.scaler_u.transform(u)
        x = torch.tensor(x).float()
        u = torch.tensor(u).float()
        xdot, y = self.model(x, u)
        if args.normalize_xdot:
            xdot = xdot.detach().cpu().numpy()
            xdot = torch.tensor(self.scaler_xdot.inverse_transform(xdot))
        if args.normalize_y:
            y = y.detach().cpu().numpy()
            y = torch.tensor(self.scaler_y.inverse_transform(y))
        return xdot, y
        

if args.dB is not None:
    a = get_noise_bound(u, args.dB)
    b = get_noise_bound(y, args.dB)
    u = u + get_uniform_white_noise(u, a)
    y = y + get_uniform_white_noise(y, b)
else:
    a = None
    b = None

X, u, xdot, y = np.concatenate([X] * args.repeat), np.concatenate([u] * args.repeat), np.concatenate(
    [xdot] * args.repeat), np.concatenate([y] * args.repeat)
X, u, xdot, y = torch.tensor(X, dtype=torch.float32), torch.tensor(u, dtype=torch.float32), torch.tensor(xdot,
                                                                                                            dtype=torch.float32), torch.tensor(
    y, dtype=torch.float32)
X_val, u_val, xdot_val, y_val = torch.tensor(X_val, dtype=torch.float32), torch.tensor(u_val,
                                                                                        dtype=torch.float32), torch.tensor(
    xdot_val, dtype=torch.float32), torch.tensor(y_val, dtype=torch.float32)

train_ds = Dataset(X, u, xdot, y)

X_val_scaled = X_val.reshape(-1, X_val.shape[-1])
u_val_scaled = u_val.reshape(-1, u_val.shape[-1])
y_val_scaled = y_val.reshape(-1, y_val.shape[-1])
xdot_val_scaled = xdot_val.reshape(-1, xdot_val.shape[-1])

if not args.no_normalize_x:
    X_val_scaled = scaler_X.transform(X_val_scaled)
if args.normalize_u:
    u_val_scaled = scaler_u.transform(u_val_scaled)
if args.normalize_y:
    y_val_scaled = scaler_y.transform(y_val_scaled)
if args.normalize_xdot:
    xdot_val_scaled = scaler_xdot.transform(xdot_val_scaled)

val_ds = Dataset(X_val_scaled, u_val_scaled, xdot_val_scaled, y_val_scaled)

models = []
train_loader = torch.utils.data.DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
val_loader = torch.utils.data.DataLoader(val_ds, batch_size=len(val_ds) if args.val_batch_size is None else args.val_batch_size, shuffle=False)

if args.baseline:
    model = Baseline(DIM, 2 * args.hidden_dim, sigs, args.depth)
else:
    model = PHNNModel(DIM, args.hidden_dim, args.depth, J=args.J, R=args.R, grad_H=args.grad_H, G=args.G,
                        excitation=args.excitation, u_dim=sigs)
print("Parameters:", sum(p.numel() for p in model.parameters()))
train(model, train_loader, val_loader, args.epochs, args.output_weight, loss_fn=args.criterion, device=args.device, weight_decay=args.weight_decay, validation_frequency=args.validation_frequency)
model = Predictor(model, scaler_X, scaler_u, scaler_xdot, scaler_y) # for eval


with mlflow.start_run(run_name=args.run_name, experiment_id=experiment_id):
    mlflow.set_tag("name", args.name)
    mlflow.set_tag("training_size", args.num_trajectories)
    mlflow.set_tag("time", args.time)
    if args.tag is not None:
        mlflow.set_tag("tag", args.tag)

    mlflow.log_params(vars(args))
    mlflow.pytorch.autolog()
    models.append(model)
    mlflow.pytorch.log_model(model, "model")
    metrics = compute_metrics(models, dt, X_val, u_val, xdot_val, y_val, integrator=args.integrator)
    print(metrics)

    if not args.no_forecast:
        steps = args.forecast_steps
        time = args.forecast_time
        dt = time / steps
        forecast_examples = args.forecast_examples
        t = np.array([dt * s for s in range(steps)])

        generator_val = simple_experiment(name, time, steps, amplitude_val, f0_val, start_seed=args.seed+10**6)
        X0_val = sample_initial_states(forecast_examples, DIM,
                                {"identifies": "uniform", "seed": args.seed + + 10**6, "scale": scale, "bias": bias})
        _, _, _, _, trajectories_val = generator_val.get_data(X0_val)
        visualize_trajectory(model, forecast_examples, steps, dt, trajectories_val, a=None, integrator=args.integrator)

        if not args.baseline:
            X = np.concatenate([xu[0] for xu in trajectories_val])
            G_true = np.stack([generator_val.G(x) for x in X])
            R_true = np.stack([generator_val.R(x) for x in X])
            J_true = np.stack([generator_val.J(x) for x in X])
            grad_H_true = np.stack([generator_val.grad_H(x) for x in X])

            X = torch.tensor(X).float()
            J, R = model.model.reparam(X)
            G = model.model.G(X)
            grad_H = model.model.grad_H(X)

            J = np.array(J.detach())
            R = np.array(R.detach())
            G = np.array(G.detach())
            grad_H = np.array(grad_H.detach())

            if G_true.shape[1] == 1:
                G_true = np.concatenate([G_true] * G.shape[-1], -1)
            if R_true.shape[1] == 1:
                R_true = np.concatenate([R_true] * R.shape[-1], -1)
            if J_true.shape[1] == 1:
                J_true = np.concatenate([J_true] * J.shape[-1], -1)

            scatter(grad_H, grad_H_true, "grad_H")
            scatter(G, G_true, "G")
            scatter(R, R_true, "R")
            scatter(J, J_true, "S")
