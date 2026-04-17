import mlflow
import numpy as np
from matplotlib import pyplot as plt
import os
from data import simple_experiment, dim_bias_scale_sigs
from utils import sample_initial_states
import torch
from scipy.optimize import curve_fit
from matplotlib.ticker import MaxNLocator, ScalarFormatter, NullFormatter

plt.rcParams.update({
    "figure.dpi": 150,
    "savefig.dpi": 300,
    "font.size": 18,
    "axes.labelsize": 24,
    "axes.titlesize": 26,
    "legend.fontsize": 16,
    "xtick.labelsize": 16,
    "ytick.labelsize": 16,
    "axes.spines.top": False,
    "axes.spines.right": False,
})

# test test
print("Creating results")

if not os.path.exists("results"):
    os.makedirs("results")

colors = {
    "True": "black",
    "baseline": "blue",
    "default": "red",
    "prior": "green",
    "kan": "purple",
    "shallow": "green",
    "shorter": "purple",
    "wrong": "turquoise",
    "generic": "magenta",
    "quadratic": "black",
    "no_noise": "red",
    "noise25": "blue",
    "noise20": "green",
    "noise35": "purple",
}
line_styles = {
    "True": "solid",
    "baseline": "--",
    "default": "--",
    "prior": "--",
    "shallow": "--",
    "shorter": "--",
    "kan": "--",
    "noise35": "dotted",
    "noise30": "solid",
    "noise25": "dotted",
    "noise20": "dashed",
    "no_noise": "--",
    "wrong": "--",
    "generic": "--",
}
thickness = {
    "True": 5,
    "baseline": 3,
    "default": 3,
    "prior": 3,
    "shallow": 3,
    "shorter": 3,
    "noise35": 3,
    "noise30": 3,
    "noise25": 3,
    "noise20": 3,
    "no_noise": 3,
    "kan": 3
}


def plot_scaling(name, i):
    print(f"scaling_{name}_{i}")
    experiment = mlflow.get_experiment_by_name(f"scaling_{name}_{i}")
    runs = mlflow.search_runs(experiment.experiment_id)
    baseline = {}
    default = {}
    prior = {}
    kan = {}
    wrong = {}
    for i, run in runs.iterrows():
        run_id = run["run_id"]
        run = mlflow.get_run(run_id)
        params = run.data.params
        metrics = run.data.metrics
        run_name = params["run_name"]
        mae_rel = metrics["mae_rel"]
        mse_rel = metrics["mse_rel"]
        num_traj = int(params["num_trajectories"])
        if run_name.startswith(f"{name}_baseline"):
            baseline.setdefault(num_traj, []).append(mae_rel)
        elif run_name.startswith(f"{name}_default"):
            default.setdefault(num_traj, []).append(mae_rel)
        elif run_name.startswith(f"{name}_prior"):
            prior.setdefault(num_traj, []).append(mae_rel)
        elif run_name.startswith(f"{name}_kan"):
            kan.setdefault(num_traj, []).append(mae_rel)
        elif run_name.startswith(f"{name}_wrong"):
            wrong.setdefault(num_traj, []).append(mae_rel)
    assert default.keys() == baseline.keys() == prior.keys()
    sorted_keys = sorted(default.keys())
    mae_rel_baseline = [np.min(baseline[k]) for k in sorted_keys]
    mae_rel_default = [np.min(default[k]) for k in sorted_keys]
    mae_rel_prior = [np.min(prior[k]) for k in sorted_keys]
    mae_rel_kan = [np.min(kan[k]) for k in sorted_keys]
    if wrong:
        mae_rel_wrong = [np.min(wrong[k]) for k in sorted_keys]

    fig, ax = plt.subplots(figsize=(14, 8.5))
    fig.patch.set_facecolor("white")
    ax.set_facecolor("#f5f5f5")

    # Cleaner axis appearance
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    # Common plotting style
    plot_kwargs = dict(
        marker="o",
        markersize=9,
        linewidth=3,
        markeredgewidth=1.5,
        alpha=0.95,
        linestyle="--"
    )

    ax.plot(
        sorted_keys, mae_rel_baseline,
        label="Baseline",
        color=colors["baseline"],
        **plot_kwargs
    )
    ax.plot(
        sorted_keys, mae_rel_default,
        label="pH",
        color=colors["default"],
        **plot_kwargs
    )
    ax.plot(
        sorted_keys, mae_rel_prior,
        label="pH prior",
        color=colors["prior"],
        **plot_kwargs
    )
    ax.plot(
        sorted_keys, mae_rel_kan,
        label="pH KAN",
        color=colors["kan"],
        **plot_kwargs
    )

    if wrong:
        ax.plot(
            sorted_keys, mae_rel_wrong,
            label="pH wrong prior",
            color=colors["wrong"],
            **plot_kwargs
        )

    # Log scales
    ax.set_xscale("log")
    ax.set_yscale("log")

    # Force x-axis ticks to be exactly your training set sizes
    ax.set_xticks(sorted_keys)
    ax.set_xticklabels([str(k) for k in sorted_keys])
    ax.xaxis.set_major_formatter(ScalarFormatter())
    ax.xaxis.set_minor_formatter(NullFormatter())

    # Labels
    ax.set_xlabel("Number of trajectories used for training", fontsize=22, labelpad=12)
    ax.set_ylabel("NMAE", fontsize=24, labelpad=12)

    # Ticks
    ax.tick_params(axis="both", which="major", labelsize=18, length=6, width=1.2)
    ax.tick_params(axis="both", which="minor", length=3, width=0.8)

    # Grid: lighter and less distracting
    ax.grid(True, which="major", linestyle="--", linewidth=0.8, alpha=0.5)
    ax.grid(True, which="minor", linestyle=":", linewidth=0.5, alpha=0.25)

    # Slight padding so lines do not touch the frame
    ax.margins(x=0.03, y=0.08)

    # Legend
    leg = ax.legend(
        fontsize=18,
        loc="best",
        frameon=True,
        fancybox=True,
        framealpha=0.95,
        borderpad=0.6
    )
    leg.get_frame().set_edgecolor("0.85")

    plt.tight_layout()
    plt.savefig(
        os.path.join("results", f"{name}_scaling.eps"),
        format="eps",
        dpi=300,
        bbox_inches="tight",
        pad_inches=0.05,
    )


def prior_comparison(i):
    experiment = mlflow.get_experiment_by_name(f"prior_comparison_{i}")
    all_runs = mlflow.search_runs(experiment.experiment_id)
    for name in ["spring"]:
        runs = all_runs[all_runs["tags.name"] == name]
        wrong = {}
        generic = {}
        default = {}
        for i, run in runs.iterrows():
            run_id = run["run_id"]
            run = mlflow.get_run(run_id)
            params = run.data.params
            metrics = run.data.metrics
            run_name = params["run_name"]
            print(run_name)
            mae_rel = metrics["mae_rel"]
            mse_rel = metrics["mse_rel"]
            num_traj = params["num_trajectories"]
            if run_name.startswith(f"wrong"):
                wrong.setdefault(num_traj, []).append(mae_rel)
            elif run_name.startswith(f"R_generic"):
                generic.setdefault(num_traj, []).append(mae_rel)
            elif run_name.startswith(f"default"):
                default.setdefault(num_traj, []).append(mae_rel)

        print(wrong.keys(), generic.keys()), default.keys()
        assert wrong.keys() == generic.keys() == default.keys()
        sorted_keys = sorted(wrong.keys())
        print(sorted_keys)
        mae_rel_wrong = [np.min(wrong[k]) for k in sorted_keys]
        mae_rel_generic = [np.min(generic[k]) for k in sorted_keys]
        mae_rel_default = [np.min(default[k]) for k in sorted_keys]

        # Create the figure and axis with an optimized size
        fig, ax = plt.subplots(figsize=(12, 8))  # Adjusted from (30, 30) to (12, 8)

        # Plotting the data with markers and increased line width for better visibility
        ax.plot(sorted_keys, mae_rel_wrong, label="pH fully linear prior", marker='o', linestyle=":", color=colors["wrong"])
        ax.plot(sorted_keys, mae_rel_default, label="pH", marker='s', linestyle=":", color=colors["default"])
        ax.plot(sorted_keys, mae_rel_generic, label="pH MLP prior", marker='^', linestyle=":", color=colors["prior"])

        # Set log scales
        ax.set_xscale("log")
        ax.set_yscale("log")

        # Set title and labels with increased font sizes
        ax.set_xlabel("Number of Trajectories Used for Training", fontsize=16)
        ax.set_ylabel("NMAE", fontsize=16)  # Added y-label for completeness

        # Customize tick parameters for better readability
        ax.tick_params(axis='both', which='major', labelsize=14)
        ax.tick_params(axis='both', which='minor', labelsize=12)

        # Enable and customize grid
        ax.grid(True, which="both", ls="--", linewidth=0.5)

        # Customize legend with larger font size and appropriate placement
        ax.legend(fontsize=16, loc='best')  # 'best' lets matplotlib decide the optimal location

        # Optional: Tight layout for better spacing
        plt.tight_layout()
        #plt.show()

        plt.savefig(os.path.join("results", f"{name}_prior_comparison.eps"),  format="eps", dpi=300)


def recipe(i):
    experiment = mlflow.get_experiment_by_name(f"recipe_{i}")
    runs = mlflow.search_runs(experiment.experiment_id)
    preds = {}
    for i, run in runs.iterrows():
        run_id = run["run_id"]
        run = mlflow.get_run(run_id)
        params = run.data.params
        run_name = params["run_name"]
        if run_name == "kan":
            continue
        artifacts_path = mlflow.artifacts.download_artifacts(run_id=run_id)

        # Example: If artifact is a file, handle it
        for root, _, files in os.walk(artifacts_path):
            for file in files:
                artifact_file_path = os.path.join(root, file)
                if artifact_file_path.endswith("X_pred.npy"):
                    X_pred = np.load(artifact_file_path)
                    X_true = np.load(artifact_file_path.replace("X_pred", "X"))
                    preds[run_name] = X_pred
    n = X_pred.shape[-1]
    idx = 9
    N=1500
    fig, ax = plt.subplots(3, 1, figsize=(30, 15))
    time = np.arange(len(X_true[idx, :N])) * 0.01
    ax[0].plot(time, X_true[idx, :N, 0], label="True", color=colors["True"], linestyle=line_styles["True"],
            linewidth=thickness["True"])
    ax[1].plot(time, X_true[idx, :N, 1], label="True", color=colors["True"], linestyle=line_styles["True"],
            linewidth=thickness["True"])
    ax[2].plot(time, X_true[idx, :N, 2], label="True", color=colors["True"], linestyle=line_styles["True"],
            linewidth=thickness["True"])
    for run_name, X_pred in preds.items():
        def rename(name):
            if name == "shallow":
                return "pH shallower"
            elif name == "shorter":
                return "pH shorter"
            elif name == "default":
                return "pH default"
            elif name == "kan":
                return "kan"

        ax[0].plot(time, X_pred[idx, :N, 0], label=rename(run_name), color=colors[run_name],
                linestyle=line_styles[run_name], linewidth=thickness[run_name])
        ax[1].plot(time, X_pred[idx, :N, 1], label=rename(run_name), color=colors[run_name],
                linestyle=line_styles[run_name], linewidth=thickness[run_name])
        ax[2].plot(time, X_pred[idx, :N, 2], label=rename(run_name), color=colors[run_name],
                linestyle=line_styles[run_name], linewidth=thickness[run_name])
        
    for j in range(3):
        ax[j].axvline(x=10., color='purple', linestyle='--', linewidth=2)
        ax[j].grid()
        ax[j].axvspan(
            0,
            10,
            color="#f0f0f0",
            alpha=0.1,  # Adjust alpha for transparency
            label='training duration'  # Optional: label for the legend
        )
    ax[2].set_xlabel("Time [s]", fontsize=32)
    ax[0].set_ylabel(f"vertical position $x_1$", fontsize=32)
    ax[1].set_ylabel(f"Momentum $x_2$", fontsize=32)
    ax[2].set_ylabel(f"magnetic flux $x_3$", fontsize=32)
    ax[2].tick_params(axis='both', which='major', labelsize=32)
    ax[2].legend(fontsize=32)
    plt.tight_layout()
    #plt.show()
    plt.savefig(os.path.join("results", "recipe.eps"),  format="eps", dpi=300)


def compare(i):
    experiment = mlflow.get_experiment_by_name(f"compare_{i}")
    runs = mlflow.search_runs(experiment.experiment_id)
    preds = {}
    for i, run in runs.iterrows():
        run_id = run["run_id"]
        run = mlflow.get_run(run_id)
        params = run.data.params
        run_name = params["run_name"]
        artifacts_path = mlflow.artifacts.download_artifacts(run_id=run_id)

        # Example: If artifact is a file, handle it
        for root, _, files in os.walk(artifacts_path):
            for file in files:
                artifact_file_path = os.path.join(root, file)
                if artifact_file_path.endswith("X_pred.npy"):
                    X_pred = np.load(artifact_file_path)
                    X_true = np.load(artifact_file_path.replace("X_pred", "X"))
                    preds[run_name] = X_pred

    idx = 10
    state = 1
    n = X_pred.shape[-1]
    fig, ax = plt.subplots(1, 1, figsize=(30, 15))
    ax.set_xlabel("Time [s]", fontsize=32)
    ax.set_ylabel("Momentum $x_2$", fontsize=32)
    t = np.arange(2000) * 0.01
    ax.plot(t, X_true[idx, :2000, state], label="True", color=colors["True"], linestyle=line_styles["True"],
            linewidth=thickness["True"])
    for run_name, X_pred in preds.items():
        def rename(name):
            if name == "prior":
                return "pH prior"
            elif name == "default":
                return "pH"
            elif name == "baseline":
                return "Baseline"

        ax.plot(t, X_pred[idx, :2000, state], label=rename(run_name), color=colors[run_name],
                linestyle=line_styles[run_name], linewidth=thickness[run_name])
    ax.axvline(x=10., color='purple', linestyle='--', linewidth=2)
    ax.tick_params(axis='both', which='major', labelsize=32)
    ax.axvspan(
        0,
        10,
        color='gray',
        alpha=0.1,  # Adjust alpha for transparency
        label='training duration'  # Optional: label for the legend
    )
    ax.grid()
    ax.legend(fontsize=32)
    #plt.show()
    plt.savefig(os.path.join("results", "compare.eps"),  format="eps", dpi=300)


def prior_vs_default(i):
    experiment = mlflow.get_experiment_by_name(f"prior_vs_default_{i}")
    runs = mlflow.search_runs(experiment.experiment_id)
    models = {}
    for i, run in runs.iterrows():
        run_id = run["run_id"]
        run = mlflow.get_run(run_id)
        params = run.data.params
        run_name = params["run_name"]
        name = params["name"]
        artifacts_path = mlflow.artifacts.download_artifacts(run_id=run_id)

        # Example: If artifact is a file, handle it
        for root, _, files in os.walk(artifacts_path):
            for file in files:
                artifact_file_path = os.path.join(root, file)
                if file == "model.pth" or file.endswith(".pt"):
                    # Assuming the model was saved with a specific extension
                    model_uri = f"runs:/{run_id}/model"
                    try:
                        model = mlflow.pytorch.load_model(model_uri)
                        models[run_name] = model
                    except Exception as e:
                        print(f"Failed to load model for run {run_id}: {e}")
        dim, scale, bias, sigs, amplitude_train, f0_train, amplitude_val, f0_val = dim_bias_scale_sigs(name)
        X = sample_initial_states(500, dim, {"identifies": "uniform", "seed": 41})
        generator = simple_experiment(name, 10, 1000, amplitude_val, f0_val, start_seed=1+10**6)
        grad_H = np.stack([generator.grad_H(x) for x in X], axis=0)
        R = np.stack([generator.R(x) for i, x in enumerate(X)], axis=0)
        H_preds = {
        }
        R_preds = {
        }
        for run_name, model in models.items():
            X = torch.tensor(X).float()
            H_pred = model.model.grad_H(X)
            R_pred = model.model.reparam(X)[1]
            H_preds[run_name] = H_pred.detach().numpy()
            R_preds[run_name] = R_pred.detach().numpy()
    

        def rename(name):
            if name == "default":
                return "pH"
            elif name == "prior":
                return "pH prior"
            return name

        def styled_scatter_compare(ax, x, y, label, color, xlabel, ylabel, title=None):
            r = np.corrcoef(x, y)[0, 1]
            print(f"Correlation for {label}: {r:.4f}")

            ax.scatter(
                x, y,
                s=90,
                alpha=0.65,
                color=color,
                edgecolor="white",
                linewidth=0.7,
                label=f"{label}  ($r={r:.3f}$)"
            )

            # reference line y = x
            lo = min(np.min(x), np.min(y))
            hi = max(np.max(x), np.max(y))
            pad = 0.05 * (hi - lo if hi > lo else 1.0)
            lo, hi = lo - pad, hi + pad
            ax.plot([lo, hi], [lo, hi], linestyle="--", linewidth=2, color="black", alpha=0.8)

            ax.set_xlim(lo, hi)
            ax.set_ylim(lo, hi)
            ax.set_aspect("equal", adjustable="box")

            ax.set_xlabel(xlabel)
            ax.set_ylabel(ylabel)
            if title is not None:
                ax.set_title(title, pad=14)

            ax.grid(True, alpha=0.25, linewidth=0.8)
            ax.xaxis.set_major_locator(MaxNLocator(nbins=5))
            ax.yaxis.set_major_locator(MaxNLocator(nbins=5))

            leg = ax.legend(frameon=True, fancybox=True, framealpha=0.9)
            leg.get_frame().set_edgecolor("0.85")


        fig, ax = plt.subplots(figsize=(12, 12), constrained_layout=True)
        ax.set_facecolor("#f0f0f0")

        idx = 3
        for run_name, H_pred in H_preds.items():
            styled_scatter_compare(
                ax=ax,
                x=grad_H[:, idx],
                y=H_pred[:, idx],
                label=rename(run_name),
                color=colors[run_name],
                xlabel=f"True $\partial_4 H(x)$",
                ylabel=f"Identified $\partial_4 H(x)$",
            )

        fig.savefig(os.path.join("results", "compare_H.eps"),  format="eps", dpi=300, bbox_inches="tight")
        plt.close(fig)
        fig, ax = plt.subplots(figsize=(12, 12), constrained_layout=True)
        ax.set_facecolor("#f0f0f0")
        
        idx1 = 1
        idx2 = 1
        for run_name, R_pred in R_preds.items():
            styled_scatter_compare(
                ax=ax,
                x=R[:, idx1, idx2],
                y=R_pred[:, idx1, idx2],
                label=rename(run_name),
                color=colors[run_name],
                xlabel=r"True $R_{22}(x)$",
                ylabel=r"Identified $R_{22}(x)$",
            )

        fig.savefig(os.path.join("results", "compare_R_22.eps"),  format="eps", dpi=300, bbox_inches="tight")
        plt.close(fig)



def prior_vs_default_ball(i):
    experiment = mlflow.get_experiment_by_name(f"prior_vs_default_ball_{i}")
    runs = mlflow.search_runs(experiment.experiment_id)
    models = {}
    for i, run in runs.iterrows():
        run_id = run["run_id"]
        run = mlflow.get_run(run_id)
        params = run.data.params
        run_name = params["run_name"]
        name = params["name"]
        artifacts_path = mlflow.artifacts.download_artifacts(run_id=run_id)

        # Example: If artifact is a file, handle it
        for root, _, files in os.walk(artifacts_path):
            for file in files:
                artifact_file_path = os.path.join(root, file)
                if file == "model.pth" or file.endswith(".pt"):
                    # Assuming the model was saved with a specific extension
                    model_uri = f"runs:/{run_id}/model"
                    try:
                        model = mlflow.pytorch.load_model(model_uri)
                        models[run_name] = model
                    except Exception as e:
                        print(f"Failed to load model for run {run_id}: {e}")
        dim, scale, bias, sigs, amplitude_train, f0_train, amplitude_val, f0_val = dim_bias_scale_sigs(name)
        X = sample_initial_states(500, dim, {"identifies": "uniform", "seed": 41})
        generator = simple_experiment(name, 10, 1000, amplitude_val, f0_val, start_seed=1+10**6)
        grad_H = np.stack([generator.grad_H(x) for x in X], axis=0)
        R = np.stack([generator.R(x) for i, x in enumerate(X)], axis=0)
        H_preds = {
        }
        R_preds = {
        }
        for run_name, model in models.items():
            X = torch.tensor(X).float()
            H_pred = model.model.grad_H(X)
            R_pred = model.model.reparam(X)[1]
            H_preds[run_name] = H_pred.detach().numpy()
            R_preds[run_name] = R_pred.detach().numpy()


        def rename(name):
            if name == "default":
                return "pH"
            elif name == "prior":
                return "pH prior"
            return name

        def styled_scatter_compare(ax, x, y, label, color, xlabel, ylabel, title=None):
            r = np.corrcoef(x, y)[0, 1]
            print(f"Correlation for {label}: {r:.4f}")

            ax.scatter(
                x, y,
                s=90,
                alpha=0.65,
                color=color,
                edgecolor="white",
                linewidth=0.7,
                label=f"{label}  ($r={r:.3f}$)"
            )

            # reference line y = x
            lo = min(np.min(x), np.min(y))
            hi = max(np.max(x), np.max(y))
            pad = 0.05 * (hi - lo if hi > lo else 1.0)
            lo, hi = lo - pad, hi + pad
            ax.plot([lo, hi], [lo, hi], linestyle="--", linewidth=2, color="black", alpha=0.8)

            ax.set_xlim(lo, hi)
            ax.set_ylim(lo, hi)
            ax.set_aspect("equal", adjustable="box")

            ax.set_xlabel(xlabel)
            ax.set_ylabel(ylabel)
            if title is not None:
                ax.set_title(title, pad=14)

            ax.grid(True, alpha=0.25, linewidth=0.8)
            ax.xaxis.set_major_locator(MaxNLocator(nbins=5))
            ax.yaxis.set_major_locator(MaxNLocator(nbins=5))

            leg = ax.legend(frameon=True, fancybox=True, framealpha=0.9)
            leg.get_frame().set_edgecolor("0.85")


        fig, ax = plt.subplots(figsize=(12, 12), constrained_layout=True)
        ax.set_facecolor("#f0f0f0")

    idx1 = 1
    idx2 = 1
    for run_name, R_pred in R_preds.items():
        styled_scatter_compare(
            ax=ax,
            x=R[:, idx1, idx2],
            y=R_pred[:, idx1, idx2],
            label=rename(run_name),
            color=colors[run_name],
            xlabel=r"True $R_{22}(x)$",
            ylabel=r"Identified $R_{22}(x)$",
        )

    fig.savefig(os.path.join("results", "compare_ball_R.eps"),  format="eps", dpi=300, bbox_inches="tight")
    plt.close(fig)
    fig, ax = plt.subplots(figsize=(12, 12), constrained_layout=True)
    ax.set_facecolor("#f0f0f0")
    
    idx = 1
    for run_name, H_pred in H_preds.items():
        styled_scatter_compare(
            ax=ax,
            x=grad_H[:, idx],
            y=H_pred[:, idx],
            label=rename(run_name),
            color=colors[run_name],
            xlabel=f"True $\partial_2 H(x)$",
            ylabel=f"Identified $\partial_2 H(x)$",
        )

    fig.savefig(os.path.join("results", "compare_ball_H.eps"),  format="eps", dpi=300, bbox_inches="tight")
    plt.close(fig)



def prior_vs_default_motor(i):
    experiment = mlflow.get_experiment_by_name(f"prior_vs_default_motor_{i}")
    runs = mlflow.search_runs(experiment.experiment_id)
    models = {}
    for i, run in runs.iterrows():
        run_id = run["run_id"]
        run = mlflow.get_run(run_id)
        params = run.data.params
        run_name = params["run_name"]
        name = params["name"]
        artifacts_path = mlflow.artifacts.download_artifacts(run_id=run_id)

        # Example: If artifact is a file, handle it
        for root, _, files in os.walk(artifacts_path):
            for file in files:
                artifact_file_path = os.path.join(root, file)
                if file == "model.pth" or file.endswith(".pt"):
                    # Assuming the model was saved with a specific extension
                    model_uri = f"runs:/{run_id}/model"
                    try:
                        model = mlflow.pytorch.load_model(model_uri)
                        models[run_name] = model
                    except Exception as e:
                        print(f"Failed to load model for run {run_id}: {e}")
        dim, scale, bias, sigs, amplitude_train, f0_train, amplitude_val, f0_val = dim_bias_scale_sigs(name)
        X = sample_initial_states(500, dim, {"identifies": "uniform", "seed": 41})
        generator = simple_experiment(name, 10, 1000, amplitude_val, f0_val, start_seed=1+10**6)
        grad_H = np.stack([generator.grad_H(x) for x in X], axis=0)
        J = np.stack([generator.J(x) for i, x in enumerate(X)], axis=0)
        H_preds = {
        }
        J_preds = {
        }
        for run_name, model in models.items():
            X = torch.tensor(X).float()
            H_pred = model.model.grad_H(X)
            J_pred = model.model.reparam(X)[0]
            H_preds[run_name] = H_pred.detach().numpy()
            J_preds[run_name] = J_pred.detach().numpy()


        def rename(name):
            if name == "default":
                return "pH"
            elif name == "prior":
                return "pH prior"
            return name

        def styled_scatter_compare(ax, x, y, label, color, xlabel, ylabel, title=None):
            r = np.corrcoef(x, y)[0, 1]
            print(f"Correlation for {label}: {r:.4f}")

            ax.scatter(
                x, y,
                s=90,
                alpha=0.65,
                color=color,
                edgecolor="white",
                linewidth=0.7,
                label=f"{label}  ($r={r:.3f}$)"
            )

            # reference line y = x
            lo = min(np.min(x), np.min(y))
            hi = max(np.max(x), np.max(y))
            pad = 0.05 * (hi - lo if hi > lo else 1.0)
            lo, hi = lo - pad, hi + pad
            ax.plot([lo, hi], [lo, hi], linestyle="--", linewidth=2, color="black", alpha=0.8)

            ax.set_xlim(lo, hi)
            ax.set_ylim(lo, hi)
            ax.set_aspect("equal", adjustable="box")

            ax.set_xlabel(xlabel)
            ax.set_ylabel(ylabel)
            if title is not None:
                ax.set_title(title, pad=14)

            ax.grid(True, alpha=0.25, linewidth=0.8)
            ax.xaxis.set_major_locator(MaxNLocator(nbins=5))
            ax.yaxis.set_major_locator(MaxNLocator(nbins=5))

            leg = ax.legend(frameon=True, fancybox=True, framealpha=0.9)
            leg.get_frame().set_edgecolor("0.85")


        fig, ax = plt.subplots(figsize=(12, 12), constrained_layout=True)
        ax.set_facecolor("#f0f0f0")

    idx1 = 1
    idx2 = 2
    for run_name, J_pred in J_preds.items():
        styled_scatter_compare(
            ax=ax,
            x=J[:, idx1, idx2],
            y=J_pred[:, idx1, idx2],
            label=rename(run_name),
            color=colors[run_name],
            xlabel=r"True $J_{31}(x)$",
            ylabel=r"Identified $J_{31}(x)$",
        )

    fig.savefig(os.path.join("results", "compare_motor_J.eps"),  format="eps", dpi=300, bbox_inches="tight")
    plt.close(fig)
    fig, ax = plt.subplots(figsize=(12, 12), constrained_layout=True)
    ax.set_facecolor("#f0f0f0")
    
    idx = 1
    for run_name, H_pred in H_preds.items():
        styled_scatter_compare(
            ax=ax,
            x=grad_H[:, idx],
            y=H_pred[:, idx],
            label=rename(run_name),
            color=colors[run_name],
            xlabel=f"True $\partial_2 H(x)$",
            ylabel=f"Identified $\partial_2 H(x)$",
        )

    fig.savefig(os.path.join("results", "compare_motor_H.eps"),  format="eps", dpi=300, bbox_inches="tight")
    plt.close(fig)



def noise(i):
    experiment = mlflow.get_experiment_by_name(f"noise_{i}")
    print(experiment)
    runs = mlflow.search_runs(experiment.experiment_id)
    preds = {}
    for i, run in runs.iterrows():
        run_id = run["run_id"]
        run = mlflow.get_run(run_id)
        params = run.data.params
        run_name = params["run_name"]
        artifacts_path = mlflow.artifacts.download_artifacts(run_id=run_id)

        # Example: If artifact is a file, handle it
        for root, _, files in os.walk(artifacts_path):
            for file in files:
                artifact_file_path = os.path.join(root, file)
                if artifact_file_path.endswith("X_pred.npy"):
                    X_pred = np.load(artifact_file_path)
                    X_true = np.load(artifact_file_path.replace("X_pred", "X"))
                    preds[run_name] = X_pred
    idx = 12
    n = X_pred.shape[-1]
    fig, ax = plt.subplots(3, 1, figsize=(30, 15))
    ax[2].set_xlabel("Time [s]", fontsize=32)
    ax[0].set_ylabel("Vertical position $x_1$", fontsize=32)
    ax[1].set_ylabel("Momentum $x_2$", fontsize=32)
    ax[2].set_ylabel("Magnetic flux $x_3$", fontsize=32)
    n = 0
    N = 2000
    t = (np.arange(N-n) + n) * 0.01
    for j in range(3):
        #ax[j].tick_params(axis='both', which='major', labelsize=32)
        ax[j].plot(t, X_true[idx, n:N, j], label="True", color=colors["True"], linestyle=line_styles["True"],
                linewidth=thickness["True"])
        for run_name in ["noise25", "noise20", "no_noise"]:
            X_pred = preds[run_name]
            print(run_name)

            def rename(name):
                if name == "noise20":
                    return "20 dB"
                elif name == "noise25":
                    return "25 dB"
                elif name == "noise30":
                    return "30 dB"
                elif name == "noise35":
                    return "35 dB"
                elif name == "no_noise":
                    return "No noise"

            ax[j].plot(t, X_pred[idx, n:N, j], label=rename(run_name), color=colors[run_name],
                    linestyle=line_styles[run_name], linewidth=thickness[run_name])
        if N >= 1000:
            x_start = 0
            x_end = 10.0 
            ax[j].axvspan(
                x_start,
                x_end,
                color="#f0f0f0",
                alpha=0.1,
                label='training duration'
            )

            ax[j].axvline(
                x=10.0,
                color='purple',
                linestyle='--',
                linewidth=2,
            )
        ax[0].legend(fontsize=32)
        ax[j].grid()
        #plt.show()
        plt.tight_layout()
    plt.savefig(os.path.join("results", f"noise_{N}.eps"),  format="eps", dpi=300)


os.makedirs("results", exist_ok=True)
#plot_scaling("spring", "paper")
#plot_scaling("ball", "paper")
#plot_scaling("motor", "paper")
recipe("f")
noise("f")
prior_vs_default("f")
prior_vs_default_ball("b")
prior_vs_default_motor("c")