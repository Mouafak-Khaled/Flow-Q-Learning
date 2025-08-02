import io

import imageio.v2 as imageio
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.special import expit, logit
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, WhiteKernel
from scipy.interpolate import griddata


class SuccessRateGaussianProcess:
    def __init__(
        self,
        real_success_rates: pd.DataFrame,
        simulated_success_rates: pd.Series,
        seed: int = 42,
    ):
        self.seed = seed
        self.real_success_rates = real_success_rates
        self.gp_data = self.real_success_rates.loc[simulated_success_rates.index]
        self.gp_data["simulated_success"] = simulated_success_rates
        self.gp_data = self.gp_data.dropna()
        self.frames = []
        self.current_idx = None
        self._fit_gp()

    def ask(self):
        assert self.current_idx is None
        X = self.real_success_rates[["alpha", "step"]].drop(self.gp_data.index).copy()
        X["alpha"] = (np.log10(X["alpha"]) - np.log10(3)) / (
            np.log10(1000) - np.log10(3)
        )
        X["step"] = X["step"] / 1_000_000

        y_pred, sigma = self.gp.predict(X, return_std=True)

        lower = expit(y_pred - sigma)
        upper = expit(y_pred + sigma)
        uncertainty = upper - lower

        self.current_idx = X.index[np.argmax(uncertainty)]
        self.gp_data.loc[self.current_idx] = self.real_success_rates.loc[
            self.current_idx
        ].copy()

        return self.gp_data.loc[self.current_idx]

    def tell(self, sim_success: float) -> None:
        assert self.current_idx is not None
        if sim_success is not None:
            self.gp_data.loc[self.current_idx, "simulated_success"] = sim_success
            self._fit_gp()
        self.current_idx = None

    def get_data(self):
        X = self.real_success_rates[["alpha", "step"]].copy()
        X["alpha"] = (np.log10(X["alpha"]) - np.log10(3)) / (
            np.log10(1000) - np.log10(3)
        )
        X["step"] = X["step"] / 1_000_000
        y_pred, _ = self.gp.predict(X, return_std=True)
        data = self.real_success_rates.copy()
        data["simulated_success"] = expit(y_pred)
        return data

    def plot(self, filename="report/success_rate_gp.gif"):
        imageio.mimsave(filename, self.frames, duration=1.0)

    def _fit_gp(self):
        X = self.gp_data[["alpha", "step"]].copy()
        X["alpha"] = (np.log10(X["alpha"]) - np.log10(3)) / (
            np.log10(1000) - np.log10(3)
        )
        X["step"] = X["step"] / 1_000_000
        eps = 1e-6
        y = logit(self.gp_data["simulated_success"].clip(eps, 1 - eps))

        kernel = RBF(length_scale=1e2, length_scale_bounds=(1e-1, 1e2)) + WhiteKernel(
            noise_level=1e-4, noise_level_bounds=(1e-10, 1)
        )
        self.gp = GaussianProcessRegressor(
            kernel=kernel, n_restarts_optimizer=10, random_state=self.seed
        )
        self.gp.fit(X, y)

        self._save_frame()

    def _save_frame(self):
        alpha_range = np.linspace(np.log10(3), np.log10(1000), 20)
        step_range = np.linspace(20_000, 1_000_000, 50)
        alpha_grid, step_grid = np.meshgrid(alpha_range, step_range)
        X_pred = np.column_stack([alpha_grid.ravel(), step_grid.ravel()])
        X_pred = pd.DataFrame(X_pred, columns=["alpha", "step"])
        X_pred["alpha"] = (X_pred["alpha"] - np.log10(3)) / (
            np.log10(1000) - np.log10(3)
        )
        X_pred["step"] = X_pred["step"] / 1_000_000

        y_pred, sigma = self.gp.predict(X_pred, return_std=True)
        y_pred = y_pred.reshape(alpha_grid.shape)

        lower = expit(y_pred - sigma.reshape(alpha_grid.shape))
        upper = expit(y_pred + sigma.reshape(alpha_grid.shape))
        y_pred = expit(y_pred)
        uncertainty = upper - lower

        fig, axes = plt.subplots(1, 3, figsize=(24, 6))
        ticks = np.linspace(0, 1, 11)

        # Group by (alpha, step) and average the success rates
        avg_success = (
            self.real_success_rates.groupby(["alpha", "step"])["success"].mean().reset_index()
        )
        # Interpolate to match the grid shape
        points = avg_success[["alpha", "step"]].values
        values = avg_success["success"].values
        grid_success = griddata(
            points,
            values,
            (10**alpha_grid, step_grid)
        )

        heatmap = axes[0].contourf(
            10**alpha_grid,
            step_grid,
            grid_success,
            levels=50,
            cmap="viridis",
            vmin=0,
            vmax=1,
        )
        fig.colorbar(heatmap, ax=axes[0], label="Real Success", ticks=ticks)
        axes[0].set_xlabel("alpha")
        axes[0].set_ylabel("step")
        axes[0].set_title("Real Success Rate")
        axes[0].set_xscale("log")
        axes[0].grid(True)

        prediction_contour = axes[1].contourf(
            10**alpha_grid,
            step_grid,
            y_pred,
            levels=50,
            cmap="viridis",
            vmin=0,
            vmax=1,
        )
        fig.colorbar(
            prediction_contour, ax=axes[1], label="Predicted Success", ticks=ticks
        )
        axes[1].set_xlabel("alpha")
        axes[1].set_ylabel("step")
        axes[1].set_title("Predicted Success on Real Environment")
        axes[1].set_xscale("log")
        axes[1].scatter(
            self.gp_data["alpha"],
            self.gp_data["step"],
            c="r",
            marker="+",
            label="Data",
        )
        axes[1].legend()
        axes[1].grid(True)

        uncertainty_contour = axes[2].contourf(
            10**alpha_grid,
            step_grid,
            uncertainty,
            levels=50,
            cmap="viridis",
            vmin=0,
            vmax=1,
        )
        fig.colorbar(
            uncertainty_contour, ax=axes[2], label="Uncertainty (σ)", ticks=ticks
        )
        axes[2].set_xlabel("alpha")
        axes[2].set_ylabel("step")
        axes[2].set_title("Prediction Uncertainty on Simulated Environment")
        axes[2].set_xscale("log")
        axes[2].scatter(
            self.gp_data["alpha"],
            self.gp_data["step"],
            c="r",
            marker="+",
            label="Data",
        )
        axes[2].legend()
        axes[2].grid(True)

        plt.tight_layout()
        buf = io.BytesIO()
        plt.savefig(buf, format="png")
        plt.close()
        buf.seek(0)

        image = imageio.imread(buf)
        self.frames.append(image)
