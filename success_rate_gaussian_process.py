import io

import imageio.v2 as imageio
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.special import expit, logit
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF


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
        self._fit_gp()
        self.frames = []
        self.current_idx = None

    def ask(self):
        assert self.current_idx is None
        X = self.real_success_rates[["alpha", "step"]].drop(self.gp_data.index).copy()
        X["alpha"] = (np.log10(X["alpha"]) - np.log10(3)) / (
            np.log10(1000) - np.log10(3)
        )
        X["step"] = X["step"] / 1_000_000

        _, sigma_real = self.gp_real.predict(X, return_std=True)
        _, sigma_sim = self.gp_sim.predict(X, return_std=True)

        self.current_idx = X.index[np.argmax(sigma_real + sigma_sim)]
        self.gp_data.loc[self.current_idx] = self.real_success_rates.loc[
            self.current_idx
        ].copy()

        return self.gp_data.loc[self.current_idx]

    def tell(self, sim_success):
        assert self.current_idx is not None
        self.gp_data.loc[self.current_idx, "simulated_success"] = sim_success
        self.current_idx = None
        self._fit_gp()

    def get_data(self):
        X = self.real_success_rates[["alpha", "step"]].copy()
        X["alpha"] = (np.log10(X["alpha"]) - np.log10(3)) / (
            np.log10(1000) - np.log10(3)
        )
        X["step"] = X["step"] / 1_000_000
        y_pred, _ = self.gp_sim.predict(X, return_std=True)
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
        y_real = logit(self.gp_data["success"].clip(eps, 1 - eps))
        y_sim = logit(self.gp_data["simulated_success"].clip(eps, 1 - eps))

        kernel = RBF()
        self.gp_real = GaussianProcessRegressor(
            kernel=kernel, n_restarts_optimizer=10, random_state=self.seed
        )
        self.gp_sim = GaussianProcessRegressor(
            kernel=kernel, n_restarts_optimizer=10, random_state=self.seed
        )
        self.gp_real.fit(X, y_real)
        self.gp_sim.fit(X, y_sim)

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

        # Real Pred
        y_pred_real, sigma_real = self.gp_real.predict(X_pred, return_std=True)
        y_pred_real = y_pred_real.reshape(alpha_grid.shape)

        lower_real = expit(y_pred_real - sigma_real.reshape(alpha_grid.shape))
        upper_real = expit(y_pred_real + sigma_real.reshape(alpha_grid.shape))
        y_pred_real = expit(y_pred_real)
        uncertainty_real = upper_real - lower_real

        # Sim Pred
        y_pred_sim, sigma_sim = self.gp_sim.predict(X_pred, return_std=True)
        y_pred_sim = y_pred_sim.reshape(alpha_grid.shape)

        lower_sim = expit(y_pred_sim - sigma_sim.reshape(alpha_grid.shape))
        upper_sim = expit(y_pred_sim + sigma_sim.reshape(alpha_grid.shape))
        y_pred_sim = expit(y_pred_sim)
        uncertainty_sim = upper_sim - lower_sim

        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        ticks = np.linspace(0, 1, 11)

        prediction_contour = axes[0, 0].contourf(
            10**alpha_grid,
            step_grid,
            y_pred_real,
            levels=50,
            cmap="viridis",
            vmin=0,
            vmax=1,
        )
        fig.colorbar(
            prediction_contour, ax=axes[0, 0], label="Predicted Success", ticks=ticks
        )
        axes[0, 0].set_xlabel("alpha")
        axes[0, 0].set_ylabel("step")
        axes[0, 0].set_title("Predicted Success on Real Environment")
        axes[0, 0].set_xscale("log")
        axes[0, 0].scatter(
            self.gp_data["alpha"],
            self.gp_data["step"],
            c="r",
            marker="+",
            label="Data",
        )
        axes[0, 0].legend()
        axes[0, 0].grid(True)

        uncertainty_contour = axes[0, 1].contourf(
            10**alpha_grid,
            step_grid,
            uncertainty_real,
            levels=50,
            cmap="viridis",
            vmin=0,
            vmax=1,
        )
        fig.colorbar(
            uncertainty_contour, ax=axes[0, 1], label="Uncertainty (σ)", ticks=ticks
        )
        axes[0, 1].set_xlabel("alpha")
        axes[0, 1].set_ylabel("step")
        axes[0, 1].set_title("Prediction Uncertainty on Real Environment")
        axes[0, 1].set_xscale("log")
        axes[0, 1].scatter(
            self.gp_data["alpha"],
            self.gp_data["step"],
            c="r",
            marker="+",
            label="Data",
        )
        axes[0, 1].legend()
        axes[0, 1].grid(True)

        prediction_contour = axes[1, 0].contourf(
            10**alpha_grid,
            step_grid,
            y_pred_sim,
            levels=50,
            cmap="viridis",
            vmin=0,
            vmax=1,
        )
        fig.colorbar(
            prediction_contour, ax=axes[1, 0], label="Predicted Success", ticks=ticks
        )
        axes[1, 0].set_xlabel("alpha")
        axes[1, 0].set_ylabel("step")
        axes[1, 0].set_title("Predicted Success on Simulated Environment")
        axes[1, 0].set_xscale("log")
        axes[1, 0].scatter(
            self.gp_data["alpha"],
            self.gp_data["step"],
            c="r",
            marker="+",
            label="Data",
        )
        axes[1, 0].legend()
        axes[1, 0].grid(True)

        uncertainty_contour = axes[1, 1].contourf(
            10**alpha_grid,
            step_grid,
            uncertainty_sim,
            levels=50,
            cmap="viridis",
            vmin=0,
            vmax=1,
        )
        fig.colorbar(
            uncertainty_contour, ax=axes[1, 1], label="Uncertainty (σ)", ticks=ticks
        )
        axes[1, 1].set_xlabel("alpha")
        axes[1, 1].set_ylabel("step")
        axes[1, 1].set_title("Prediction Uncertainty on Simulated Environment")
        axes[1, 1].set_xscale("log")
        axes[1, 1].scatter(
            self.gp_data["alpha"],
            self.gp_data["step"],
            c="r",
            marker="+",
            label="Data",
        )
        axes[1, 1].legend()
        axes[1, 1].grid(True)

        plt.tight_layout()
        buf = io.BytesIO()
        plt.savefig(buf, format="png")
        plt.close()
        buf.seek(0)

        # Read image from buffer and append
        image = imageio.imread(buf)
        self.frames.append(image)
