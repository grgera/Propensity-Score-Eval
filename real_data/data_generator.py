import os
from abc import ABC, abstractmethod

import numpy as np
from scipy import stats
from scipy.stats import wasserstein_distance, wasserstein_distance_nd, gaussian_kde
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.metrics import mean_squared_error
from propensity_reweighting.src.preprocessing import ProScoreVectorizer

from .utils import compute_geom_loss


def performance_from_column(data, column):
    """
    In real data performance is already computed, we need to extract it by name of column.
    """
    return data[column].tolist()


class Setup:
    def __init__(self, data_a, data_b, performance_a, performance_b, emd, mse):
        """
        A class to hold the generated setup.
        """
        self.data_a = data_a
        self.data_b = data_b
        self.performance_a = performance_a
        self.performance_b = performance_b
        self.emd = emd  # Earth Mover's Distance
        self.mse = mse


class DataGenerator(ABC):
    def __init__(self, config):
        """
        Initializes the DataGenerator with configuration.
        Config specifies original and target distribution parameters and performance function.
        """
        self.config = config
        self.vectorizer = ProScoreVectorizer(config["vectorizer"])

    @abstractmethod
    def get_data(self, **kwargs):
        pass

    def plot_distributions(self, data_a, data_b, model_performance, setup, output_dir):
        """
        Plot the KDE of data_a and data_b, as well as the model performance function.
        Save or display the plot based on the config.
        """
        # Handle the plot ID and check max setups
        existing_plots = [int(fi.split('-')[-1].split('.')[0]) for fi in os.listdir(output_dir) if fi.endswith('.pdf')]
        if len(existing_plots) >= self.config.get('max_setup_plots', 2):
            return
        else:
            new_id = 0
            if len(existing_plots) > 0:
                new_id = max(existing_plots) + 1
            save_file = os.path.join(output_dir, f"setup_config-{new_id}.pdf")

        # Create figure and aesthetic setup
        plt.figure(figsize=(10, 6))
        sns.set(style="whitegrid")

        # Plot KDEs for data_a and data_b
        sns.kdeplot(data_a, color='tab:blue', label='Density A (Original)', fill=True, alpha=0.35, linewidth=0.5)
        sns.kdeplot(data_b, color='tab:green', label='Density B (Target)', fill=True, alpha=0.35, linewidth=0.5)

        # Overlay the model performance function
        x_range = np.linspace(min(min(data_a), min(data_b)) - 1, max(max(data_a), max(data_b)) + 1, 100)
        y_range = model_performance(x_range)
        plt.plot(x_range, y_range, label='Model Performance', color='tab:red', linestyle='--', alpha=0.7, linewidth=2)

        # Add text for mean performances
        plt.text(
            0.05, 0.95,
            f"Performance A: {np.mean(setup.performance_a):.4f}\nPerformance B: {np.mean(setup.performance_b):.4f}",
            transform=plt.gca().transAxes, fontsize=12, verticalalignment='top', bbox=dict(facecolor='white', alpha=0.7)
        )

        # Add titles and labels
        plt.title(f'Distributions and Model Performance\n EMD: {setup.emd:.4f}',
                  fontsize=14)
        plt.xlabel('Input Space', fontsize=12)
        plt.ylabel('Performance (and density)', fontsize=12)

        # Remove top and right spines for a cleaner look
        sns.despine()

        # Customize legend (no frame)
        plt.legend(frameon=False, fontsize=10)

        # Save or show the plot
        if save_file:
            plt.savefig(save_file, bbox_inches='tight')
        else:
            plt.show()

class MultivariateDataGenerator(DataGenerator, ABC):
    def plot_distributions(self, data_a, data_b, model_performance, setup, output_dir):
        """
        Plot the 2D data_a and data_b (or reduce to 2D via PCA), and overlay the model performance function.
        Save or display the plot based on the config.
        """
        # Handle the plot ID and check max setups
        existing_plots = [int(fi.split('-')[-1].split('.')[0]) for fi in os.listdir(output_dir) if fi.endswith('.pdf')]
        if len(existing_plots) >= self.config.get('max_setup_plots', 2):
            return
        else:
            new_id = 0
            if len(existing_plots) > 0:
                new_id = max(existing_plots) + 1
            save_file = os.path.join(output_dir, f"setup_config-{new_id}.pdf")

        if data_a.shape[1] == 1:
            return DataGenerator.plot_distributions(self, data_a, data_b, model_performance, setup, output_dir)
        
        # Reduce data to 2D if it is higher dimensional
        if data_a.shape[1] > 2:
            pca = PCA(n_components=2)
            data_a = pca.fit_transform(data_a)
            data_b = pca.fit_transform(data_b)

        # Create figure and aesthetic setup
        plt.figure(figsize=(10, 6))
        sns.set(style="whitegrid")

        # Plot scatter points for data_a and data_b
        plt.scatter(data_a[:, 0], data_a[:, 1], color='tab:blue', label='Data A (Original)', alpha=0.4, edgecolor='k',
                    s=30)
        plt.scatter(data_b[:, 0], data_b[:, 1], color='tab:green', label='Data B (Target)', alpha=0.4, edgecolor='k',
                    s=30)

        # Estimate density using gaussian_kde for both data sets
        kde_a = gaussian_kde(data_a.T)
        kde_b = gaussian_kde(data_b.T)

        # Create a meshgrid for density plotting
        x_min, x_max = min(data_a[:, 0].min(), data_b[:, 0].min()) - 1, max(data_a[:, 0].max(), data_b[:, 0].max()) + 1
        y_min, y_max = min(data_a[:, 1].min(), data_b[:, 1].min()) - 1, max(data_a[:, 1].max(), data_b[:, 1].max()) + 1
        x_grid, y_grid = np.meshgrid(np.linspace(x_min, x_max, 100), np.linspace(y_min, y_max, 100))
        positions = np.vstack([x_grid.ravel(), y_grid.ravel()])

        # Calculate densities for data_a and data_b
        density_a = np.reshape(kde_a(positions).T, x_grid.shape)
        density_b = np.reshape(kde_b(positions).T, x_grid.shape)

        # Plot density contours
        plt.contour(x_grid, y_grid, density_a, cmap="Blues", alpha=0.6, levels=5)
        plt.contour(x_grid, y_grid, density_b, cmap="Greens", alpha=0.6, levels=5)

        # Overlay the model performance function (as heatmap over the input space)
        x_flat = np.linspace(x_min, x_max, 100)
        y_flat = np.linspace(y_min, y_max, 100)
        X, Y = np.meshgrid(x_flat, y_flat)
        positions_flat = np.vstack([X.ravel(), Y.ravel()]).T

        # We apply the performance model on 2D (so it is not exact depiction of reality)
        tmp_model_performance = globals()[self.config['model_performance']]
        performance_vals = tmp_model_performance(positions_flat, max_perf=np.mean(data_a, axis=0))
        performance_vals = performance_vals.reshape(X.shape)

        plt.contour(X, Y, performance_vals, cmap="Reds", alpha=0.5, levels=10)

        # Add text for mean performances
        plt.text(
            0.05, 0.95,
            f"Performance A: {np.mean(setup.performance_a):.4f}\nPerformance B: {np.mean(setup.performance_b):.4f}",
            transform=plt.gca().transAxes, fontsize=12, verticalalignment='top', bbox=dict(facecolor='white', alpha=0.8)
        )

        # Add titles and labels
        plt.title(f'Distributions and Model Performance (2D)\n EMD: {setup.emd:.4f}', fontsize=14)
        plt.xlabel('Component 1', fontsize=12)
        plt.ylabel('Component 2', fontsize=12)

        # Remove top and right spines for a cleaner look
        sns.despine()

        # Customize legend (no frame)
        plt.legend(frameon=False, fontsize=10)

        # Save or show the plot
        if save_file:
            plt.savefig(save_file, bbox_inches='tight')
        else:
            plt.show()

class WmtInputGenerator(MultivariateDataGenerator):
    def get_data(self, original_theme, target_theme, path_to_data, source, output_dir=None):
        """
        Get the original and target distributions along with model performance from certain collection.
        """
        full_data = pd.read_csv(path_to_data)
        data_orig = full_data[full_data["theme"] == original_theme]
        data_targ = full_data[full_data["theme"] == target_theme]

        data_a = self.vectorizer.vectorize_texts(data_orig[source].tolist())
        data_b = self.vectorizer.vectorize_texts(data_targ[source].tolist())

        column_for_performance = self.config['model_performance']
        performance_a = performance_from_column(data_orig, column_for_performance)
        performance_b = performance_from_column(data_targ, column_for_performance)

        # EMD and MSE
        emd = compute_geom_loss(data_a, data_b)
        # emd = wasserstein_distance_nd(data_a, data_b)
        mse = mean_squared_error([np.mean(performance_a)], [np.mean(performance_b)])

        setup = Setup(data_a, data_b, performance_a, performance_b, emd, mse)

        if output_dir:
            self.plot_distributions(data_a, data_b, performance_from_column, setup, output_dir)

        return setup
    
class ParserInputGenerator(MultivariateDataGenerator):
    def get_data(self, original_theme, target_theme, path_to_data, source, output_dir=None):
        """
        Get the original and target distributions along with model performance from certain collection.
        """
        full_data = pd.read_csv(path_to_data, sep="\t")
        data_orig = full_data[full_data["treebank"] == original_theme]
        data_targ = full_data[full_data["treebank"] == target_theme]

        data_a = self.vectorizer.vectorize_texts(data_orig[source].tolist())
        data_b = self.vectorizer.vectorize_texts(data_targ[source].tolist())

        column_for_performance = self.config['model_performance']
        performance_a = performance_from_column(data_orig, column_for_performance)
        performance_b = performance_from_column(data_targ, column_for_performance)

        # EMD and MSE
        emd = compute_geom_loss(data_a, data_b)
        # emd = wasserstein_distance_nd(data_a, data_b)
        mse = mean_squared_error([np.mean(performance_a)], [np.mean(performance_b)])

        setup = Setup(data_a, data_b, performance_a, performance_b, emd, mse)

        if output_dir:
            self.plot_distributions(data_a, data_b, performance_from_column, setup, output_dir)

        return setup
    
class QaInputGenerator(MultivariateDataGenerator):
    def get_data(self, original_theme, target_theme, path_to_data, source, output_dir=None):
        """
        Get the original and target distributions along with model performance from certain collection.
        """
        full_data = pd.read_pickle(path_to_data)
        data_orig = full_data[full_data["Answer Type"] == original_theme]
        data_targ = full_data[full_data["Answer Type"] == target_theme]

        data_a = self.vectorizer.vectorize_texts(data_orig[source].tolist())
        data_b = self.vectorizer.vectorize_texts(data_targ[source].tolist())

        column_for_performance = self.config['model_performance']
        performance_a = performance_from_column(data_orig, column_for_performance)
        performance_b = performance_from_column(data_targ, column_for_performance)

        # EMD and MSE
        emd = compute_geom_loss(data_a, data_b)
        # emd = wasserstein_distance_nd(data_a, data_b)
        mse = mean_squared_error([np.mean(performance_a)], [np.mean(performance_b)])

        setup = Setup(data_a, data_b, performance_a, performance_b, emd, mse)

        if output_dir:
            self.plot_distributions(data_a, data_b, performance_from_column, setup, output_dir)

        return setup