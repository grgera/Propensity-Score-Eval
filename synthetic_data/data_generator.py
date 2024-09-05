import os
from abc import ABC, abstractmethod

import numpy as np
from scipy.stats import wasserstein_distance
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mean_squared_error


def quadratic_1d(x, max_perf):
    """
    Quadratic performance function that peaks at max_perf.
    f(x) = max(0, 1 - 0.1 * (x - max_perf)^2)
    """
    return np.maximum(0, 1 - 0.08 * (x - max_perf) ** 2)


def gaussian_1d(x, max_perf=2, sigma=1):
    """
    Gaussian performance function peaking at max_perf.
    f(x) = exp(-0.5 * ((x - max_perf) / sigma) ^ 2)
    """
    return np.exp(-0.5 * ((x - max_perf) / sigma) ** 2)


def sigmoid_1d(x, max_perf=2, steepness=1):
    """
    Sigmoid performance function with steepest point at max_perf.
    f(x) = 1 / (1 + exp(-steepness * (x - max_perf)))
    """
    return 1 / (1 + np.exp(-steepness * (x - max_perf)))


def exp_decay_1d(x, max_perf=2, decay_rate=0.5):
    """
    Exponential decay performance function starting from max_perf.
    f(x) = exp(-decay_rate * (x - max_perf))
    """
    return np.minimum(1, np.exp(-decay_rate * x))


AVAILABLE_PERF_MODELS = {
    'quadratic_1d': quadratic_1d,
    'gaussian_1d': gaussian_1d,
    'sigmoid_1d': sigmoid_1d,
    'exp_decay_1d': exp_decay_1d
}


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

    @abstractmethod
    def generate_data(self, **kwargs):
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


class GaussianInputGenerator(DataGenerator):
    def generate_data(self, output_dir=None, n_samples=1000):
        """
        Generate the original and target distributions along with model performance.
        """
        # Sample means and scales from their respective ranges
        mean_range = self.config['mean_range']
        loc_range = self.config['loc_range']

        mu_a = np.random.uniform(mean_range[0], mean_range[1])
        mu_b = np.random.uniform(mean_range[0], mean_range[1])

        loc_a = np.random.uniform(loc_range[0], loc_range[1])
        loc_b = np.random.uniform(loc_range[0], loc_range[1])

        # Sample from original distribution (Normal with mu_a and loc_a)
        data_a = np.random.normal(mu_a, loc_a, size=n_samples)

        # Sample from target distribution (Normal with mu_b and loc_b)
        data_b = np.random.normal(mu_b, loc_b, size=n_samples)

        # Generate model performance M_{\rho, f}: x -> R+
        tmp_model_performance = globals()[self.config['model_performance']]
        model_performance = (lambda x: tmp_model_performance(x, max_perf=mu_a))

        performance_a = model_performance(data_a)
        performance_b = model_performance(data_b)

        # Calculate distribution shift (absolute difference in means)
        distribution_shift = np.abs(mu_a - mu_b)

        # Calculate Earth Mover's Distance (Wasserstein Distance) between the two distributions
        emd = wasserstein_distance(data_a, data_b)
        mse = mean_squared_error([np.mean(performance_a)], [np.mean(performance_b)])

        # Create the Setup instance
        setup = Setup(data_a, data_b, performance_a, performance_b, emd, mse)

        # If plotting is specified, plot and save/show
        if output_dir:
            self.plot_distributions(data_a, data_b, model_performance, setup, output_dir)

        return setup


class ExponentialInputGenerator(DataGenerator):
    def generate_data(self, output_dir=None, n_samples=1000):
        """
        Generate the original and target distributions along with model performance using Exponential distributions.
        """
        # Sample rates from their respective ranges
        rate_range = self.config['rate_range']

        rate_a = np.random.uniform(rate_range[0], rate_range[1])
        rate_b = np.random.uniform(rate_range[0], rate_range[1])

        # Sample from original distribution (Exponential with rate_a)
        data_a = np.random.exponential(scale=1 / rate_a, size=n_samples)

        # Sample from target distribution (Exponential with rate_b)
        data_b = np.random.exponential(scale=1 / rate_b, size=n_samples)

        # Generate model performance M_{\rho, f}: x -> R+
        tmp_model_performance = globals()[self.config['model_performance']]
        model_performance = (lambda x: tmp_model_performance(x, max_perf=1 / rate_a))

        performance_a = model_performance(data_a)
        performance_b = model_performance(data_b)

        # Calculate distribution shift (absolute difference in rates)
        distribution_shift = np.abs(1 / rate_a - 1 / rate_b)

        # Calculate Earth Mover's Distance (Wasserstein Distance) between the two distributions
        emd = wasserstein_distance(data_a, data_b)
        mse = mean_squared_error([np.mean(performance_a)], [np.mean(performance_b)])

        # Create the Setup instance
        setup = Setup(data_a, data_b, performance_a, performance_b, emd, mse)

        # If plotting is specified, plot and save/show
        if output_dir:
            self.plot_distributions(data_a, data_b, model_performance, setup, output_dir)

        return setup


class BetaInputGenerator(DataGenerator):
    def generate_data(self, output_dir=None, n_samples=1000):
        """
        Generate the original and target distributions along with model performance using Beta distributions.
        """
        # Sample alpha and beta parameters from their respective ranges
        alpha_range = self.config['alpha_range']
        beta_range = self.config['beta_range']

        alpha_a = np.random.uniform(alpha_range[0], alpha_range[1])
        beta_a = np.random.uniform(beta_range[0], beta_range[1])

        alpha_b = np.random.uniform(alpha_range[0], alpha_range[1])
        beta_b = np.random.uniform(beta_range[0], beta_range[1])

        # Sample from original distribution (Beta with alpha_a and beta_a)
        data_a = np.random.beta(alpha_a, beta_a, size=n_samples)

        # Sample from target distribution (Beta with alpha_b and beta_b)
        data_b = np.random.beta(alpha_b, beta_b, size=n_samples)

        # Generate model performance M_{\rho, f}: x -> R+
        tmp_model_performance = globals()[self.config['model_performance']]
        model_performance = (lambda x: tmp_model_performance(x, max_perf=alpha_a + beta_a))

        performance_a = model_performance(data_a)
        performance_b = model_performance(data_b)

        # Calculate distribution shift (absolute difference in means)
        mean_a = alpha_a / (alpha_a + beta_a)
        mean_b = alpha_b / (alpha_b + beta_b)
        distribution_shift = np.abs(mean_a - mean_b)

        # Calculate Earth Mover's Distance (Wasserstein Distance) between the two distributions
        emd = wasserstein_distance(data_a, data_b)
        mse = mean_squared_error([np.mean(performance_a)], [np.mean(performance_b)])

        # Create the Setup instance
        setup = Setup(data_a, data_b, performance_a, performance_b, emd, mse)

        # If plotting is specified, plot and save/show
        if output_dir:
            self.plot_distributions(data_a, data_b, model_performance, setup, output_dir)

        return setup


class StudentTInputGenerator(DataGenerator):
    def generate_data(self, output_dir=None, n_samples=1000):
        """
        Generate the original and target distributions along with model performance using Student's t distributions.
        """
        # Sample degrees of freedom from their respective ranges
        df_range = self.config['df_range']

        df_a = np.random.uniform(df_range[0], df_range[1])
        df_b = np.random.uniform(df_range[0], df_range[1])

        # Sample from original distribution (Student's t with df_a)
        data_a = np.random.standard_t(df_a, size=n_samples)

        # Sample from target distribution (Student's t with df_b)
        data_b = np.random.standard_t(df_b, size=n_samples)

        # Generate model performance M_{\rho, f}: x -> R+
        tmp_model_performance = globals()[self.config['model_performance']]
        model_performance = (lambda x: tmp_model_performance(x, max_perf=df_a))

        performance_a = model_performance(data_a)
        performance_b = model_performance(data_b)

        # Calculate distribution shift (absolute difference in degrees of freedom)
        distribution_shift = np.abs(df_a - df_b)

        # Calculate Earth Mover's Distance (Wasserstein Distance) between the two distributions
        emd = wasserstein_distance(data_a, data_b)
        mse = mean_squared_error([np.mean(performance_a)], [np.mean(performance_b)])

        # Create the Setup instance
        setup = Setup(data_a, data_b, performance_a, performance_b, emd, mse)

        # If plotting is specified, plot and save/show
        if output_dir:
            self.plot_distributions(data_a, data_b, model_performance, setup, output_dir)

        return setup


class WeibullInputGenerator(DataGenerator):
    def generate_data(self, output_dir=None, n_samples=1000):
        """
        Generate the original and target distributions along with model performance using Weibull distributions.
        """
        # Sample shape and scale parameters from their respective ranges
        shape_range = self.config['shape_range']
        scale_range = self.config['scale_range']

        shape_a = np.random.uniform(shape_range[0], shape_range[1])
        scale_a = np.random.uniform(scale_range[0], scale_range[1])

        shape_b = np.random.uniform(shape_range[0], shape_range[1])
        scale_b = np.random.uniform(scale_range[0], scale_range[1])

        # Sample from original distribution (Weibull with shape_a and scale_a)
        data_a = np.random.weibull(shape_a, size=n_samples) * scale_a

        # Sample from target distribution (Weibull with shape_b and scale_b)
        data_b = np.random.weibull(shape_b, size=n_samples) * scale_b

        # Generate model performance M_{\rho, f}: x -> R+
        tmp_model_performance = globals()[self.config['model_performance']]
        model_performance = (lambda x: tmp_model_performance(x, max_perf=shape_a))

        performance_a = model_performance(data_a)
        performance_b = model_performance(data_b)

        # Calculate distribution shift (absolute difference in shape parameters)
        distribution_shift = np.abs(shape_a - shape_b)

        # Calculate Earth Mover's Distance (Wasserstein Distance) between the two distributions
        emd = wasserstein_distance(data_a, data_b)
        mse = mean_squared_error([np.mean(performance_a)], [np.mean(performance_b)])

        # Create the Setup instance
        setup = Setup(data_a, data_b, performance_a, performance_b, emd, mse)

        # If plotting is specified, plot and save/show
        if output_dir:
            self.plot_distributions(data_a, data_b, model_performance, setup, output_dir)

        return setup