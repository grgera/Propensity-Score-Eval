from abc import ABC, abstractmethod
import numpy as np
from matplotlib import pyplot as plt
from scipy.stats import gaussian_kde
from sklearn.calibration import CalibratedClassifierCV, calibration_curve
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KernelDensity
from sklearn.preprocessing import StandardScaler
from scipy.stats import multivariate_normal
import torch
import torch.nn as nn
import torch.optim as optim
from hep_ml import reweight
from propensity_reweighting.src.reweight import PropensityReweighter
from propensity_reweighting.src.preprocessing import ProScoreVectorizer


class Reweigher(ABC):
    def __init__(self):
        """
        """
        ## TODO: redefine without hot value
        config={"model": "FacebookAI/xlm-roberta-base", 'tokenizer': "FacebookAI/xlm-roberta-base"}
        self.vectorizer = ProScoreVectorizer(config)

    @abstractmethod
    def learn_weights(self, data_a, data_b):
        """
        Learns the reweighting scheme from dataset A to make it look like dataset B.
        """
        pass

    @abstractmethod
    def reweigh(self, data_a):
        """
        Returns reweighting factors for dataset A.
        """
        pass


class DummyReweigher(Reweigher):
    def learn_weights(self, data_a, data_b):
        pass

    def reweigh(self, data_a):
        return np.ones(len(data_a))


class ImportanceSamplingReweigher(Reweigher):
    def __init__(self):
        self.mu_a = None
        self.mu_b = None
        self.cov_a = None
        self.cov_b = None
        self.trained = False

    def learn_weights(self, data_a, data_b):
        # Estimate the mean vector and covariance matrix for the original distribution (data_a)
        self.mu_a = np.mean(data_a, axis=0)  # Mean vector for data_a
        self.cov_a = np.cov(data_a, rowvar=False)  # Covariance matrix for data_a

        # Estimate the mean vector and covariance matrix for the target distribution (data_b)
        self.mu_b = np.mean(data_b, axis=0)  # Mean vector for data_b
        self.cov_b = np.cov(data_b, rowvar=False)  # Covariance matrix for data_b

        self.trained = True

    def reweigh(self, data_a):
        if not self.trained:
            raise RuntimeError("Weights have not been learned. Call 'learn_weights' first.")

        # Compute the probability density of data_a under the original distribution N(mu_a, cov_a)
        p_a = multivariate_normal.pdf(data_a, mean=self.mu_a, cov=self.cov_a)

        # Compute the probability density of data_a under the target distribution N(mu_b, cov_b)
        p_b = multivariate_normal.pdf(data_a, mean=self.mu_b, cov=self.cov_b)

        # The importance weights are the ratio p_b / p_a
        weights = p_b / p_a
        weights = np.maximum(weights, 1e-10)

        return weights


class ClassifierReweigher(Reweigher):
    def __init__(self):
        """
        Initializes the reweighter. It does not require distribution parameters.
        """
        self.classifier = LogisticRegression()
        self.scaler = StandardScaler()
        self.trained = False

    def learn_weights(self, data_a, data_b):
        """
        Learns the reweighting scheme by training a classifier to distinguish between data_a and data_b.
        """
        # Create labels for the data
        labels_a = np.zeros(len(data_a))
        labels_b = np.ones(len(data_b))

        # Combine data and labels
        combined_data = np.vstack([data_a, data_b])
        combined_labels = np.concatenate([labels_a, labels_b])

        # Standardize the data
        # Flatten data to ensure proper scaling
        combined_data_flat = combined_data.reshape(-1, combined_data.shape[
            -1]) if combined_data.ndim > 1 else combined_data.reshape(-1, 1)
        combined_data_scaled = self.scaler.fit_transform(combined_data_flat)

        # Train the classifier
        self.classifier.fit(combined_data_scaled, combined_labels)
        self.trained = True

    def reweigh(self, data_a):
        """
        Returns the reweighting factors for dataset A based on the classifier's probability estimates.
        """
        if not self.trained:
            raise RuntimeError("Weights have not been learned. Call 'learn_weights' first.")

        # Standardize data_a
        # Flatten data_a to ensure proper scaling
        data_a_flat = data_a.reshape(-1, data_a.shape[-1]) if data_a.ndim > 1 else data_a.reshape(-1, 1)
        data_a_standardized = self.scaler.transform(data_a_flat)

        # Get predicted probabilities for data_a being from class 1 (data_b)
        prob_b = self.classifier.predict_proba(data_a_standardized)[:, 1]

        # Weights are proportional to the probability of being from class B
        return prob_b


class CalibratedClassifierReweigher(Reweigher):
    def __init__(self):
        """
        Initializes the reweighter. It does not require distribution parameters.
        """
        self.classifier = LogisticRegression()  # Base classifier
        self.scaler = StandardScaler()  # Data standardizer
        self.calibrated_classifier = None  # Placeholder for calibrated classifier
        self.trained = False

    def learn_weights(self, data_a, data_b, plot_calibration_curve=False):
        """
        Learns the reweighting scheme by training a classifier to distinguish between data_a and data_b.
        Optionally plots the calibration curve if requested.
        """
        # Create labels for the data
        labels_a = np.zeros(len(data_a))  # Class 0 for data_a
        labels_b = np.ones(len(data_b))  # Class 1 for data_b

        # Combine data and labels
        combined_data = np.vstack([data_a, data_b])
        combined_labels = np.concatenate([labels_a, labels_b])

        # Standardize the data
        # Flatten data to ensure proper scaling
        combined_data_flat = combined_data.reshape(-1, combined_data.shape[
            -1]) if combined_data.ndim > 1 else combined_data.reshape(-1, 1)
        combined_data_scaled = self.scaler.fit_transform(combined_data_flat)

        # Train the base classifier (logistic regression)
        self.classifier.fit(combined_data_scaled, combined_labels)

        # Calibrate the classifier using Platt scaling (sigmoid)
        self.calibrated_classifier = CalibratedClassifierCV(estimator=self.classifier, method='sigmoid',
                                                            cv='prefit')
        self.calibrated_classifier.fit(combined_data_scaled, combined_labels)
        self.trained = True

        # Optionally plot the calibration curve
        if plot_calibration_curve:
            self._plot_calibration_curve(combined_data_scaled, combined_labels)

    def reweigh(self, data_a):
        """
        Returns the reweighting factors for dataset A based on the calibrated classifier's probability estimates.
        """
        if not self.trained:
            raise RuntimeError("Weights have not been learned. Call 'learn_weights' first.")

        # Standardize data_a
        # Flatten data_a to ensure proper scaling
        data_a_flat = data_a.reshape(-1, data_a.shape[-1]) if data_a.ndim > 1 else data_a.reshape(-1, 1)
        data_a_standardized = self.scaler.transform(data_a_flat)

        # Get predicted probabilities for data_a being from class 1 (data_b) using the calibrated classifier
        prob_b = self.calibrated_classifier.predict_proba(data_a_standardized)[:, 1]

        # Weights are proportional to the probability of being from class B
        return prob_b

    def _plot_calibration_curve(self, data, labels):
        """
        Plots the calibration curve (reliability diagram) for the calibrated classifier.
        """
        prob_true, prob_pred = calibration_curve(labels, self.calibrated_classifier.predict_proba(data)[:, 1],
                                                 n_bins=10)

        plt.plot(prob_pred, prob_true, marker='o', label='Calibrated Classifier')
        plt.plot([0, 1], [0, 1], linestyle='--', label='Perfect Calibration')
        plt.xlabel('Predicted Probability')
        plt.ylabel('True Probability')
        plt.title('Calibration Curve (Reliability Diagram)')
        plt.legend()
        plt.show()


class KDEReweigher(Reweigher):
    def __init__(self):
        self.kde_a = None
        self.kde_b = None
        self.trained = False

    def learn_weights(self, data_a, data_b):
        # Transpose the data to fit gaussian_kde's expected input shape (features as rows)
        self.kde_a = gaussian_kde(data_a.T)  # KDE for original distribution (data_a)
        self.kde_b = gaussian_kde(data_b.T)  # KDE for target distribution (data_b)
        self.trained = True

    def reweigh(self, data_a):
        if not self.trained:
            raise RuntimeError("Weights have not been learned. Call 'learn_weights' first.")

        # Transpose data_a for KDE evaluation
        data_a_T = data_a.T

        # Evaluate the KDEs on the transposed data
        p_a = self.kde_a.evaluate(data_a_T)  # Probability under original distribution
        p_b = self.kde_b.evaluate(data_a_T)  # Probability under target distribution

        # Avoid division by zero
        p_a = np.maximum(p_a, 1e-10)

        # Weights are the ratio of the KDEs
        weights = p_b / p_a

        weights = np.maximum(weights, 1e-10)

        return weights


class DensityRatioReweigher(Reweigher):
    def __init__(self):
        self.kde_a = None
        self.kde_b = None
        self.trained = False

    def learn_weights(self, data_a, data_b):
        # Fit the kernel density estimator for the original distribution (data_a)
        self.kde_a = KernelDensity(kernel='gaussian', bandwidth=1.0).fit(data_a)

        # Fit the kernel density estimator for the target distribution (data_b)
        self.kde_b = KernelDensity(kernel='gaussian', bandwidth=1.0).fit(data_b)

        self.trained = True

    def reweigh(self, data_a):
        if not self.trained:
            raise RuntimeError("Weights have not been learned. Call 'learn_weights' first.")

        # Evaluate the log density of data_a under both KDEs
        log_density_a = self.kde_a.score_samples(data_a)
        log_density_b = self.kde_b.score_samples(data_a)

        # The importance weights are the exponentiated density ratio
        weights = np.exp(log_density_b - log_density_a)
        weights = np.maximum(weights, 1e-10)
        return weights


class Discriminator(nn.Module):
    def __init__(self, input_dim):
        super(Discriminator, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(input_dim, 10),
            nn.ReLU(),
            nn.Linear(10, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.fc(x)


class AdversarialReweigher(Reweigher):
    def __init__(self):
        self.discriminator = None
        self.optimizer = None
        self.trained = False

    def learn_weights(self, data_a, data_b):
        # Pass input_dim to Discriminator
        input_dim = data_a.shape[1]

        self.discriminator = Discriminator(input_dim)
        self.optimizer = optim.Adam(self.discriminator.parameters(), lr=0.001)

        # Convert data to tensors, ensure data is 2D with shape (batch_size, input_dim)
        data_a = torch.tensor(data_a, dtype=torch.float32)
        data_b = torch.tensor(data_b, dtype=torch.float32)

        if data_a.ndim == 1:
            data_a = data_a.view(-1, 1)
            data_b = data_b.view(-1, 1)

        # Training loop
        for epoch in range(100):
            self.optimizer.zero_grad()

            # Discriminator prediction
            pred_a = self.discriminator(data_a)
            pred_b = self.discriminator(data_b)

            # Loss functions (MSE Loss)
            loss_a = torch.mean((pred_a - 0) ** 2)
            loss_b = torch.mean((pred_b - 1) ** 2)

            loss = loss_a + loss_b
            loss.backward()
            self.optimizer.step()

        self.trained = True

    def reweigh(self, data_a):
        if not self.trained:
            raise RuntimeError("Weights have not been learned. Call 'learn_weights' first.")

        # Convert data to tensor, ensure 2D shape
        data_a = torch.tensor(data_a, dtype=torch.float32)
        if data_a.ndim == 1:
            data_a = data_a.view(-1, 1)

        # Get weights from the discriminator's output
        with torch.no_grad():
            weights = self.discriminator(data_a).numpy().flatten()

        return weights


class FoldingReweighter:
    def __init__(self):
        self.reweighter = None

    def learn_weights(self, data_a, data_b):
        reweighter_base = reweight.GBReweighter(gb_args={'subsample': 0.9})

        self.reweighter = reweight.FoldingReweighter(reweighter_base, n_folds=10, verbose=False)
        self.reweighter.fit(data_a, data_b)

        self.trained = False

    def reweigh(self, data_a):
        weights_pred = self.reweighter.predict_weights(data_a)
        return weights_pred
    
class ModifiedHepReweighter(Reweigher):
    def __init__(self, config):
        self.config = config
        self.pr = PropensityReweighter(config)
        self.grid_values = {'n_estimators': [30, 100], 
                   'learning_rate':[0.01, 0.001, 0.009],
                   'max_depth': [5, 10],
                   'min_samples_leaf': [1, 10, 100]}
        
        self.best_predictor = None

    def learn_weights(self, data_a, data_b):
        self.best_predictor = self.pr.fit_gridsearch(data_a, data_b, self.grid_values, vectorized=True)
        self.trained = True

    def reweigh(self, data_a):
        if not self.trained:
            raise RuntimeError("Weights have not been learned. Call 'learn_weights' first.")

        weights = self.best_predictor.predict(data_a, vetorized=True)

        return weights


AVAILABLE_REWEIGHERS = {
    'dummy': DummyReweigher(),
    'importance': ImportanceSamplingReweigher(),
    'classifier': ClassifierReweigher(),
    'calib-classifier': CalibratedClassifierReweigher(),
    'kde': KDEReweigher(),
    'density-ratio': DensityRatioReweigher(),
    'adversarial': AdversarialReweigher(),
    'folding-reweighter': FoldingReweighter(),
    'modifiedhep-reweighter': ModifiedHepReweighter
}
