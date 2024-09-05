from abc import ABC, abstractmethod
import numpy as np
from scipy.optimize import linprog
from scipy.stats import norm, gaussian_kde
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KernelDensity
from sklearn.preprocessing import StandardScaler
import torch
import torch.nn as nn
import torch.optim as optim


class Reweigher(ABC):
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
        self.loc_a = None
        self.loc_b = None
        self.trained = False

    def learn_weights(self, data_a, data_b):
        # Estimate the parameters (mean and standard deviation) for original distribution (data_a)
        self.mu_a = np.mean(data_a)
        self.loc_a = np.std(data_a)

        # Estimate the parameters (mean and standard deviation) for target distribution (data_b)
        self.mu_b = np.mean(data_b)
        self.loc_b = np.std(data_b)

        self.trained = True

    def reweigh(self, data_a):
        if not self.trained:
            raise RuntimeError("Weights have not been learned. Call 'learn_weights' first.")

        # Compute the probability density of data_a under the original distribution N(mu_a, loc_a)
        p_a = norm.pdf(data_a, loc=self.mu_a, scale=self.loc_a)

        # Compute the probability density of data_a under the target distribution N(mu_b, loc_b)
        p_b = norm.pdf(data_a, loc=self.mu_b, scale=self.loc_b)

        # The importance weights are the ratio p_b / p_a
        weights = p_b / p_a

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
        combined_data = np.concatenate([data_a, data_b])
        combined_labels = np.concatenate([labels_a, labels_b])

        # Standardize the data
        combined_data = self.scaler.fit_transform(combined_data.reshape(-1, 1))

        # Train the classifier
        self.classifier.fit(combined_data, combined_labels)
        self.trained = True

    def reweigh(self, data_a):
        """
        Returns the reweighting factors for dataset A based on the classifier's probability estimates.
        """
        if not self.trained:
            raise RuntimeError("Weights have not been learned. Call 'learn_weights' first.")

        # Standardize data_a
        data_a_standardized = self.scaler.transform(data_a.reshape(-1, 1))

        # Get predicted probabilities for data_a being from class 1 (data_b)
        prob_b = self.classifier.predict_proba(data_a_standardized)[:, 1]

        # Weights are proportional to the probability of being from class B
        return prob_b


class KDEReweigher(Reweigher):
    def __init__(self):
        self.kde_a = None
        self.kde_b = None
        self.trained = False

    def learn_weights(self, data_a, data_b):
        self.kde_a = gaussian_kde(data_a)
        self.kde_b = gaussian_kde(data_b)
        self.trained = True

    def reweigh(self, data_a):
        if not self.trained:
            raise RuntimeError("Weights have not been learned. Call 'learn_weights' first.")

        # Evaluate the KDEs
        p_a = self.kde_a.evaluate(data_a)
        p_b = self.kde_b.evaluate(data_a)

        # Avoid division by zero
        p_a = np.maximum(p_a, 1e-10)

        # Weights are the ratio of the KDEs
        weights = p_b / p_a
        return weights


class OptimalTransportReweigher(Reweigher):
    def __init__(self):
        self.weights = None
        self.trained = False

    def learn_weights(self, data_a, data_b):
        # Sort the data
        data_a_sorted = np.sort(data_a)
        data_b_sorted = np.sort(data_b)

        n = len(data_a_sorted)
        m = len(data_b_sorted)

        # Create the cost matrix
        cost_matrix = np.abs(data_a_sorted[:, np.newaxis] - data_b_sorted)

        # Define the objective function
        c = np.ones(n)

        # Define the equality constraints
        # We need to define constraints that ensure that the total weights sum up to 1
        # and that weights are non-negative.
        A_eq = np.vstack([
            np.ones(n),  # Sum of weights should be 1
            np.eye(n)  # Each weight should be non-negative
        ])
        b_eq = np.array([
            1.0,  # Total sum of weights is 1
            *np.zeros(n)  # Each weight is non-negative
        ])

        # Bounds for each weight
        bounds = [(0, None)] * n

        # Solve the linear programming problem
        result = linprog(c, A_eq=A_eq, b_eq=b_eq, bounds=bounds, method='highs')

        if result.success:
            self.weights = result.x
            self.trained = True
        else:
            raise ValueError("Linear programming failed to find a solution.")

    def reweigh(self, data_a):
        if not self.trained:
            raise RuntimeError("Weights have not been learned. Call 'learn_weights' first.")

        # Interpolate weights for the given data_a
        weights = np.interp(data_a, np.sort(data_a), self.weights)
        return weights


class DensityRatioReweigher(Reweigher):
    def __init__(self):
        self.kde_a = None
        self.kde_b = None
        self.trained = False

    def learn_weights(self, data_a, data_b):
        self.kde_a = KernelDensity(kernel='gaussian', bandwidth=1.0).fit(data_a.reshape(-1, 1))
        self.kde_b = KernelDensity(kernel='gaussian', bandwidth=1.0).fit(data_b.reshape(-1, 1))
        self.trained = True

    def reweigh(self, data_a):
        if not self.trained:
            raise RuntimeError("Weights have not been learned. Call 'learn_weights' first.")

        log_density_a = self.kde_a.score_samples(data_a.reshape(-1, 1))
        log_density_b = self.kde_b.score_samples(data_a.reshape(-1, 1))

        # The importance weights are the exponentiated density ratio
        weights = np.exp(log_density_b - log_density_a)
        return weights


class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(1, 10),
            nn.ReLU(),
            nn.Linear(10, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.fc(x)


class AdversarialReweigher(Reweigher):
    def __init__(self):
        self.discriminator = Discriminator()
        self.optimizer = optim.Adam(self.discriminator.parameters(), lr=0.001)
        self.trained = False

    def learn_weights(self, data_a, data_b):
        data_a = torch.tensor(data_a, dtype=torch.float32).view(-1, 1)
        data_b = torch.tensor(data_b, dtype=torch.float32).view(-1, 1)

        # Training loop
        for epoch in range(100):
            self.optimizer.zero_grad()

            # Discriminator prediction
            pred_a = self.discriminator(data_a)
            pred_b = self.discriminator(data_b)

            # Loss functions
            loss_a = torch.mean((pred_a - 0) ** 2)
            loss_b = torch.mean((pred_b - 1) ** 2)

            loss = loss_a + loss_b
            loss.backward()
            self.optimizer.step()

        self.trained = True

    def reweigh(self, data_a):
        if not self.trained:
            raise RuntimeError("Weights have not been learned. Call 'learn_weights' first.")

        # Get weights from the discriminator's output
        data_a = torch.tensor(data_a, dtype=torch.float32).view(-1, 1)
        with torch.no_grad():
            weights = self.discriminator(data_a).numpy().flatten()

        return weights


AVAILABLE_REWEIGHERS = {
    'dummy': DummyReweigher(),
    'importance': ImportanceSamplingReweigher(),
    'classifier': ClassifierReweigher(),
    'kde': KDEReweigher(),
    'OT': OptimalTransportReweigher(),
    'density-ratio': DensityRatioReweigher(),
    'adversarial': AdversarialReweigher()
}
