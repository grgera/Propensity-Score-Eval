from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import mean_squared_error
import numpy as np
import pandas as pd

def replace_nan_inf(lst):
    for i in range(len(lst)):
        if np.isnan(lst[i]):
            lst[i] = 1 if lst[i] > 0 else 0
        elif np.isinf(lst[i]):
            lst[i] = 1 if lst[i] > 0 else 0
    return lst

def bootstrap_resampling(data, statistic_function, with_weights=None, num_samples=1000, alpha=0.05, seed=42): 
    sample_statistics = []

    for _ in range(num_samples):
        # Generate a bootstrap sample with replacement
        if with_weights:
            normalized_weights = np.array(with_weights) / sum(with_weights)
            bootstrap_sample = np.random.choice(data, size=len(data), replace=True, p=normalized_weights)
        else:
            bootstrap_sample = np.random.choice(data, size=len(data), replace=True)

        # Calculate the statistic on the bootstrap sample
        sample_statistic = statistic_function.predict(np.array(bootstrap_sample).reshape(-1,1))
        sample_statistics.append(sample_statistic)

    # Calculate the confidence interval based on alpha
    mean = np.mean(sample_statistics)
    lower_bound = np.percentile(sample_statistics, 100 * (alpha / 2))
    upper_bound = np.percentile(sample_statistics, 100 * (1 - alpha / 2))

    confidence_interval = (lower_bound, upper_bound)
    return mean, confidence_interval  

def compute_metrics(X_train, y_train, X_test, y_test, target_data):
    regressor = LinearRegression().fit(np.array(X_train.values).reshape(-1, 1), y_train)
    first_mse = mean_squared_error(y_test, regressor.predict(np.array(X_test.values).reshape(-1,1)))
    second_mse = mean_squared_error(target_data['Y'], regressor.predict(np.array(target_data['X']).reshape(-1,1)))
    
    sample_from_old = np.random.choice(y_test, size=70, replace=False)
    sample_from_new = np.random.choice(target_data["Y"], size=70, replace=False) 
    X_logreg = np.concatenate((sample_from_old, sample_from_new), axis=0)
    t = [1] * len(sample_from_old) + [0] * len(sample_from_new)
    X_train_lg, _, y_train_lg, _ = train_test_split(X_logreg, t, test_size=0.0001, random_state=None, shuffle=True)
    rf = RandomForestClassifier().fit(X_train_lg.reshape(-1, 1), y_train_lg)
    weights = [rf.predict_proba(item.reshape(1, -1))[0][0] for item in X_test.values]
    
    true_weights = target_data["X"] / X_test.values
    true_weights = replace_nan_inf(true_weights.values)
    true_weights_norm = (np.array(true_weights) - np.min(true_weights)) / (np.max(true_weights) - np.min(true_weights))

    third_mse = mean_squared_error(target_data['Y'], regressor.predict(np.array(target_data['X']).reshape(-1,1)), sample_weight=weights)

    first_ci = bootstrap_resampling(target_data['X'], regressor) 
    regressor_weighted = LinearRegression().fit(np.array(X_train.values).reshape(-1, 1), 
                                            y_train, 
                                            sample_weight=[rf.predict_proba(item.reshape(1, -1))[0][0] for item in X_train.values])
    second_ci = bootstrap_resampling(target_data['X'], regressor_weighted, weights)
    
    return [(first_mse, second_mse, third_mse), (first_ci, second_ci), (weights, true_weights_norm)]