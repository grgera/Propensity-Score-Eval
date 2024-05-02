from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import mean_squared_error
from sklearn.calibration import CalibratedClassifierCV
import numpy as np
import pandas as pd


def bootstrap_resampling(data, statistic_function, with_weights=None, num_samples=1000, alpha=0.05, seed=42):
    np.random.seed(seed) 
    sample_statistics = []

    for _ in range(num_samples):
        # Generate a bootstrap sample with replacement
        if with_weights:
            normalized_weights = np.array(with_weights) / sum(with_weights)
            bootstrap_ind = np.random.choice(len(data), size=len(data), replace=True, p=normalized_weights)
        else:
            bootstrap_ind = np.random.choice(len(data), size=len(data), replace=True)

        bootstrap_sample = data[bootstrap_ind]
        # Calculate the statistic on the bootstrap sample
        sample_statistic = statistic_function.predict(bootstrap_sample)
        sample_statistics.append(sample_statistic)

    mean = np.mean(sample_statistics)
    lower_bound = np.percentile(sample_statistics, 100 * (alpha / 2))
    upper_bound = np.percentile(sample_statistics, 100 * (1 - alpha / 2))

    confidence_interval = (lower_bound, upper_bound)
    return mean, confidence_interval  

def compute_metrics(X_train, y_train, X_test, X_shifted_train):
    regressor = LinearRegression().fit(X_train, y_train)
                                       
    sample_from_old = np.random.choice(len(X_train), size=250, replace=False)
    sample_from_new = np.random.choice(len(X_shifted_train), size=250, replace=False) 

    X_logreg = np.concatenate((X_train[sample_from_old], X_shifted_train[sample_from_new]), axis=0)
    t = [1] * len(sample_from_old) + [0] * len(sample_from_new)

    X_train_lg, _, y_train_lg, _ = train_test_split(X_logreg, t, test_size=0.0001, random_state=None, shuffle=True)
    ccv = CalibratedClassifierCV(RandomForestClassifier(n_estimators=300, min_samples_leaf=40, max_depth=3), cv=3)

    calibr = ccv.fit(X_train_lg, y_train_lg)
            
    weights = [calibr.predict_proba(item.reshape(1, -1))[0][0] / (1 - calibr.predict_proba(item.reshape(1, -1))[0][0]) for item in X_test]
    
    first_ci = bootstrap_resampling(X_test, regressor) 
    regressor_weighted = LinearRegression().fit(X_train, 
                                            y_train, 
                                            sample_weight=[calibr.predict_proba(item.reshape(1, -1))[0][0] / (1- calibr.predict_proba(item.reshape(1, -1))[0][0]) for item in X_train])
    second_ci = bootstrap_resampling(X_test, regressor_weighted, weights)
    
    return (first_ci, second_ci)