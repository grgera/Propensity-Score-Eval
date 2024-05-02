from .preprocessing import ProScoreVectorizer
from hep_ml.metrics_utils import ks_2samp_weighted
from concurrent.futures import ThreadPoolExecutor
from tqdm.auto import tqdm
from hep_ml import reweight
import itertools
import numpy as np
import sys
import os

class PropensityReweighter:
    def __init__(self, config):
        """
        """
        self.preprocess = ProScoreVectorizer(config)
        self.config = config["PropensityReweighter"]
        self.logging = self.config["logging"]
        
        self.n_estimators = self.config["GBparams"]["n_estimators"]
        self.learning_rate = self.config["GBparams"]["learning_rate"]
        self.max_depth = self.config["GBparams"]["max_depth"]
        self.min_samples_leaf = self.config["GBparams"]["min_samples_leaf"]
        self.n_folds = self.config["GBparams"]["n_folds"]
        
        self.reweighter = None
        
    def _k2_logger(self, orig, targ, weights_pred, type_data='non-weighted', printing=True):
        """
        """
        k2 = []
        for i in range(self.preprocess.model.config.hidden_size):
            k2.append(ks_2samp_weighted(orig[:, i], 
                                                targ[:, i], 
                                                weights1=weights_pred, 
                                                weights2=np.ones(len(targ), dtype=float)
                                               )
                             )
        if printing:
            print(f"Mean and median k2 on {type_data} test data:", np.mean(k2), "-", np.median(k2))
            
        return np.mean(k2)
    
    def _check_params(self, params_gb):
        """
        """
        params = {'n_estimators': self.n_estimators, 
                  'learning_rate': self.learning_rate, 
                  'max_depth': self.max_depth, 
                  'min_samples_leaf':self.min_samples_leaf}

        if params_gb:
            for item in params:
                if item in params_gb.keys():
                    params[item] = params_gb[item]

        return params

    def fit(self, original, target, params_gb={}, vectorized=False):
        """
        """
        if not vectorized:
            original = self.preprocess.vectorize_texts(original)
            target = self.preprocess.vectorize_texts(target)

        if self.logging:
            self._k2_logger(original, target, np.ones(len(original)))

        params = self._check_params(params_gb)
        reweighter_base = reweight.GBReweighter(gb_args={'subsample': 0.9}, **params)

        self.reweighter = reweight.FoldingReweighter(reweighter_base, n_folds=self.n_folds, verbose=False);
        self.reweighter.fit(original, target);

        if self.logging:
            wieghts_pred = self.reweighter.predict_weights(original)
            self._k2_logger(original, target, wieghts_pred, type_data='weighted')

        return self
        
    def predict(self, test):
        """
        """
        vctr_test = self.preprocess.vectorize_texts(test)
        return self.reweighter.predict_weights(vctr_test)
    
    def process_param_set(self, args):
        params, vct_original, vct_target, k2_non_weighted = args

        current_logging = self.logging
        self.logging = False

        model = self.fit(vct_original, vct_target, params, vectorized=True)
        wieghts_pred = model.reweighter.predict_weights(vct_original)
        
        self.logging = current_logging

        mean_k2 = self._k2_logger(vct_original, 
                        vct_target, 
                        wieghts_pred, 
                        type_data='weighted',
                        printing=False)

        error = 0 if mean_k2 > k2_non_weighted else (k2_non_weighted - mean_k2)

        return params, error
    
    def fit_gridsearch(self, original, target, grid_params, parallel=-1):
        max_score = -sys.maxsize
        best_params = {}
        results = []

        vctr_original = self.preprocess.vectorize_texts(original)
        vctr_target = self.preprocess.vectorize_texts(target)

        if parallel == -1:
            parallel_units = os.cpu_count()
        elif paralle_units > os.cpu_count():
            raise RuntimeError(f"Maximum CPU available: {os.cpu_count()}")

        param_set = [dict(zip(grid_params.keys(), values)) for values in itertools.product(*grid_params.values())]
        
        k2_non_weighted = self._k2_logger(vctr_original, vctr_target, np.ones(len(vctr_original)), printing=False)

        try:
            with ThreadPoolExecutor(max_workers=parallel_units) as executor:
                for params in tqdm(param_set):
                    future = executor.submit(self.process_param_set,
                                             (params, vctr_original, vctr_target, k2_non_weighted))

                    _, total_error = future.result()

                    avg_error = total_error / self.n_folds
                    results.append((params, avg_error))

                    if avg_error > max_score:
                        max_score = avg_error
                        best_params = params

        except KeyboardInterrupt:
            print("Ça ne marche pas")
            executor.shutdown(wait=False)
                              
        print("Results:")
        sorted_res = sorted(results, key=lambda x: x[1], reverse=True)
        for res in sorted_res[:5]:
            print(res)
        print("Best Parameters:", best_params)
        print("Error:", max_score)
        print()
        return self.fit(vctr_original, vctr_target, best_params, vectorized=True)
    
    