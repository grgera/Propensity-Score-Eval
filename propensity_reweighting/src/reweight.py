from .preprocessing import ProScoreVectorizer
from hep_ml.metrics_utils import ks_2samp_weighted
from hep_ml import reweight
import numpy as np

class PropensityReweighter:
    def __init__(self, config):
        """
        """
        self.preprocess = ProScoreVectorizer(config)
        self.config = config["PropensityReweighter"]
        self.logging = self.config["logging"]
        
        self.reweighter = None
        
    def _k2_logger(self, orig, targ, weights_pred, type_data='non-weighted'):
        k2 =[]
        for i in range(self.preprocess.model.config.hidden_size):
            k2.append(ks_2samp_weighted(orig[:, i], 
                                                targ[:, i], 
                                                weights1=weights_pred, 
                                                weights2=np.ones(len(targ), dtype=float)
                                               )
                             )
        print(f"Mean and median k2 on {type_data} test data:", np.mean(k2), "-", np.median(k2))
        
    def fit(self, original, target):
        """
        """
        vctr_original = self.preprocess.vectorize_texts(original)
        vctr_target = self.preprocess.vectorize_texts(target)
        
        if self.logging:
            self._k2_logger(vctr_original, vctr_target, np.ones(len(vctr_original)))
            
        ## todo: make a gridSearch here to choose hyperparams
        reweighter_base = reweight.GBReweighter(n_estimators=30, learning_rate=0.009, max_depth=20, min_samples_leaf=1, 
                                   gb_args={'subsample': 0.9})
        self.reweighter = reweight.FoldingReweighter(reweighter_base, n_folds=3);
        self.reweighter.fit(vctr_original, vctr_target);
        
        if self.logging:
            wieghts_pred = self.reweighter.predict_weights(vctr_original)
            self._k2_logger(vctr_original, vctr_target, wieghts_pred, type_data='weighted')
        
    def predict(self, test):
        """
        """
        vctr_test = self.preprocess.vectorize_texts(test)
        return self.reweighter.predict_weights(vctr_test)
    
    