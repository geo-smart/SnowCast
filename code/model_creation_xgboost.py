from sklearn.ensemble import ExtraTreesRegressor

from model_creation_rf import RandomForestHole


class XGBoostHole(RandomForestHole):

    def get_model(self):
        """
        rfc_pipeline = Pipeline(steps = [
          ('data_scaling', StandardScaler()),
          ('model', RandomForestRegressor(max_depth = 15,
                                           min_samples_leaf = 0.004,
                                           min_samples_split = 0.008,
                                           n_estimators = 25))])
        #return rfc_pipeline
          """
        etmodel = ExtraTreesRegressor(bootstrap=False, ccp_alpha=0.0, criterion='mse',
                                      max_depth=None, max_features='auto', max_leaf_nodes=None,
                                      max_samples=None, min_impurity_decrease=0.0,
                                      # min_impurity_split=None,
                                      min_samples_leaf=1,
                                      min_samples_split=2, min_weight_fraction_leaf=0.0,
                                      n_estimators=100, n_jobs=-1, oob_score=False,
                                      random_state=123, verbose=0, warm_start=False)
        return etmodel
