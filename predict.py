from joblib import load

import train

class Predict:
    def __init__(self, model_file, k_neighbors = -1, max_conflicts = -1, search_distance = -1):
        if k_neighbors < 0 or max_conflicts < 0 or search_distance < 0:
            raise PredictException("Specify legal values for all of: k_neighbors, max_conflicts, search_distance")

        self.load_model(model_file)
        self.set_k_neighbors(k_neighbors)
        self.set_search_distance(search_distance)
        self.set_max_conflicts(max_conflicts)

    @staticmethod
    def delete_extremas(df, indices):
        df.loc[indices, "Extrema"] = 0

    @staticmethod
    def feature_attributes(*args, **kwargs):
        return train.split_data(*args, **kwargs)[0]

    def load_model(self, file):
        self._model = load(file)

    def predict_with_model(self, df_features):
        predict = self._model.predict(df_features)

        # Copy it to get rid of pandas error of setting on copy of a slice
        df_features = df_features.copy()
        df_features["Extrema"] = predict
        return df_features

    # Returns number of removed extremas from the originally predicted data based on
    # the decision tree model
    #
    # Note that the dataframe df is changed in place with predicted extrema results
    def predict_full(self, df, df_features, model_predict_only = False):
        model_predicted = self.predict_with_model(df_features)
        self._merge_data(df, model_predicted)

        if model_predict_only:
            return 0
        
        bad_extremas = self.validate_extremas(df)
        Predict.delete_extremas(df, bad_extremas)

        return len(bad_extremas)

    @staticmethod
    def preprocess_data(*args, **kwargs):
        return train.preprocess_data(*args, **kwargs)

    def set_k_neighbors(self, k):
        self._k = k

    # Leftward/backward search distance starting at a given data point p
    def set_search_distance(self, d):
        self._distance = d

    def set_max_conflicts(self, max_conflicts):
        self._max_conflicts = max_conflicts

    # Returns list of indices of extremas that should be deleted due to having conflicts or not
    # having the minimum k neighbors of same extrema type as predicted by the model
    #
    # We want to search df in its unmodified state for our heuristic checks, rather than
    # modifying extrema data in place while still searching through the data frame
    def validate_extremas(self, df):
        bad_indices = []

        for i, row in df.iterrows():
            extrema_type = row["Extrema"]
            if extrema_type == 0:
                continue

            search_region = df.iloc[i - self._distance:i]
            if self._has_conflicts(search_region, extrema_type) or not self._has_k_neighbors(search_region, extrema_type):
                bad_indices.append(i)

        return bad_indices

    def _has_conflicts(self, region, extrema_type):
        # Note that (extrema_type * -1) is the opposite extrema type, where 1 = maxima, -1 = minima
        return len(region[region["Extrema"] == extrema_type * -1]) > self._max_conflicts

    def _has_k_neighbors(self, region, extrema_type):
        return len(region[region["Extrema"] == extrema_type]) >= self._k

    def _merge_data(self, df, predicted_extremas):
        # TODO: is there a better way to do this?
        for i, row in predicted_extremas.iterrows():
            df.loc[i, "Extrema"] = row["Extrema"]

class PredictException(Exception):
    pass
