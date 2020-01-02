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

    # Returns the predicted extrema at the lastest point in time in the dataframe df
    #
    # Wrapper function that calls predict_point
    #
    # Meant for use with real time live trading data in some form of automated
    # trading to predict the latest point in time
    def predict_latest(self, df, df_features, model_predict_only = False):
        return self.predict_point(df, df_features, len(df) - 1, model_predict_only)

    # Returns the predicted extrema at any given point based on its index in df
    #
    # Generates a copy of the dataframe df with only the required search distance
    # sliced from the original dataframe to perform the prediction on; saves
    # extra calculation time over computing predictions on the entire dataframe
    #
    # Original data dataframe df is not modified with extrema data
    def predict_point(self, df, df_features, index, model_predict_only = False):
        # We do not add 1 to the calculated start_i, e.g. index - dist + 1, as we want
        # a slice of size dist + 1 rather than dist: this gives a search region of size
        # dist followed by the point we want to determine
        start_i = index - self._distance
        region = slice(start_i if start_i >= 0 else 0, index + 1)

        df_point = df.iloc[region].copy()
        df_point_features = df_features.iloc[region].copy()

        self.predict_full(df_point, df_point_features, model_predict_only)
        return df_point.iloc[-1]["Extrema"]

    def predict_with_model(self, df_features):
        predict = self._model.predict(df_features)

        # Copy it to get rid of pandas error of setting on copy of a slice
        df_features = df_features.copy()
        df_features["Extrema"] = predict
        return df_features

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

        # Offset is used to make predict_point work where the sliced dataframes do not
        # start with an index of 0
        #
        # This will be 0 when using predict_full on properly indexed data
        start_offset = df.first_valid_index()

        for i, row in df.iterrows():
            extrema_type = row["Extrema"]
            if extrema_type == 0:
                continue

            search_region = df.iloc[i - start_offset - self._distance:i - start_offset]
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
