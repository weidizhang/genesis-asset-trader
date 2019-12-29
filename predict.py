from joblib import load

class Predict:
    def __init__(self, model_file, k_neighbors = 5, max_conflicts = 0, search_distance = 24): # remove default values
        self.load_model(model_file)
        self.set_k_neighbors(k_neighbors)
        self.set_search_distance(search_distance)
        self.set_max_conflicts(max_conflicts)

    def load_model(self, file):
        self._model = load(file)

    def predict_with_model(self, df_features):
        predict = self._model.predict(df_features)

        # Copy it to get rid of pandas error of setting on copy of a slice
        df_features = df_features.copy()
        df_features["Extrema"] = predict
        return df_features

    def predict_full(self, df, df_features, model_predict_only = False):
        model_predicted = self.predict_with_model(df_features)
        self._merge_data(df, model_predicted)

        if model_predict_only:
            return df # TODO: ??

        # for each extrema, verify them with additional rules
        
        bad_extremas = self.validate_extremas(df)
        removed = len(list(bad_extremas))

        print("Removed extremas:", removed)
        # just conflicts: 206
        # with k neighbors: 

        return df

    def set_k_neighbors(self, k):
        self._k = k

    # Leftward/backward search distance starting at a given data point p
    def set_search_distance(self, d):
        self._distance = d

    def set_max_conflicts(self, max_conflicts):
        self._max_conflicts = max_conflicts

    # Yields indices of extremas that should be deleted due to having conflicts or not
    # having the minimum k neighbors of same extrema type as predicted by the model
    #
    # A generator is used as we want to search the df in its unmodified state for
    # our heuristic checks
    def validate_extremas(self, df):
        for i, row in df.iterrows():
            extrema_type = row["Extrema"]
            if extrema_type == 0:
                continue

            search_region = df.iloc[i - self._distance:i]
            if self._has_conflicts(search_region, extrema_type) or not self._has_k_neighbors(search_region, extrema_type):
                yield i

    def _has_conflicts(self, region, extrema_type):
        # Note that (extrema_type * -1) is the opposite extrema type, where 1 = maxima, -1 = minima
        return len(region[region["Extrema"] == extrema_type * -1]) > self._max_conflicts

    def _has_k_neighbors(self, region, extrema_type):
        return len(region[region["Extrema"] == extrema_type]) >= self._k

    def _merge_data(self, df, predicted_extremas):
        # Delete real extrema data and replace with predicted data
        # TODO: make sure extrema data is empty by setting a flag in data processor
        df["Extrema"] = 0

        # TODO: is there a better way to do this?
        for i, row in predicted_extremas.iterrows():
            df.loc[i, "Extrema"] = row["Extrema"]
