import data_processor
import predict

class Trader:
    def __init__(self, data_adapter, predictor,
            processor_data_condensed = True, processor_data_hourly = True, processor_extrema_n = 20):
        # Allow the data adapter and predictor to be changeable
        self.set_adapter(data_adapter)
        self.set_predictor(predictor)

        # Other attributes are set once and should not need to be modified; the data processor
        # should act consistently
        #
        # The default values in the constructor are also the ones used for training the model
        # by default with regards to the data processor
        self._data_condensed = processor_data_condensed
        self._data_hourly = processor_data_hourly
        self._extrema_n = processor_extrema_n

    # The signal is the extrema at the lastest point in time in the processed dataframe
    # Predicted using the predictor object; note that: -1 (minima), 0 (none), 1 (maxima)
    def current_signal(self):
        return self._predictor.predict_latest(self._df, Predict.feature_attributes(self._df))

    def update_data(self, data):
        # Adapt the data first into a dataframe processable by our data processor
        adapted_data = self._adapter(data)

        self._df = data_processor.read_data_from_df(adapted_data, self._data_condensed, self._data_hourly, self._extrema_n)
        Predict.preprocess_data(self._df)

    def set_adapter(self, adapter):
        self._adapter = adapter

    def set_predictor(self, predictor):
        self._predictor = predictor
