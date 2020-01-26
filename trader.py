import data_processor
import predict

# Trader is an abstract class that is meant to be inherited in order to create an
# automated trading software for specific trading platforms
#
# It provides the base functions necessary, i.e. current extrema signals, updating the data
#
# Sample flow / code:
#
# class ABCTrader(Trader): ...
# predictor = Predictor( ... )
# trader = ABCTrader( ... )
# while True:
#     trader.update_data( ... some newly received or updated data ... )
#     trader.make_transaction()

class Trader:
    def __init__(self, predictor,
            processor_data_condensed = True, processor_data_hourly = True, processor_extrema_n = 20):
        # Allow the data adapter and predictor to be changeable
        self.set_predictor(predictor)

        # Other attributes are set once and should not need to be modified; the data processor
        # should act consistently
        #
        # The default values in the constructor are also the ones used for training the model
        # by default with regards to the data processor
        self._data_condensed = processor_data_condensed
        self._data_hourly = processor_data_hourly
        self._extrema_n = processor_extrema_n

    # To be implemented by the child class
    #
    # Should convert the data into a dataframe that is compatible with the data processor module
    def adapt_data(self, data):
        raise NotImplementedError

    # The signal is the extrema at the lastest point in time in the processed dataframe
    # Predicted using the predictor object; note that: -1 (minima), 0 (none), 1 (maxima)
    def current_signal(self):
        return self._predictor.predict_latest(self._df, Predict.feature_attributes(self._df))

    # To be implemented by the child class
    def make_buy_transaction(self):
        raise NotImplementedError

    # To be implemented by the child class
    def make_sell_transaction(self):
        raise NotImplementedError

    # An oversimplification of transaction logic that does not have any external data sources
    # or other strategies that are used alongside the received signal to make trades
    #
    # In this sample function: Buy at minima, sell at maxima
    #
    # Should be overriden by the child class for more complex logic
    def make_transaction(self):
        extrema = self.current_signal()
        if extrema == -1:
            self.make_buy_transaction()
        elif extrema == 1:
            self.make_sell_transaction()

    def set_predictor(self, predictor):
        self._predictor = predictor

    def update_data(self, data):
        # Adapt the data first before processing the data for use
        adapted_data = self.adapt_data(data)

        self._df = data_processor.read_data_from_df(adapted_data, self._data_condensed, self._data_hourly, self._extrema_n)
        return Predict.preprocess_data(self._df)
