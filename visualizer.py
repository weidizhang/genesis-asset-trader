from pandas.plotting import register_matplotlib_converters
import matplotlib.pyplot as plt

class Visualizer:
    def __init__(self, title, num_subplot_rows, height_ratios):
        self._configure()
        self._generate_figure(title, num_subplot_rows, height_ratios)
        self._current_axis = 0

    def fill_next_axis(self, callback, title = None, legend = None):
        if self._current_axis > len(self._axes):
            raise VisualizerException("Exceeded number of axes in figure")

        next_axis = self._axes[self._current_axis]
        if title is not None:
            next_axis.title.set_text(title)

        # Any data drawn, e.g. via scatter or plot functions, should be done
        # by the callback function
        callback(next_axis)

        if legend is not None:
            next_axis.legend(legend)

        self._current_axis += 1

    def get_fig(self):
        return self._fig

    def get_axes(self):
        return self._axes

    # Should be called on completion of filling all axes
    def show(self):
        plt.show()

    def show_last_x_axis_only(self):
        # Typically, we will have multiple graphs stacked on top of each other
        # in rows with the same x axis on all graphs
        #
        # For better visuals, we would only want to show the bottom most x axis
        for i in range(len(self._axes) - 1):
            self._axes[i].get_xaxis().set_visible(False)

    def _configure(self):
        register_matplotlib_converters()
        plt.rcParams["figure.figsize"] = (10, 6)
        plt.rcParams["axes.titlepad"] = 3

    def _generate_figure(self, title, num_subplot_rows, height_ratios):
        self._fig, self._axes = plt.subplots(num_subplot_rows, gridspec_kw = { "height_ratios": height_ratios })
        self._fig.suptitle(title)

class VisualizerException(Exception):
    pass
