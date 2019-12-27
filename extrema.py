import numpy as np

def local_minima(df, n):
    return local_extrema(df, n, np.less_equal)

def local_maxima(df, n):
    return local_extrema(df, n, np.greater_equal)

# df         = DataFrame
# n          = number of rows before and after current data position to consider
# comparator = comparator function to call to determine if minima or maxima
#
# A point is considered an extrema (minima or maxima) when the comparator function
# evaluates to true with the point and the *n* elements before and after it, i.e.
# the point is less than the surrounding 2n points with point at index i as the center
def local_extrema(df, n, comparator):
    indices = np.zeros(len(df))

    for i in range(len(df)):
        before = df.iloc[i-n:i].to_numpy()
        after = df.iloc[i+1:i+n+1].to_numpy()

        if np.all(comparator(df.iloc[i], before)) and np.all(comparator(df.iloc[i], after)):
            indices[i] = 1

    return np.nonzero(indices)[0]
