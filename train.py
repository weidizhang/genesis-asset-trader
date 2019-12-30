from joblib import dump
from sklearn.tree import DecisionTreeClassifier, export_graphviz

import graphviz

import data_processor

def make_decision_tree(x, y):
    dtree = DecisionTreeClassifier()
    dtree = dtree.fit(x, y)
    return dtree

def save_decision_tree(dtree, file = "models/model.joblib"):
    dump(dtree, file)

def visualize_decision_tree(dtree, features):
    class_names = dtree.classes_.astype(str)

    graph = graphviz.Source(
        export_graphviz(
            dtree, out_file = None,
            feature_names = features.columns,
            class_names = class_names
        )
    )
    graph.format = "png"
    graph.render("trained-tree", view = True)

def split_data(df, 
            feature_columns = [
                "HLCAverage", "RSI", "EMACrossDifference", "EMACrossDirection", "MACDCrossDifference", "MACDCrossDirection"
            ]
        ):
    # We want to classify into the target class of Extrema to determine if
    # it is a high or low point in price
    features = df[feature_columns]
    target = df["Extrema"]

    return features, target

def preprocess_data(df):
    # NaN values are not supported by sklearn; workaround this by using zeros instead
    df.fillna(0, inplace = True)

def main():
    df = data_processor.main(False)
    preprocess_data(df)

    # Feature attributes to train on, target attribute to classify
    features, target = split_data(df)

    dtree = make_decision_tree(features, target)
    save_decision_tree(dtree)
    visualize_decision_tree(dtree, features)

    return dtree


if __name__ == "__main__":
    main()
