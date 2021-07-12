"""
Exercise description
--------------------

In the context of Mercadolibre's Marketplace an algorithm is needed to
predict if an item listed in the markeplace is new or used.

Your task to design a machine learning model to predict if an item is new or
used and then evaluate the model over held-out test data.

To assist in that task a dataset is provided in `MLA_100k.jsonlines` and a
function to read that dataset in `build_dataset`.

For the evaluation you will have to choose an appropiate metric and also
elaborate an argument on why that metric was chosen.

The deliverables are:
    - This file including all the code needed to define and evaluate a model.
    - A text file with a short explanation on the criteria applied to choose
      the metric and the performance achieved on that metric.


"""
import json
import pickle
import pandas as pd
from sklearn.model_selection import train_test_split
from catboost import CatBoostClassifier
from meli import full_pipe, make_report

# You can safely assume that `build_dataset` is correctly implemented
def build_dataset():
    data = [json.loads(x) for x in open("MLA_100k.jsonlines")]
    target = lambda x: x.get("condition")
    N = -10000
    X_train = data[:N]
    X_test = data[N:]
    y_train = [target(x) for x in X_train]
    y_test = [target(x) for x in X_test]
    for x in X_test:
        del x["condition"]
    for x in X_train:
        del x["condition"]
    return X_train, y_train, X_test, y_test


if __name__ == "__main__":
    print("Loading dataset...")
    # Train and test data following sklearn naming conventions
    # X_train (X_test too) is a list of dicts with information about each item.
    # y_train (y_test too) contains the labels to be predicted (new or used).
    # The label of X_train[i] is y_train[i].
    # The label of X_test[i] is y_test[i].
    X_train, y_train, X_test, y_test = build_dataset()

    # Insert your code below this line:
    X_train = pd.DataFrame(X_train)
    X_test = pd.DataFrame(X_test)
    y_train = pd.DataFrame(y_train)
    y_test = pd.DataFrame(y_test)

    # Step 1: Create a validation set 
    X_train, X_val, y_train, y_val = train_test_split(X_train,
                                                      y_train,
                                                      shuffle=True,
                                                      test_size=10_000)

    # Step 2: Implement all transformations required for feature extraction
    X_train = full_pipe.fit_transform(X_train)
    X_val = full_pipe.transform(X_val)
    X_test = full_pipe.transform(X_test)

    # Step 3: instantiate and fit a model
    # Note: I'm using CatBoost because it was the best model according
    # to the notebooks where I trained the gridsearch.
    # It's also quite faster than training a whole gridsearch.
    cat = CatBoostClassifier(verbose=False)
    cat.fit(X=X_train, y=y_train, eval_set=(X_val, y_val))

    # Write reports to markdown
    make_report(cat,
                X_val,
                y_val,
                title='Cross-Validation Metrics',
                filename='cross_val_report.md')

    make_report(cat,
                X_test,
                y_test,
                title='Test Metrics',
                filename='test_report.md')

    # Save to disk just in case
    with open('model.pkl', 'wb') as f:
        pickle.dump(cat, f)