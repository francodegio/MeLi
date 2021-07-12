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
import os
import json
import pickle
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from catboost import CatBoostClassifier
from sklearn.model_selection import train_test_split
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import OneHotEncoder
from sklearn.base import TransformerMixin, BaseEstimator
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import make_pipeline, make_union, Pipeline
from sklearn.metrics import (
    classification_report, 
    confusion_matrix,
    plot_roc_curve
)


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


class FeatureGeneration(TransformerMixin, BaseEstimator):
    
    def __init__(self):
        self.columns = ['warranty', 'deals', 'methods', 'listing_type_id',
                        'tags', 'tags1', 'free_shipping', 'mode',
                        'Acordar con el comprador', 'Contra reembolso',
                        'Transferencia bancaria', 'Tarjeta de crédito',
                        'Giro postal', 'Cheque certificado',
                        'MercadoPago', 'Efectivo', 'debit_cards',
                        'accepts_mercadopago', 'title']


    def fit(self, X, *_):
        return self


    def transform(self, X, *_):
        self.X = X.copy()
        df = X.copy()

        self._create_shipping()
        self._create_payments()

        df['warranty'] = df['warranty'].notna().astype(int)
        df['accepts_mercadopago'] = df['accepts_mercadopago'].astype(int)
        df['deals'] = df['deal_ids'].apply(lambda x: 1 if x else 0)
        df = pd.concat([df, self.shipping, self.payments], axis=1)
        df['tags'] = df['tags'].apply(lambda x: x[0] if x else 'null')

        return df[self.columns]


    def _create_shipping(self):
        shipping = pd.DataFrame(self.X['shipping'].to_dict()).T
        shipping['free_shipping'] = shipping['free_shipping'].astype(int)
        shipping['local_pick_up'] = shipping['local_pick_up'].astype(int)
        
        shipping['methods'] = self._create_methods(shipping['free_methods'])
        shipping.rename(columns={'tags':'tags1'}, inplace=True)
        shipping['tags1'] = shipping['tags1'].apply(
            lambda x: x[0] if x else 'null'
        )
        self.shipping = shipping[['methods', 'tags1', 'free_shipping', 'mode']]
        
        
    def _create_payments(self):
        payments = self.X['non_mercado_pago_payment_methods'].apply(
            lambda lista: [item.get('description') for item in lista]\
                      if lista else None
        )
        
        payments = payments.apply(self._payment_mapper)
        credit_cards = ['American Express', 'MasterCard', 'Diners', 'Visa']
        debit_cards = ['Mastercard Maestro', 'Visa Electron']
        payments['credit_cards'] = payments[credit_cards].sum(axis=1)
        payments['debit_cards'] = payments[debit_cards].sum(axis=1)
        
        mask = payments['Tarjeta de crédito'] == 0
        mask = mask & (payments['credit_cards']==1)
                
        payments.loc[mask, 'Tarjeta de crédito'] = 1
        cols = debit_cards + credit_cards + ['credit_cards']
        self.payments = payments.drop(columns=cols)


    @classmethod
    def _create_methods(cls, serie):
        methods = serie.copy()
        return methods.apply(lambda x: x[0].get('id') \
                             if isinstance(x, list) \
                             else 0)


    @classmethod
    def _payment_mapper(cls, pmt_list):
        pmt = {'Acordar con el comprador', 'American Express',
               'Cheque certificado', 'Contra reembolso',
               'Diners', 'Efectivo', 'Giro postal',
               'MasterCard', 'Mastercard Maestro', 'MercadoPago',
               'Tarjeta de crédito', 'Transferencia bancaria',
               'Visa', 'Visa Electron'}
        
        result = pd.Series(0, index=pmt)
        if pmt_list:
            result[pmt_list] = 1

        return result


class ColumnSelector(BaseEstimator, TransformerMixin):
    
    def __init__(self, columns):
        self.columns = columns
        
    def fit(self, X, *_):
        return self
    
    def transform(self, X, *_):
        return X[self.columns]
    
    
cat_cols = ['listing_type_id', 'tags', 'tags1', 'mode']

other_cols = ['warranty', 'deals', 'methods', 'free_shipping',
              'Acordar con el comprador', 'Contra reembolso',
              'Transferencia bancaria', 'Tarjeta de crédito',
              'Giro postal', 'Cheque certificado',
              'MercadoPago', 'Efectivo', 'debit_cards',
              'accepts_mercadopago']

title = 'title'

feats = make_pipeline(FeatureGeneration())
cats = make_pipeline(
    ColumnSelector(cat_cols),
    OneHotEncoder(handle_unknown='ignore')
)
nlp = Pipeline(
    [
        ('col',ColumnSelector(title)),
        ('vect', TfidfVectorizer()),
        ('dim', TruncatedSVD(100))
    ]
)
others = make_pipeline(ColumnSelector(other_cols))
pipe = make_union(cats, others, nlp)
full_pipe = make_pipeline(feats, pipe)


def make_report(
        model,
        X: pd.DataFrame,
        y: pd.Series,
        title: str='Train Metrics',
        filename: str='report.md'
    ) -> None:

    y_hat = model.predict(X)
    report = classification_report(y, y_hat, output_dict=True)
    
    report['accuracy'] = {
        'precision':'',
        'recall':'',
        'f1-score': report.get('accuracy', ''),
        'support': report.get('macro avg', {}).get('support')
    }
    report = pd.DataFrame(report).T
    
    plt.title(f"\nConfusion Matrix\n")
    fig = sns.heatmap(
        confusion_matrix(y, y_hat),
        annot=True,
        fmt='.2f'
    )
    base_path = os.path.splitext(filename)[0]
    cm_image_path =f'{base_path}_cf.png' 
    plt.savefig(cm_image_path)
    fig.clear()
    del fig
    
    roc_image_path = f'{base_path}_roc.png'
    roc_curve = plot_roc_curve(model, X, y)
    plt.savefig(roc_image_path)
    del roc_curve
    
    try:
        features = pd.Series(model.feature_importances_,
                             index=model.feature_names_)
        features = features.sort_values(ascending=False)
        features = features.head(20)
    except Exception as e:
        features = None
    
    full_report = f"# Model report\n"
    full_report += f"## {title}\n"
    full_report += f"\n### Algorithm information:\n\n{model.__str__()}\n"
    full_report += f"\n### Classification_report:\n\n"
    full_report += report.to_markdown()
    full_report += "\n\n### Confusion Matrix"
    full_report += f"\n\n![ ]({cm_image_path})"
    full_report += "\n\n### ROC AUC Curve"
    full_report += f"\n\n![ ]({roc_image_path})"
    
    if isinstance(features, pd.Series):
        plt.title(f'\nFeature Importances')
        fig = sns.barplot(x=features.values, y=features.index)
        fi_image_path = f'{base_path}_fi.png'
        plt.savefig(fi_image_path)
        full_report += "\n\n### Feature Importances"
        full_report += f"\n\n![ ]({fi_image_path})"
        
    with open(filename, 'w') as f:
        f.write(full_report)


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