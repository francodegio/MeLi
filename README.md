# MeLi

Challenge solution submition by Franco De Giovannini

## Installation
### 1. Clone repo and change directory

```bash
$ git clone https://github.com/francodegio/MeLi.git && cd MeLi
```

### 2. Create virtual environment (optional)
```bash
$ python -m venv meli
```
Activate such environment
```bash
$ source meli/bin/activate
```

### 2. Run setup.sh
```bash
$ sh setup.sh
```
This will connect to Google Drive in order to download the dataset,
install the requirements and install the package `meli` that contains
all functions and objects required to train the model and export the metrics.

### 3. Execute the script
If everything went smoothly, you can run the script with the following commands:
```bash
$ cd src
$ python new_or_used.py
```
This will save a pickle with a trained model, as well as a couple of markdown files
that contain a report with the relevant metrics.

## Notes on models
You can find all the transformations and development of the models and code
inside the `notebooks` directory.

The model contains 2 kinds of features:
- Intrinsic Features: those that are generated from different classifieds' attributes.
- Text Features: a 100-dimension vector generated from text features extracted from
the titles.

Highest `accuracy` scores obtained were:
- 0.91 with trained data (80.000 observations)
- 0.86 with cross validation data (10.000 observations)
- 0.86 with test data (10.000 observations).

The metric used was `accuracy` since the dataset was fairly balanced and to my understanding,
there are not any kind of issues that disproportionally affect a class over the other.
Additionally, both precision and recall metrics are quite balanced as well for each class.

Finally, the best `roc_auc` score obtained was: