from sklearn.feature_extraction.text import HashingVectorizer
from sklearn.naive_bayes import BernoulliNB
from sklearn.metrics import accuracy_score
import os
import pickle
from utils import init_logger
import logging

init_logger()

# Load data
logging.info("Loading data")
try:
    data = pickle.load(open("./data/sklearn-data.pickle", "rb"))
except FileNotFoundError:
    raise FileNotFoundError("Place data files in ./data/sklearn-data.pickle")

x_train, y_train = data["x_train"], data["y_train"]
x_test, y_test = data["x_test"], data["y_test"]

# Transformers reviews to feature-vectors. Removes stop words and only looks if word is present or not.
vectorizer = HashingVectorizer(stop_words='english', binary=True, n_features=2**9)

# Load vectorized reviews from file or vectorize them and save for later
x_train_path = './data/x_train_vec.pkl'
y_train_path = "./data/y_train_vec.pkl"
if os.path.isfile(x_train_path) and os.path.isfile(y_train_path):
    logging.info("Loading vectorized data")
    x_train = pickle.load(open(x_train_path, "rb"))
    x_test = pickle.load(open(y_train_path, "rb"))
else:
    logging.info("Vectorizing reviews")
    # Transform x_train and x_test to vectors
    x_train = vectorizer.fit_transform(x_train)
    x_test = vectorizer.transform(x_test)

    # Save to pickle for later use
    logging.info("Saving vectorized reviews to file")
    with open(x_train_path, 'wb') as f:
        pickle.dump(x_train, f)
    with open(y_train_path, 'wb') as f:
        pickle.dump(x_test, f)

# Fit a classifier
logging.info("Fitting classifier")
classifier = BernoulliNB()
classifier.fit(X=x_train, y=y_train)

# Predict and report score
logging.info("Predicting reviews")
train_pred = classifier.predict(x_train.toarray())
test_pred = classifier.predict(x_test.toarray())

train_acc = accuracy_score(y_train, train_pred)
test_acc = accuracy_score(y_test, test_pred)

logging.info("Train_Acc: {}, Test_Acc: {}".format(train_acc, test_acc))
