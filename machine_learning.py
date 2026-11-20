# machinelearning.py

from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn import tree
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
import pandas as pd


# Step 1: Import dataset
legitimate_df = pd.read_csv("structured_data_legitimate.csv")
phishing_df = pd.read_csv("structured_data_phishing.csv")

# Step 2: Combine the data
df = pd.concat([legitimate_df, phishing_df], axis=0)
df = df.sample(frac=1)  # Shuffle dataset

df = df.drop('URL', axis=1)  # Drop URL column
df = df.drop_duplicates()  # Remove duplicates

X = df.drop('label', axis=1)  # Features
Y = df['label']  # Labels

# Step 3: Split data into train and test sets
x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=10)

# Step 4: Initialize models
svm_model = svm.LinearSVC()
rf_model = RandomForestClassifier(n_estimators=60)
dt_model = tree.DecisionTreeClassifier()
ab_model = AdaBoostClassifier(n_estimators=50, algorithm="SAMME")
nb_model = GaussianNB()
nn_model = MLPClassifier(alpha=1)
kn_model = KNeighborsClassifier()

# Step 5: Train each model once
# The original code looped over 5 folds, but each .fit() call starts from scratch,
# so only the last fold's training was ever kept. Training once on that same split
# gives identical models in about a fifth of the time.
K = 5
index = int(X.shape[0] / K)
X_train_final = X.iloc[:index * (K - 1)]
Y_train_final = Y.iloc[:index * (K - 1)]

for model in (svm_model, rf_model, dt_model, ab_model, nb_model, nn_model, kn_model):
    model.fit(X_train_final, Y_train_final)
