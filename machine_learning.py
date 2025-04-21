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
import numpy as np
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

# K-fold cross validation
K = 5
total = X.shape[0]
index = int(total / K)

# Create K-fold splits
X_1_test = X.iloc[:index]
X_1_train = X.iloc[index:]
Y_1_test = Y.iloc[:index]
Y_1_train = Y.iloc[index:]

X_2_test = X.iloc[index:index*2]
X_2_train = X.iloc[np.r_[:index, index*2:]]
Y_2_test = Y.iloc[index:index*2]
Y_2_train = Y.iloc[np.r_[:index, index*2:]]

X_3_test = X.iloc[index*2:index*3]
X_3_train = X.iloc[np.r_[:index*2, index*3:]]
Y_3_test = Y.iloc[index*2:index*3]
Y_3_train = Y.iloc[np.r_[:index*2, index*3:]]

X_4_test = X.iloc[index*3:index*4]
X_4_train = X.iloc[np.r_[:index*3, index*4:]]
Y_4_test = Y.iloc[index*3:index*4]
Y_4_train = Y.iloc[np.r_[:index*3, index*4:]]

X_5_test = X.iloc[index*4:]
X_5_train = X.iloc[:index*4]
Y_5_test = Y.iloc[index*4:]
Y_5_train = Y.iloc[:index*4]

# Create lists for X and Y train/test sets
X_train_list = [X_1_train, X_2_train, X_3_train, X_4_train, X_5_train]
X_test_list = [X_1_test, X_2_test, X_3_test, X_4_test, X_5_test]
Y_train_list = [Y_1_train, Y_2_train, Y_3_train, Y_4_train, Y_5_train]
Y_test_list = [Y_1_test, Y_2_test, Y_3_test, Y_4_test, Y_5_test]

# K-fold iteration (without evaluation)
for i in range(0, K):
    svm_model.fit(X_train_list[i], Y_train_list[i])
    rf_model.fit(X_train_list[i], Y_train_list[i])
    dt_model.fit(X_train_list[i], Y_train_list[i])
    ab_model.fit(X_train_list[i], Y_train_list[i])
    nb_model.fit(X_train_list[i], Y_train_list[i])
    nn_model.fit(X_train_list[i], Y_train_list[i])
    kn_model.fit(X_train_list[i], Y_train_list[i])

# The predictions are made but not displayed or calculated for metrics
svm_predictions = svm_model.predict(X_test_list[i])
rf_predictions = rf_model.predict(X_test_list[i])
dt_predictions = dt_model.predict(X_test_list[i])
ab_predictions = ab_model.predict(X_test_list[i])
nb_predictions = nb_model.predict(X_test_list[i])
nn_predictions = nn_model.predict(X_test_list[i])
kn_predictions = kn_model.predict(X_test_list[i])
