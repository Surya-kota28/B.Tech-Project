import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn import svm, tree
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix, roc_curve, auc, precision_recall_curve, accuracy_score, f1_score
from imblearn.over_sampling import SMOTE  # Install if needed: pip install imbalanced-learn

# Load datasets
legitimate_df = pd.read_csv("structured_data_legitimate.csv")
phishing_df = pd.read_csv("structured_data_phishing.csv")

# Combine and shuffle datasets
df = pd.concat([legitimate_df, phishing_df], axis=0)
df = df.sample(frac=1, random_state=42)

# Remove URL column and duplicates
df = df.drop('URL', axis=1, errors='ignore')
df = df.drop_duplicates()

# Split data into features (X) and labels (Y)
X = df.drop('label', axis=1)
Y = df['label']

# Scale features for better model performance
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Address class imbalance using SMOTE
smote = SMOTE(random_state=10)
X_resampled, Y_resampled = smote.fit_resample(X_scaled, Y)

# Split dataset into training and testing sets
x_train, x_test, y_train, y_test = train_test_split(X_resampled, Y_resampled, test_size=0.2, random_state=10)

# Initialize models with hyperparameter tuning
models = {
    "Logistic Regression": LogisticRegression(max_iter=5000),
    "Random Forest": RandomForestClassifier(n_estimators=200, random_state=10),
    "Decision Tree": tree.DecisionTreeClassifier(max_depth=10, min_samples_split=5),
    "AdaBoost": AdaBoostClassifier(algorithm="SAMME"),
    "Naive Bayes": GaussianNB(),
    "Neural Network": MLPClassifier(alpha=1, max_iter=1000),
    "KNN": KNeighborsClassifier(n_neighbors=5, weights="distance"),
    "SVM": svm.LinearSVC(max_iter=5000)
}

# Train models and store results
results = {}
for name, model in models.items():
    model.fit(x_train, y_train)
    preds = model.predict(x_test)
    probs = model.predict_proba(x_test)[:, 1] if hasattr(model, "predict_proba") else None
    acc = accuracy_score(y_test, preds)
    f1 = f1_score(y_test, preds)
    cm = confusion_matrix(y_test, preds)
    cv_score = cross_val_score(model, X_resampled, Y_resampled, cv=5).mean()
    
    results[name] = {"model": model, "preds": preds, "probs": probs, "acc": acc, "f1": f1, "cm": cm, "cv": cv_score}

# Plot ROC Curves
plt.figure(figsize=(10, 5))
for name, result in results.items():
    if result["probs"] is not None:
        fpr, tpr, _ = roc_curve(y_test, result["probs"])
        plt.plot(fpr, tpr, label=f'{name} (AUC: {auc(fpr, tpr):.2f})')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve')
plt.legend()
plt.show()

# Plot Precision-Recall Curves
plt.figure(figsize=(10, 5))
for name, result in results.items():
    if result["probs"] is not None:
        precision, recall, _ = precision_recall_curve(y_test, result["probs"])
        plt.plot(recall, precision, label=name)
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('Precision-Recall Trade-off')
plt.legend()
plt.show()

# Accuracy and F1 Score Bar Chart
model_names = list(results.keys())
accuracies = [results[name]["acc"] for name in model_names]
f1_scores = [results[name]["f1"] for name in model_names]
cv_scores = [results[name]["cv"] for name in model_names]

fig, ax = plt.subplots(figsize=(8, 5))
x = np.arange(len(model_names))
width = 0.3
ax.bar(x - width, accuracies, width, label='Accuracy', color='blue')
ax.bar(x, f1_scores, width, label='F1 Score', color='green')
ax.bar(x + width, cv_scores, width, label='Cross-Val Score', color='orange')
ax.set_xticks(x)
ax.set_xticklabels(model_names, rotation=45, ha='right')
ax.set_ylabel("Scores")
ax.set_title("Model Performance Metrics")
ax.legend()
plt.ylim(0, 1)
plt.show()

# Plot Confusion Matrices
fig, axes = plt.subplots(2, 4, figsize=(15, 8))
axes = axes.ravel()
for i, (name, result) in enumerate(results.items()):
    cm = result["cm"]
    ax = axes[i]
    im = ax.imshow(cm, cmap='Blues')
    ax.set_title(name)
    ax.set_xlabel('Predicted')
    ax.set_ylabel('Actual')
    fig.colorbar(im, ax=ax)
plt.tight_layout()
plt.show()
