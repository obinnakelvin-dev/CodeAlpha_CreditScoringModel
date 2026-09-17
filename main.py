import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    classification_report,
    confusion_matrix
)

import joblib

# ==========================================
# 1. LOAD THE DATASET
# ==========================================

df = pd.read_csv("data/credit_risk_dataset.csv")

print("Dataset loaded successfully!")

print("\nFirst 5 rows:")
print(df.head())


# ==========================================
# 2. CHECK THE DATA
# ==========================================

print("\nMissing values:")
print(df.isnull().sum())

print("\nDuplicate rows:")
print(df.duplicated().sum())


# ==========================================
# 3. CLEAN THE DATA
# ==========================================

# Remove duplicate rows
df = df.drop_duplicates()

# Fill missing values with the median
df["person_emp_length"] = df["person_emp_length"].fillna(
    df["person_emp_length"].median()
)

df["loan_int_rate"] = df["loan_int_rate"].fillna(
    df["loan_int_rate"].median()
)

print("\nData cleaning completed!")

print("\nMissing values after cleaning:")
print(df.isnull().sum())

print("\nDuplicate rows after cleaning:")
print(df.duplicated().sum())


# ==========================================
# 4. SEPARATE FEATURES AND TARGET
# ==========================================

# loan_status is the target variable
# 0 = No default
# 1 = Default

X = df.drop("loan_status", axis=1)
y = df["loan_status"]

print("\nFeatures and target separated.")

print("Features shape:", X.shape)
print("Target shape:", y.shape)


# ==========================================
# 5. ENCODE CATEGORICAL DATA
# ==========================================

# Find columns containing text
categorical_columns = X.select_dtypes(
    include=["object"]
).columns

print("\nCategorical columns:")
print(categorical_columns)

# Convert categorical columns into numerical columns
X = pd.get_dummies(
    X,
    columns=categorical_columns,
    drop_first=True
)

print("\nEncoding completed!")

print("New feature shape:", X.shape)


# ==========================================
# 6. SPLIT DATA INTO TRAINING AND TESTING
# ==========================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

print("\nTraining data shape:")
print(X_train.shape)

print("\nTesting data shape:")
print(X_test.shape)

print("\nTraining target shape:")
print(y_train.shape)

print("\nTesting target shape:")
print(y_test.shape)


# ==========================================
# 7. MODEL 1 - BASELINE LOGISTIC REGRESSION
# ==========================================

model_1 = Pipeline([
    ("scaler", StandardScaler()),
    (
        "classifier",
        LogisticRegression(
            max_iter=1000,
            random_state=42
        )
    )
])

# Train Model 1
model_1.fit(X_train, y_train)

# Make predictions
y_pred_1 = model_1.predict(X_test)

# Get probability of default
y_prob_1 = model_1.predict_proba(X_test)[:, 1]


# ==========================================
# 8. EVALUATE MODEL 1
# ==========================================

accuracy_1 = accuracy_score(y_test, y_pred_1)

precision_1 = precision_score(
    y_test,
    y_pred_1,
    zero_division=0
)

recall_1 = recall_score(
    y_test,
    y_pred_1,
    zero_division=0
)

f1_1 = f1_score(
    y_test,
    y_pred_1,
    zero_division=0
)

roc_auc_1 = roc_auc_score(
    y_test,
    y_prob_1
)


print("\n==========================================")
print("       MODEL 1 - BASELINE")
print("==========================================")

print("Accuracy:", accuracy_1)
print("Precision:", precision_1)
print("Recall:", recall_1)
print("F1-Score:", f1_1)
print("ROC-AUC:", roc_auc_1)


# ==========================================
# 9. MODEL 2 - BALANCED LOGISTIC REGRESSION
# ==========================================

model_2 = Pipeline([
    ("scaler", StandardScaler()),
    (
        "classifier",
        LogisticRegression(
            max_iter=1000,
            random_state=42,
            class_weight="balanced"
        )
    )
])

# Train Model 2
model_2.fit(X_train, y_train)

# Make predictions
y_pred_2 = model_2.predict(X_test)

# Get probability of default
y_prob_2 = model_2.predict_proba(X_test)[:, 1]


# ==========================================
# 10. EVALUATE MODEL 2
# ==========================================

accuracy_2 = accuracy_score(y_test, y_pred_2)

precision_2 = precision_score(
    y_test,
    y_pred_2,
    zero_division=0
)

recall_2 = recall_score(
    y_test,
    y_pred_2,
    zero_division=0
)

f1_2 = f1_score(
    y_test,
    y_pred_2,
    zero_division=0
)

roc_auc_2 = roc_auc_score(
    y_test,
    y_prob_2
)


print("\n==========================================")
print("       MODEL 2 - BALANCED")
print("==========================================")

print("Accuracy:", accuracy_2)
print("Precision:", precision_2)
print("Recall:", recall_2)
print("F1-Score:", f1_2)
print("ROC-AUC:", roc_auc_2)


# ==========================================
# 11. COMPARE BOTH MODELS
# ==========================================

comparison = pd.DataFrame({
    "Metric": [
        "Accuracy",
        "Precision",
        "Recall",
        "F1-Score",
        "ROC-AUC"
    ],
    "Baseline Model": [
        accuracy_1,
        precision_1,
        recall_1,
        f1_1,
        roc_auc_1
    ],
    "Balanced Model": [
        accuracy_2,
        precision_2,
        recall_2,
        f1_2,
        roc_auc_2
    ]
})


print("\n==========================================")
print("          MODEL COMPARISON")
print("==========================================")

print(comparison.to_string(index=False))


# ==========================================
# 12. CLASSIFICATION REPORT
# ==========================================

print("\n==========================================")
print("    BASELINE CLASSIFICATION REPORT")
print("==========================================")

print(
    classification_report(
        y_test,
        y_pred_1,
        zero_division=0
    )
)


print("\n==========================================")
print("    BALANCED CLASSIFICATION REPORT")
print("==========================================")

print(
    classification_report(
        y_test,
        y_pred_2,
        zero_division=0
    )
)


# ==========================================
# 13. CONFUSION MATRIX - BASELINE
# ==========================================

print("\n==========================================")
print("    BASELINE CONFUSION MATRIX")
print("==========================================")

print(confusion_matrix(y_test, y_pred_1))


# ==========================================
# 14. CONFUSION MATRIX - BALANCED
# ==========================================

print("\n==========================================")
print("    BALANCED CONFUSION MATRIX")
print("==========================================")

print(confusion_matrix(y_test, y_pred_2))


# ==========================================
# 15. FINISHED
# ==========================================

print("\n==========================================")
print("       CREDIT SCORING PROJECT COMPLETE")
print("==========================================")

# ==========================================
# 16. SAVE THE BALANCED MODEL
# ==========================================

joblib.dump(
    model_2,
    "model/credit_scoring_model.pkl"
)

print("\nModel saved successfully!")
print("Saved to: model/credit_scoring_model.pkl")