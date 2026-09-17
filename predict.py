import pandas as pd
import joblib


# ==========================================
# 1. LOAD THE TRAINED MODEL
# ==========================================

model = joblib.load("model/credit_scoring_model.pkl")

print("Credit Scoring Model loaded successfully!")


# ==========================================
# 2. GET USER INFORMATION
# ==========================================

print("\nEnter the borrower's information:")

person_age = int(input("Age: "))

person_income = float(input("Annual income: "))

person_home_ownership = input(
    "Home ownership (RENT, OWN, MORTGAGE, OTHER): "
).upper()

person_emp_length = float(
    input("Employment length in years: ")
)

loan_intent = input(
    "Loan intent (PERSONAL, EDUCATION, MEDICAL, VENTURE, HOMEIMPROVEMENT, DEBTCONSOLIDATION): "
).upper()

loan_grade = input(
    "Loan grade (A, B, C, D, E, F, G): "
).upper()

loan_amnt = float(
    input("Loan amount: ")
)

loan_int_rate = float(
    input("Loan interest rate (%): ")
)

loan_percent_income = float(
    input("Loan percent of income (e.g. 0.20): ")
)

cb_person_default_on_file = input(
    "Previous default on file? (Y/N): "
).upper()

cb_person_cred_hist_length = int(
    input("Credit history length in years: ")
)


# ==========================================
# 3. CREATE INPUT DATAFRAME
# ==========================================

input_data = pd.DataFrame({
    "person_age": [person_age],
    "person_income": [person_income],
    "person_home_ownership": [person_home_ownership],
    "person_emp_length": [person_emp_length],
    "loan_intent": [loan_intent],
    "loan_grade": [loan_grade],
    "loan_amnt": [loan_amnt],
    "loan_int_rate": [loan_int_rate],
    "loan_percent_income": [loan_percent_income],
    "cb_person_default_on_file": [
        cb_person_default_on_file
    ],
    "cb_person_cred_hist_length": [
        cb_person_cred_hist_length
    ]
})


# ==========================================
# 4. ENCODE CATEGORICAL DATA
# ==========================================

categorical_columns = input_data.select_dtypes(
    include=["object"]
).columns

input_data = pd.get_dummies(
    input_data,
    columns=categorical_columns,
    drop_first=True
)


# ==========================================
# 5. MATCH TRAINING FEATURES
# ==========================================

# Load the original dataset to get the
# exact feature columns used during training

training_data = pd.read_csv(
    "data/credit_risk_dataset.csv"
)

training_data = training_data.drop(
    "loan_status",
    axis=1
)

training_data = pd.get_dummies(
    training_data,
    columns=training_data.select_dtypes(
        include=["object"]
    ).columns,
    drop_first=True
)

input_data = input_data.reindex(
    columns=training_data.columns,
    fill_value=0
)


# ==========================================
# 6. MAKE PREDICTION
# ==========================================

prediction = model.predict(input_data)[0]

probability = model.predict_proba(
    input_data
)[0][1]


# ==========================================
# 7. DISPLAY RESULT
# ==========================================

print("\n==========================================")
print("          CREDIT RISK RESULT")
print("==========================================")

print(
    f"Probability of default: "
    f"{probability * 100:.2f}%"
)

if prediction == 1:
    print("\nPrediction: HIGHER DEFAULT RISK")
else:
    print("\nPrediction: LOWER DEFAULT RISK")

print("\nPrediction completed.")