# Anshul Hudda
# CU24250002
# Sec - 'B'

# ============================================================
# EXPERIMENT 10
# End-to-End Data Analytics Project:
# Business Intelligence and Predictive Analytics
# Dataset: IBM HR Analytics Employee Attrition
# ============================================================

# -----------------------------
# 1. IMPORT LIBRARIES
# -----------------------------
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from scipy.stats import chi2_contingency, ttest_ind

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
    roc_auc_score,
    roc_curve
)

# -----------------------------
# 2. LOAD DATASET
# -----------------------------
file_path = "/content/WA_Fn-UseC_-HR-Employee-Attrition.csv"

df = pd.read_csv(file_path)

print("=" * 70)
print("IBM HR ANALYTICS - END-TO-END DATA ANALYTICS PROJECT")
print("=" * 70)

print("\nDataset Shape:")
print(df.shape)

print("\nFirst 5 Rows:")
display(df.head())

# -----------------------------
# 3. DATA INFORMATION
# -----------------------------
print("\nColumn Names:")
print(df.columns.tolist())

print("\nData Types:")
print(df.dtypes)

print("\nDataset Information:")
df.info()

# -----------------------------
# 4. DATA QUALITY CHECK
# -----------------------------
print("\n" + "=" * 70)
print("DATA QUALITY CHECK")
print("=" * 70)

print("\nMissing Values:")
missing = df.isnull().sum()
print(missing[missing > 0] if missing.sum() > 0 else "No missing values found.")

print("\nDuplicate Rows:")
print(df.duplicated().sum())

# -----------------------------
# 5. REMOVE IRRELEVANT COLUMNS
# -----------------------------
# These columns do not provide useful predictive information
# for employee attrition.
irrelevant_columns = [
    "EmployeeCount",
    "EmployeeNumber",
    "Over18",
    "StandardHours"
]

df = df.drop(
    columns=[c for c in irrelevant_columns if c in df.columns],
    errors="ignore"
)

print("\nShape after removing irrelevant columns:")
print(df.shape)

# -----------------------------
# 6. BASIC STATISTICS
# -----------------------------
print("\n" + "=" * 70)
print("DESCRIPTIVE STATISTICS")
print("=" * 70)

display(df.describe().T)

# -----------------------------
# 7. TARGET DISTRIBUTION
# -----------------------------
print("\nAttrition Distribution:")
print(df["Attrition"].value_counts())

print("\nAttrition Percentage:")
print(
    (df["Attrition"].value_counts(normalize=True) * 100).round(2)
)

# -----------------------------
# 8. OUTLIER DETECTION
# -----------------------------
numeric_columns = df.select_dtypes(include=np.number).columns

outlier_summary = {}

for col in numeric_columns:
    Q1 = df[col].quantile(0.25)
    Q3 = df[col].quantile(0.75)
    IQR = Q3 - Q1

    lower = Q1 - 1.5 * IQR
    upper = Q3 + 1.5 * IQR

    count = ((df[col] < lower) | (df[col] > upper)).sum()
    outlier_summary[col] = count

outlier_df = pd.DataFrame(
    list(outlier_summary.items()),
    columns=["Column", "Outlier_Count"]
)

print("\nOutlier Summary:")
display(outlier_df.sort_values("Outlier_Count", ascending=False))

# -----------------------------
# 9. OUTLIER VISUALIZATION
# -----------------------------
plt.figure(figsize=(12, 6))
sns.boxplot(data=df[numeric_columns])
plt.xticks(rotation=90)
plt.title("Boxplot of Numerical Variables")
plt.tight_layout()
plt.show()

# -----------------------------
# 10. EDA - ATTRITION
# -----------------------------
plt.figure(figsize=(6, 4))
sns.countplot(data=df, x="Attrition")
plt.title("Employee Attrition Distribution")
plt.xlabel("Attrition")
plt.ylabel("Number of Employees")
plt.show()

# -----------------------------
# 11. ATTRITION BY DEPARTMENT
# -----------------------------
plt.figure(figsize=(8, 5))
sns.countplot(data=df, x="Department", hue="Attrition")
plt.title("Employee Attrition by Department")
plt.xticks(rotation=20)
plt.show()

# -----------------------------
# 12. ATTRITION BY JOB ROLE
# -----------------------------
plt.figure(figsize=(11, 6))
sns.countplot(
    data=df,
    y="JobRole",
    hue="Attrition"
)
plt.title("Employee Attrition by Job Role")
plt.show()

# -----------------------------
# 13. ATTRITION BY OVERTIME
# -----------------------------
plt.figure(figsize=(7, 5))
sns.countplot(data=df, x="OverTime", hue="Attrition")
plt.title("Employee Attrition vs Overtime")
plt.show()

# -----------------------------
# 14. ATTRITION BY BUSINESS TRAVEL
# -----------------------------
plt.figure(figsize=(8, 5))
sns.countplot(
    data=df,
    x="BusinessTravel",
    hue="Attrition"
)
plt.title("Employee Attrition vs Business Travel")
plt.xticks(rotation=15)
plt.show()

# -----------------------------
# 15. AGE DISTRIBUTION
# -----------------------------
plt.figure(figsize=(8, 5))
sns.histplot(
    data=df,
    x="Age",
    hue="Attrition",
    kde=True,
    bins=20
)
plt.title("Age Distribution by Attrition")
plt.show()

# -----------------------------
# 16. MONTHLY INCOME VS ATTRITION
# -----------------------------
plt.figure(figsize=(8, 5))
sns.boxplot(
    data=df,
    x="Attrition",
    y="MonthlyIncome"
)
plt.title("Monthly Income vs Employee Attrition")
plt.show()

# -----------------------------
# 17. YEARS AT COMPANY
# -----------------------------
plt.figure(figsize=(8, 5))
sns.boxplot(
    data=df,
    x="Attrition",
    y="YearsAtCompany"
)
plt.title("Years at Company vs Attrition")
plt.show()

# -----------------------------
# 18. JOB SATISFACTION
# -----------------------------
plt.figure(figsize=(8, 5))
sns.countplot(
    data=df,
    x="JobSatisfaction",
    hue="Attrition"
)
plt.title("Job Satisfaction vs Attrition")
plt.xlabel("Job Satisfaction Level")
plt.show()

# -----------------------------
# 19. CORRELATION ANALYSIS
# -----------------------------
corr = df.select_dtypes(include=np.number).corr()

plt.figure(figsize=(14, 10))
sns.heatmap(
    corr,
    annot=False,
    cmap="coolwarm",
    center=0
)
plt.title("Correlation Heatmap")
plt.tight_layout()
plt.show()

# -----------------------------
# 20. STATISTICAL ANALYSIS
# -----------------------------
print("\n" + "=" * 70)
print("STATISTICAL ANALYSIS")
print("=" * 70)

# Chi-Square Test: Overtime vs Attrition
table = pd.crosstab(
    df["OverTime"],
    df["Attrition"]
)

chi2, p_value, dof, expected = chi2_contingency(table)

print("\nChi-Square Test: Overtime vs Attrition")
print("Chi-Square Statistic:", round(chi2, 4))
print("P-value:", round(p_value, 6))

if p_value < 0.05:
    print("Result: Statistically significant relationship.")
else:
    print("Result: No statistically significant relationship.")

# T-Test: Age vs Attrition
age_yes = df[df["Attrition"] == "Yes"]["Age"]
age_no = df[df["Attrition"] == "No"]["Age"]

t_stat, t_p = ttest_ind(
    age_yes,
    age_no,
    equal_var=False
)

print("\nIndependent T-Test: Age vs Attrition")
print("T-statistic:", round(t_stat, 4))
print("P-value:", round(t_p, 6))

if t_p < 0.05:
    print("Result: Mean age differs significantly between groups.")
else:
    print("Result: No statistically significant difference in mean age.")

# -----------------------------
# 21. BUSINESS KPIs
# -----------------------------
total_employees = len(df)
employees_left = (df["Attrition"] == "Yes").sum()
employees_stayed = (df["Attrition"] == "No").sum()

attrition_rate = employees_left / total_employees * 100

avg_income = df["MonthlyIncome"].mean()
avg_age = df["Age"].mean()
avg_years_company = df["YearsAtCompany"].mean()

print("\n" + "=" * 70)
print("BUSINESS KPIs")
print("=" * 70)

print(f"Total Employees      : {total_employees}")
print(f"Employees Left       : {employees_left}")
print(f"Employees Stayed     : {employees_stayed}")
print(f"Attrition Rate       : {attrition_rate:.2f}%")
print(f"Average Monthly Income: {avg_income:.2f}")
print(f"Average Employee Age : {avg_age:.2f}")
print(f"Average Years at Company: {avg_years_company:.2f}")

# -----------------------------
# 22. PREPARE DATA FOR ML
# -----------------------------
model_df = df.copy()

model_df["Attrition"] = model_df["Attrition"].map({
    "Yes": 1,
    "No": 0
})

X = model_df.drop("Attrition", axis=1)
y = model_df["Attrition"]

categorical_features = X.select_dtypes(
    include=["object"]
).columns.tolist()

numerical_features = X.select_dtypes(
    include=np.number
).columns.tolist()

print("\nCategorical Features:")
print(categorical_features)

print("\nNumerical Features:")
print(numerical_features)

# -----------------------------
# 23. PREPROCESSING PIPELINE
# -----------------------------
preprocessor = ColumnTransformer(
    transformers=[
        (
            "num",
            StandardScaler(),
            numerical_features
        ),
        (
            "cat",
            OneHotEncoder(
                handle_unknown="ignore",
                drop="first"
            ),
            categorical_features
        )
    ]
)

# -----------------------------
# 24. TRAIN TEST SPLIT
# -----------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

print("\nTraining Records:", X_train.shape[0])
print("Testing Records :", X_test.shape[0])

# -----------------------------
# 25. LOGISTIC REGRESSION MODEL
# -----------------------------
model = Pipeline(
    steps=[
        ("preprocessor", preprocessor),
        (
            "classifier",
            LogisticRegression(
                max_iter=2000,
                class_weight="balanced"
            )
        )
    ]
)

model.fit(X_train, y_train)

# -----------------------------
# 26. PREDICTION
# -----------------------------
y_pred = model.predict(X_test)
y_prob = model.predict_proba(X_test)[:, 1]

# -----------------------------
# 27. MODEL EVALUATION
# -----------------------------
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
roc_auc = roc_auc_score(y_test, y_prob)

print("\n" + "=" * 70)
print("MODEL PERFORMANCE")
print("=" * 70)

print(f"Accuracy  : {accuracy:.4f}")
print(f"Precision : {precision:.4f}")
print(f"Recall    : {recall:.4f}")
print(f"F1-Score  : {f1:.4f}")
print(f"ROC-AUC   : {roc_auc:.4f}")

print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# -----------------------------
# 28. CONFUSION MATRIX
# -----------------------------
cm = confusion_matrix(y_test, y_pred)

plt.figure(figsize=(6, 5))
sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    cmap="Blues"
)
plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()

# -----------------------------
# 29. ROC CURVE
# -----------------------------
fpr, tpr, thresholds = roc_curve(
    y_test,
    y_prob
)

plt.figure(figsize=(7, 5))
plt.plot(
    fpr,
    tpr,
    label=f"ROC-AUC = {roc_auc:.3f}"
)
plt.plot(
    [0, 1],
    [0, 1],
    linestyle="--"
)
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve - Logistic Regression")
plt.legend()
plt.show()

# -----------------------------
# 30. MODEL COEFFICIENTS
# -----------------------------
trained_preprocessor = model.named_steps["preprocessor"]
trained_classifier = model.named_steps["classifier"]

feature_names = trained_preprocessor.get_feature_names_out()
coefficients = trained_classifier.coef_[0]

importance_df = pd.DataFrame({
    "Feature": feature_names,
    "Coefficient": coefficients,
    "Absolute_Importance": np.abs(coefficients)
})

importance_df = importance_df.sort_values(
    "Absolute_Importance",
    ascending=False
)

print("\nTop 15 Predictive Features:")
display(
    importance_df.head(15)
)

plt.figure(figsize=(10, 7))

top_features = importance_df.head(15).sort_values(
    "Coefficient"
)

sns.barplot(
    data=top_features,
    x="Coefficient",
    y="Feature"
)

plt.title("Top Predictive Features for Employee Attrition")
plt.tight_layout()
plt.show()

# -----------------------------
# 31. BUSINESS INSIGHTS
# -----------------------------
print("\n" + "=" * 70)
print("KEY BUSINESS INSIGHTS")
print("=" * 70)

overtime_attrition = pd.crosstab(
    df["OverTime"],
    df["Attrition"],
    normalize="index"
) * 100

print("\nAttrition percentage by overtime:")
print(overtime_attrition.round(2))

department_attrition = pd.crosstab(
    df["Department"],
    df["Attrition"],
    normalize="index"
) * 100

print("\nAttrition percentage by department:")
print(department_attrition.round(2))

role_attrition = pd.crosstab(
    df["JobRole"],
    df["Attrition"],
    normalize="index"
) * 100

print("\nAttrition percentage by job role:")
print(role_attrition.round(2))

print("\n" + "=" * 70)
print("END-TO-END ANALYTICS PROJECT COMPLETED")
print("=" * 70)



# # Data Story / Project Summary

# ## Business Problem

# Employee attrition is an important business problem because high employee turnover can increase recruitment costs, training costs, workload, and productivity loss.

# The objective of this project is to analyze employee data and identify factors associated with employee attrition. A predictive model is also developed to estimate whether an employee is likely to leave the organization.

# ## Data Understanding

# The IBM HR Analytics Employee Attrition dataset contains employee demographic, job, compensation, satisfaction, and career-related information.

# The target variable is:

# - **Attrition:** Yes / No

# ## Data Preprocessing

# The dataset was examined for:

# - Missing values
# - Duplicate records
# - Data types
# - Irrelevant columns
# - Numerical and categorical variables
# - Potential outliers

# Irrelevant columns such as EmployeeNumber, EmployeeCount, Over18, and StandardHours were removed from the predictive analysis.

# Categorical variables were converted using One-Hot Encoding, while numerical variables were standardized before model training.

# ## Exploratory Data Analysis

# EDA was performed using:

# - Attrition distribution
# - Department analysis
# - Job role analysis
# - Overtime analysis
# - Business travel analysis
# - Age distribution
# - Monthly income analysis
# - Job satisfaction analysis
# - Years at company analysis
# - Correlation heatmap

# ## Statistical Analysis

# Two statistical tests were performed:

# 1. Chi-Square Test – Overtime vs Attrition
# 2. Independent T-Test – Age vs Attrition

# A significance level of 0.05 was used to determine statistical significance.

# ## Predictive Analytics

# Logistic Regression was selected because the target variable is binary:

# - 1 = Employee leaves
# - 0 = Employee stays

# The model was evaluated using:

# - Accuracy
# - Precision
# - Recall
# - F1-Score
# - ROC-AUC

# ## Business Insight

# The analysis helps identify employee groups and workplace factors associated with higher attrition. These insights can support HR teams in developing targeted employee retention strategies.

# ## Overall Story

# Employee data → Data Cleaning → EDA → Statistical Analysis → Predictive Modeling → Model Evaluation → Business Insights → HR Recommendations
