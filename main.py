import pandas as pd
import seaborn as sns
import math #idk if i need it yet
import matplotlib.pyplot as plt
import numpy as np
import sklearn.preprocessing
from scipy.stats import chi2_contingency
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, roc_auc_score
from collections import Counter

# getting the df into our code
df = pd.read_csv("WA_Fn-UseC_-Telco-Customer-Churn-main.csv")

# removing columns that clearly provide no use
df.drop(columns=["gender", "customerID"], inplace=True)

# let's remind outselves that our goal is to predict if our classmates may churn from their own phone plans
# clearly we'll have no way to tell if they end up churning our not, but the experiment is more social/fun than scientific. 
# for this objective however, we need to make sure that our TelCo IBM Kaggle dataset lines up roughly with our classmates
# thus, we'll remove columns they might not know on the spot, as well as columns irrelevant to them such as 'OnlineSecurity'
df.drop(columns=["OnlineSecurity", "OnlineBackup", "DeviceProtection", "TechSupport", "StreamingTV", "StreamingMovies", 
                   "PaperlessBilling", "TotalCharges"], inplace=True)
# of course we can assume our classmates are not Senior Citizens, and I'd like to hope most don't have kids in undergrad, 
# but leaving those variables in can help us make the most of our dataset, and we can pre-configure them to false when we ask them.

# renaming tenure for clarity's sake
df = df.rename(columns={"tenure": "Months_Tenure"})

# Manually map certain columns to have their 'Yes' and 'No' to be converted to 1 and 0, respectively
yes_no_columns = [
    "Partner",
    "Dependents",
    "PhoneService",
    "MultipleLines",
    "InternetService",
    "Churn"
]

# Replace 'Yes' with 1 and 'No' with 0 in each specified column
for column in yes_no_columns:
    df[column].replace({'Yes': 1, 'No': 0}, inplace=True)

# consolidate the no/yes values for simplification's sake
df["MultipleLines"].replace({"No phone service": 0}, inplace=True)
df["InternetService"].replace({'DSL': 1, 'Fiber optic': 1}, inplace=True)

# Define function to categorize tenure
def categorize_tenure(Months_Tenure):
    if Months_Tenure <= 6:
        return '0-6'
    elif Months_Tenure <= 18:
        return '7-18'
    elif Months_Tenure <= 36:
        return '19-36'
    elif Months_Tenure <= 72:
        return '37-72'

# Apply the function to the 'Months_Tenure' column and overwrite it with categorical values
df['Months_Tenure'] = df['Months_Tenure'].apply(categorize_tenure)

# Define function to categorize MonthlyCharges
def categorize_monthly_charges(charge):
    if charge < 25:
        return '0-25'
    elif charge < 50:
        return '25-50'
    elif charge < 75:
        return '50-75'
    elif charge < 100:
        return '75-100'
    else:
        return '100+'
    
# Apply the function to the 'MonthlyCharges' column and overwrite it with categorical values
df['MonthlyCharges'] = df['MonthlyCharges'].apply(categorize_monthly_charges)

# checking all the unique values in our data
#unique_per_column = {col: df[col].unique() for col in df.columns}
#for column, values in unique_per_column.items():
#    print(f"{column}: {values}")

# seperating the types of columns before we encode the categorical variables for easier work later
categorical_columns = ["Months_Tenure", "Contract", "PaymentMethod", "MonthlyCharges"]
numerical_columns = df.drop(columns=categorical_columns)

# Initialize encoder
encoder = sklearn.preprocessing.OneHotEncoder(sparse_output=False)

# Fit and transform
encoded_array = encoder.fit_transform(df[categorical_columns])

# Convert back to DataFrame
encoded_df = pd.DataFrame(encoded_array, columns=encoder.get_feature_names_out(categorical_columns))

# combining the numerical and categorical dfs
df = pd.concat([numerical_columns, encoded_df], axis=1)

# covering all dtype to int for clarity
df = df.astype(int)

# Function to calculate Cramér's V, which will help us analyze which variables are important 
# (through their strength of associaton in a heatmap)
def cramers_v(x, y):
    contingency_table = pd.crosstab(x, y)
    chi2 = chi2_contingency(contingency_table)[0]
    n = contingency_table.sum().sum()
    return np.sqrt(chi2 / (n * (min(contingency_table.shape) - 1)))

# Calculate Cramér's V for all variables against Churn (they're all binary)
binary_columns = ['SeniorCitizen', 'Partner', 'Dependents', 'PhoneService',
       'MultipleLines', 'InternetService', 'Months_Tenure_0-6',
       'Months_Tenure_19-36', 'Months_Tenure_37-72', 'Months_Tenure_7-18',
       'Contract_Month-to-month', 'Contract_One year', 'Contract_Two year',
       'PaymentMethod_Bank transfer (automatic)',
       'PaymentMethod_Credit card (automatic)',
       'PaymentMethod_Electronic check', 'PaymentMethod_Mailed check',
       'MonthlyCharges_0-25', 'MonthlyCharges_100+', 'MonthlyCharges_25-50',
       'MonthlyCharges_50-75', 'MonthlyCharges_75-100'] 
cramers_results = {col: cramers_v(df['Churn'], df[col]) for col in binary_columns}

# Convering the results into a df and displaying it as a barplot shows us 
CramerDf = pd.DataFrame(cramers_results.items(), columns=["Variable", "Cramer's V"])
CramerDf = CramerDf.sort_values(by="Cramer's V")
#sns.catplot(x="Cramer's V", y="Variable", data=CramerDf, kind="bar")
#plt.show()

# the plot shows us that several variables are relatively irrelevant and worth dropping
# we also need to drop at least one category from each categorical variable we performed onehotencoding on
df.drop(columns=["MonthlyCharges_25-50", "MonthlyCharges_50-75", "PhoneService","MultipleLines", "Months_Tenure_19-36", "PaymentMethod_Mailed check"])

# now we can finally prepare our logistic regression to measure the probability of churn given our binary variables.

X = df [['SeniorCitizen', 'Partner', 'Dependents', 'InternetService', 'Months_Tenure_0-6', 
    'Months_Tenure_37-72', 'Months_Tenure_7-18', 'Contract_Month-to-month', 'Contract_One year', 
    'Contract_Two year', 'PaymentMethod_Bank transfer (automatic)', 'PaymentMethod_Credit card (automatic)',
    'PaymentMethod_Electronic check', 'MonthlyCharges_0-25', 
    'MonthlyCharges_100+', 'MonthlyCharges_75-100']]
y = df['Churn']

# Step 1: Split the Data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 2: Check Class Distribution
print(f"Class distribution before SMOTE: {Counter(y_train)}")

# Step 3: Apply SMOTE to Balance the Dataset (i added this 2nd, after testing default logistic regression)
smote = SMOTE(random_state=42)
X_train_sm, y_train_sm = smote.fit_resample(X_train, y_train)
print(f"Class distribution after SMOTE: {Counter(y_train_sm)}")

# Step 4: Train Logistic Regression Model
log_model = LogisticRegression()
log_model.fit(X_train_sm, y_train_sm)

# Step 5: Make Predictions
# Predicted probabilities
y_pred_probs = log_model.predict_proba(X_test)[:, 1]

# Predicted classes with a custom threshold (I added this 3rd, to balance the model a little more)
threshold = 0.6
y_pred_custom = (y_pred_probs >= threshold).astype(int)

# Step 6: Evaluate the Model
# Default threshold evaluation
print("Logistic Regression Metrics (Default Threshold):")
print(f"Accuracy: {accuracy_score(y_test, log_model.predict(X_test))}")
print(f"AUC: {roc_auc_score(y_test, y_pred_probs)}")
print(f"Confusion Matrix:\n{confusion_matrix(y_test, log_model.predict(X_test))}")

# Custom threshold evaluation. In our business case, we'd want to be a less strict than the model. 
print("\nLogistic Regression Metrics (Custom Threshold 0.6):")
print(f"Confusion Matrix:\n{confusion_matrix(y_test, y_pred_custom)}")
print(classification_report(y_test, y_pred_custom))

# Step 7: Model Coefficients
print("\nLogistic Regression Coefficients:")
print("Intercept:", log_model.intercept_)
print("Coefficients:", log_model.coef_)