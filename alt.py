import pandas as pd
import seaborn as sns
import math #idk if i need it yet
import matplotlib as mpl # idk if i need it yet
import scipy
import sklearn.preprocessing

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
unique_per_column = {col: df[col].unique() for col in df.columns}
for column, values in unique_per_column.items():
    print(f"{column}: {values}")

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


print(df)

print(df.dtypes)