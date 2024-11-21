import pandas as pd
data = pd.read_csv("WA_Fn-UseC_-Telco-Customer-Churn-main.csv")

#drop irrelevant/unimportant columns 
data.drop()

# Manually map 'Yes' to 1 and 'No' to 0 for specific columns
yes_no_columns = [
    'Partner',
    'Dependents',
    'PaperlessBilling',
    'Churn',
    'OnlineSecurity',
    'OnlineBackup',
    'DeviceProtection',
    'TechSupport',
    'StreamingMovies'
]

# Replace 'Yes' with 1 and 'No' with 0 in each specified column
for column in yes_no_columns:
    data[column] = data[column].replace({'Yes': 1, 'No': 0})

# create binary columns for each category in 'InternetService'
data['InternetService_DSL'] = (data['InternetService'] == 'DSL').astype(int)
data['InternetService_FiberOptic'] = (data['InternetService'] == 'Fiber optic').astype(int)
data['InternetService_No'] = (data['InternetService'] == 'No').astype(int)

# create binary columns for each category in 'Contract'
data['Contract_MonthToMonth'] = (data['Contract'] == 'Month-to-month').astype(int)
data['Contract_OneYear'] = (data['Contract'] == 'One year').astype(int)
data['Contract_TwoYear'] = (data['Contract'] == 'Two year').astype(int)

# create binary columns for each category in 'PaymentMethod'
data['PaymentMethod_BankTransfer'] = (data['PaymentMethod'] == 'Bank transfer (automatic)').astype(int)
data['PaymentMethod_CreditCard'] = (data['PaymentMethod'] == 'Credit card (automatic)').astype(int)
data['PaymentMethod_ElectronicCheck'] = (data['PaymentMethod'] == 'Electronic check').astype(int)
data['PaymentMethod_MailedCheck'] = (data['PaymentMethod'] == 'Mailed check').astype(int)

# Specify the columns you want to keep in the dataset
columns_to_keep = [col for col in data.columns if col not in ['InternetService', 'Contract', 'PaymentMethod']]

# Reassign data to only include the specified columns
data = data[columns_to_keep]

# Display the first few rows to verify the new binary columns
print(data.head())

# Define function to categorize tenure
def categorize_tenure(tenure):
    if tenure < 10:
        return '0-10'
    elif tenure < 30:
        return '10-30'
    elif tenure < 50:
        return '30-50'
    elif tenure < 70:
        return '50-70'
    elif tenure < 100:
        return '70-100'
    else:
        return '100+'

# Apply the function to the 'tenure' column and overwrite it with categorical values
data['tenure'] = data['tenure'].apply(categorize_tenure)

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
data['MonthlyCharges'] = data['MonthlyCharges'].apply(categorize_monthly_charges)