import pandas as pd
import tkinter as tk
from tkinter import filedialog
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler, Normalizer
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from catboost import CatBoostClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from imblearn.over_sampling import SMOTE
import openpyxl

def load_dataset():
    root = tk.Tk()
    root.withdraw()  # Hide the root window
    file_path = filedialog.askopenfilename(title="Select file", filetypes=[("CSV files", "*.csv")])
    if not file_path:
        raise ValueError("No file selected or operation cancelled.")
    return pd.read_csv(file_path)

# Load the dataset
df = load_dataset()

# Display basic information about the dataset
print(df.info())
print(df.describe())

# Check for missing values
missing_values = df.isnull().sum()
print(missing_values)

# Handle missing values (example: dropping rows with missing values)
df = df.dropna()
# Alternatively, you can fill missing values
# df = df.fillna(df.mean())

# Convert categorical variables to numeric using one-hot encoding
df = pd.get_dummies(df, drop_first=True)

# Split the data into features (X) and target (y)
X = df.drop('Attrition_Yes', axis=1)  # Assuming 'Attrition_Yes' is the target variable
y = df['Attrition_Yes']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize the models
models = {
    "KNN": KNeighborsClassifier(),
    "Random Forest": RandomForestClassifier(),
    "Decision Tree": DecisionTreeClassifier(),
    "SVM": SVC(probability=True),
    "Naive Bayes": GaussianNB(),
    "Gradient Boosting": GradientBoostingClassifier(),
    "AdaBoost": AdaBoostClassifier(),
    "CatBoost": CatBoostClassifier(verbose=0)
}

# Train and evaluate the models with raw data
results = {}
for model_name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)[:, 1]
    results[model_name] = {
        "Accuracy": accuracy_score(y_test, y_pred),
        "Precision": precision_score(y_test, y_pred),
        "Recall": recall_score(y_test, y_pred),
        "F1-score": f1_score(y_test, y_pred),
        "AUC-ROC": roc_auc_score(y_test, y_pred_proba)
    }

# Display the results for raw data
results_df = pd.DataFrame(results).T
print("Results with Raw Data")
print(results_df)

# Initialize the scalers and normalizers
scalers = {
    "Min-Max Scaling": MinMaxScaler(),
    "Standardization": StandardScaler(),
    "L2 Normalization": Normalizer(norm='l2')
}

# Apply the scalers and normalizers
scaled_data = {}
for scaler_name, scaler in scalers.items():
    scaled_data[scaler_name] = scaler.fit_transform(X)

# Combine L1 and L2 Normalization data
combined_L1_L2_data = (scaled_data["L2 Normalization"]) / 1

# Store the scaled data in a dictionary
scaled_data_dict = {name: pd.DataFrame(data, columns=X.columns) for name, data in scaled_data.items()}
scaled_data_dict["L1 & L2 Normalization"] = pd.DataFrame(combined_L1_L2_data, columns=X.columns)

# Save the scaled data to separate sheets in an Excel file
with pd.ExcelWriter('scaled_data.xlsx') as writer:
    for scaler_name, data in scaled_data_dict.items():
        data.to_excel(writer, sheet_name=scaler_name)

# Perform Correlation Analysis without displaying the results
correlation_results = {}
for scaler_name, data in scaled_data_dict.items():
    correlation_results[scaler_name] = data.corr()

# SMOTE Analysis
smote = SMOTE()
X_smote, y_smote = smote.fit_resample(X_train, y_train)

# Train and evaluate the models with scaled/normalized data
for scaler_name, data in scaled_data_dict.items():
    if scaler_name in ["L2 Normalization", "L1 Normalization"]:
        continue
    X_train_scaled, X_test_scaled, y_train_scaled, y_test_scaled = train_test_split(data, y, test_size=0.2, random_state=42)
        
    results_scaled = {}
    for model_name, model in models.items():
        model.fit(X_train_scaled, y_train_scaled)
        y_pred_scaled = model.predict(X_test_scaled)
        y_pred_proba_scaled = model.predict_proba(X_test_scaled)[:, 1]
        results_scaled[model_name] = {
            "Accuracy": accuracy_score(y_test_scaled, y_pred_scaled),
            "Precision": precision_score(y_test_scaled, y_pred_scaled),
            "Recall": recall_score(y_test_scaled, y_pred_scaled),
            "F1-score": f1_score(y_test_scaled, y_pred_scaled),
            "AUC-ROC": roc_auc_score(y_test_scaled, y_pred_proba_scaled)
        }
    
    # Display the results for scaled/normalized data
    results_df_scaled = pd.DataFrame(results_scaled).T
    print(f'Results for {scaler_name}' if scaler_name != "L1 & L2 Normalization" else 'Results for L1 & L2 Normalization')
    print(results_df_scaled)
