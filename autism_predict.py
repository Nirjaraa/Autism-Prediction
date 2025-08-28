

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import LabelEncoder
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split, cross_val_score, RandomizedSearchCV
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

import pickle

# read csv data to a pandas dataframe
df = pd.read_csv("data/train.csv")

df.shape

df.head()

df.info()

# convert age column datatype
df["age"] = df["age"].astype(int)

df.head(2)

pd.set_option('display.max_columns', None)

for col in df.columns:
    numerical_features = ["ID", "age", "result"]
    if col not in numerical_features:
        print(col, df[col].unique())
        print("-"*50)

# dropping ID and age_desc column
df = df.drop(columns=["ID", "age_desc"])

df.shape

df["contry_of_res"].unique()

# define the mapping dic for country names
mapping = {
    "Viet Nam": "Vietnam",
    "Hong Kong": "China",
    "AmericanSamoa": "United States"

}
# replace value in the country column
df["contry_of_res"] = df["contry_of_res"].replace(mapping)

df["contry_of_res"].unique()

df["Class/ASD"].value_counts()

df.columns

df.describe()

sns.set_theme(style="darkgrid")

# distribution plot
sns.histplot(df["age"], kde=True)
plt.title("Distribution of Age")

# calculate mean and median
age_mean = df["age"].mean()
age_median = df["age"].median()
print("mean:", age_mean)
print("median", age_median)
plt.axvline(age_mean, color="red", linestyle="dashed", label="mean")
plt.axvline(age_median, color="green", linestyle="dashed", label="median")
plt.legend()

plt.show()

# distribution plot
sns.histplot(df["result"], kde=True)
plt.title("Distribution of result")

# calculate mean and median
result_mean = df["result"].mean()
result_median = df["result"].median()
print("mean:", result_mean)
print("median", result_median)
plt.axvline(result_mean, color="red", linestyle="dashed", label="mean")
plt.axvline(result_median, color="green", linestyle="dashed", label="median")
plt.legend()

plt.show()

sns.boxplot(x=df["age"])
plt.title("Box plot for age")
plt.xlabel("Age")
plt.show()

sns.boxplot(x=df["result"])
plt.title("Box plot for result")
plt.xlabel("result")
plt.show()

Q1 = df["age"].quantile(0.25)
Q3 = df["age"].quantile(0.75)
IQR = Q3-Q1
lower_bound = Q1-1.5*IQR
upper_bound = Q3+1.5*IQR
age_outliers = df[(df["age"] < lower_bound) | (df["age"] > upper_bound)]
print("Number of age outliers:", len(age_outliers))

Q1 = df["result"].quantile(0.25)
Q3 = df["result"].quantile(0.75)
IQR = Q3-Q1
lower_bound = Q1-1.5*IQR
upper_bound = Q3+1.5*IQR
age_outliers = df[(df["result"] < lower_bound) | (df["result"] > upper_bound)]
print("Number of result outliers:", len(age_outliers))

df.columns


# countplot for target column (CLASS/ASD)
sns.countplot(x=df["Class/ASD"])
plt.title("Countplot for target column")
plt.xlabel("Class/ASD")
plt.ylabel("Count")
plt.show()

df["Class/ASD"].value_counts()

df["ethnicity"] = df["ethnicity"].replace({"?": "Others", "others": "Others"})

df["relation"] = df["relation"].replace(
    {"?": "Others", "Relative": "Others", "Parent": "Others", "Health care professional": "Others"})

df.head()


# Identify categorical columns to encode (excluding the target variable if it's already numerical)
categorical_cols_to_encode = ['gender', 'ethnicity', 'jaundice',
                              'austim', 'contry_of_res', 'used_app_before', 'relation']

# Apply Label Encoding
label_encoders = {}
for col in categorical_cols_to_encode:
    label_encoders[col] = LabelEncoder()
    df[col] = label_encoders[col].fit_transform(df[col])

print(df.head())

filename = 'label_encoders.pkl'

# Save the label_encoders dictionary to the pickle file
with open(filename, 'wb') as f:
    pickle.dump(label_encoders, f)

print(f"Label encoders saved to {filename}")

plt.figure(figsize=(20, 15))
sns.heatmap(df.corr(), annot=True, fmt=".2f")
plt.title("Correlation matrix")
plt.savefig("plot_name.png")
plt.close()


def replace_outliers_with_median(df, column):
    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)

    IQR = Q3-Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    median = df[column].median()
    # replace outliers with median
    df[column] = df[column].apply(
        lambda x: median if x < lower_bound or x > upper_bound else x)
    return df


df = replace_outliers_with_median(df, "age")
df = replace_outliers_with_median(df, "result")

df.shape

df.columns

X = df.drop(columns=["Class/ASD"])
y = df["Class/ASD"]
print(X, y)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42)

print(y_train.shape)
print(y_test.shape)

y_train.value_counts()

y_test.value_counts()

smote = SMOTE(random_state=42)
X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)

print(y_train_resampled.value_counts())

models = {
    "Decision Tree": DecisionTreeClassifier(random_state=42),
    "Random Forest": RandomForestClassifier(random_state=42),
    "XGBoost": XGBClassifier(random_state=42)
}

cv_scores = {}

for model_name, model in models.items():
    print(f"Training {model_name}...")
    scores = cross_val_score(model, X_train_resampled, y_train_resampled, cv=5)
    cv_scores[model_name] = scores
    print(f"{model_name} CV scores: {np.mean(scores):.2f}")

decision_tree_model = DecisionTreeClassifier(random_state=42)
random_forest_model = RandomForestClassifier(random_state=42)
xgboost_model = XGBClassifier(random_state=42)

param_grid_dt = {
    "criterion": ["gini", "entropy"],
    "max_depth": [None, 10, 20, 30, 50, 70],
    "min_samples_split": [2, 5, 10],
    "min_samples_leaf": [1, 2, 4]
}

param_grid_rf = {
    "n_estimators": [50, 100, 200, 300, 400],
    "max_depth": [None, 10, 20, 30, 40, 50],
    "min_samples_split": [2, 5, 10],
    "min_samples_leaf": [1, 2, 4]
}

param_grid_xgb = {
    "n_estimators": [50, 100, 200, 400, 500],
    "max_depth": [3, 5, 7, 9],
    "learning_rate": [0.01, 0.1, 0.2, 0.3],
    "subsample": [0.5, 0.7, 1.0],
    "colsample_bytree": [0.5, 0.7, 1.0]
}

random_search_dt = RandomizedSearchCV(
    estimator=decision_tree_model, param_distributions=param_grid_dt, n_iter=10, cv=5, n_jobs=-1, verbose=2, random_state=42)
random_search_rf = RandomizedSearchCV(
    estimator=random_forest_model, param_distributions=param_grid_rf, n_iter=10, cv=5, n_jobs=-1, verbose=2, random_state=42)
random_search_xgb = RandomizedSearchCV(
    estimator=xgboost_model, param_distributions=param_grid_xgb, n_iter=10, cv=5, n_jobs=-1, verbose=2, random_state=42)

random_search_dt.fit(X_train_resampled, y_train_resampled)
random_search_rf.fit(X_train_resampled, y_train_resampled)
random_search_xgb.fit(X_train_resampled, y_train_resampled)


best_model = None
best_score = 0

if random_search_dt.best_score_ > best_score:
    best_model = random_search_dt.best_estimator_
    best_score = random_search_dt.best_score_

if random_search_rf.best_score_ > best_score:
    best_model = random_search_rf.best_estimator_
    best_score = random_search_rf.best_score_

if random_search_xgb.best_score_ > best_score:
    best_model = random_search_xgb.best_estimator_
    best_score = random_search_xgb.best_score_

print("Best Model:", best_model)
print(f"Best Score::{best_score:.2f}")

with open("best_model.pkl", "wb") as f:
    pickle.dump(best_model, f)

y_test_pred = best_model.predict(X_test)
print(accuracy_score(y_test, y_test_pred))
print(confusion_matrix(y_test, y_test_pred))
print(classification_report(y_test, y_test_pred))
