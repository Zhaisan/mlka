import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.experimental import enable_hist_gradient_boosting  # Enable HistGradientBoostingClassifier
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import accuracy_score
from sklearn.impute import SimpleImputer

# Step 1: Load the training and test datasets
train_data = pd.read_csv("train.csv")
test_data = pd.read_csv("test.csv")

# Step 2: Data Preprocessing
# Check for missing values in the data
# You can also handle outliers and feature engineering here if needed

# Step 3: Feature Selection/Engineering
# You can select relevant features or perform feature engineering here

# Step 4: Split the Training Data
X = train_data.drop(columns=['target'])
y = train_data['target']

X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 2.5: Handle Missing Values (Impute)
imputer = SimpleImputer(strategy='mean')  # You can choose other strategies as well

X_train = imputer.fit_transform(X_train)
X_val = imputer.transform(X_val)

# Step 5: Model Selection and Training
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_val = scaler.transform(X_val)

model = HistGradientBoostingClassifier(random_state=42)
model.fit(X_train, y_train)

# Step 6: Model Evaluation
y_pred = model.predict(X_val)
accuracy = accuracy_score(y_val, y_pred)
print("Validation Accuracy:", accuracy)

# Step 7: Predictions on Test Data
X_test = test_data.drop(columns=['Id'])
X_test = scaler.transform(X_test)

# Handle missing values in the test data
X_test = imputer.transform(X_test)

test_predictions = model.predict(X_test)

# Step 8: Create the "answers.csv" file
answers_df = pd.DataFrame({
    "Id": test_data["Id"],
    "target": test_predictions
})

answers_df.to_csv("answers.csv", index=False)

# Step 10: Calculate Your Score (Based on the accuracy of your predictions)
score = min(2, (accuracy - 0.55) / 0.17)
print("Score:", score)
