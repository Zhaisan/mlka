import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import accuracy_score
from sklearn.impute import SimpleImputer

# Load the training and test datasets
train_data = pd.read_csv("train.csv")
test_data = pd.read_csv("test.csv")

# Define features and target
X = train_data.drop(columns=['target'])
y = train_data['target']

# Split the training data
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# Impute missing values
imputer = SimpleImputer(strategy='mean')
X_train = imputer.fit_transform(X_train)
X_val = imputer.transform(X_val)

# Scale features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_val_scaled = scaler.transform(X_val)

# Train the model
model = HistGradientBoostingClassifier(random_state=42)
model.fit(X_train_scaled, y_train)

# Evaluate the model
y_pred = model.predict(X_val_scaled)
accuracy = accuracy_score(y_val, y_pred)
print(f"Validation Accuracy: {accuracy}")

# Predict on test data
X_test = test_data.drop(columns=['Id'])
X_test = imputer.transform(X_test)  # Impute missing values
X_test_scaled = scaler.transform(X_test)  # Scale features

test_predictions = model.predict(X_test_scaled)

# Create the "answers.csv" file
answers_df = pd.DataFrame({
    "Id": test_data["Id"],
    "target": test_predictions
})
answers_df.to_csv("answers.csv", index=False)

# Calculate and print the score
score = max(0, min(2, (accuracy - 0.55) / 0.17))
print(f"Score: {score}")
