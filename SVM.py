import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

# Load the dataset
data_path = 'sph6004_assignment1_data.csv'  
data = pd.read_csv(data_path)

# Preprocess the data
# Assuming preprocessing steps and feature selection have already been defined
features = ['gender', 'admission_age', 'race', 'heart_rate_mean', 'sbp_mean', 'dbp_mean', 'lactate_min', 'lactate_max']
target = 'aki'
X = data[features]
y = data[target]
categorical_features = ['gender', 'race']
numerical_features = ['admission_age', 'heart_rate_mean', 'sbp_mean', 'dbp_mean', 'lactate_min', 'lactate_max']

# Preprocessing for numerical and categorical data
preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numerical_features),
        ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)])

# Splitting the dataset into training and testing sets
X_prepared = preprocessor.fit_transform(X)
X_train, X_test, y_train, y_test = train_test_split(X_prepared, y, test_size=0.2, random_state=42)

# Initialize and train the SVM model
svm_model = LinearSVC(random_state=42, max_iter=10000, C=0.1)
svm_model.fit(X_train, y_train)

# Predict on the testing set
y_pred_svm = svm_model.predict(X_test)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred_svm)
precision, recall, f1, _ = precision_recall_fscore_support(y_test, y_pred_svm, average='weighted')

print(f"Accuracy: {accuracy}")
print(f"Precision: {precision}")
print(f"Recall: {recall}")
print(f"F1-Score: {f1}")
