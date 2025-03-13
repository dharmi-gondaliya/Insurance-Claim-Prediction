import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns
from imblearn.over_sampling import SMOTE
import re

# Load the dataset
# Assuming the data is saved as 'insurance_data.csv'
# If you're copying this data directly, you'll need to save it as a CSV file first
data = pd.read_csv('Insurance claims data.csv')

# Function to extract numerical values from torque and power
def extract_number(value):
    if pd.isna(value):
        return np.nan
    match = re.search(r'(\d+\.?\d*)', str(value))
    if match:
        return float(match.group(1))
    return np.nan

# Exploratory Data Analysis (EDA)
print("Data shape:", data.shape)
print("\nData types:")
print(data.dtypes)
print("\nMissing values:")
print(data.isnull().sum())
print("\nTarget variable distribution:")
print(data['claim_status'].value_counts(normalize=True))

# Feature Engineering
# Extract numerical values from torque and power
data['torque_value'] = data['max_torque'].apply(lambda x: extract_number(x))
data['power_value'] = data['max_power'].apply(lambda x: extract_number(x))

# Extract numerical displacement value
data['displacement'] = pd.to_numeric(data['displacement'], errors='coerce')

# Convert boolean-like features to binary
bool_features = ['is_esc', 'is_adjustable_steering', 'is_tpms', 'is_parking_sensors', 
                 'is_parking_camera', 'is_front_fog_lights', 'is_rear_window_wiper',
                 'is_rear_window_washer', 'is_rear_window_defogger', 'is_brake_assist',
                 'is_power_door_locks', 'is_central_locking', 'is_power_steering',
                 'is_driver_seat_height_adjustable', 'is_day_night_rear_view_mirror',
                 'is_ecw', 'is_speed_alert']

for feature in bool_features:
    data[feature] = data[feature].map({'Yes': 1, 'No': 0})

# Prepare data for modeling
# Define features and target variable
X = data.drop(['claim_status', 'policy_id', 'max_torque', 'max_power'], axis=1)
y = data['claim_status']

# Define numerical and categorical features
numerical_features = ['subscription_length', 'vehicle_age', 'customer_age', 'region_density',
                     'torque_value', 'power_value', 'airbags', 'displacement', 'cylinder',
                     'turning_radius', 'length', 'width', 'gross_weight', 'ncap_rating']

categorical_features = ['region_code', 'segment', 'model', 'fuel_type', 'engine_type',
                       'rear_brakes_type', 'transmission_type', 'steering_type']

# Create boolean features list (already converted to 0/1)
boolean_features = bool_features

# Define preprocessing for numerical and categorical features
numerical_transformer = Pipeline(steps=[
    ('scaler', StandardScaler())
])

categorical_transformer = Pipeline(steps=[
    ('onehot', OneHotEncoder(handle_unknown='ignore'))
])

# Combine preprocessing steps
preprocessor = ColumnTransformer(
    transformers=[
        ('num', numerical_transformer, numerical_features),
        ('cat', categorical_transformer, categorical_features),
        ('bool', 'passthrough', boolean_features)
    ])

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Check for class imbalance
print("\nClass distribution in training set:")
print(y_train.value_counts(normalize=True))

# Apply SMOTE for handling class imbalance (if needed)
if len(y_train.unique()) > 1 and y_train.value_counts(normalize=True).min() < 0.3:
    smote = SMOTE(random_state=42)
    # We need to preprocess before applying SMOTE
    X_train_processed = preprocessor.fit_transform(X_train)
    X_train_resampled, y_train_resampled = smote.fit_resample(X_train_processed, y_train)
    # Now we can't use the preprocessor in the pipeline since data is already processed
    rf_pipeline = RandomForestClassifier(random_state=42)
    gb_pipeline = GradientBoostingClassifier(random_state=42)
    # Set the flag to indicate SMOTE was used
    smote_used = True
else:
    # Create model pipelines
    rf_pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('classifier', RandomForestClassifier(random_state=42))
    ])
    
    gb_pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('classifier', GradientBoostingClassifier(random_state=42))
    ])
    # Set the flag to indicate SMOTE was not used
    smote_used = False
    X_train_resampled = X_train
    y_train_resampled = y_train

# Train and evaluate Random Forest
print("\nTraining Random Forest model...")
rf_pipeline.fit(X_train_resampled, y_train_resampled)

# If SMOTE was used, we need to preprocess the test data separately
if smote_used:
    X_test_processed = preprocessor.transform(X_test)
    y_pred_rf = rf_pipeline.predict(X_test_processed)
else:
    y_pred_rf = rf_pipeline.predict(X_test)

print("\nRandom Forest Results:")
print("Accuracy:", accuracy_score(y_test, y_pred_rf))
print("\nClassification Report:")
print(classification_report(y_test, y_pred_rf))

# Train and evaluate Gradient Boosting
print("\nTraining Gradient Boosting model...")
gb_pipeline.fit(X_train_resampled, y_train_resampled)

# Predict with Gradient Boosting
if smote_used:
    y_pred_gb = gb_pipeline.predict(X_test_processed)
else:
    y_pred_gb = gb_pipeline.predict(X_test)

print("\nGradient Boosting Results:")
print("Accuracy:", accuracy_score(y_test, y_pred_gb))
print("\nClassification Report:")
print(classification_report(y_test, y_pred_gb))

# Feature importance for Random Forest (only if SMOTE was not used)
if not smote_used:
    feature_names = (
        numerical_features + 
        list(preprocessor.transformers_[1][1].named_steps['onehot'].get_feature_names_out(categorical_features)) +
        boolean_features
    )
    
    importances = rf_pipeline.named_steps['classifier'].feature_importances_
    indices = np.argsort(importances)[::-1]
    
    # Print feature ranking
    print("\nFeature ranking:")
    for f in range(min(20, len(feature_names))):
        try:
            print(f"{f+1}. {feature_names[indices[f]]} ({importances[indices[f]]})")
        except IndexError:
            print(f"Error with index {indices[f]} (max index: {len(feature_names)-1})")

# Function to predict insurance eligibility for new data
def predict_eligibility(new_data, model=rf_pipeline, preprocessor=preprocessor, smote_used=smote_used):
    """
    Predicts insurance eligibility for new data.
    
    Parameters:
    new_data: DataFrame with the same structure as training data (excluding target)
    model: Trained model (default: Random Forest)
    preprocessor: Data preprocessor
    smote_used: Boolean indicating if SMOTE was used
    
    Returns:
    Array of predictions (0 = not eligible, 1 = eligible)
    """
    if smote_used:
        new_data_processed = preprocessor.transform(new_data)
        predictions = model.predict(new_data_processed)
    else:
        predictions = model.predict(new_data)
    
    return predictions

# Example of how to use the prediction function
print("\nExample of prediction for first 3 samples:")
example_data = X.iloc[:3]
predictions = predict_eligibility(example_data)
print(predictions)

# Save the model (optional)
import joblib
joblib.dump(rf_pipeline, 'insurance_eligibility_model.pkl')
joblib.dump(preprocessor, 'insurance_preprocessor.pkl')
print("\nModel and preprocessor saved to disk.")