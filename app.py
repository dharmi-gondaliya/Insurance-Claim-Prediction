import os
import re
import pandas as pd
import numpy as np
import joblib
from flask import Flask, render_template, request, jsonify

app = Flask(__name__)

# Load the model and preprocessor
# Check if the model exists, otherwise provide instructions
try:
    model = joblib.load('./model/insurance_eligibility_model.pkl')
    preprocessor = joblib.load('./model/insurance_preprocessor.pkl')
    model_loaded = True
except FileNotFoundError:
    model_loaded = False

# Function to extract numerical values from torque and power
def extract_number(value):
    if pd.isna(value):
        return np.nan
    match = re.search(r'(\d+\.?\d*)', str(value))
    if match:
        return float(match.group(1))
    return np.nan

# Function to safely convert values to float
def safe_float_conversion(value, default=0.0):
    if value is None:
        return default
    try:
        # First, strip any alphabetic characters
        value_str = str(value)
        # Remove all non-numeric characters except decimal point
        numeric_only = re.sub(r'[^0-9.]', '', value_str)
        # If nothing left, return default
        if not numeric_only:
            return default
        return float(numeric_only)
    except (ValueError, TypeError):
        return default

# Function to safely convert values to int
def safe_int_conversion(value, default=0):
    if value is None:
        return default
    try:
        # Convert to float first, then to int
        float_value = safe_float_conversion(value, default=float(default))
        return int(float_value)
    except (ValueError, TypeError):
        return default

# Function to prepare data for prediction
def prepare_data(form_data):
    # Create a dictionary to store the data
    data = {}
    
    # Print form data for debugging
    print("Form data received:", form_data)
    
    # Numerical features with safe conversion
    data['subscription_length'] = safe_float_conversion(form_data.get('subscription_length'))
    data['vehicle_age'] = safe_float_conversion(form_data.get('vehicle_age'))
    data['customer_age'] = safe_int_conversion(form_data.get('customer_age'))
    data['region_density'] = safe_int_conversion(form_data.get('region_density'))
    data['airbags'] = safe_int_conversion(form_data.get('airbags'))
    data['displacement'] = safe_float_conversion(form_data.get('displacement'))
    data['cylinder'] = safe_int_conversion(form_data.get('cylinder'))
    data['turning_radius'] = safe_float_conversion(form_data.get('turning_radius'))
    data['length'] = safe_float_conversion(form_data.get('length'))
    data['width'] = safe_float_conversion(form_data.get('width'))
    data['gross_weight'] = safe_float_conversion(form_data.get('gross_weight'))
    data['ncap_rating'] = safe_int_conversion(form_data.get('ncap_rating'))
    
    # Extract numerical values from torque and power
    max_torque = form_data.get('max_torque', '')
    max_power = form_data.get('max_power', '')
    data['torque_value'] = extract_number(max_torque)
    data['power_value'] = extract_number(max_power)
    
    # Categorical features
    data['region_code'] = str(form_data.get('region_code', ''))
    data['segment'] = str(form_data.get('segment', ''))
    data['model'] = str(form_data.get('model', ''))
    data['fuel_type'] = str(form_data.get('fuel_type', ''))
    data['engine_type'] = str(form_data.get('engine_type', ''))
    data['rear_brakes_type'] = str(form_data.get('rear_brakes_type', ''))
    data['transmission_type'] = str(form_data.get('transmission_type', ''))
    data['steering_type'] = str(form_data.get('steering_type', ''))
    
    # Boolean features (convert Yes/No to 1/0)
    boolean_features = [
        'is_esc', 'is_adjustable_steering', 'is_tpms', 'is_parking_sensors',
        'is_parking_camera', 'is_front_fog_lights', 'is_rear_window_wiper',
        'is_rear_window_washer', 'is_rear_window_defogger', 'is_brake_assist',
        'is_power_door_locks', 'is_central_locking', 'is_power_steering',
        'is_driver_seat_height_adjustable', 'is_day_night_rear_view_mirror',
        'is_ecw', 'is_speed_alert'
    ]
    
    for feature in boolean_features:
        data[feature] = 1 if form_data.get(feature) == 'Yes' else 0
    
    # Print processed data for debugging
    print("Processed data:", data)
    
    # Convert to DataFrame
    df = pd.DataFrame([data])
    
    return df

# Route for the home page
@app.route('/', methods=['GET'])
def home():
    return render_template('index.html', model_loaded=model_loaded)

# Route for prediction
@app.route('/predict', methods=['POST'])
def predict():
    if not model_loaded:
        return jsonify({
            'error': 'Model not loaded. Please train the model first.',
            'instructions': 'Run the training script to generate the model file.'
        }), 400
    
    try:
        # Get form data
        form_data = request.form.to_dict()
        
        # Log all form data
        print("Raw form data:", form_data)
        
        # Check for problematic values and log them
        problematic_fields = []
        for key, value in form_data.items():
            if key in ['subscription_length', 'vehicle_age', 'customer_age', 'region_density', 
                      'airbags', 'displacement', 'cylinder', 'turning_radius', 'length', 
                      'width', 'gross_weight', 'ncap_rating']:
                if not isinstance(value, (int, float)) and not (isinstance(value, str) and value.strip().replace('.', '', 1).isdigit()):
                    problematic_fields.append(f"{key}: {value}")
        
        if problematic_fields:
            print(f"Warning: Problematic numeric fields detected: {', '.join(problematic_fields)}")
        
        # Prepare data for prediction
        data = prepare_data(form_data)
        
        # Apply the preprocessor if available
        if 'preprocessor' in globals() and preprocessor is not None:
            try:
                # Use the preprocessor to transform the data
                processed_data = preprocessor.transform(data)
                # Make prediction on processed data
                prediction = model.predict(processed_data)[0]
                
                # Get prediction probability if available
                try:
                    probability = model.predict_proba(processed_data)[0][1] * 100
                    probability_message = f"Confidence: {probability:.2f}%"
                except:
                    probability_message = ""
            except Exception as e:
                print(f"Error during preprocessing: {str(e)}")
                # Fall back to direct prediction if preprocessing fails
                prediction = model.predict(data)[0]
                probability_message = ""
        else:
            # Direct prediction without preprocessing
            prediction = model.predict(data)[0]
            
            # Get prediction probability if available
            try:
                probability = model.predict_proba(data)[0][1] * 100
                probability_message = f"Confidence: {probability:.2f}%"
            except:
                probability_message = ""
        
        # Determine eligibility message
        eligible = prediction == 1
        message = "Eligible for insurance" if eligible else "Not eligible for insurance"
        
        # Return prediction
        return render_template(
            'result.html',
            eligible=eligible,
            message=message,
            probability_message=probability_message,
            form_data=form_data
        )
    except Exception as e:
        import traceback
        error_traceback = traceback.format_exc()
        print(f"Error processing request: {str(e)}")
        print(f"Traceback: {error_traceback}")
        
        return jsonify({
            'error': f'Error processing request: {str(e)}',
            'details': 'Please check input data format and try again.',
            'traceback': error_traceback if app.debug else None
        }), 400

if __name__ == '__main__':
    app.run(debug=True)