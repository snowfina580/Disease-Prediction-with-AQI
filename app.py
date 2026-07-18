from flask import Flask, render_template, request, jsonify
import pickle
import numpy as np
import pandas as pd
import os

app = Flask(__name__)

# ==============================================================================
# LOAD TRAINED MODELS
# ==============================================================================
print("🔄 Loading trained models...")

try:
    # Load AQI model
    aqi_model = pickle.load(open('model/aqi_classifier.pkl', 'rb'))
    aqi_cat_encoder = pickle.load(open('model/aqi_category_encoder.pkl', 'rb'))
    
    # Load disease model
    disease_model = pickle.load(open('model/disease_classifier.pkl', 'rb'))
    disease_encoder = pickle.load(open('model/disease_encoder.pkl', 'rb'))
    
    # Load risk model
    risk_model = pickle.load(open('model/risk_regressor.pkl', 'rb'))
    risk_scaler = pickle.load(open('model/risk_scaler.pkl', 'rb'))
    gender_encoder = pickle.load(open('model/gender_encoder.pkl', 'rb'))
    city_encoder = pickle.load(open('model/city_encoder.pkl', 'rb'))
    
    # Load AQI data for city lookup
    aqi_data = pd.read_csv('data/aqi.csv')
    
    print("✅ All models loaded successfully!")
    models_loaded = True
except Exception as e:
    print(f"❌ Error loading models: {e}")
    models_loaded = False

# ==============================================================================
# SAMPLE DATA FOR DROPDOWN
# ==============================================================================
CITIES = {
    'Delhi': {'AQI': 380, 'PM2.5': 180, 'PM10': 320},
    'Mumbai': {'AQI': 220, 'PM2.5': 95, 'PM10': 160},
    'Bangalore': {'AQI': 140, 'PM2.5': 55, 'PM10': 95},
    'Chennai': {'AQI': 160, 'PM2.5': 70, 'PM10': 120},
    'Kolkata': {'AQI': 300, 'PM2.5': 140, 'PM10': 250},
    'Hyderabad': {'AQI': 180, 'PM2.5': 80, 'PM10': 130},
    'Pune': {'AQI': 150, 'PM2.5': 65, 'PM10': 110},
    'Ahmedabad': {'AQI': 200, 'PM2.5': 90, 'PM10': 150}
}

HEALTH_ADVICE = {
    'Very Poor': {
        'Asthma': [
            '🚫 Avoid all outdoor activities',
            '😷 Wear N95/FFP3 mask when going outside',
            '🏥 Consult doctor immediately if symptoms worsen',
            '💨 Stay indoors with air purifier'
        ],
        'Bronchitis': [
            '🚫 Avoid strenuous physical activities',
            '💊 Consult respiratory specialist',
            '😷 Wear protective mask at all times',
            '🌬️ Use air purifier indoors'
        ],
        'default': [
            '🚫 Minimize outdoor exposure',
            '😷 Use N95 masks when outdoors',
            '🏥 Consult healthcare provider',
            '🌡️ Monitor symptoms closely'
        ]
    },
    'Poor': {
        'Asthma': [
            '⚠️ Limit outdoor activities',
            '😷 Wear N95 mask for outdoor activities',
            '🏥 Keep rescue inhaler accessible',
            '💧 Stay hydrated'
        ],
        'default': [
            '⚠️ Reduce outdoor time',
            '😷 Wear protective mask',
            '🥗 Eat antioxidant-rich foods',
            '💧 Drink plenty of water'
        ]
    },
    'Moderate': {
        'default': [
            '✓ Can do outdoor activities with care',
            '🚴 Exercise during cooler hours',
            '🥕 Maintain healthy diet',
            '💪 Regular physical activity recommended'
        ]
    },
    'Good': {
        'default': [
            '✓ Excellent day for outdoor activities',
            '🏃 Ideal for exercise and sports',
            '🌳 Enjoy outdoor recreation',
            '😊 Good conditions for all age groups'
        ]
    }
}

# ==============================================================================
# ROUTES
# ==============================================================================

@app.route('/')
def index():
    """Home page"""
    return render_template('index.html', cities=list(CITIES.keys()))

@app.route('/predict', methods=['POST'])
def predict():
    """Make prediction based on user input"""
    
    if not models_loaded:
        return jsonify({'error': 'Models not loaded. Please train models first.'}), 500
    
    try:
        data = request.json
        
        # Extract user input
        city = data.get('city', '')
        age = int(data.get('age', 30))
        gender = data.get('gender', 'Male')
        cough = int(data.get('cough', 0))
        breathlessness = int(data.get('breathlessness', 0))
        fever = int(data.get('fever', 0))
        chest_pain = int(data.get('chest_pain', 0))
        fatigue = int(data.get('fatigue', 0))
        smoking = int(data.get('smoking', 0))
        
        # Get city AQI data
        if city in CITIES:
            city_data = CITIES[city]
            aqi_value = city_data['AQI']
            pm25 = city_data['PM2.5']
            pm10 = city_data['PM10']
        else:
            return jsonify({'error': 'Invalid city selected'}), 400
        
        # Determine AQI Category
        if aqi_value <= 50:
            aqi_category_pred = 'Good'
        elif aqi_value <= 100:
            aqi_category_pred = 'Satisfactory'
        elif aqi_value <= 200:
            aqi_category_pred = 'Moderately Polluted'
        elif aqi_value <= 300:
            aqi_category_pred = 'Poor'
        else:
            aqi_category_pred = 'Very Poor'
        
        # Prepare features for disease classification with correct column order
        # Column order MUST match training data: ['Cough', 'Breathlessness', 'Wheezing', 
        # 'Shortness of Breath', 'Fever', 'Chest Pain', 'Headache', 'Fatigue', 'Sneezing', 'Itchy Eyes']
        symptoms_df_features = pd.DataFrame([[
            cough,                    # Cough
            breathlessness,           # Breathlessness
            np.random.randint(0, 1),  # Wheezing (random)
            breathlessness,           # Shortness of Breath
            fever,                    # Fever
            chest_pain,               # Chest Pain
            np.random.randint(0, 1),  # Headache (random)
            fatigue,                  # Fatigue
            np.random.randint(0, 1),  # Sneezing (random)
            np.random.randint(0, 1)   # Itchy Eyes (random)
        ]], columns=['Cough', 'Breathlessness', 'Wheezing', 'Shortness of Breath', 'Fever',
                     'Chest Pain', 'Headache', 'Fatigue', 'Sneezing', 'Itchy Eyes'])
        
        # Predict disease
        disease_pred_encoded = disease_model.predict(symptoms_df_features)[0]
        disease_pred = disease_encoder.inverse_transform([disease_pred_encoded])[0]
        
        # Prepare features for risk regression with proper column names
        gender_encoded = 0 if gender == 'Male' else 1
        
        risk_features_df = pd.DataFrame([[
            age, 
            gender_encoded, 
            smoking, 
            0,                        # drinking (default)
            150,                      # Physical Activity (150 min/week)
            7,                        # Sleep Hours (7 hours)
            5,                        # Stress Level (5/10)
            24,                       # BMI (24)
            cough, 
            breathlessness, 
            chest_pain, 
            fever, 
            fatigue,
            aqi_value, 
            pm25
        ]], columns=['Age', 'Gender_encoded', 'Smoking', 'Drinking', 'Physical Activity',
                     'Sleep Hours', 'Stress Level', 'BMI', 'Cough', 'Breathlessness',
                     'Chest Pain', 'Fever', 'Fatigue', 'AQI', 'PM2.5'])
        
        risk_features_scaled = risk_scaler.transform(risk_features_df)
        risk_pred = risk_model.predict(risk_features_scaled)[0]
        risk_pred = max(0, min(100, risk_pred))  # Clamp between 0-100
        
        # Determine risk level
        if risk_pred > 70:
            risk_level = 'HIGH'
        elif risk_pred > 50:
            risk_level = 'MEDIUM'
        else:
            risk_level = 'LOW'
        
        # Get health advice
        if aqi_category_pred in HEALTH_ADVICE and disease_pred in HEALTH_ADVICE[aqi_category_pred]:
            advice = HEALTH_ADVICE[aqi_category_pred][disease_pred]
        elif aqi_category_pred in HEALTH_ADVICE:
            advice = HEALTH_ADVICE[aqi_category_pred]['default']
        else:
            advice = HEALTH_ADVICE['Moderate']['default']
        
        return jsonify({
            'city': city,
            'aqi': aqi_value,
            'aqi_category': aqi_category_pred,
            'pm25': round(pm25, 2),
            'pm10': round(pm10, 2),
            'predicted_disease': disease_pred,
            'risk_percent': round(risk_pred, 1),
            'risk_level': risk_level,
            'health_advice': advice
        })
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/health')
def health():
    """Health check endpoint"""
    return jsonify({
        'status': 'ok',
        'models_loaded': models_loaded
    })

# ==============================================================================
# ERROR HANDLERS
# ==============================================================================

@app.errorhandler(404)
def not_found(error):
    return jsonify({'error': 'Page not found'}), 404

@app.errorhandler(500)
def server_error(error):
    return jsonify({'error': 'Internal server error'}), 500

# ==============================================================================
# MAIN
# ==============================================================================

if __name__ == '__main__':
    print("\n" + "=" * 80)
    print("🌐 FLASK WEB APPLICATION STARTING")
    print("=" * 80)
    print("📍 Server: http://localhost:5000")
    print("=" * 80 + "\n")
    
    app.run(debug=True, host='0.0.0.0', port=5000)
