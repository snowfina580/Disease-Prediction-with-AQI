import pandas as pd
import numpy as np
import os
from datetime import datetime, timedelta

# Create data directory if it doesn't exist
os.makedirs('data', exist_ok=True)

# Set random seed for reproducibility
np.random.seed(42)

# ==============================================================================
# 1. AQI DATASET (Air Quality Index India 2023-2025)
# ==============================================================================
print("Generating AQI Dataset...")

cities = ['Delhi', 'Mumbai', 'Bangalore', 'Chennai', 'Kolkata', 'Hyderabad', 'Pune', 'Ahmedabad']
dates = pd.date_range(start='2023-01-01', end='2025-12-31', freq='D')

aqi_data = []
for city in cities:
    for date in dates:
        aqi_value = np.random.randint(50, 450)  # AQI ranges from 0-500+
        pm25 = np.random.uniform(5, 300)          # PM2.5 in µg/m³
        pm10 = np.random.uniform(20, 400)         # PM10 in µg/m³
        no2 = np.random.uniform(10, 150)          # NO2 in ppb
        so2 = np.random.uniform(5, 100)           # SO2 in ppb
        
        # Determine AQI category
        if aqi_value <= 50:
            aqi_category = 'Good'
        elif aqi_value <= 100:
            aqi_category = 'Satisfactory'
        elif aqi_value <= 200:
            aqi_category = 'Moderately Polluted'
        elif aqi_value <= 300:
            aqi_category = 'Poor'
        else:
            aqi_category = 'Very Poor'
        
        aqi_data.append({
            'City': city,
            'Date': date,
            'Year': date.year,
            'Month': date.month,
            'Day': date.day,
            'AQI': aqi_value,
            'PM2.5': pm25,
            'PM10': pm10,
            'NO2': no2,
            'SO2': so2,
            'AQI_Category': aqi_category
        })

aqi_df = pd.DataFrame(aqi_data)
aqi_df.to_csv('data/aqi.csv', index=False)
print(f"✅ AQI Dataset created: {len(aqi_df)} records")

# ==============================================================================
# 2. SYMPTOMS-DISEASE DATASET
# ==============================================================================
print("Generating Symptoms-Disease Dataset...")

diseases = ['Asthma', 'Bronchitis', 'Allergic Rhinitis', 'Heart Disease', 'Diabetes']
symptoms_list = ['Cough', 'Breathlessness', 'Fever', 'Chest Pain', 'Headache', 
                 'Fatigue', 'Wheezing', 'Sneezing', 'Itchy Eyes', 'Shortness of Breath']

symptoms_data = []
for disease in diseases:
    # Create different symptom patterns for each disease
    for i in range(200):
        symptom_combination = {}
        symptom_combination['Disease'] = disease
        
        # Disease-specific symptom probabilities
        if disease == 'Asthma':
            symptom_combination['Cough'] = np.random.choice([0, 1], p=[0.3, 0.7])
            symptom_combination['Breathlessness'] = np.random.choice([0, 1], p=[0.2, 0.8])
            symptom_combination['Wheezing'] = np.random.choice([0, 1], p=[0.3, 0.7])
            symptom_combination['Shortness of Breath'] = np.random.choice([0, 1], p=[0.2, 0.8])
        elif disease == 'Bronchitis':
            symptom_combination['Cough'] = np.random.choice([0, 1], p=[0.1, 0.9])
            symptom_combination['Breathlessness'] = np.random.choice([0, 1], p=[0.4, 0.6])
            symptom_combination['Fever'] = np.random.choice([0, 1], p=[0.3, 0.7])
            symptom_combination['Chest Pain'] = np.random.choice([0, 1], p=[0.5, 0.5])
        else:
            # Random for other diseases
            for symptom in symptoms_list:
                symptom_combination[symptom] = np.random.choice([0, 1])
        
        # Fill missing symptoms with random values
        for symptom in symptoms_list:
            if symptom not in symptom_combination:
                symptom_combination[symptom] = np.random.choice([0, 1])
        
        symptoms_data.append(symptom_combination)

symptoms_df = pd.DataFrame(symptoms_data)
symptoms_df.to_csv('data/symptoms.csv', index=False)
print(f"✅ Symptoms-Disease Dataset created: {len(symptoms_df)} records")

# ==============================================================================
# 3. MEDICAL DISEASE PREDICTION DATASET
# ==============================================================================
print("Generating Medical Disease Prediction Dataset...")

medical_data = []
for i in range(1000):
    age = np.random.randint(5, 85)
    city = np.random.choice(cities)
    
    # Get average AQI for the city
    city_aqi = aqi_df[aqi_df['City'] == city]['AQI'].mean()
    
    record = {
        'Age': age,
        'Gender': np.random.choice([0, 1]),  # 0=Male, 1=Female
        'Smoking': np.random.choice([0, 1]),
        'Drinking': np.random.choice([0, 1]),
        'Physical Activity': np.random.randint(0, 300),  # minutes per week
        'Sleep Hours': np.random.randint(4, 12),
        'Stress Level': np.random.randint(1, 10),
        'BMI': np.random.uniform(15, 40),
        'Cough': np.random.choice([0, 1]),
        'Breathlessness': np.random.choice([0, 1]),
        'Chest Pain': np.random.choice([0, 1]),
        'Fever': np.random.choice([0, 1]),
        'Fatigue': np.random.choice([0, 1]),
        'City': city,
        'AQI': city_aqi,
        'PM2.5': aqi_df[aqi_df['City'] == city]['PM2.5'].mean()
    }
    
    # Predict disease risk based on features
    risk_score = 0
    if record['Cough'] == 1:
        risk_score += 20
    if record['Breathlessness'] == 1:
        risk_score += 25
    if record['Smoking'] == 1:
        risk_score += 30
    if city_aqi > 200:
        risk_score += 30
    if age > 50:
        risk_score += 15
    if record['Stress Level'] > 7:
        risk_score += 10
    
    risk_score = min(100, risk_score)
    
    # Assign disease
    if risk_score > 70:
        disease = 'High Risk'
    elif risk_score > 50:
        disease = 'Medium Risk'
    else:
        disease = 'Low Risk'
    
    record['Disease_Risk_Percent'] = risk_score
    record['Disease_Category'] = disease
    
    medical_data.append(record)

medical_df = pd.DataFrame(medical_data)
medical_df.to_csv('data/medical.csv', index=False)
print(f"✅ Medical Disease Prediction Dataset created: {len(medical_df)} records")

print("\n✅ All datasets generated successfully!")
print(f"   - aqi.csv: {len(aqi_df)} records")
print(f"   - symptoms.csv: {len(symptoms_df)} records")
print(f"   - medical.csv: {len(medical_df)} records")
