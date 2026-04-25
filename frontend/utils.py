import pandas as pd
import os

def load_data():
    """Load customer data from CSV instead of PostgreSQL."""
    # Assuming we run from the project root or frontend/
    csv_path = 'data/Telco-Customer-Churn.csv'
    if not os.path.exists(csv_path):
        # try relative path from frontend/
        csv_path = '../data/Telco-Customer-Churn.csv'
        
    df = pd.read_csv(csv_path)
    
    # Clean column names (trim any whitespace)
    df.columns = df.columns.str.strip()

    # Rename columns to match the database schema
    df = df.rename(columns={
        'customerID': 'customer_id',
        'SeniorCitizen': 'senior_citizen',
        'PhoneService': 'phone_service',
        'MultipleLines': 'multiple_lines',
        'InternetService': 'internet_service',
        'OnlineSecurity': 'online_security',
        'OnlineBackup': 'online_backup',
        'DeviceProtection': 'device_protection',
        'TechSupport': 'tech_support',
        'StreamingTV': 'streaming_tv',
        'StreamingMovies': 'streaming_movies',
        'PaperlessBilling': 'paperless_billing',
        'MonthlyCharges': 'monthly_charges',
        'TotalCharges': 'total_charges',
        'PaymentMethod': 'payment_method',
        'Churn': 'churn',
        'Partner': 'partner',
        'Dependents': 'dependents',
        'Contract': 'contract'
    })

    # Trim whitespace in all string columns
    for col in df.select_dtypes(include=['object']).columns:
        df[col] = df[col].str.strip()

    # Convert 'total_charges' to numeric
    df['total_charges'] = pd.to_numeric(df['total_charges'], errors='coerce')
    df = df.dropna(subset=['total_charges'])

    # Convert data types
    df['senior_citizen'] = df['senior_citizen'].astype(int)
    
    return df
