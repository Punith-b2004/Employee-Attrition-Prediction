from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import pandas as pd
import numpy as np
import joblib
import shap
from sklearn.preprocessing import StandardScaler
import logging

# Set up logging
# ...existing code...

# Set up logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)
# ...existing code...
app = FastAPI()

# Load model and preprocessors
try:
    model = joblib.load('models/xgb_model_smote.pkl')
    scaler = joblib.load('models/scaler.pkl')
    feature_columns = joblib.load('models/feature_columns.pkl')
    logger.info("Model and preprocessors loaded successfully")
except Exception as e:
    logger.error(f"Error loading model/preprocessors: {str(e)}")
    raise

# Define categorical options
categorical_options = {
    'BusinessTravel': ['Travel_Rarely', 'Travel_Frequently', 'Non-Travel'],
    'Department': ['Sales', 'Research & Development', 'Human Resources'],
    'EducationField': ['Life Sciences', 'Other', 'Medical', 'Marketing', 'Technical Degree', 'Human Resources'],
    'Gender': ['Female', 'Male'],
    'JobRole': [
        'Sales Executive', 'Research Scientist', 'Laboratory Technician', 'Manufacturing Director',
        'Healthcare Representative', 'Manager', 'Sales Representative', 'Research Director', 'Human Resources'
    ],
    'MaritalStatus': ['Single', 'Married', 'Divorced'],
    'OverTime': ['Yes', 'No']
}

# Define input schema
class EmployeeInput(BaseModel):
    Age: int
    BusinessTravel: str
    DailyRate: int
    Department: str
    DistanceFromHome: int
    Education: int
    EducationField: str
    EnvironmentSatisfaction: int
    Gender: str
    HourlyRate: int
    JobInvolvement: int
    JobLevel: int
    JobRole: str
    JobSatisfaction: int
    MaritalStatus: str
    MonthlyIncome: int
    MonthlyRate: int
    NumCompaniesWorked: int
    OverTime: str
    PercentSalaryHike: int
    PerformanceRating: int
    RelationshipSatisfaction: int
    StockOptionLevel: int
    TotalWorkingYears: int
    TrainingTimesLastYear: int
    WorkLifeBalance: int
    YearsAtCompany: int
    YearsInCurrentRole: int
    YearsSinceLastPromotion: int
    YearsWithCurrManager: int

# Preprocess input data
def preprocess_input(data: EmployeeInput, scaler: StandardScaler, feature_columns):
    try:
        logger.debug("Starting preprocessing")
        input_dict = data.dict()
        # Convert OverTime to numerical
        input_dict['OverTime'] = 1 if input_dict['OverTime'] == 'Yes' else 0
        df = pd.DataFrame([input_dict])

        # Calculate PromotionGap
        df['PromotionGap'] = df['YearsAtCompany'] - df['YearsSinceLastPromotion']
        logger.debug("PromotionGap calculated")

        # One-hot encode categorical variables (excluding OverTime)
        categorical_cols = ['BusinessTravel', 'Department', 'EducationField', 'Gender', 'JobRole', 'MaritalStatus']
        for col in categorical_cols:
            if input_dict[col] not in categorical_options[col]:
                logger.error(f"Invalid value for {col}: {input_dict[col]}")
                raise HTTPException(
                    status_code=400,
                    detail=f"Invalid value for {col}. Choose from {categorical_options[col]}"
                )
        df = pd.get_dummies(df, columns=categorical_cols, drop_first=True)
        logger.debug(f"One-hot encoded columns: {df.columns.tolist()}")

        # Create TenureBand
        df['TenureBand'] = pd.cut(
            df['YearsAtCompany'],
            bins=[-float('inf'), 2, 5, float('inf')],
            labels=['Short', 'Medium', 'Long']
        )
        df = pd.get_dummies(df, columns=['TenureBand'], drop_first=True)
        logger.debug("TenureBand created")

        # Calculate WorkLifeBalanceIndex
        df['WorkLifeBalanceIndex'] = (
            df['WorkLifeBalance'] + df['EnvironmentSatisfaction'] + df['JobSatisfaction'] + df['OverTime']
        ) / 4
        logger.debug("WorkLifeBalanceIndex calculated")

        # Scale numerical features
        numerical_cols = [
            'Age', 'DailyRate', 'DistanceFromHome', 'Education', 'EnvironmentSatisfaction',
            'HourlyRate', 'JobInvolvement', 'JobLevel', 'JobSatisfaction', 'MonthlyIncome',
            'MonthlyRate', 'NumCompaniesWorked', 'PercentSalaryHike', 'PerformanceRating',
            'RelationshipSatisfaction', 'StockOptionLevel', 'TotalWorkingYears',
            'TrainingTimesLastYear', 'WorkLifeBalance', 'YearsAtCompany',
            'YearsInCurrentRole', 'YearsSinceLastPromotion', 'YearsWithCurrManager',
            'PromotionGap', 'WorkLifeBalanceIndex', 'OverTime'
        ]
        df[numerical_cols] = scaler.transform(df[numerical_cols])
        logger.debug("Numerical features scaled")

        # Align with training features
        for col in feature_columns:
            if col not in df.columns:
                df[col] = 0
        df = df[feature_columns]
        logger.debug("Features aligned with training columns")

        return df
    except Exception as e:
        logger.error(f"Preprocessing error: {str(e)}")
        raise

@app.get("/")
async def root():
    return {
        "message": "Welcome to the Employee Attrition Prediction API. Use POST /predict to make predictions."
    }

@app.post("/predict")
async def predict(data: EmployeeInput):
    try:
        logger.debug("Received predict request")
        # Preprocess input
        df = preprocess_input(data, scaler, feature_columns)

        # Make prediction
        prob = model.predict_proba(df)[:, 1][0]
        logger.debug(f"Prediction probability: {prob}")

        # SHAP explanation
        explainer = shap.Explainer(model)
        shap_values = explainer(df)
        shap_dict = {feature: float(shap_values.values[0][i]) for i, feature in enumerate(df.columns)}
        logger.debug("SHAP values computed")

        return {
            "attrition_probability": float(prob),
            "shap_values": shap_dict
        }
    except Exception as e:
        logger.error(f"Predict endpoint error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))