# Employee Attrition Prediction Capstone Project

This project builds a machine learning classification model to predict employee attrition, focusing on accuracy, interpretability, and fairness for HR decision-making. The project uses the IBM HR Analytics Employee Attrition Dataset, performing exploratory data analysis (EDA), feature engineering, model building, evaluation, explainability, fairness analysis, and deployment via a FastAPI backend and Gradio dashboard.

## Objective
The goal is to predict whether an employee is likely to leave (attrition) or stay, providing actionable insights for HR to reduce turnover. The model emphasizes interpretability using SHAP and fairness by analyzing bias across gender, age, and department.

## Dataset
- **Source**: IBM HR Analytics Employee Attrition Dataset (available on Kaggle).
- **Target Variable**: `Attrition` (Yes/No, binary classification).
- **Features**: 35 features including demographic (Age, Gender), work environment (JobSatisfaction, WorkLifeBalance), compensation (MonthlyIncome, DailyRate), and performance-related (PerformanceRating, YearsAtCompany) variables.
- **Class Distribution**: Imbalanced (~16% Yes, ~84% No).

## Project Tasks

### 1. Exploratory Data Analysis (EDA)
- **Data Inspection**: Loaded dataset, checked data types, and confirmed no missing values.
- **Univariate Analysis**:
  - Visualized distributions of `Gender`, `Department`, `JobRole`, and `Attrition`.
  - Plotted `Age` distribution (histogram with KDE).
- **Bivariate Analysis**:
  - Analyzed `Attrition` vs. `Gender`, `Department`, `MonthlyIncome`, and `JobSatisfaction` using count plots and box plots.
  - Created a correlation heatmap for numerical features.
- **Key Findings**:
  - Attrition rate: ~16% (Yes), confirming class imbalance.
  - Higher attrition observed in Sales department and among employees with frequent business travel.
  - Lower `MonthlyIncome` and `JobSatisfaction` correlated with higher attrition.

### 2. Feature Engineering
- **Outlier Removal**: Removed outliers in `MonthlyIncome`, `YearsAtCompany`, and `DistanceFromHome` using IQR method.
- **Encoding**:
  - Label-encoded `Attrition` (Yes=1, No=0) and `OverTime` (Yes=1, No=0).
  - One-hot encoded categorical features: `BusinessTravel`, `Department`, `EducationField`, `Gender`, `JobRole`, `MaritalStatus`.
- **Derived Features**:
  - **PromotionGap**: `YearsAtCompany` - `YearsSinceLastPromotion`.
  - **TenureBand**: Categorized `YearsAtCompany` into Short (<2 years), Medium (2-5 years), and Long (>5 years).
  - **WorkLifeBalanceIndex**: Average of `WorkLifeBalance`, `EnvironmentSatisfaction`, `JobSatisfaction`, and `OverTime`.
- **Scaling**: Standardized numerical features using `StandardScaler`.
- **Dropped Columns**: Removed irrelevant columns (`EmployeeNumber`, `Over18`, `EmployeeCount`, `StandardHours`).

### 3. Model Building
- **Train-Test Split**: Stratified split (80-20) to preserve attrition ratio.
- **Baseline Models**:
  - Logistic Regression (standard and class-weighted).
  - Decision Tree.
- **Advanced Models**:
  - Random Forest.
  - XGBoost (with SMOTE and isotonic calibration).
  - LightGBM.
  - CatBoost.
- **Class Imbalance Handling**:
  - Applied SMOTE to oversample the minority class (Attrition=Yes).
  - Used class weighting in Logistic Regression.
- **Hyperparameter Tuning**:
  - Performed GridSearchCV for Random Forest, XGBoost, LightGBM, and CatBoost.
  - Best parameters:
    - Random Forest: `{'max_depth': None, 'min_samples_split': 2, 'n_estimators': 200}`
    - XGBoost: `{'learning_rate': 0.1, 'max_depth': 6, 'n_estimators': 200}`
    - CatBoost: `{'depth': 6, 'iterations': 200, 'learning_rate': 0.1}`

### 4. Model Evaluation
- **Metrics**:
  - Confusion Matrix, Accuracy, Precision, Recall, F1 Score, ROC-AUC, PR-AUC.
  - Cost-sensitive evaluation: False Negatives (FN) cost=5, False Positives (FP) cost=1.
- **Results**:
  - **Logistic Regression (Weighted)**: Best cost performance (Total Cost: 110, F1: 0.55, ROC-AUC: 0.81).
  - **Tuned Random Forest**: High precision (0.73) but lower recall (0.24), cost: 174.
  - **XGBoost with SMOTE**: Balanced performance (F1: 0.30, ROC-AUC: 0.77, cost: 186).
  - **Tuned CatBoost**: Strong ROC-AUC (0.78), cost: 158.
  - **Decision Tree**: Poor performance (F1: 0.32, ROC-AUC: 0.59, cost: 185).
- **Trade-offs**:
  - Weighted Logistic Regression minimizes cost but sacrifices precision.
  - Tuned Random Forest offers high precision but misses many true positives (low recall).
  - XGBoost with SMOTE balances interpretability and performance.

### 5. Explainability & Fairness
- **SHAP Analysis**:
  - Top influential features: `OverTime`, `DistanceFromHome`, `BusinessTravel_Travel_Frequently`.
  - Insights:
    - Employees with `OverTime=Yes` have a 9.76% attrition rate.
    - Employees with low `DistanceFromHome` (bottom 25%) have a 4.55% attrition rate.
    - Frequent travelers (`BusinessTravel_Travel_Frequently=1`) have a 9.76% attrition rate.
- **Bias Analysis**:
  - **Gender**: Disparate Impact (DI) = 0.69 (bias detected, <0.8 threshold).
    - Attrition rate: Males (4.9%) vs. Females (7.1%).
  - **Age**: DI = inf (bias detected, Young vs. Senior).
    - Attrition rate: Young (15.1%), Middle (1.0%), Senior (0%).
  - **Department**: DI = 2.58 (bias detected).
    - Attrition rate: Sales (8.9%) vs. R&D (3.4%).
- **Mitigation Strategies**:
  - Gender: Adjust prediction thresholds, reweight samples, or remove `Gender_Male` feature.
  - Age: Use fairness-aware algorithms (e.g., Fairlearn), add age-based reweighting.
  - Department: Reweight samples by department, apply post-processing to adjust predictions.

### 6. Deployment
- **Backend**: FastAPI API (`app.py`) for predictions and SHAP explanations.
  - Endpoint: `POST /predict` accepts employee data, returns attrition probability and SHAP values.
  - Preprocessing: Aligns input with training features, applies scaling, and computes derived features.
- **Frontend**: Gradio dashboard (`gradio_app.py`) for interactive predictions.
  - Features sliders and dropdowns for all input variables.
  - Displays attrition probability and a SHAP bar plot for the top 10 influential features.
- **Model Storage**: Saved XGBoost model, scaler, and feature columns using `joblib`.

## Files
- **`employee_attrition_prediction.ipynb`**: Jupyter notebook with EDA, feature engineering, model building, evaluation, and fairness analysis.
- **`app.py`**: FastAPI backend for model predictions and SHAP explanations.
- **`gradio_app.py`**: Gradio dashboard for interactive predictions.
- **`models/`**: Directory containing saved model (`xgb_model_smote.pkl`), scaler (`scaler.pkl`), and feature columns (`feature_columns.pkl`).
- **`shap_plot.png`**: Generated SHAP summary plot (example output).

## Setup Instructions
1. **Clone the Repository**:
   ```bash
   git clone <repository-url>
   cd <repository-directory>

  ## Install Dependencies:
bashpip install -r requirements.txt
Required packages: pandas, numpy, scikit-learn, xgboost, lightgbm, catboost, imblearn, shap, fastapi, uvicorn, gradio, requests, seaborn, matplotlib.
Run the FastAPI Backend:
bashuvicorn app:app --reload
The API will run at http://127.0.0.1:8000.
Launch the Gradio Dashboard:
Run gradio_app.py:
bashpython gradio_app.py
Access the dashboard at the provided local URL (e.g., http://127.0.0.1:7860).
Make Predictions:

Use the Gradio interface to input employee details and view predictions.
Alternatively, send a POST request to http://127.0.0.1:8000/predict with JSON data (see EmployeeInput schema in app.py).



## Usage

## Gradio Dashboard: Adjust sliders and dropdowns to input employee data, click "Predict Attrition" to view the attrition probability and SHAP plot.
API Request Example:
bashcurl -X POST "http://127.0.0.1:8000/predict" -H "Content-Type: application/json" -d @input.json
## Sample input.json:
json{
    "Age": 25,
    "BusinessTravel": "Travel_Frequently",
    "DailyRate": 300,
    "Department": "Sales",
    "DistanceFromHome": 25,
    "Education": 2,
    "EducationField": "Life Sciences",
    "EnvironmentSatisfaction": 1,
    "Gender": "Male",
    "HourlyRate": 35,
    "JobInvolvement": 1,
    "JobLevel": 1,
    "JobSatisfaction": 1,
    "MaritalStatus": "Single",
    "MonthlyIncome": 2500,
    "MonthlyRate": 4000,
    "NumCompaniesWorked": 4,
    "OverTime": "Yes",
    "PercentSalaryHike": 11,
    "PerformanceRating": 3,
    "RelationshipSatisfaction": 1,
    "StockOptionLevel": 0,
    "TotalWorkingYears": 3,
    "TrainingTimesLastYear": 0,
    "WorkLifeBalance": 1,
    "YearsAtCompany": 1,
    "YearsInCurrentRole": 0,
    "YearsSinceLastPromotion": 1,
    "YearsWithCurrManager": 0
}


## Results

## Best Model: Weighted Logistic Regression (lowest cost: 110, highest F1: 0.55).
## Key Insights:

 Overtime, long commute distances, and frequent business travel significantly increase attrition risk.
Employees in Sales and younger employees (<30 years) are more likely to leave.


Fairness Concerns: Bias detected in gender, age, and department predictions, requiring mitigation.
Deployment: The Gradio dashboard and FastAPI API enable HR teams to predict attrition and understand key drivers.

## Future Improvements

Explore additional fairness-aware algorithms (e.g., Fairlearn, AIF360).
Incorporate time-series features (e.g., recent performance trends).
Enhance the dashboard with real-time bias metrics and mitigation options.
Test alternative oversampling techniques (e.g., ADASYN) or ensemble methods.

## License
This project is licensed under the MIT License. See the LICENSE file for details.
Acknowledgments

Dataset: IBM HR Analytics Employee Attrition Dataset (Kaggle).
Libraries: scikit-learn, xgboost, lightgbm, catboost, shap, fastapi, gradio.
