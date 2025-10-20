import gradio as gr
import requests
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import json
import logging

# Set up logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

# Default employee (high attrition case)
default_employee = {
    "Age": 25,
    "DailyRate": 300,
    "DistanceFromHome": 25,
    "Education": 2,
    "EnvironmentSatisfaction": 1,
    "HourlyRate": 35,
    "JobInvolvement": 1,
    "JobLevel": 1,
    "JobSatisfaction": 1,
    "MonthlyIncome": 2500,
    "MonthlyRate": 4000,
    "NumCompaniesWorked": 4,
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
    "YearsWithCurrManager": 0,
    "OverTime": "Yes",
    "BusinessTravel": "Travel_Frequently",
    "Department": "Sales",
    "EducationField": "Life Sciences",
    "Gender": "Male",
    "JobRole": "Sales Executive",
    "MaritalStatus": "Single"
}

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

# Define numerical ranges
numerical_ranges = {
    'Age': (18, 60),
    'DailyRate': (100, 1500),
    'DistanceFromHome': (1, 29),
    'Education': (1, 5),
    'EnvironmentSatisfaction': (1, 4),
    'HourlyRate': (30, 100),
    'JobInvolvement': (1, 4),
    'JobLevel': (1, 5),
    'JobSatisfaction': (1, 4),
    'MonthlyIncome': (1000, 20000),
    'MonthlyRate': (2000, 27000),
    'NumCompaniesWorked': (0, 9),
    'PercentSalaryHike': (11, 25),
    'PerformanceRating': (3, 4),
    'RelationshipSatisfaction': (1, 4),
    'StockOptionLevel': (0, 3),
    'TotalWorkingYears': (0, 40),
    'TrainingTimesLastYear': (0, 6),
    'WorkLifeBalance': (1, 4),
    'YearsAtCompany': (0, 40),
    'YearsInCurrentRole': (0, 18),
    'YearsSinceLastPromotion': (0, 15),
    'YearsWithCurrManager': (0, 17)
}

# Function to create Gradio input components with default values
def create_input_components():
    inputs = []
    for field, options in categorical_options.items():
        inputs.append(
            gr.Dropdown(choices=options, label=field, value=default_employee[field])
        )
    for field, (min_val, max_val) in numerical_ranges.items():
        inputs.append(
            gr.Slider(minimum=min_val, maximum=max_val, step=1, label=field, value=default_employee[field])
        )
    return inputs

# Function to make API call
def predict_attrition(*inputs):
    try:
        # Map inputs to EmployeeInput schema
        input_data = {}
        input_idx = 0
        for field in categorical_options.keys():
            input_data[field] = inputs[input_idx]
            input_idx += 1
        for field in numerical_ranges.keys():
            input_data[field] = int(inputs[input_idx])
            input_idx += 1

        # Send POST request to FastAPI backend
        response = requests.post("http://127.0.0.1:8000/predict", json=input_data)
        response.raise_for_status()
        result = response.json()

        # Extract prediction and SHAP values
        probability = result['attrition_probability']
        shap_values = result['shap_values']

        # Create SHAP summary plot
        shap_df = pd.DataFrame({
            'Feature': list(shap_values.keys()),
            'SHAP Value': list(shap_values.values())
        })
        shap_df = shap_df.sort_values(by='SHAP Value', ascending=False).head(10)

        plt.figure(figsize=(10, 6))
        sns.barplot(x='SHAP Value', y='Feature', data=shap_df, palette='viridis')
        plt.title('Top 10 Features Contributing to Attrition Prediction')
        plt.tight_layout()
        plt.savefig('shap_plot.png')
        plt.close()

        return (
            f"Attrition Probability: {probability:.2%}",
            'shap_plot.png'
        )

    except requests.exceptions.RequestException as e:
        logger.error(f"API request failed: {str(e)}")
        return f"Error: Could not connect to the API. Details: {str(e)}", None
    except Exception as e:
        logger.error(f"Prediction error: {str(e)}")
        return f"Error: {str(e)}", None

# Build the Gradio UI
with gr.Blocks() as demo:
    gr.Markdown("# 🧠 Employee Attrition Prediction Dashboard")
    gr.Markdown("Use the sliders and dropdowns to explore how employee factors influence attrition risk.")

    inputs = create_input_components()
    predict_button = gr.Button("🔍 Predict Attrition Risk")
    output_text = gr.Textbox(label="Prediction Result")
    output_plot = gr.Image(label="Top Influential Features (SHAP)")

    predict_button.click(
        fn=predict_attrition,
        inputs=inputs,
        outputs=[output_text, output_plot]
    )

# Launch Gradio app
demo.launch()
# import gradio as gr
# import requests
# import pandas as pd
# import matplotlib.pyplot as plt
# import seaborn as sns
# import json
# import logging

# # Set up logging
# logging.basicConfig(level=logging.DEBUG)
# logger = logging.getLogger(__name__)

# # Define categorical options (same as in app.py)
# categorical_options = {
#     'BusinessTravel': ['Travel_Rarely', 'Travel_Frequently', 'Non-Travel'],
#     'Department': ['Sales', 'Research & Development', 'Human Resources'],
#     'EducationField': ['Life Sciences', 'Other', 'Medical', 'Marketing', 'Technical Degree', 'Human Resources'],
#     'Gender': ['Female', 'Male'],
#     'JobRole': [
#         'Sales Executive', 'Research Scientist', 'Laboratory Technician', 'Manufacturing Director',
#         'Healthcare Representative', 'Manager', 'Sales Representative', 'Research Director', 'Human Resources'
#     ],
#     'MaritalStatus': ['Single', 'Married', 'Divorced'],
#     'OverTime': ['Yes', 'No']
# }

# # Define numerical ranges based on dataset insights
# numerical_ranges = {
#     'Age': (18, 60),
#     'DailyRate': (100, 1500),
#     'DistanceFromHome': (1, 29),
#     'Education': (1, 5),
#     'EnvironmentSatisfaction': (1, 4),
#     'HourlyRate': (30, 100),
#     'JobInvolvement': (1, 4),
#     'JobLevel': (1, 5),
#     'JobSatisfaction': (1, 4),
#     'MonthlyIncome': (1000, 20000),
#     'MonthlyRate': (2000, 27000),
#     'NumCompaniesWorked': (0, 9),
#     'PercentSalaryHike': (11, 25),
#     'PerformanceRating': (3, 4),
#     'RelationshipSatisfaction': (1, 4),
#     'StockOptionLevel': (0, 3),
#     'TotalWorkingYears': (0, 40),
#     'TrainingTimesLastYear': (0, 6),
#     'WorkLifeBalance': (1, 4),
#     'YearsAtCompany': (0, 40),
#     'YearsInCurrentRole': (0, 18),
#     'YearsSinceLastPromotion': (0, 15),
#     'YearsWithCurrManager': (0, 17)
# }

# # ✅ Default low attrition profile
# default_values = {
#     "Age": 40,
#     "DailyRate": 1100,
#     "DistanceFromHome": 5,
#     "Education": 4,
#     "EnvironmentSatisfaction": 4,
#     "HourlyRate": 80,
#     "JobInvolvement": 4,
#     "JobLevel": 3,
#     "JobSatisfaction": 4,
#     "MonthlyIncome": 15000,
#     "MonthlyRate": 20000,
#     "NumCompaniesWorked": 2,
#     "PercentSalaryHike": 20,
#     "PerformanceRating": 4,
#     "RelationshipSatisfaction": 4,
#     "StockOptionLevel": 2,
#     "TotalWorkingYears": 15,
#     "TrainingTimesLastYear": 3,
#     "WorkLifeBalance": 4,
#     "YearsAtCompany": 10,
#     "YearsInCurrentRole": 7,
#     "YearsSinceLastPromotion": 3,
#     "YearsWithCurrManager": 6,
#     "OverTime": "No",
#     "BusinessTravel": "Travel_Rarely",
#     "Department": "Research & Development",
#     "JobRole": "Research Scientist",
#     "MaritalStatus": "Married",
#     "Gender": "Female"
# }

# # Function to create Gradio input components
# def create_input_components():
#     inputs = []

#     # Add categorical inputs with default values
#     for field, options in categorical_options.items():
#         inputs.append(
#             gr.Dropdown(choices=options, label=field, value=default_values.get(field, options[0]))
#         )

#     # Add numerical inputs with default values
#     for field, (min_val, max_val) in numerical_ranges.items():
#         inputs.append(
#             gr.Slider(minimum=min_val, maximum=max_val, step=1, label=field, value=default_values.get(field, min_val))
#         )

#     return inputs

# # Function to make API call and generate SHAP plot
# def predict_attrition(*inputs):
#     try:
#         # Map inputs to the EmployeeInput schema
#         input_data = {}
#         input_idx = 0
#         for field in categorical_options.keys():
#             input_data[field] = inputs[input_idx]
#             input_idx += 1
#         for field in numerical_ranges.keys():
#             input_data[field] = int(inputs[input_idx])
#             input_idx += 1

#         # Send POST request to FastAPI backend
#         response = requests.post("http://127.0.0.1:8000/predict", json=input_data)
#         response.raise_for_status()
#         result = response.json()

#         # Extract prediction and SHAP values
#         probability = result['attrition_probability']
#         shap_values = result['shap_values']

#         # Create SHAP summary plot
#         shap_df = pd.DataFrame({
#             'Feature': list(shap_values.keys()),
#             'SHAP Value': list(shap_values.values())
#         })
#         shap_df = shap_df.sort_values(by='SHAP Value', ascending=False).head(10)

#         plt.figure(figsize=(10, 6))
#         sns.barplot(x='SHAP Value', y='Feature', data=shap_df, palette='viridis')
#         plt.title('Top 10 Features Contributing to Attrition Prediction')
#         plt.tight_layout()
#         plt.savefig('shap_plot.png')
#         plt.close()

#         # Return probability and SHAP plot
#         return (
#             f"Attrition Probability: {probability:.2%}",
#             'shap_plot.png'
#         )

#     except requests.exceptions.RequestException as e:
#         logger.error(f"API request failed: {str(e)}")
#         return f"Error: Could not connect to the API. Details: {str(e)}", None
#     except Exception as e:
#         logger.error(f"Prediction error: {str(e)}")
#         return f"Error: {str(e)}", None


# # ✅ Create Gradio interface
# with gr.Blocks() as demo:
#     gr.Markdown("# 🧩 Employee Attrition Prediction Dashboard")
#     gr.Markdown("Enter or adjust employee details to predict attrition risk and visualize SHAP-based feature importance.")

#     inputs = create_input_components()
#     predict_button = gr.Button("🔍 Predict Attrition")
#     output_text = gr.Textbox(label="Prediction Result")
#     output_plot = gr.Image(label="Top 10 SHAP Feature Influences")

#     predict_button.click(
#         fn=predict_attrition,
#         inputs=inputs,
#         outputs=[output_text, output_plot]
#     )

# # Launch interface
# demo.launch()
