# Medical Insurance Cost Predictor

A machine learning web application that predicts medical insurance costs using various regression models. Built with Streamlit and scikit-learn.

## Features

- **Multiple ML Models**: Compare 9 different regression algorithms
- **Interactive Web Interface**: User-friendly Streamlit dashboard
- **Real-time Predictions**: Instant cost predictions based on user input
- **Model Performance Metrics**: Comprehensive evaluation with MSE, RMSE, R², and MAE
- **Feature Importance Visualization**: See which factors most influence insurance costs
- **Model Comparison**: Side-by-side comparison of different algorithms

## Available Models

- Linear Regression
- Ridge Regression
- Lasso Regression
- Decision Tree
- Random Forest
- Gradient Boosting
- XGBoost
- Support Vector Regression (SVR)
- K-Nearest Neighbors

## Installation

1. Clone or download this repository
2. Install required dependencies:

```bash
pip install streamlit pandas numpy matplotlib seaborn scikit-learn xgboost
```

## Usage

1. Navigate to the project directory:
```bash
cd medical_predicition
```

2. Run the Streamlit application:
```bash
streamlit run medical_proj.py
```

3. Open your web browser and go to the displayed local URL (typically `http://localhost:8501`)

## Input Parameters

The application requires the following information to predict insurance costs:

- **Age**: 18-100 years
- **Sex**: Male or Female
- **BMI**: Body Mass Index (15.0-50.0)
- **Children**: Number of dependents (0-10)
- **Smoker**: Yes or No
- **Region**: Northeast, Northwest, Southeast, or Southwest

## Dataset

The application uses the `medical_insurance (1).csv` dataset containing historical insurance data with the following features:
- Age, sex, BMI, number of children, smoking status, region, and insurance charges

## Model Performance

Each model provides:
- **MSE**: Mean Squared Error
- **RMSE**: Root Mean Squared Error
- **R²**: Coefficient of Determination
- **MAE**: Mean Absolute Error
- **Training Time**: Time taken to train the model

## Features

### Model Selection
- Choose single or multiple models for comparison
- Default selection includes Random Forest and Linear Regression

### Visualization
- Feature importance plots for tree-based models
- Model comparison charts
- Dataset exploration tools

### Dataset Information
- View dataset overview and statistics
- Sample data preview
- Feature distribution plots

## File Structure

```
medical_predicition/
├── medical_proj.py          # Main application file
├── medical_insurance (1).csv # Dataset
└── README.md               # This file
```

## Requirements

- Python 3.7+
- streamlit
- pandas
- numpy
- matplotlib
- seaborn
- scikit-learn
- xgboost

## How It Works

1. **Data Loading**: Loads the insurance dataset
2. **Preprocessing**: Applies StandardScaler for numerical features and OneHotEncoder for categorical features
3. **Model Training**: Trains selected models using scikit-learn pipelines
4. **Prediction**: Makes predictions based on user input
5. **Evaluation**: Calculates performance metrics on test data
6. **Visualization**: Displays results and comparisons

## Contributing

Feel free to fork this project and submit pull requests for improvements.

## License

This project is open source and available under the MIT License.