# NYC Yellow Cab Intelligence: Forecasting Fares, Time & Demand

## 📌 Project Overview
This project develops a predictive analytics framework for NYC Yellow Taxi operations. It utilizes machine learning to forecast three key operational metrics:
1. **Fare Prediction:** Regression models to estimate total fare amount.
2. **Trip Duration Prediction:** Predicting travel time based on pickup/drop-off data.
3. **Demand Forecasting:** Time-series forecasting for taxi demand at JFK Airport.

## 📂 Repository Structure
- `notebooks/`: Jupyter notebooks containing EDA, preprocessing, and model training.
- `reports/`: Final project report and presentation slides.

## 🛠️ Technologies & Tools
- **Cloud Infrastructure:** AWS S3 (Data Storage), AWS SageMaker (Development)
- **Languages:** Python
- **Libraries:** Pandas, Scikit-Learn, XGBoost, SHAP, Prophet, Statsmodels
- **Techniques:** Cyclical Encoding, One-Hot Encoding, StandardScaler, GridSearch

## 🧠 Methodologies
### 1. Fare & Duration Prediction (Regression)
- **Models:** Linear Regression, Random Forest, XGBoost.
- **Key Features:** Trip distance, pickup/drop-off zones, time slots (cyclical encoding for hour/day).
- **Performance:**
  - **XGBoost** performed best for both tasks (R² ~0.99 for Fare, ~0.93 for Duration).
  - SHAP analysis revealed `trip_distance` and `fare_per_mile` as top predictors.

### 2. Demand Forecasting (Time Series)
- **Objective:** Forecast hourly pickups at JFK Airport for the next 48 hours.
- **Models:** Facebook Prophet vs. SARIMA.
- **Performance:** **Prophet** (RMSE 5.36) outperformed SARIMA (RMSE 6.35) due to better handling of seasonality.

## 👥 Authors
- Maithili Lingala
- Sandeep Pandellapalli
- Shobha Ganapati Bhat
