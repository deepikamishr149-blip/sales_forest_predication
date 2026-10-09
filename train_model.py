import pandas as pd
from sklearn.ensemble import RandomForestRegressor
import pickle

# Dataset load
df = pd.read_csv("Sales Dataset.csv")

# Date convert
df["Order Date"] = pd.to_datetime(df["Order Date"])

# Month wise sales
monthly_sales = df.groupby(
    df["Order Date"].dt.to_period("M")
)["Amount"].sum().reset_index()

monthly_sales["Month"] = monthly_sales["Order Date"].astype(str)

# Features
monthly_sales["Month_Number"] = range(len(monthly_sales))

X = monthly_sales[["Month_Number"]]
y = monthly_sales["Amount"]

# Model
model = RandomForestRegressor(random_state=42)
model.fit(X, y)

# Save model
pickle.dump(model, open("sales_forecast_model.pkl", "wb"))

print("Sales Forecast Model Created Successfully!")
from sklearn.metrics import mean_absolute_error, r2_score

# Model prediction
y_pred = model.predict(X)

print("MAE:", mean_absolute_error(y, y_pred))
print("R2 Score:", r2_score(y, y_pred))


"""
from sklearn.metrics import mean_absolute_error, r2_score

# Prediction on test data
y_pred = model.predict(X_test)

print("MAE:", mean_absolute_error(y_test, y_pred))
print("R2 Score:", r2_score(y_test, y_pred))


from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
import numpy as np

y_pred = model.predict(X_test)

print("R2 Score :", round(r2_score(y_test, y_pred),3))
print("MAE :", round(mean_absolute_error(y_test, y_pred),2))
print("RMSE :", round(np.sqrt(mean_squared_error(y_test, y_pred)),2))

importance = pd.DataFrame({
    "Feature": X.columns,
    "Importance": model.feature_importances_
})

print("\nFeature Importance")
print(importance.sort_values(by="Importance", ascending=False))"""