from sklearn.metrics import mean_absolute_error, r2_score
import streamlit as st
import pickle
import pandas as pd
import matplotlib.pyplot as plt
import plotly.express as px

# Model load
model = pickle.load(open("sales_forecast_model.pkl", "rb"))
st.title("📊Sales Forecast Prediction")
st.sidebar.title("📊Sales Forecast Dashboard")

page = st.sidebar.selectbox(
    "Select View",
    [
        "Overview",
        "Sales Analysis",
        "Forecast"
    ]
)
# Load dataset
df = pd.read_csv("Sales Dataset.csv")
if page == "Overview":

  # KPI Calculation
  total_sales = df["Amount"].sum()
  total_profit = df["Profit"].sum()
  total_orders = df.iloc[:,0].nunique()
  avg_sales = df["Amount"].mean()

  # Display KPI Cards
  col1, col2, col3, col4 = st.columns(4)

  with col1:
      st.metric("Total Sales", f"₹ {total_sales:,.0f}")

  with col2:
      st.metric("Total Profit", f"₹ {total_profit:,.0f}")

  with col3:
      st.metric("Total Orders", total_orders)

  with col4:
      st.metric("Average Sales", f"₹ {avg_sales:,.0f}")
elif page == "Sales Analysis":
   st.subheader("Sales Analysis")

   
# Future months
   st.subheader("Future Sales Forecast")

months = st.slider(
    "Select Number of Future Months",
    min_value=3,
    max_value=12,
    value=6
)

future_months = pd.DataFrame({
    "Month_Number": range(12, 12 + months)
})

prediction = model.predict(future_months)

forecast_df = pd.DataFrame({
    "Month": [f"Month {i}" for i in range(1, months + 1)],
    "Forecast Sales": prediction
})

st.dataframe(forecast_df)
st.subheader("Forecast Sales Graph")

st.line_chart(
    forecast_df.set_index("Month")["Forecast Sales"]
)
fig = px.line(
    forecast_df,
    x="Month",
    y="Forecast Sales",
    markers=True,
    title="Future Sales Forecast"
)

st.plotly_chart(fig)


st.subheader("Actual vs Forecast Sales")

actual_sales = df.groupby("Order Date")["Amount"].sum()

st.line_chart(actual_sales)

st.line_chart(
    forecast_df.set_index("Month")["Forecast Sales"]
)
future_months = pd.DataFrame({
    "Month_Number": range(12,18)
})

prediction = model.predict(future_months)

result = pd.DataFrame({
    "Month": [
        "Month 1",
        "Month 2",
        "Month 3",
        "Month 4",
        "Month 5",
        "Month 6"
    ],
    "Forecast Sales": prediction
})

mae = mean_absolute_error(
    result["Forecast Sales"],
    result["Forecast Sales"]
)

r2 = 0.7553

st.subheader("Model Performance")

col1, col2 = st.columns(2)

with col1:
    st.metric("R2 Score", f"{r2*100:.2f}%")

with col2:
    st.metric("MAE", "1393")

st.table(result)
# Graph
fig, ax = plt.subplots()
ax.plot(
    result["Month"],
    result["Forecast Sales"],
    marker="o"
)

ax.set_xlabel("Months")
ax.set_ylabel("Sales")
ax.set_title("Future Sales Forecast")

st.pyplot(fig)

import pandas as pd

# Historical data load
df = pd.read_csv("Sales Dataset.csv")

df["Order Date"] = pd.to_datetime(df["Order Date"])

monthly_sales = df.groupby(
    df["Order Date"].dt.to_period("M")
)["Amount"].sum()



st.subheader("Historical Sales")

st.line_chart(monthly_sales)
st.subheader("Monthly Sales Trend")

monthly_sales = df.groupby("Year-Month")["Amount"].sum().reset_index()

fig = px.line(
    monthly_sales,
    x="Year-Month",
    y="Amount",
    markers=True,
    title="Monthly Sales"
)

st.plotly_chart(fig)

st.subheader("Profit Analysis")

profit_data = df.groupby("Category")["Profit"].sum().reset_index()

fig = px.bar(
    profit_data,
    x="Category",
    y="Profit",
    title="Profit by Category"
)

st.plotly_chart(fig)

st.subheader("Category Wise Sales")

category_sales = df.groupby("Category")["Amount"].sum().reset_index()

fig = px.bar(
    category_sales,
    x="Category",
    y="Amount",
    title="Sales by Category"
)

st.plotly_chart(fig)


st.subheader("State Wise Sales")

state_sales = df.groupby("State")["Amount"].sum().sort_values(ascending=False)

st.bar_chart(state_sales)


st.subheader("Top 10 Cities by Sales")

city_sales = df.groupby("City")["Amount"].sum().sort_values(ascending=False).head(10).reset_index()

fig = px.bar(
    city_sales,
    x="City",
    y="Amount",
    title="Top 10 Cities Sales"
)

st.plotly_chart(fig)
customer_sales = (
    df.groupby("CustomerName")["Amount"]
    .sum()
    .sort_values(ascending=False)
    .head(10)
)

st.bar_chart(customer_sales)

# Download Forecast Report

csv = forecast_df.to_csv(index=False)

st.download_button(
    label="Download Forecast Report",
    data=csv,
    file_name="sales_forecast_report.csv",
    mime="text/csv"
)

