import streamlit as st
import pandas as pd
import numpy as np
import requests
from io import BytesIO
from datetime import datetime
import os

# Visualization and modeling
import matplotlib
os.environ.setdefault('MPLCONFIGDIR', '/tmp/matplotlib')
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

# --- configuration ---
st.set_page_config(layout="wide", page_title="Spanish Heritage APP")

# Custom CSS for top-right name
st.markdown(
    """
    <style>
    .top-right {
        position: absolute;
        top: 10px;
        right: 25px;
        font-size: 16px;
        font-weight: bold;
        color: #444;
    }
    </style>
    <div class="top-right">Siron Barker</div>
    """,
    unsafe_allow_html=True
)

st.title("Spanish Heritage APP — 70-year historical series")

# Countries (3 wealthiest in Latin America by GDP)
COUNTRIES = {
    "Brazil": "BR",
    "Mexico": "MX",
    "Argentina": "AR"
}

# Map user categories to World Bank indicators
INDICATOR_MAP = {
    "Population": {"code": "SP.POP.TOTL","units": "people","source": "World Bank (Population, total)"},
    "Unemployment rate": {"code": "SL.UEM.TOTL.ZS","units": "%","source": "World Bank (Unemployment, total % of labor force)"},
    "Education levels from 0-25": {"code": "SE.SCH.LIFE","units": "years (rescaled 0-25)","source": "World Bank (School life expectancy)"},
    "Life expectancy": {"code": "SP.DYN.LE00.IN","units": "years","source": "World Bank (Life expectancy at birth, total)"},
    "Average wealth": {"code": "NY.GDP.PCAP.CD","units": "current US$","source": "World Bank (GDP per capita)"},
    "Average income": {"code": "NY.GDP.PCAP.CD","units": "current US$","source": "World Bank (GDP per capita)"},
    "Birth rate": {"code": "SP.DYN.CBRT.IN","units": "per 1,000 people","source": "World Bank (Crude birth rate)"},
    "Immigration out of the country": {"code": "SM.POP.NETM","units": "people","source": "World Bank (Net migration)"},
    "Murder Rate": {"code": "VC.IHR.PSRC.P5","units": "per 100,000","source": "World Bank (Intentional homicides)"}
}

# --- helper functions ---
def wb_fetch(country_code, indicator, start, end):
    url = f"https://api.worldbank.org/v2/country/{country_code}/indicator/{indicator}?date={start}:{end}&format=json&per_page=2000"
    resp = requests.get(url)
    if resp.status_code != 200:
        return pd.DataFrame({"year": [], "value": []})
    data = resp.json()
    if not isinstance(data, list) or len(data) < 2:
        return pd.DataFrame({"year": [], "value": []})
    rows = []
    for r in data[1]:
        if r.get("date") and r.get("value") is not None:
            rows.append({"year": int(r["date"]), "value": float(r["value"])})    
    return pd.DataFrame(rows).dropna().sort_values("year")

def regression_analysis(df, degree, extrapolate_years):
    poly = PolynomialFeatures(degree=degree, include_bias=False)
    X = poly.fit_transform(df[["year"]])
    model = LinearRegression()
    model.fit(X, df["value"])
    y_pred = model.predict(X)

    eq = f"{model.intercept_:.3f}"
    for i, c in enumerate(model.coef_):
        eq += f" + {c:.3f}*x^{i+1}"
    eq = eq.replace("+ -", "- ")

    r2 = r2_score(df["value"], y_pred)
    mae = mean_absolute_error(df["value"], y_pred)
    rmse = np.sqrt(mean_squared_error(df["value"], y_pred))

    x_min, x_max = df["year"].min(), df["year"].max() + extrapolate_years
    x_smooth = np.linspace(x_min, x_max, 300)
    y_smooth = model.predict(poly.transform(x_smooth.reshape(-1, 1)))

    return model, eq, r2, mae, rmse, x_smooth, y_smooth

def analyze_function(model, df, degree):
    poly = PolynomialFeatures(degree=degree, include_bias=False)
    X = poly.fit_transform(df[["year"]])
    years = df["year"].values
    values = model.predict(X)

    analysis = []
    diffs = np.diff(values)
    for i, d in enumerate(diffs, start=1):
        if np.sign(d) != np.sign(diffs[i-1]) and i > 1:
            analysis.append(f"Local extremum around {years[i]} with value ≈ {values[i]:.2f}.")
    max_slope_idx = np.argmax(diffs)
    analysis.append(f"Fastest growth around {years[max_slope_idx]} with Δ≈{diffs[max_slope_idx]:.2f}.")
    return analysis

# --- sidebar controls ---
with st.sidebar:
    country = st.selectbox("Country", list(COUNTRIES.keys()), index=0)
    category = st.selectbox("Category", list(INDICATOR_MAP.keys()), index=0)
    degree = st.slider("Polynomial degree (≥3)", 3, 8, 3)
    step_years = st.slider("Graph increment (years)", 1, 10, 1)
    extrapolate_years = st.number_input("Extrapolate forward (years)", 0, 50, 5)

# --- main app ---
now = datetime.now().year
start, end = now - 69, now - 1
df = wb_fetch(COUNTRIES[country], INDICATOR_MAP[category]["code"], start, end)

if df.empty:
    st.error("No data available.")
    st.stop()

st.subheader(f"Raw Data: {category} for {country}")
st.caption(f"Source: {INDICATOR_MAP[category]['source']} | Units: {INDICATOR_MAP[category]['units']}")
edited_df = st.data_editor(df, num_rows="dynamic")
df = edited_df.dropna().sort_values("year")

if len(df) < degree + 1:
    st.error("Not enough points for chosen polynomial degree.")
    st.stop()

# Regression
model, eq, r2, mae, rmse, x_smooth, y_smooth = regression_analysis(df, degree, extrapolate_years)

# Plot
fig, ax = plt.subplots(figsize=(10, 6))
ax.scatter(df["year"], df["value"], label="Data")
ax.plot(x_smooth, y_smooth, color="red", label="Regression fit")
ax.set_xlabel("Year")
ax.set_ylabel(INDICATOR_MAP[category]["units"])
ax.legend()
st.pyplot(fig)

# Model equation
st.subheader("Model Equation")
st.code(f"f(x) = {eq}")

# Metrics
st.markdown(f"- R² = {r2:.3f}")
st.markdown(f"- MAE = {mae:.3f}")
st.markdown(f"- RMSE = {rmse:.3f}")

# Function analysis
st.subheader("Function Analysis")
analysis = analyze_function(model, df, degree)
for line in analysis:
    st.write("•", line)

# Extrapolation example
future_year = df["year"].max() + extrapolate_years
future_val = model.predict(PolynomialFeatures(degree=degree, include_bias=False).fit_transform([[future_year]]))[0]
st.markdown(f"According to the model, in **{future_year}** the {category.lower()} will be about **{future_val:.2f} {INDICATOR_MAP[category]['units']}**.")

# --- Download results ---
report = f"""
Analysis Report
---------------
Country: {country}
Category: {category}
Model Equation: f(x) = {eq}

Metrics:
- R² = {r2:.3f}
- MAE = {mae:.3f}
- RMSE = {rmse:.3f}

Function Analysis:
{chr(10).join(analysis)}

Future Projection:
Year {future_year} → {future_val:.2f} {INDICATOR_MAP[category]['units']}
"""

# Download buttons
st.download_button(
    label="📥 Download Analysis Report (.txt)",
    data=report,
    file_name=f"analysis_{country}_{category}.txt",
    mime="text/plain"
)

csv_data = df.to_csv(index=False).encode("utf-8")
st.download_button(
    label="📊 Download Raw Dataset (.csv)",
    data=csv_data,
    file_name=f"dataset_{country}_{category}.csv",
    mime="text/csv"
)
