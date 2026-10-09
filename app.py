import streamlit as st
import pandas as pd
import plotly.express as px
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error

st.title("EC2 Instance EDA Dashboard")

df = pd.read_csv("ec2dataset.csv")

st.write("Dataset Preview")
st.dataframe(df)    #part 2

st.subheader("Dataset Information")

col1, col2, col3 = st.columns(3)

col1.metric("Number of Instances", len(df))
col2.metric("Number of Columns", len(df.columns))
col3.metric("Missing Values", df.isna().sum().sum())  #part 3

st.subheader("Dataset Structure")

st.write("Columns:")
st.write(df.columns.tolist())

st.subheader("Data Types")

dtype_df = pd.DataFrame({
    "Column": df.columns,
    "Data Type": df.dtypes.astype(str).values
})

st.dataframe(dtype_df, use_container_width=True) #part 4

df["Memory_GiB"] = (
    df["Instance Memory"]
    .str.extract(r"([\d.]+)")
    .astype(float)
)   #part 5

df["vCPU_Count"] = (
    df["vCPUs"]
    .str.extract(r"(\d+)")
    .astype(float)
)   

def clean_price(value):
    if pd.isna(value):
        return None

    value = str(value)

    if "unavailable" in value.lower():
        return None

    return float(
        value.replace("$", "")
        .replace(" hourly", "")
        .strip()
    )

price_columns = [
    "On Demand",
    "Linux Reserved cost",
    "Linux Spot Minimum cost",
    "Windows On Demand cost",
    "Windows Reserved cost"
]

for column in price_columns:
    df[column + "_USD"] = df[column].apply(clean_price)  #Part 7


df["Monthly_On_Demand"] = df["On Demand_USD"] * 730

df["Cost_Per_GiB"] = (
    df["On Demand_USD"] /
    df["Memory_GiB"]
)

df["Cost_Per_vCPU"] = (
    df["On Demand_USD"] /
    df["vCPU_Count"]
)

st.subheader("Monthly On-Demand Cost")

st.dataframe(
    df[
        [
            "Name",
            "API Name",
            "Memory_GiB",
            "vCPU_Count",
            "On Demand_USD",
            "Monthly_On_Demand"
        ]
    ]
)               #part 8

#FILTERS
st.sidebar.header("Filters")

max_memory = float(df["Memory_GiB"].max())

memory_filter = st.sidebar.slider(
    "Maximum Memory (GiB)",
    min_value=0.5,
    max_value=max_memory,
    value=max_memory
)
cpu_values = sorted(
    df["vCPU_Count"].dropna().unique()
)

selected_cpu = st.sidebar.multiselect(
    "vCPU Count",
    options=cpu_values,
    default=cpu_values
)

#Network Performance Filter
network_values = sorted(
    df["Network Performance"].dropna().unique()
)

selected_network = st.sidebar.multiselect(
    "Network Performance",
    options=network_values,
    default=network_values
)

#maximum Hourly Cost Filter
max_hourly_cost = float(
    df["On Demand_USD"].dropna().max()
)

hourly_cost_filter = st.sidebar.slider(
    "Maximum Hourly Cost ($)",
    min_value=0.0,
    max_value=max_hourly_cost,
    value=max_hourly_cost
)

filtered_df = df[
    (df["Memory_GiB"] <= memory_filter) &
    (df["vCPU_Count"].isin(selected_cpu)) &
    (df["Network Performance"].isin(selected_network)) &
    (df["On Demand_USD"] <= hourly_cost_filter)
]
st.subheader("Filtered EC2 Instances")

st.write(
    f"{len(filtered_df)} instances found"
)

st.dataframe(filtered_df)    #part 9 & 10
# PART 11 - KPI CARDS

st.subheader("EC2 Summary")

col1, col2, col3, col4 = st.columns(4)

col1.metric(
    "Instances",
    len(filtered_df)
)

col2.metric(
    "Avg Memory",
    f"{filtered_df['Memory_GiB'].mean():.2f} GiB"
)

col3.metric(
    "Avg vCPUs",
    f"{filtered_df['vCPU_Count'].mean():.1f}"
)

col4.metric(
    "Avg Hourly Cost",
    f"${filtered_df['On Demand_USD'].mean():.4f}"
)
      #Part 11

st.subheader("Memory Distribution")

fig = px.histogram(
    filtered_df,
    x="Memory_GiB",
    nbins=30,
    title="Distribution of EC2 Memory"
)

st.plotly_chart(fig, use_container_width=True)


st.subheader("vCPU Distribution")

fig = px.histogram(
    filtered_df,
    x="vCPU_Count",
    title="Distribution of vCPUs"
)

st.plotly_chart(fig, use_container_width=True)

st.subheader("Memory vs vCPUs")

fig = px.scatter(
    filtered_df,
    x="vCPU_Count",
    y="Memory_GiB",
    hover_name="API Name",
    hover_data=["On Demand_USD"],
    title="EC2 Memory vs vCPU"
)

st.plotly_chart(fig, use_container_width=True)

st.subheader("Memory vs On-Demand Cost")

fig = px.scatter(
    filtered_df,
    x="Memory_GiB",
    y="On Demand_USD",
    hover_name="API Name",
    size="vCPU_Count",
    title="Memory vs EC2 On-Demand Cost"
)

st.plotly_chart(fig, use_container_width=True)

st.subheader("Lowest-Cost EC2 Instances")

cheapest = (
    filtered_df
    .sort_values("On Demand_USD")
    [
        [
            "Name",
            "API Name",
            "Memory_GiB",
            "vCPU_Count",
            "On Demand_USD",
            "Monthly_On_Demand"
        ]
    ]
    .head(10)
)

st.dataframe(cheapest)

st.subheader("Highest-Cost EC2 Instances")

most_expensive = (
    filtered_df
    .sort_values(
        "On Demand_USD",
        ascending=False
    )
    [
        [
            "Name",
            "API Name",
            "Memory_GiB",
            "vCPU_Count",
            "On Demand_USD",
            "Monthly_On_Demand"
        ]
    ]
    .head(10)
)

st.dataframe(most_expensive)

pricing = filtered_df[
    [
        "Name",
        "API Name",
        "On Demand_USD",
        "Linux Reserved cost_USD",
        "Linux Spot Minimum cost_USD",
        "Windows On Demand cost_USD",
        "Windows Reserved cost_USD"
    ]
]

st.subheader("EC2 Pricing Comparison")
st.dataframe(pricing)

selected_instance = st.selectbox(
    "Select an EC2 Instance",
    filtered_df["API Name"].unique()
)

instance = filtered_df[
    filtered_df["API Name"] == selected_instance
].iloc[0]

pricing_data = pd.DataFrame({
    "Pricing Model": [
        "On Demand",
        "Linux Reserved",
        "Linux Spot"
    ],

    "Hourly Cost": [
        instance["On Demand_USD"],
        instance["Linux Reserved cost_USD"],
        instance["Linux Spot Minimum cost_USD"]
    ]
})

pricing_data = pricing_data.dropna()

fig = px.bar(
    pricing_data,
    x="Pricing Model",
    y="Hourly Cost",
    title=f"Pricing Comparison: {selected_instance}"
)

st.plotly_chart(
    fig,
    use_container_width=True
)

pricing_data["Monthly Cost"] = (
    pricing_data["Hourly Cost"] * 730
)

st.dataframe(pricing_data)

# Cost efficiency calculations

df["Cost_Per_GiB"] = (
    df["On Demand_USD"] /
    df["Memory_GiB"]
)

df["Cost_Per_vCPU"] = (
    df["On Demand_USD"] /
    df["vCPU_Count"]
)

st.subheader("Cost per GiB of Memory")

efficiency = (
    filtered_df
    .sort_values("Cost_Per_GiB")
    [
        [
            "Name",
            "API Name",
            "Memory_GiB",
            "vCPU_Count",
            "On Demand_USD",
            "Cost_Per_GiB"
        ]
    ]
    .head(15)
)

st.dataframe(efficiency)

csv = filtered_df.to_csv(index=False)

st.download_button(
    label="Download Filtered Dataset",
    data=csv,
    file_name="filtered_ec2_instances.csv",
    mime="text/csv"
)

#completing the lab 
# ============================================================
# ACTIVITY - ANALYZING AMAZON EC2 INSTANCE COSTS
# ============================================================

st.divider()
st.header("Activity: Analyzing Amazon EC2 Instance Costs")
# ------------------------------------------------------------
# ACTIVITY PART 1 - STEP 4: SUMMARY ANALYSIS
# ------------------------------------------------------------

st.subheader("Cost Summary Statistics")

activity_cost_columns = [
    "On Demand_USD",
    "Linux Reserved cost_USD",
    "Linux Spot Minimum cost_USD",
    "Windows On Demand cost_USD",
    "Windows Reserved cost_USD"
]

cost_summary = df[activity_cost_columns].describe()

st.dataframe(cost_summary)

# ------------------------------------------------------------
# ACTIVITY PART 1 - STEP 5: COST BOXPLOT
# ------------------------------------------------------------

st.subheader("Cost Comparison of Amazon EC2 Instances")

fig_box, ax = plt.subplots(figsize=(12, 6))

sns.boxplot(
    data=df[activity_cost_columns],
    ax=ax
)

ax.set_title(
    "Cost Comparison of Amazon EC2 Instances (Hourly)"
)

ax.set_ylabel("Cost (USD)")

ax.tick_params(
    axis="x",
    rotation=45
)

plt.tight_layout()

st.pyplot(fig_box)

plt.close(fig_box)


# ------------------------------------------------------------
# ACTIVITY PART 1 - STEP 6: OUTLIER DETECTION
# ------------------------------------------------------------

st.subheader("On-Demand Cost Outliers")

def detect_outliers(dataframe, column):

    Q1 = dataframe[column].quantile(0.25)
    Q3 = dataframe[column].quantile(0.75)

    IQR = Q3 - Q1

    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR

    return dataframe[
        (dataframe[column] < lower_bound) |
        (dataframe[column] > upper_bound)
    ]


outliers_on_demand = detect_outliers(
    df,
    "On Demand_USD"
)

st.write(
    f"Number of On-Demand cost outliers: "
    f"{len(outliers_on_demand)}"
)

st.dataframe(
    outliers_on_demand[
        [
            "Name",
            "API Name",
            "Memory_GiB",
            "vCPU_Count",
            "On Demand_USD"
        ]
    ]
)


# ------------------------------------------------------------
# ACTIVITY PART 1 - STEP 7: RESERVED VS ON-DEMAND
# ------------------------------------------------------------

st.subheader("Reserved vs On-Demand Cost")

cost_comparison = (
    df[
        [
            "Name",
            "API Name",
            "On Demand_USD",
            "Linux Reserved cost_USD"
        ]
    ]
    .dropna()
    .sort_values("On Demand_USD")
)

st.write("Top 10 Lowest-Cost Instances")

st.dataframe(
    cost_comparison.head(10)
)

# ------------------------------------------------------------
# ACTIVITY PART 1 - STEP 8: INSTANCE FAMILY ANALYSIS
# ------------------------------------------------------------

st.subheader("T2 vs T3 Instance Family Analysis")


def filter_instance_family(family):

    return df[
        df["Name"]
        .astype(str)
        .str.startswith(family, na=False)
    ]


t2_instances = filter_instance_family("T2")
t3_instances = filter_instance_family("T3")

st.write("T2 Instance Costs Summary")

t2_summary = (
    t2_instances[activity_cost_columns]
    .describe()
)

st.dataframe(t2_summary)


st.write("T3 Instance Costs Summary")

t3_summary = (
    t3_instances[activity_cost_columns]
    .describe()
)

st.dataframe(t3_summary)

st.subheader("T2 Cost Distribution")

fig_t2, ax = plt.subplots(figsize=(12, 6))

sns.boxplot(
    data=t2_instances[activity_cost_columns],
    showmeans=True,
    ax=ax
)

ax.set_title("Cost Distribution for T2 Instances")
ax.set_ylabel("Cost (USD)")
ax.tick_params(axis="x", rotation=45)

plt.tight_layout()

st.pyplot(fig_t2)

plt.close(fig_t2)

st.subheader("T3 Cost Distribution")

fig_t3, ax = plt.subplots(figsize=(12, 6))

sns.boxplot(
    data=t3_instances[activity_cost_columns],
    showmeans=True,
    ax=ax
)

ax.set_title("Cost Distribution for T3 Instances")
ax.set_ylabel("Cost (USD)")
ax.tick_params(axis="x", rotation=45)

plt.tight_layout()

st.pyplot(fig_t3)

plt.close(fig_t3)

st.subheader(
    "T2 and T3 On-Demand vs Reserved Cost"
)

comparison = pd.concat([
    t2_instances[
        [
            "Name",
            "API Name",
            "On Demand_USD",
            "Linux Reserved cost_USD"
        ]
    ],

    t3_instances[
        [
            "Name",
            "API Name",
            "On Demand_USD",
            "Linux Reserved cost_USD"
        ]
    ]
])

comparison_sorted = (
    comparison
    .dropna()
    .sort_values("On Demand_USD")
)

st.dataframe(
    comparison_sorted.head(10)
)


# ============================================================
# ACTIVITY PART 2 - REGRESSION ANALYSIS
# ============================================================

st.divider()

st.header(
    "Predicting EC2 Instance Costs Using Regression Analysis"
)

# ------------------------------------------------------------
# PART 2 - STEPS 3 & 4: FEATURES AND MISSING DATA
# ------------------------------------------------------------

regression_data = df[
    [
        "Memory_GiB",
        "vCPU_Count",
        "On Demand_USD"
    ]
].dropna()

st.subheader("Regression Dataset")

st.dataframe(
    regression_data.head()
)

st.write(
    f"Records available for regression: "
    f"{len(regression_data)}"
)

# Predictor variables
X = regression_data[
    [
        "Memory_GiB",
        "vCPU_Count"
    ]
]

# Target variable
y = regression_data[
    "On Demand_USD"
]


X_train, X_test, y_train, y_test = (
    train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42
    )
)

st.subheader("Training and Testing Data")

col1, col2 = st.columns(2)

col1.metric(
    "Training Samples",
    len(X_train)
)

col2.metric(
    "Testing Samples",
    len(X_test)
)

# ------------------------------------------------------------
# PART 2 - STEP 6: TRAIN MODEL
# ------------------------------------------------------------

model = LinearRegression()

model.fit(
    X_train,
    y_train
)

st.subheader("Linear Regression Model")

st.write(
    f"Intercept: {model.intercept_:.6f}"
)

coefficients = pd.DataFrame({
    "Feature": [
        "Memory_GiB",
        "vCPU_Count"
    ],

    "Coefficient": model.coef_
})

st.dataframe(coefficients)

# ------------------------------------------------------------
# PART 2 - STEP 7: MODEL EVALUATION
# ------------------------------------------------------------

y_pred = model.predict(
    X_test
)

mae = mean_absolute_error(
    y_test,
    y_pred
)

mse = mean_squared_error(
    y_test,
    y_pred
)

rmse = mse ** 0.5


st.subheader("Model Performance")

col1, col2, col3 = st.columns(3)

col1.metric(
    "MAE",
    f"{mae:.4f}"
)

col2.metric(
    "MSE",
    f"{mse:.4f}"
)

col3.metric(
    "RMSE",
    f"{rmse:.4f}"
)

# ------------------------------------------------------------
# PART 2 - STEP 8: ACTUAL VS PREDICTED
# ------------------------------------------------------------

st.subheader(
    "Actual vs Predicted On-Demand Costs"
)

fig_reg, ax = plt.subplots(
    figsize=(8, 6)
)

ax.scatter(
    y_test,
    y_pred,
    alpha=0.7
)

minimum = min(
    y_test.min(),
    y_pred.min()
)

maximum = max(
    y_test.max(),
    y_pred.max()
)

ax.plot(
    [minimum, maximum],
    [minimum, maximum],
    linestyle="--"
)

ax.set_title(
    "Actual vs Predicted On-Demand Costs"
)

ax.set_xlabel(
    "Actual On-Demand Cost"
)

ax.set_ylabel(
    "Predicted On-Demand Cost"
)

plt.tight_layout()

st.pyplot(fig_reg)

plt.close(fig_reg)

# ------------------------------------------------------------
# PART 2 - STEP 9: MAKE A PREDICTION
# ------------------------------------------------------------

st.subheader(
    "Predict Cost for a New EC2 Configuration"
)

new_instance = pd.DataFrame({
    "Memory_GiB": [4],
    "vCPU_Count": [2]
})

predicted_cost = model.predict(
    new_instance
)

st.success(
    f"Predicted On-Demand Cost for "
    f"4 GiB memory and 2 vCPUs: "
    f"${predicted_cost[0]:.4f} per hour"
)