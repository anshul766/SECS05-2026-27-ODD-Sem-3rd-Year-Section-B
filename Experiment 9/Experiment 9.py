# Anshul Hudda
# CU24250002
# Sec - 'B'
# ============================================================
# EXPERIMENT NO. 9
# Data Storytelling and Business Insight Generation
# using Interactive Visualizations
# ============================================================

# -----------------------------
# 1. Import Libraries
# -----------------------------
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go

from IPython.display import display

print("=" * 75)
print("DATA STORYTELLING AND BUSINESS INSIGHT GENERATION")
print("=" * 75)


# -----------------------------
# 2. Load Dataset
# -----------------------------
df = pd.read_csv("/content/superstore.csv")

print("\n1. DATASET OVERVIEW")
print("-" * 50)

print("Dataset Shape:", df.shape)

print("\nFirst 5 Records:")
display(df.head())

print("\nColumn Names:")
print(df.columns.tolist())


# -----------------------------
# 3. Clean Column Names
# -----------------------------
df.columns = (
    df.columns
    .str.strip()
    .str.replace(" ", "_")
    .str.replace("-", "_")
)

print("\nCleaned Column Names:")
print(df.columns.tolist())


# -----------------------------
# 4. Basic Data Information
# -----------------------------
print("\n2. DATA INFORMATION")
print("-" * 50)

print("\nData Types:")
print(df.dtypes)

print("\nMissing Values:")
print(df.isnull().sum())

print("\nDuplicate Rows:", df.duplicated().sum())


# -----------------------------
# 5. Convert Date Columns
# -----------------------------
possible_order_dates = [
    "Order_Date",
    "Order_Date_",
    "OrderDate",
    "Order_Date"
]

possible_ship_dates = [
    "Ship_Date",
    "ShipDate"
]

for col in possible_order_dates:
    if col in df.columns:
        df[col] = pd.to_datetime(df[col], errors="coerce")
        order_date_col = col
        break
else:
    order_date_col = None

for col in possible_ship_dates:
    if col in df.columns:
        df[col] = pd.to_datetime(df[col], errors="coerce")
        ship_date_col = col
        break
else:
    ship_date_col = None


# -----------------------------
# 6. Identify Important Columns
# -----------------------------
def find_column(possible_names):
    for name in possible_names:
        if name in df.columns:
            return name
    return None


sales_col = find_column(["Sales", "sales"])
profit_col = find_column(["Profit", "profit"])
quantity_col = find_column(["Quantity", "quantity"])
discount_col = find_column(["Discount", "discount"])
category_col = find_column(["Category", "category"])
subcategory_col = find_column(["Sub_Category", "SubCategory"])
region_col = find_column(["Region", "region"])
country_col = find_column(["Country", "country"])
segment_col = find_column(["Segment", "segment"])
customer_col = find_column(["Customer_Name", "CustomerName"])
state_col = find_column(["State", "state"])
city_col = find_column(["City", "city"])


print("\n3. IMPORTANT COLUMNS DETECTED")
print("-" * 50)

print("Sales       :", sales_col)
print("Profit      :", profit_col)
print("Quantity    :", quantity_col)
print("Discount    :", discount_col)
print("Category    :", category_col)
print("Sub-Category:", subcategory_col)
print("Region      :", region_col)
print("Country     :", country_col)
print("Segment     :", segment_col)


# -----------------------------
# 7. Basic Statistical Analysis
# -----------------------------
print("\n4. DESCRIPTIVE STATISTICS")
print("-" * 50)

numeric_columns = df.select_dtypes(include=np.number).columns

display(df[numeric_columns].describe())


# -----------------------------
# 8. KPI Calculation
# -----------------------------
print("\n5. KEY PERFORMANCE INDICATORS")
print("-" * 50)

total_sales = df[sales_col].sum() if sales_col else 0
total_profit = df[profit_col].sum() if profit_col else 0
total_quantity = df[quantity_col].sum() if quantity_col else 0

profit_margin = (
    (total_profit / total_sales) * 100
    if total_sales != 0
    else 0
)

total_orders = (
    df["Order_ID"].nunique()
    if "Order_ID" in df.columns
    else len(df)
)

print(f"Total Sales     : ${total_sales:,.2f}")
print(f"Total Profit    : ${total_profit:,.2f}")
print(f"Total Quantity  : {total_quantity:,.0f}")
print(f"Total Orders    : {total_orders:,}")
print(f"Profit Margin   : {profit_margin:.2f}%")


# -----------------------------
# 9. Monthly Sales Trend
# -----------------------------
if order_date_col and sales_col:

    monthly_sales = (
        df.dropna(subset=[order_date_col])
        .set_index(order_date_col)
        .resample("ME")[sales_col]
        .sum()
        .reset_index()
    )

    fig = px.line(
        monthly_sales,
        x=order_date_col,
        y=sales_col,
        markers=True,
        title="Monthly Sales Trend"
    )

    fig.update_layout(
        xaxis_title="Date",
        yaxis_title="Sales"
    )

    fig.show()


# -----------------------------
# 10. Sales by Category
# -----------------------------
if category_col and sales_col:

    category_sales = (
        df.groupby(category_col)[sales_col]
        .sum()
        .sort_values(ascending=False)
        .reset_index()
    )

    fig = px.bar(
        category_sales,
        x=category_col,
        y=sales_col,
        text_auto=".2s",
        title="Sales by Category"
    )

    fig.update_layout(
        xaxis_title="Category",
        yaxis_title="Total Sales"
    )

    fig.show()


# -----------------------------
# 11. Profit by Category
# -----------------------------
if category_col and profit_col:

    category_profit = (
        df.groupby(category_col)[profit_col]
        .sum()
        .sort_values(ascending=False)
        .reset_index()
    )

    fig = px.bar(
        category_profit,
        x=category_col,
        y=profit_col,
        text_auto=".2s",
        title="Profit by Category"
    )

    fig.update_layout(
        xaxis_title="Category",
        yaxis_title="Total Profit"
    )

    fig.show()


# -----------------------------
# 12. Sales by Region
# -----------------------------
if region_col and sales_col:

    region_sales = (
        df.groupby(region_col)[sales_col]
        .sum()
        .sort_values(ascending=False)
        .reset_index()
    )

    fig = px.bar(
        region_sales,
        x=region_col,
        y=sales_col,
        text_auto=".2s",
        title="Sales by Region"
    )

    fig.update_layout(
        xaxis_title="Region",
        yaxis_title="Total Sales"
    )

    fig.show()


# -----------------------------
# 13. Profit by Region
# -----------------------------
if region_col and profit_col:

    region_profit = (
        df.groupby(region_col)[profit_col]
        .sum()
        .sort_values(ascending=False)
        .reset_index()
    )

    fig = px.bar(
        region_profit,
        x=region_col,
        y=profit_col,
        text_auto=".2s",
        title="Profit by Region"
    )

    fig.update_layout(
        xaxis_title="Region",
        yaxis_title="Total Profit"
    )

    fig.show()


# -----------------------------
# 14. Sales by Customer Segment
# -----------------------------
if segment_col and sales_col:

    segment_sales = (
        df.groupby(segment_col)[sales_col]
        .sum()
        .sort_values(ascending=False)
        .reset_index()
    )

    fig = px.pie(
        segment_sales,
        names=segment_col,
        values=sales_col,
        hole=0.4,
        title="Sales Distribution by Customer Segment"
    )

    fig.show()


# -----------------------------
# 15. Top 10 Sub-Categories
# -----------------------------
if subcategory_col and sales_col:

    top_subcategories = (
        df.groupby(subcategory_col)[sales_col]
        .sum()
        .sort_values(ascending=False)
        .head(10)
        .reset_index()
    )

    fig = px.bar(
        top_subcategories,
        x=sales_col,
        y=subcategory_col,
        orientation="h",
        text_auto=".2s",
        title="Top 10 Sub-Categories by Sales"
    )

    fig.show()


# -----------------------------
# 16. Top 10 Countries by Sales
# -----------------------------
if country_col and sales_col:

    top_countries = (
        df.groupby(country_col)[sales_col]
        .sum()
        .sort_values(ascending=False)
        .head(10)
        .reset_index()
    )

    fig = px.bar(
        top_countries,
        x=country_col,
        y=sales_col,
        text_auto=".2s",
        title="Top 10 Countries by Sales"
    )

    fig.show()


# -----------------------------
# 17. Discount vs Profit
# -----------------------------
if discount_col and profit_col:

    fig = px.scatter(
        df,
        x=discount_col,
        y=profit_col,
        size=sales_col if sales_col else None,
        hover_data=[category_col] if category_col else None,
        title="Discount vs Profit"
    )

    fig.update_layout(
        xaxis_title="Discount",
        yaxis_title="Profit"
    )

    fig.show()


# -----------------------------
# 18. Correlation Heatmap
# -----------------------------
print("\n6. CORRELATION ANALYSIS")
print("-" * 50)

numeric_df = df.select_dtypes(include=np.number)

if len(numeric_df.columns) > 1:

    correlation = numeric_df.corr()

    plt.figure(figsize=(10, 7))

    sns.heatmap(
        correlation,
        annot=True,
        fmt=".2f",
        cmap="coolwarm",
        linewidths=0.5
    )

    plt.title("Correlation Heatmap")
    plt.show()


# -----------------------------
# 19. Identify Best/Worst Areas
# -----------------------------
print("\n7. KEY BUSINESS INSIGHTS")
print("-" * 50)

if category_col and sales_col:
    best_category = (
        df.groupby(category_col)[sales_col]
        .sum()
        .idxmax()
    )
    print("Highest Sales Category:", best_category)

if category_col and profit_col:
    best_profit_category = (
        df.groupby(category_col)[profit_col]
        .sum()
        .idxmax()
    )
    print("Highest Profit Category:", best_profit_category)

if region_col and sales_col:
    best_region = (
        df.groupby(region_col)[sales_col]
        .sum()
        .idxmax()
    )
    print("Highest Sales Region:", best_region)

if region_col and profit_col:
    best_profit_region = (
        df.groupby(region_col)[profit_col]
        .sum()
        .idxmax()
    )
    print("Highest Profit Region:", best_profit_region)

if subcategory_col and profit_col:
    best_subcategory = (
        df.groupby(subcategory_col)[profit_col]
        .sum()
        .idxmax()
    )

    worst_subcategory = (
        df.groupby(subcategory_col)[profit_col]
        .sum()
        .idxmin()
    )

    print("Most Profitable Sub-Category:", best_subcategory)
    print("Least Profitable Sub-Category:", worst_subcategory)


print("\n" + "=" * 75)
print("EXPERIMENT 9 ANALYSIS COMPLETED")
print("=" * 75)



# # Questions and Answers

# ### Q1. What is data storytelling? How is it different from data visualization?

# **Answer:**  
# Data storytelling is the process of combining data, visualizations, and narrative to communicate meaningful insights. Data visualization mainly presents information through charts and graphs, while data storytelling explains what the data means and why it matters for decision-making.

# ---

# ### Q2. Why is storytelling important in business analytics and decision-making?

# **Answer:**  
# Storytelling makes complex analytical information easier to understand. It helps business stakeholders identify important trends, understand problems, and make informed decisions based on evidence.

# ---

# ### Q3. What are the essential components of an effective data story?

# **Answer:**  
# The main components are:

# 1. A clear business problem or objective.
# 2. Relevant and reliable data.
# 3. Meaningful visualizations.
# 4. Important insights and findings.
# 5. A logical narrative.
# 6. Actionable recommendations.
# 7. A clear conclusion.

# ---

# ### Q4. How do Key Performance Indicators (KPIs) enhance business reporting?

# **Answer:**  
# KPIs provide measurable indicators of business performance. They help organizations monitor important metrics such as sales, profit, revenue, growth, and customer performance. KPIs make it easier to compare performance and identify areas requiring improvement.

# ---

# ### Q5. Why should visualizations be arranged in a logical sequence while presenting analytical findings?

# **Answer:**  
# A logical sequence helps the audience understand the story step by step. It allows viewers to move from the business problem to the analysis, findings, and recommendations without confusion.

# ---

# ### Q6. What factors should be considered while selecting visualizations for a business presentation?

# **Answer:**  
# The type of data, purpose of the analysis, audience, readability, simplicity, and amount of information should be considered. For example, line charts are suitable for trends, bar charts for comparisons, pie charts for simple proportions, and scatter plots for relationships between variables.

# ---

# ### Q7. Explain how dashboards and storytelling complement each other in Business Intelligence.

# **Answer:**  
# Dashboards provide interactive visualizations and KPIs in one place, while storytelling explains the meaning of those visualizations. Together, they help stakeholders explore data and understand the insights required for decision-making.

# ---

# ### Q8. What challenges may arise while communicating analytical insights to non-technical stakeholders?

# **Answer:**  
# Common challenges include technical terminology, complex charts, excessive information, lack of statistical knowledge, and difficulty understanding analytical concepts. These challenges can be reduced by using simple language, clear visualizations, and focusing on business impact.

# ---

# ### Q9. Give two real-world examples where data storytelling has influenced business or policy decisions.

# **Answer:**  

# 1. **Business:** Retail companies use sales dashboards and customer data to identify purchasing trends and make decisions about inventory, pricing, and marketing.

# 2. **Public Policy:** Government organizations use data visualizations to communicate information about population, healthcare, education, and economic trends to support policy planning.

# ---

# ### Q10. How can effective data storytelling improve strategic planning and organizational performance?

# **Answer:**  
# Effective data storytelling converts complex data into understandable insights. It helps managers identify opportunities, risks, trends, and performance gaps. This supports better strategic planning, efficient resource allocation, faster decision-making, and improved organizational performance.
