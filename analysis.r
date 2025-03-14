import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load Data (Assuming a CSV file with Transaction Data)
df = pd.read_csv('customer_purchases.csv')

# Preview Data
df.head()

# Data Cleaning (Convert Dates, Handle Missing Values)
df['Date'] = pd.to_datetime(df['Date'])
df.dropna(inplace=True)

# RFM Analysis
rfm = df.groupby('CustomerID').agg({
    'Date': lambda x: (df['Date'].max() - x.max()).days,  # Recency
    'TransactionID': 'count',  # Frequency
    'TotalAmount': 'sum'  # Monetary
})
rfm.columns = ['Recency', 'Frequency', 'Monetary']
print("\nRFM Analysis:\n", rfm.describe())

# Top-Selling Products
product_sales = df.groupby('Product')['TotalAmount'].sum().sort_values(ascending=False)
print("\nTop-Selling Products:\n", product_sales.head())

# Sales Trend Over Time
sales_trend = df.groupby(df['Date'].dt.to_period('M'))['TotalAmount'].sum()
plt.figure(figsize=(10, 5))
sales_trend.plot(kind='line', marker='o', color='b')
plt.title("Monthly Sales Trend")
plt.xlabel("Month")
plt.ylabel("Total Sales")
plt.grid(True)
plt.show()

# Customer Churn Detection
customer_last_purchase = df.groupby('CustomerID')['Date'].max()
churn_threshold = df['Date'].max() - pd.Timedelta(days=90)  # 90-day threshold
churned_customers = customer_last_purchase[customer_last_purchase < churn_threshold]
print(f"\nChurned Customers (Not Purchased in last 90 days): {len(churned_customers)}")

# Visualizing Customer Purchase Frequency
plt.figure(figsize=(10, 5))
sns.histplot(rfm['Frequency'], bins=20, kde=True, color='g')
plt.title("Customer Purchase Frequency Distribution")
plt.xlabel("Number of Transactions")
plt.ylabel("Count of Customers")
plt.show()

print("Analysis Completed ✅")
