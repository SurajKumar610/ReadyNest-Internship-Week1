import os
import numpy as np
import pandas as pd

def main():
    os.makedirs(os.path.join("data", "raw"), exist_ok=True)
    os.makedirs(os.path.join("data", "cleaned"), exist_ok=True)
    os.makedirs("notebooks", exist_ok=True)
    os.makedirs("models", exist_ok=True)
    os.makedirs(os.path.join("reports", "pdf"), exist_ok=True)
    os.makedirs(os.path.join("reports", "excel"), exist_ok=True)
    os.makedirs(os.path.join("reports", "ppt"), exist_ok=True)
    
    np.random.seed(42)
    rows = 200
    dates = pd.date_range(start='2024-01-01', periods=rows, freq='D')
    categories = ['Electronics', 'Clothing', 'Home Decor', 'Office Supplies', 'Fitness']
    regions = ['North', 'East', 'South', 'West']
    states = ['NY', 'CA', 'TX', 'FL', 'IL', 'WA', 'MA', 'CO']
    
    sales = np.random.exponential(scale=200, size=rows) + np.random.normal(loc=50, scale=10, size=rows)
    sales = np.clip(sales, 10, None)
    profit = sales * np.random.uniform(0.1, 0.4, size=rows)
    
    # Random missing values in Sales
    sales_with_nan = sales.copy()
    sales_with_nan[np.random.choice(rows, size=10, replace=False)] = np.nan
    
    df = pd.DataFrame({
        'Date': dates,
        'Sales': sales_with_nan,
        'Profit': profit,
        'Category': np.random.choice(categories, size=rows),
        'Region': np.random.choice(regions, size=rows),
        'State': np.random.choice(states, size=rows),
        'Customer_Age': np.random.randint(18, 70, size=rows),
        'Churned': np.random.choice([0, 1], size=rows, p=[0.8, 0.2]),
        'AB_Group': np.random.choice(['A', 'B'], size=rows),
        'Converted': np.random.choice([0, 1], size=rows, p=[0.88, 0.12])
    })
    
    out_path = os.path.join("data", "raw", "sales.csv")
    df.to_csv(out_path, index=False)
    print(f"Generated sample dataset: {out_path}")

if __name__ == "__main__":
    main()
