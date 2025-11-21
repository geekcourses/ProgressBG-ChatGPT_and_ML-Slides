import pandas as pd

# Create a DataFrame
data = {
    "Name": ["Alice", "Bob", "Charlie"],
    "Age": [25, 30, 35],
    "City": ["New York", "London", "Paris"],
}
df = pd.DataFrame(data)

# View first few rows
print(df.head())

# Calculate mean age
print(f"Mean Age: {df['Age'].mean()}")
