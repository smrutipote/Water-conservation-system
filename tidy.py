import pandas as pd

# Step 1: Load the dataset from the GitHub raw URL
url = "https://raw.githubusercontent.com/Sushma7870-git/Tidy_Tuesday/refs/heads/main/du-bois-challenge-2022-graph-3-data.csv"
df = pd.read_csv(url)

# Step 2: Check the column names and preview the data
print(df.columns)  # Check the column names to ensure everything is correct
print(df.head())  # Preview the data

# Step 3: Clean 'black_population' column and calculate the midpoint
def extract_midpoint(pop_range):
    # Handle cases like "UNDER" or "OVER"
    if 'UNDER' in pop_range:
        return 10000  # Assign a default value like 10,000 for "UNDER" cases
    if 'NEGROES AND OVER' in pop_range:
        return 750000  # You can also assign a placeholder value here (e.g., 1000000)

    # Handle cases like "NEGROES AND OVER" where we just want the number (e.g., 750,000)
    if 'NEGROES AND OVER' in pop_range:
        # Extract the number before "NEGROES AND OVER"
        parts = pop_range.split('NEGROES AND OVER')[0].strip().replace(',', '')
        
        return 750000
        # except ValueError:
        #     return None  # If there's an error converting, return None

    # Remove commas and split the range for other cases
    try:
        parts = pop_range.replace(',', '').split('-')
        if len(parts) != 2:
            return None  # Handle unexpected format
        low = int(parts[0].strip())
        high = int(parts[1].strip())
        return (low + high) / 2
    except ValueError:
        return None  # If there's any error in conversion, return None

# Step 4: Create a new 'midpoint' column (use 'black_population' here instead of 'population')
df['midpoint'] = df['population'].apply(extract_midpoint)

# Step 5: Optional — sort by midpoint or preview cleaned data
df_sorted = df.sort_values(by='midpoint', ascending=False)

# Step 6: Preview the processed data
print(df_sorted[['state', 'population', 'midpoint']].head())

# Optional — export to CSV for use in Vega-Lite or visualization
df_sorted.to_csv("processed_black_population.csv", index=False)
