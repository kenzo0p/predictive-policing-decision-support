import pandas as pd
import matplotlib.pyplot as plt

# Load processed data
df = pd.read_csv("data/processed/clean_crime_data.csv")

print(df.head())
print(df.shape)


# Choose one state to visualize
sample_state = "Andhra Pradesh"   # change if needed

# Filter data for the selected state
state_data = df[df["state"] == sample_state]

# Sort by total_crimes to make the chart readable
state_data = state_data.sort_values("total_crimes", ascending=False)

# Plot a bar chart of crimes by district
plt.figure(figsize=(12, 6))  # Make figure larger for district names
plt.bar(state_data["district"], state_data["total_crimes"], color='skyblue')
plt.xlabel("District")
plt.ylabel("Total IPC Crimes")
plt.title(f"District-wise Crime Distribution in {sample_state} (2013)")
plt.xticks(rotation=90)  # Rotate x-axis labels to avoid overlap
plt.tight_layout()       # Adjust layout to prevent clipping
plt.show()

print(f"Plotting done for {sample_state}")
print("Available states:", df["state"].unique())
