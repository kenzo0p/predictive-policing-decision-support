import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Load cleaned data
df = pd.read_csv("data/processed/clean_crime_data.csv")

# Normalize state names
df["state"] = df["state"].str.strip().str.upper()

print("=" * 60)
print("BIAS ANALYSIS FOR SINGLE-YEAR DATA (2013)")
print("=" * 60)

# ============================================================
# Analysis 1: State-wise Crime Distribution (Identify Outliers)
# ============================================================

# Aggregate to state level
state_df = df.groupby("state")["total_crimes"].sum().reset_index()
state_df = state_df.sort_values("total_crimes", ascending=False)

# Calculate statistics for bias detection
mean_crimes = state_df["total_crimes"].mean()
std_crimes = state_df["total_crimes"].std()

# Flag potential outliers (states with unusually high or low crime counts)
state_df["z_score"] = (state_df["total_crimes"] - mean_crimes) / std_crimes
outliers = state_df[abs(state_df["z_score"]) > 2]

print("\n📊 State-wise Crime Statistics:")
print(f"   Mean crimes per state: {mean_crimes:,.0f}")
print(f"   Std deviation: {std_crimes:,.0f}")

print("\n⚠️ Potential Outlier States (Z-score > 2):")
if outliers.empty:
    print("   No extreme outliers detected.")
else:
    print(outliers[["state", "total_crimes", "z_score"]].to_string(index=False))

# Plot 1: Top 15 States by Crime Count
plt.figure(figsize=(14, 6))
top_states = state_df.head(15)
bars = plt.bar(top_states["state"], top_states["total_crimes"], color='steelblue')

# Highlight outliers in red
for i, (state, crimes, z) in enumerate(zip(top_states["state"], top_states["total_crimes"], top_states["z_score"])):
    if abs(z) > 2:
        bars[i].set_color('crimson')

plt.xlabel("State/UT")
plt.ylabel("Total IPC Crimes")
plt.title("Top 15 States by Total Crimes (2013) - Outliers Highlighted in Red")
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.show()

# ============================================================
# Analysis 2: District-level Variance within States
# ============================================================

# Calculate coefficient of variation for each state (measure of reporting consistency)
state_variance = df.groupby("state")["total_crimes"].agg(['mean', 'std', 'count'])
state_variance["cv"] = (state_variance["std"] / state_variance["mean"]) * 100  # Coefficient of Variation
state_variance = state_variance.sort_values("cv", ascending=False)

print("\n📈 States with Highest District-level Variance (Potential Reporting Inconsistency):")
print("   High CV% suggests uneven crime reporting across districts")
print(state_variance.head(10).to_string())

# Plot 2: Coefficient of Variation by State
plt.figure(figsize=(14, 6))
cv_data = state_variance.head(20).reset_index()
plt.barh(cv_data["state"], cv_data["cv"], color='darkorange')
plt.xlabel("Coefficient of Variation (%)")
plt.ylabel("State/UT")
plt.title("District-level Reporting Variance by State (Higher = More Inconsistent)")
plt.tight_layout()
plt.show()

# ============================================================
# Analysis 3: Compare a specific state's districts
# ============================================================

sample_state = "BIHAR"
state_data = df[df["state"] == sample_state].copy()
state_data = state_data.sort_values("total_crimes", ascending=False)

# Calculate district-level z-scores within the state
state_mean = state_data["total_crimes"].mean()
state_std = state_data["total_crimes"].std()
state_data["z_score"] = (state_data["total_crimes"] - state_mean) / state_std

print(f"\n📍 District Analysis for {sample_state}:")
print(f"   Mean crimes per district: {state_mean:,.0f}")
print(f"   Districts with unusual values (|z| > 1.5):")
unusual_districts = state_data[abs(state_data["z_score"]) > 1.5]
print(unusual_districts[["district", "total_crimes", "z_score"]].to_string(index=False))

# Plot 3: District-wise distribution for selected state
plt.figure(figsize=(14, 6))
colors = ['crimson' if abs(z) > 1.5 else 'steelblue' for z in state_data["z_score"]]
plt.bar(state_data["district"], state_data["total_crimes"], color=colors)
plt.xlabel("District")
plt.ylabel("Total IPC Crimes")
plt.title(f"District-wise Crime Distribution in {sample_state} (2013) - Outliers in Red")
plt.xticks(rotation=90)
plt.tight_layout()
plt.show()

print("\n✅ Bias analysis complete!")
print("Note: With only 2013 data, year-over-year trend analysis is not possible.")
print("Consider adding multi-year data for temporal bias detection.")
