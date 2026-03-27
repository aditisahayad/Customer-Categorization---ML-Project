"""Quick script to test the model with various inputs and find all 3 clusters."""
import pickle
import pandas as pd

# Load model
with open("models/model.pkl", "rb") as f:
    model = pickle.load(f)

# Test cases with wide range of values
test_cases = [
    {"Age": 25, "Income": 10000, "Total_Spending": 50, "Children": 3, "Education": 0},
    {"Age": 25, "Income": 15000, "Total_Spending": 100, "Children": 2, "Education": 1},
    {"Age": 30, "Income": 30000, "Total_Spending": 200, "Children": 2, "Education": 2},
    {"Age": 35, "Income": 40000, "Total_Spending": 500, "Children": 1, "Education": 2},
    {"Age": 40, "Income": 50000, "Total_Spending": 800, "Children": 1, "Education": 3},
    {"Age": 45, "Income": 55000, "Total_Spending": 1200, "Children": 1, "Education": 3},
    {"Age": 50, "Income": 70000, "Total_Spending": 1500, "Children": 0, "Education": 4},
    {"Age": 55, "Income": 90000, "Total_Spending": 2000, "Children": 0, "Education": 4},
    {"Age": 60, "Income": 100000, "Total_Spending": 2500, "Children": 0, "Education": 4},
    {"Age": 35, "Income": 80000, "Total_Spending": 100, "Children": 0, "Education": 2},
    {"Age": 28, "Income": 20000, "Total_Spending": 1500, "Children": 0, "Education": 2},
    {"Age": 65, "Income": 120000, "Total_Spending": 3000, "Children": 0, "Education": 4},
    {"Age": 30, "Income": 25000, "Total_Spending": 50, "Children": 3, "Education": 0},
    {"Age": 45, "Income": 60000, "Total_Spending": 2000, "Children": 0, "Education": 3},
    {"Age": 22, "Income": 8000, "Total_Spending": 20, "Children": 0, "Education": 0},
    {"Age": 70, "Income": 150000, "Total_Spending": 5000, "Children": 0, "Education": 4},
]

LABELS = {0: "Low Value", 1: "Medium Value", 2: "High Value"}

print(f"{'Age':>4} {'Income':>8} {'Spending':>9} {'Kids':>5} {'Edu':>4} | {'Cluster':>8} {'Category'}")
print("-" * 75)
for tc in test_cases:
    df = pd.DataFrame([tc])
    cluster = int(model.predict(df)[0])
    label = LABELS.get(cluster, "Unknown")
    print(f"{tc['Age']:>4} {tc['Income']:>8} {tc['Total_Spending']:>9} {tc['Children']:>5} {tc['Education']:>4} | {cluster:>8} {label}")
