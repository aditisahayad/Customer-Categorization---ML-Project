import pickle
import pandas as pd

model = pickle.load(open("models/model.pkl", "rb"))
labels = {0: "Low", 1: "Medium", 2: "High"}

tests = [
    (25, 10000, 50, 3, 0),
    (30, 30000, 200, 2, 2),
    (35, 40000, 500, 1, 2),
    (45, 55000, 1200, 1, 3),
    (55, 90000, 2000, 0, 4),
    (65, 120000, 3000, 0, 4),
    (22, 8000, 20, 0, 0),
    (28, 20000, 1500, 0, 2),
    (70, 150000, 5000, 0, 4),
    (35, 80000, 100, 0, 2),
    (40, 60000, 2000, 0, 3),
    (50, 70000, 1500, 0, 4),
    (30, 25000, 50, 3, 0),
]

results = []
for t in tests:
    df = pd.DataFrame([{"Age": t[0], "Income": t[1], "Total_Spending": t[2], "Children": t[3], "Education": t[4]}])
    c = int(model.predict(df)[0])
    results.append(f"Age={t[0]} Inc={t[1]} Spend={t[2]} Kids={t[3]} Edu={t[4]} => Cluster {c} ({labels[c]})")

with open("results.txt", "w", encoding="utf-8") as f:
    f.write("\n".join(results))
