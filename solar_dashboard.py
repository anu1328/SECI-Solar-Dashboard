import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
import numpy as np

# Load CSV
df = pd.read_csv("solar_output.csv")

# 📊 Bar chart of daily output
daily_output = df.groupby("Date")["SolarOutput_kWh"].sum()
daily_output.plot(kind="bar", title="Daily Solar Energy Output (kWh)", ylabel="kWh", color='orange')
plt.savefig("daily_output.png")
plt.show()

# 📈 Line graph of hourly output
one_day = df[df["Date"] == "2025-07-06"]
plt.plot(one_day["Hour"], one_day["SolarOutput_kWh"], marker='o', linestyle='-', color='green')
plt.title("Hourly Solar Output – 2025-07-06")
plt.xlabel("Hour")
plt.ylabel("Output (kWh)")
plt.grid(True)
plt.savefig("hourly_output.png")
plt.show()

# 🧠 ML Prediction: Predict output for hour 19
X = one_day[["Hour"]]  # Features
y = one_day["SolarOutput_kWh"]  # Target

model = LinearRegression()
model.fit(X, y)

predicted_output = model.predict(pd.DataFrame({'Hour': [19]}))  # Predict for hour 19

print(f"\n🔮 Predicted Solar Output for Hour 19: {predicted_output[0]:.2f} kWh")
