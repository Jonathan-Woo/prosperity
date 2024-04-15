import os
import numpy as np
import pandas as pd

with open(os.path.join(os.path.dirname(__file__), "starfruit.csv")) as f:
    data = pd.read_csv(f, sep=";")
data = data[data["product"] == "STARFRUIT"]
data["mid_price"] = data["bid_price_1"] + data["ask_price_1"] / 2

############################
# CHANGE THIS
time_window = 10
############################

segments = []
for i in range(len(data) - (time_window - 1)):
    segments.append(list(data["mid_price"][i : i + time_window]))

segments = np.array(segments)
X, y = segments[:, :-1], segments[:, -1]
X = np.hstack([X, np.ones((X.shape[0], 1))])
b = np.linalg.inv(X.T @ X) @ X.T @ y
print(f"Coefficients: {b}")
