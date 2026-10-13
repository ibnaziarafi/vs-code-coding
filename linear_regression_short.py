"""Short linear regression example using NumPy."""

import numpy as np


sizes = [50, 60, 70, 80, 90]  # House sizes in square metres
prices = [150, 180, 210, 240, 270]  # Prices in thousands of dollars

slope, starting_price = np.polyfit(sizes, prices, 1)
prediction = slope * 100 + starting_price

print(f"Slope: {slope:.2f}")
print(f"Starting price: {starting_price:.2f}")
print(f"Estimated price for a 100 m² house: ${prediction:.0f},000")
