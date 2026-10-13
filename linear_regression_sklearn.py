"""Short linear regression example using scikit-learn."""

from sklearn.linear_model import LinearRegression


sizes = [[50], [60], [70], [80], [90]]  # House sizes in square metres
prices = [150, 180, 210, 240, 270]  # Prices in thousands of dollars

model = LinearRegression().fit(sizes, prices)
prediction = model.predict([[100]])[0]

print(f"Slope: {model.coef_[0]:.2f}")
print(f"Starting price: {model.intercept_:.2f}")
print(f"Estimated price for a 100 m² house: ${prediction:.0f},000")
