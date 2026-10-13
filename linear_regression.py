"""
Linear regression written from scratch.

The program learns this simple rule:

    house price = slope * house size + starting price

It uses only normal Python. No NumPy, pandas, or scikit-learn is needed.
"""


class LinearRegression:
    """A very small linear regression model."""

    def __init__(self, learning_rate=0.01, steps=10000):
        # The learning rate controls how big each learning step is.
        self.learning_rate = learning_rate
        self.steps = steps

        # These are the two numbers in: y = slope * x + starting_price
        self.slope = 0.0
        self.starting_price = 0.0

    def fit(self, x_values, y_values):
        """Learn from examples of x and y."""
        if len(x_values) == 0 or len(x_values) != len(y_values):
            raise ValueError("x_values and y_values must have the same non-zero length")

        number_of_examples = len(x_values)

        for step in range(self.steps):
            slope_change = 0.0
            starting_price_change = 0.0

            slope_change_list = []
            starting_price_change_list = []

            # Check every example and see how wrong our guesses are.
            for x, correct_y in zip(x_values, y_values):
                guessed_y = self.predict_one(x)
                mistake = guessed_y - correct_y

                # These formulas tell us how to improve the two numbers.
                slope_change += mistake * x
                starting_price_change += mistake
                slope_change_list.append(slope_change)
                starting_price_change_list.append(starting_price_change)

            # Print the first round so we can see the values without flooding
            # the screen with the same information from every learning round.
            if step < 2:
                print("Slope changes:", slope_change_list)
                print("Starting price changes:", starting_price_change_list)
                print("")

            # Average the changes so the numbers do not jump too far.
            slope_change = slope_change / number_of_examples
            starting_price_change = starting_price_change / number_of_examples

            self.slope -= self.learning_rate * slope_change
            self.starting_price -= self.learning_rate * starting_price_change

    def predict_one(self, x):
        """Guess one y value from one x value."""
        return self.slope * x + self.starting_price

    def predict(self, x_values):
        """Guess y values for a list of x values."""
        return [self.predict_one(x) for x in x_values]


def main():
    # x = house size in square metres
    house_sizes = [50, 60, 70, 80, 90]

    # y = house price in thousands of dollars
    house_prices = [150, 180, 210, 240, 270]

    model = LinearRegression(learning_rate=0.00001, steps=10000)
    model.fit(house_sizes, house_prices)

    print("The computer learned:")
    print(f"  slope = {model.slope:.2f}")
    print(f"  starting price = {model.starting_price:.2f}")

    new_house_size = 100
    guessed_price = model.predict_one(new_house_size)

    print()
    print(f"A {new_house_size} square metre house may cost")
    print(f"about ${guessed_price:.0f},000.")


if __name__ == "__main__":
    main()
