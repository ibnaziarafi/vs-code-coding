"""
One tiny neural network from scratch.

It learns to answer:
    "Will a student pass?"

Inputs:
    study hours and practice questions

Output:
    a number close to 0 (fail) or 1 (pass)
"""

import math


def sigmoid(number):
    """Turn any number into a value between 0 and 1."""
    return 1 / (1 + math.exp(-number))


# Each row is: [study hours, practice questions]
students = [
    [1, 1],
    [2, 1],
    [3, 2],
    [4, 3],
    [5, 4],
]

# 0 means fail and 1 means pass.
answers = [0, 0, 1, 1, 1]


# The network starts with simple guesses for its weights and bias.
weight_hours = 0.0
weight_practice = 0.0
bias = 0.0
learning_rate = 0.1


for _ in range(10000):
    for inputs, correct_answer in zip(students, answers):
        hours, practice_questions = inputs

        # The neuron combines the inputs and makes a guess.
        score = (
            hours * weight_hours
            + practice_questions * weight_practice
            + bias
        )
        guess = sigmoid(score)

        # Find how far the guess is from the correct answer.
        error = guess - correct_answer

        # Improve the weights and bias a little.
        weight_hours -= learning_rate * error * hours
        weight_practice -= learning_rate * error * practice_questions
        bias -= learning_rate * error


def predict(hours, practice_questions):
    """Use the trained network to predict pass or fail."""
    score = (
        hours * weight_hours
        + practice_questions * weight_practice
        + bias
    )
    return sigmoid(score)


probability = predict(2, 1)
print(f"Chance of passing: {probability:.0%}")
print("Prediction:", "Pass" if probability >= 0.5 else "Fail")
