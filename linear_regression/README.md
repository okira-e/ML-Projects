# Student Performance Prediction

This project is a Python script that uses linear regression to predict student performance based on a dataset of student information. The script reads the data from a CSV file called "student-mat.csv", performs data preprocessing, trains a linear regression model, makes predictions on a test set, and evaluates the model's performance.

## Explanation

1. The script reads the data from the "student-mat.csv" file using the `read_csv` function from the pandas library and stores it in the `data` variable.
2. The numpy library's `set_printoptions` function is used to display the full array when printing.
3. The script separates the features (x1, x2, x3) and the target variable (y) from the dataset using the `iloc` method.
4. The features are concatenated into a single DataFrame called `x` using the `concat` function from pandas.
5. The `ColumnTransformer` class from scikit-learn is used to apply one-hot encoding to categorical features in `x`.
6. The transformed features are converted to a numpy array using `fit_transform` from the `ColumnTransformer` object.
7. The data is split into training and test sets using the `train_test_split` function from scikit-learn.
8. A linear regression model is created using the `LinearRegression` class from scikit-learn.
9. The model is trained on the training set using the `fit` method.
10. Predictions are made on the test set using the `predict` method.
11. The numpy `set_printoptions` function is used to set the precision of the printed arrays to two decimal places.
12. The predicted results, actual results, and average error margin are printed.
13. A plot is generated to visualize the predicted values against the test set using the `plot` function from matplotlib.


## Prerequisites

- Pythn 3.x
- Pandas
- Numpy
- Matplotlib
- Scikit-learn

## Installation

1. Clone this repository to your local machine using:
2. Install the required packages using:
```
pip install pandas numpy scikit-learn matplotlib
```

## Usage

```shell
python main.py
```

---
Feel free to modify the script according to your needs or integrate it into a larger project.

