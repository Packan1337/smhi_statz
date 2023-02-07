import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# Class that extracts data from .csv file and is able to generate matrix with the data.
class InputData:
    df = None
    x = None
    y = None

    # Generate Pandas Dataframe based of .csv data.
    @classmethod
    def csv_to_df(cls, csv_file):
        cls.df = pd.read_csv(csv_file)

    # Generate matrix based of Pandas dataframe data.
    @classmethod
    def generate_matrix(cls, df_x, df_y):
        cls.x = np.array(df_x)

        for data_type in df_y:
            if isinstance(data_type, str):
                cls.y = np.array(range(1, len(df_y) + 1), dtype=float)

                return print(f"The data is of type 'string', filling array with floats for range(len(df_y)).")
            elif isinstance(data_type, float) or isinstance(data_type, int):
                cls.y = np.array(df_y, dtype=float)

                return print(f"The data is of type: {type(data_type)}, converted values to dtype float.")


# Class that uses Gauss elimination equations to get desired result.
class Gauss:
    rows = None
    cols = None
    ab_norm = None
    k = None
    m = None

    @classmethod
    # Takes x and y from the raw data as parameters to create matrices that we can do calculations on.
    def premade_arrays(cls, x, y):
        a = np.array([x, np.ones(len(x))], dtype=float).T
        b = y.reshape(-1, 1)

        # Create normalization.
        a_trans = a.T
        a = np.dot(a_trans, a)
        b = np.dot(a_trans, b) 

        cls.ab_norm = np.concatenate((a, b), axis=1)
        cls.rows = np.shape(cls.ab_norm)[0]
        cls.cols = np.shape(cls.ab_norm)[1]

    # Solves for k and m using Gauss and Least Square method.
    @classmethod
    def gauss_and_least_square(cls):
        solution_vector = np.zeros(cls.cols - 1)
        for i in range(cls.cols - 1):
            for j in range(i + 1, cls.rows):
                cls.ab_norm[j, :] = -(cls.ab_norm[j, i] / cls.ab_norm[i, i]) * cls.ab_norm[i, :] + cls.ab_norm[j, :]

        # Backwards substitution
        for i in np.arange(cls.rows - 1, -1, -1):
            solution_vector[i] = (cls.ab_norm[i, -1] - np.dot(cls.ab_norm[i, 0:cls.cols - 1], solution_vector)) \
                                 / cls.ab_norm[i, i]

        cls.k = np.round(solution_vector[0], 3)
        cls.m = np.round(solution_vector[1], 3)


# Generates a graph GUI using matplotlib.
def generate_graph(x_raw, y_raw, k, m):
    plt.title("Medeltemperaturer 2022")
    plt.xlabel("Temperature")
    plt.ylabel("Month")
    plt.scatter(x_raw, y_raw, label="Medeltemperatur")
    plt.plot(x_raw, k * x_raw + m, "-r",
             label=f"Best line fit\nwhere;\nK = {k}\nm = {m}")
    plt.legend()
    return plt.show()