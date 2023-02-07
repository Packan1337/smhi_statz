from gauss import *

InputData.csv_to_df("smhi-data-int.csv")
InputData.generate_matrix(InputData.df.temp, InputData.df.month)
Gauss.premade_arrays(InputData.x, InputData.y)
Gauss.gauss_and_least_square()
generate_graph(InputData.x, InputData.y, Gauss.k, Gauss.m) 