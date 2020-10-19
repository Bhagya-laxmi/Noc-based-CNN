import csv
import pandas as pd
import numpy as np

csv_file = input('Location of the input file: ')
txt_file = input('Location of the output text file: ')
position = int(input("Enter the position to read from: "))

data = pd.read_csv(csv_file, skiprows = position-1, nrows=1)

label = data.pop('Labels')
print("Label is: ",label.values)

np.savetxt(txt_file, data.values, fmt='%d')
