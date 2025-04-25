import numpy as np
import sys
import pandas as pd
#import matplotlib.pyplot as plt

from funskjoner import read_excel_file, create_y_matrix, generate_Y_DC
from OPF_DC import OPF_DC


filename = 'Problem_2_data.xlsx'

sheet_1 = 'Problem 2.2 - Base case'
sheet_2 = 'Problem 2.3 - Generators'
sheet_3 = 'Problem 2.4 - Loads'
sheet_4 = 'Problem 2.5 - Environmental'

generator, load, transmission = read_excel_file(filename, sheet_1)
num_buses = 3  # Set the number of buses in your system, have to do this manually due to the set up in excel file
Y_bus = create_y_matrix(num_buses, transmission)
Y_DC = generate_Y_DC(generator, transmission)

OPF_DC(generator, load, transmission)


