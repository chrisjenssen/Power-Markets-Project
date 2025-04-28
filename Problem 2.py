import numpy as np
import sys
import pandas as pd
#import matplotlib.pyplot as plt

from funskjoner import read_excel_file
from OPF_DC import OPF_DC


filename = 'Problem_2_data.xlsx'

sheet_1 = 'Problem 2.2 - Base case'
sheet_2 = 'Problem 2.3 - Generators'
sheet_3 = 'Problem 2.4 - Loads'
sheet_4 = 'Problem 2.5 - Environmental'

generator, load, transmission = read_excel_file(filename, sheet_1)

OPF_DC(generator, load, transmission)


