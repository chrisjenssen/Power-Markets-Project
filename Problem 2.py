import numpy as np
import sys
import pandas as pd
#import matplotlib.pyplot as plt

from funskjoner import read_excel_file
from OPF_DC import OPF_DC
from Problem_2_4 import OPF_DC_2_4
from Problem_2_4_WTP import OPF_DC_2_4_WTP, read_excel_file_2_4_WTP
from Problem_2_5 import OPF_DC_2_5, read_excel_file_2_5


filename = 'Problem_2_data.xlsx'

sheet_1 = 'Problem 2.2 - Base case'
sheet_2 = 'Problem 2.3 - Generators'
sheet_3 = 'Problem 2.4 - Loads'
sheet_4 = 'Problem 2.5 - Environmental'

generator, load, transmission = read_excel_file(filename, sheet_2)

# OPF_DC(generator, load, transmission, sheetname, WTP = 0, CO2_reduction)
OPF_DC(generator, load, transmission,)


