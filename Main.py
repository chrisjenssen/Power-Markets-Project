from funskjoner import read_excel_file
from Problem_2_2_to_2_4 import OPF_DC_2_2_to_2_4
from Problem_2_4_WTP import OPF_DC_2_4_WTP, read_excel_file_2_4_WTP
from Problem_2_5 import OPF_DC_2_5, read_excel_file_2_5

def Task_2_2():
    filename = 'Problem_2_data.xlsx'
    sheet = 'Problem 2.2 - Base case'
    generator, load, transmission = read_excel_file(filename, sheet)

    OPF_DC_2_2_to_2_4(generator, load, transmission)

def Task_2_3():
    filename = 'Problem_2_data.xlsx'
    sheet = 'Problem 2.3 - Generators'
    generator, load, transmission = read_excel_file(filename, sheet)

    OPF_DC_2_2_to_2_4(generator, load, transmission)

def Task_2_4_without_WTP():
    filename = 'Problem_2_data.xlsx'
    sheet = 'Problem 2.4 - Loads'
    generator, load, transmission = read_excel_file(filename, sheet)

    OPF_DC_2_2_to_2_4(generator, load, transmission)

def Task_2_4_with_WTP():
    filename = 'Problem_2_data.xlsx'
    sheet = 'Problem 2.4 - Loads'
    generator, load, transmission = read_excel_file_2_4_WTP(filename, sheet)

    OPF_DC_2_4_WTP(generator, load, transmission)

def Task_2_5_CES():
    filename = 'Problem_2_data.xlsx'
    sheet = 'Problem 2.5 - Environmental'
    generator, load, transmission = read_excel_file_2_5(filename, sheet)

    OPF_DC_2_5(generator, load, transmission, 1)

def Task_2_5_cap_and_trade():
    filename = 'Problem_2_data.xlsx'
    sheet = 'Problem 2.5 - Environmental'
    generator, load, transmission = read_excel_file_2_5(filename, sheet)

    OPF_DC_2_5(generator, load, transmission)

"""
===========================================================
To run the tasks uncomment the task you want to test bellow 
===========================================================
"""

Task_2_2()
#Task_2_3()
#Task_2_4_without_WTP()
#Task_2_4_with_WTP()
#Task_2_5_CES()
#Task_2_5_cap_and_trade()