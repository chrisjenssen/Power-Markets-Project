import pandas as pd
import numpy as np

def readexcelfile(fileName, sheetName):
    df = pd.read_excel(fileName,sheet_name=sheetName, header=2)
    generator_data = df.loc[:,'Generator':'Slack bus']
    load_data = df.loc[:,'Load unit':'Location']
    transmission_data = df.loc[:,'Line':'Susceptance [p.u]']
    return(generator_data, load_data, transmission_data)

filename = 'Problem_2_data.xlsx'
sheetname = 'Problem 2.2 - Base case'

generator, load, transmission = readexcelfile(filename,sheetname)
print(transmission)


def create_y_bus_matrix(num_buses, susceptance):

    Y_bus = np.zeros((num_buses, num_buses), dtype=complex)

    # Fill the Y-bus matrix
    for (i, j), b_ij in susceptance:
        if i != j:
            Y_bus[i, j] -= b_ij
            Y_bus[j, i] -= b_ij
            Y_bus[i, i] += b_ij
            Y_bus[j, j] += b_ij

    return Y_bus

Ybus = create_y_bus_matrix(len(transmission['Line']), transmission['Susceptance [p.u]'])
print(Ybus)
