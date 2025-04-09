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
    for i in susceptance:
        # Extract bus indices from the 'Line' column
        bus_i, bus_j = map(int, row['Line'].split('-'))
        bus_i -= 1  # Convert to 0-based index
        bus_j -= 1  # Convert to 0-based index
        b_ij = row['Susceptance [p.u]'] * 1j  # Convert susceptance to complex form

        if bus_i != bus_j:
            Y_bus[bus_i, bus_j] -= b_ij
            Y_bus[bus_j, bus_i] -= b_ij
            Y_bus[bus_i, bus_i] += b_ij
            Y_bus[bus_j, bus_j] += b_ij

    return Y_bus

Ybus = create_y_bus_matrix(len(transmission['Line']), transmission['Susceptance [p.u]'])
print(Ybus)

"""
    for (i, j), b_ij in susceptance:
        if i != j:
            Y_bus[i, j] -= b_ij
            Y_bus[j, i] -= b_ij
            Y_bus[i, i] += b_ij
            Y_bus[j, j] += b_ij
"""