import pandas as pd
import numpy as np

import numpy as np
import pandas as pd

def read_excel_file(fileName, sheetName):
    """
    Reads an Excel file and returns the generator, load, and transmission data.
    Returns in PD dataframes
    """
    df = pd.read_excel(fileName, sheet_name=sheetName, header=2)
    generator_data = df.loc[:, 'Generator':'Slack bus']
    load_data = df.loc[:, 'Load unit':'Location']
    transmission_data = df.loc[:, 'Line':'Susceptance [p.u]']
    return generator_data, load_data, transmission_data

def create_y_bus_matrix(num_buses, susceptance_df):
    """
    Create a Y-bus matrix given the number of buses and a DataFrame of susceptance values.
    Returns an np.ndarray
    """
    # Initialize the Y-bus matrix with zeros
    Y_bus = np.zeros((num_buses, num_buses), dtype=complex)

    # Fill the Y-bus matrix using the DataFrame
    for _, row in susceptance_df.iterrows():
        # Extract bus indices from the 'Line' column
        bus_i_str, bus_j_str = row['Line'].split('-')
        bus_i = int(bus_i_str.split()[-1]) - 1  # Convert to 0-based index
        bus_j = int(bus_j_str) - 1  # Convert to 0-based index
        b_ij = row['Susceptance [p.u]'] * 1j  # Convert susceptance to complex form

        if bus_i != bus_j:
            Y_bus[bus_i, bus_j] -= b_ij
            Y_bus[bus_j, bus_i] -= b_ij
            Y_bus[bus_i, bus_i] += b_ij
            Y_bus[bus_j, bus_j] += b_ij

    return Y_bus

filename = 'Problem_2_data.xlsx'
sheetname = 'Problem 2.2 - Base case'

generator, load, transmission = read_excel_file(filename, sheetname)
print("Transmission Data:")
print(transmission)

num_buses = 3  # Set the number of buses in your system
Y_bus = create_y_bus_matrix(num_buses, transmission)

# Display the Y-bus matrix
print("Y-bus Matrix:")
print(Y_bus)

