import pandas as pd
import numpy as np

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

def create_y_matrix(num_buses, susceptance_df):
    """
    Create a Y-bus matrix given the number of buses and a DataFrame of susceptance values.
    Returns an np.ndarray
    """
    # Initialize the Y-bus matrix with zeros
    Y_bus = np.zeros((num_buses, num_buses), dtype=complex)

    # Fill the Y-bus matrix using the DataFrame
    for _, row in susceptance_df.iterrows():
        # Extract bus indices from the 'Line' column
        bus_i_str, bus_j_str = row['Line'].split('-') #Since the excel file says Line 1-2, we split it to get the bus numbers
        bus_i = int(bus_i_str.split()[-1]) - 1  # Convert to 0-based index
        bus_j = int(bus_j_str) - 1  # Convert to 0-based index
        b_ij = row['Susceptance [p.u]'] * 1j  # Convert susceptance to complex form

        if bus_i != bus_j:
            Y_bus[bus_i, bus_j] -= b_ij
            Y_bus[bus_j, bus_i] -= b_ij
            Y_bus[bus_i, bus_i] += b_ij
            Y_bus[bus_j, bus_j] += b_ij

    return Y_bus

def generate_Y_DC(generator_df, Ybus):
    """
    Generate the Y-bus matrix for DC power flow by removing the slack bus.
    Returns np.ndarray
    """
    # Find the index of the slack bus
    slack_bus = generator_df[generator_df['Slack bus'] == True]
    if not slack_bus.empty:
        slack_index = slack_bus.index[0]

        # Remove slack row and column in Y_bus
        Y_reduced = np.delete(Ybus, slack_index, axis=0)  # Remove row
        Y_reduced = np.delete(Y_reduced, slack_index, axis=1)  # Remove column

        # Neglect all real parts of elements and remove symbol 'j'
        B_fnutt = Y_reduced.imag
        Ydc = B_fnutt * (-1)
        return Ydc
    else:
        print("Slack bus not found in generator data.")
        return None


#Run to test the functions:
"""
filename = 'Problem_2_data.xlsx'
sheetname = 'Problem 2.2 - Base case'

generator, load, transmission = read_excel_file(filename, sheetname)
print("Generation Data:")
print(generator)

num_buses = 3  # Set the number of buses in your system, have to do this manually due to the set up in excel file
Y_bus = create_y_bus_matrix(num_buses, transmission)
print("Y-bus Matrix:")
print(Y_bus)

Y_DC = generate_Y_DC(generator, transmission)
print("Y-bus Matrix for DC Power Flow:")
print(Y_DC)
"""