def readexcelfile(fileName, sheetname):
    df = pd.read_excel(fileName,sheet_name=sheetname, header=2)
    generator_data = df.loc[:,'Generator':'Slack bus']
    load_data = df.loc[:,'Load unit':'Location']
    transmission_data = df.loc[:,'Line':'Susceptance [p.u]']
    return(generator_data, load_data, transmission_data)

fileName = 'Problem_2_data.xlsx'
sheetname = 'Problem 2.2 - Base case'

def create_y_bus_matrix(num_buses, susceptance_dict):

    Y_bus = np.zeros((num_buses, num_buses), dtype=complex)

    # Fill the Y-bus matrix
    for (i, j), b_ij in susceptance_dict.items():
        if i != j:
            Y_bus[i, j] -= b_ij
            Y_bus[j, i] -= b_ij
            Y_bus[i, i] += b_ij
            Y_bus[j, j] += b_ij

    return Y_bus