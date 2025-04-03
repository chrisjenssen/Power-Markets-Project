def readexcelfile(fileName, sheetname):
    df = pd.read_excel(fileName,sheet_name=sheetname, header=2)
    generator_data = df.loc[:,'Generator':'Slack bus']
    load_data = df.loc[:,'Load unit':'Location']
    transmission_data = df.loc[:,'Line':'Susceptance [p.u]']
    return(generator_data, load_data, transmission_data)

fileName = 'Problem_2_data.xlsx'
sheetname = 'Problem 2.2 - Base case'
