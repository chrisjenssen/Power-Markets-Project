import pandas as pd
import numpy as np

def read_excel_file(fileName, sheetName):
    # Read everything in
    df = pd.read_excel(fileName, sheet_name=sheetName, header=2)

    # ---- Generators ----
    gen = df.loc[:, 'Generator':'Slack bus'].copy()
    gen = gen[gen['Generator'].notna()]

    # ---- Loads ----
    ld = df.loc[:, 'Load unit':'Location.1'].copy()
    ld = ld[ld['Load unit'].notna()]

    # ---- Transmission lines ----
    tr = df.loc[:, 'Line':'Susceptance [p.u]'].copy()
    tr = tr[tr['Line'].notna()]

    return gen, ld, tr

