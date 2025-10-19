# Boilerplate
import numpy as np
import pandas as pd
import os
import warnings
import tmv3_class as tmv3c
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend

# Params
reQuery = False
makeFigs = True
savePath = r'./figures/ODT_aPriori/Publication/singleRlz/'
if not os.path.exists(savePath):
    os.makedirs(savePath)

if reQuery:
    # Locate data file and specify which realization to use (arbitrary choice)
    dataPath = r'./data'
    rlzName = r'/tjet_1_dat10_dmp20.dat'

    # Load in the table
    tables = tmv3c.load('tables')

    # Aliases for functions
    T_func   = tables.phi_mvhc_funcs['T']  
    hr_func  = tables.phi_mvhc_funcs['hr'] 
    CO_func  = tables.phi_mvhc_funcs['CO'] 
    OH_func  = tables.phi_mvhc_funcs['OH'] 
    CO2_func = tables.phi_mvhc_funcs['CO2']

    # Load in data
    assert os.path.isfile(dataPath + rlzName)
    SORdata = np.loadtxt(dataPath + rlzName)

    with open(dataPath + rlzName, 'r') as f:
        last_line = None
        for line in f:
            if line[0] == '#':
                last_line = line
    SORheader = last_line.split()[1:]
    SORdf = pd.DataFrame(SORdata, columns = SORheader)
    SORdf['30_progVar'] = SORdf['23_y_CO'] + SORdf['24_y_CO2'] + SORdf['12_y_H2'] + SORdf['17_y_H2O']

    # Sometimes mixture fraction values are very small and negative. 
    # Round these to zero.
    SORdf['9_mixf'] = SORdf['9_mixf'].apply(lambda x: 0 if x < 0 else x)

    # Set up columns to hold queried values
    SORdf['hr_queried'] = np.zeros(len(SORdf['2_posf']))
    SORdf['temp_queried'] = np.zeros(len(SORdf['2_posf']))
    SORdf['CO_queried'] = np.zeros(len(SORdf['2_posf']))
    SORdf['OH_queried'] = np.zeros(len(SORdf['2_posf']))
    SORdf['CO2_queried'] = np.zeros(len(SORdf['2_posf']))

    # Query the table
    for i in range(len(SORdf['2_posf'])):
        xim = SORdf['9_mixf'][i]
        h = SORdf['29_enth'][i]
        c = SORdf['30_progVar'][i]

        # Temporarily suppress warnings
        with warnings.catch_warnings():
            SORdf.loc[i, 'hr_queried'] =   hr_func(xim, 0, h, c, useStoredSolution = True, solver = 'gammachi')
            SORdf.loc[i, 'temp_queried'] =  T_func(xim, 0, h, c, useStoredSolution = True, solver = 'gammachi', minVal = 300)
            SORdf.loc[i, 'CO_queried'] =   CO_func(xim, 0, h, c, useStoredSolution = True, solver = 'gammachi', minVal = 0)
            SORdf.loc[i, 'OH_queried'] =   OH_func(xim, 0, h, c, useStoredSolution = True, solver = 'gammachi', minVal = 0)
            SORdf.loc[i, 'CO2_queried'] = CO2_func(xim, 0, h, c, useStoredSolution = True, solver = 'gammachi', minVal = 0)
            
        if i%30 == 0:
            print(f"Finished row {i}/{len(SORdf['2_posf'])}")

    # Save results to a new file
    SORdf.to_csv(savePath + r'singleRlz_data.csv', index = False)
    print("Results saved.")

if makeFigs:
    # Reload data
    SORdf = pd.read_csv(savePath + r'singleRlz_data.csv')
    print("Data reloaded.")

    # Figure maker
    plt.rcParams.update({'font.size': 14})
    def plotSingleODT(phi='temp', ylabel = 'Temperature', units = 'K', splity = False):
        print("Creating figures for " + phi + "...")
        # Determine column names in dataframe
        ODT_label = None
        queried_label = None
        for item in SORdf.columns:
            last_element = item.split('_')[-1]
            if last_element.lower() == phi.lower():
                ODT_label = item
            if last_element == 'queried' and item.split('_')[-2].lower() == phi.lower():
                queried_label = item
        if ODT_label == None:
            raise ValueError(f"ODT data does not contain {phi}.")
        if queried_label == None:
            raise ValueError(f"Queried unavailable for {phi}.")
        
        # Plot comparison in physical space
        plt.figure()
        plt.plot(SORdf['2_posf'], SORdf[ODT_label], 'bo', label = "ODT")
        if splity:
            plt.ylabel(f"ODT {ylabel} ({units})")
            plt.twinx()
        else:
            plt.ylabel(f"{ylabel} ({units})")
        plt.plot(SORdf['2_posf'], SORdf[queried_label], 'r.', label = "Queried")
        plt.xlabel("Position (m)")
        plt.legend()

        plt.tight_layout()
        plt.savefig(savePath + f"{phi}_physicalspace.pdf", dpi = 300)
        plt.close()

        # Plot comparison in mixture fraction space
        plt.figure()
        plt.plot(SORdf['9_mixf'], SORdf[ODT_label], 'bo', label = "ODT")
        if splity:
            plt.ylabel(f"ODT {ylabel} ({units})")
            plt.twinx()
        else:
            plt.ylabel(f"{ylabel} ({units})")
        plt.plot(SORdf['9_mixf'], SORdf[queried_label], 'r.', label = "Queried")
        plt.xlabel("Mixture Fraction")
        #plt.xlim(-0.05,1.05)
        plt.legend()

        plt.tight_layout()
        plt.savefig(savePath + f"{phi}_mixfspace.pdf", dpi = 300)

        plt.close()
        return None
        
    # Make figures
    print("Making figures...")
    plotSingleODT('temp', ylabel = 'Temperature', units = 'K')
    plotSingleODT('CO2', ylabel = r'Y$_{CO_2}$', units = r'$kg/m^3$')
    plotSingleODT('CO', ylabel = r'Y$_{CO}$', units = r'$kg/m^3$')
    plotSingleODT('OH', ylabel = r'Y$_{OH}$', units = r'$kg/m^3$')
    plotSingleODT('hr', ylabel = r'HRR$', units = r'$W/m^3$')
    print("Figures complete.")
