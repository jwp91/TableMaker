# Taken from tmv3c_workspace.ipynb

# Boilerplate
import numpy as np
import tmv3_class as tmv3c
import os

# Load table
tables = tmv3c.load('tables')

# Set cases
cases = ['tjet_Le1B_LGLE',]

# Verify functions are callable and returning reasonable values
print("Confirm functions were generated correctly:")
print("  T: ", tables.phi_mvhc_funcs['T']  (0.06, 0.005, -1e6, 0.15))
print(" hr: ", tables.phi_mvhc_funcs['hr'] (0.06, 0.005, -1e6, 0.15))
print(" CO: ", tables.phi_mvhc_funcs['CO'] (0.06, 0.005, -1e6, 0.15))
print(" OH: ", tables.phi_mvhc_funcs['OH'] (0.06, 0.005, -1e6, 0.15))
print("CO2: ", tables.phi_mvhc_funcs['CO2'](0.06, 0.005, -1e6, 0.15))

# Aliases for functions
T_func   = tables.phi_mvhc_funcs['T']  
hr_func  = tables.phi_mvhc_funcs['hr'] 
CO_func  = tables.phi_mvhc_funcs['CO'] 
OH_func  = tables.phi_mvhc_funcs['OH'] 
CO2_func = tables.phi_mvhc_funcs['CO2']

# Function for a priori testing
def apriori(phiData, ximData, xivData, hData, cData, table, phiName = None, confirm = False):
    if confirm and input(f"Are you sure you want to run a priori testing?") != 'y':
        return None
    else:
        print(f"Running {phiName} a priori testing...")
        minPhi = np.min(phiData)
        maxPhi = np.max(phiData)
        #testThreshold = (maxPhi-minPhi)*1e-4
        testThreshold = 0.0
        numRads = len(phiData)
        numTimes = len(phiData[0])
        phiQueried = np.zeros(phiData.shape)
        for radInd in range(numRads):
            fracCompleted = radInd/numRads*100
            if fracCompleted%25 <= 0.1:
                print(f"Working on row {radInd}/{numRads} ({fracCompleted:.1f}% complete)")
            for timeInd in range(numTimes):
                # For each radius and time point, grab the ODT xim, xiv, h, and c values
                h = hData[radInd][timeInd]
                c = cData[radInd][timeInd]
                xim = ximData[radInd][timeInd]
                xiv = xivData[radInd][timeInd]

                # Avoid errors due to very small negative values (e.g. -1e-8)
                if xim < 0:
                    xim = np.abs(xim)
                if xiv < 0:
                    xiv = np.abs(xiv)
                xivmax = xim*(1-xim)
                if xiv > xivmax:
                    if xiv-xivmax < 1e-6:
                        print(f"Corrected xiv at rad {radInd}, time {timeInd} from {xiv} to {xivmax}")
                        xiv = xivmax

                # A Priori testing
                if np.abs(phiData[radInd][timeInd] - minPhi) < testThreshold:
                    # Flame is essentially not present. Do not query.
                    phiQueried[radInd][timeInd] = phiData[radInd][timeInd]
                else:
                    # Query the table
                    if radInd == 0 and timeInd == 0:
                        phiQueried[radInd][timeInd] = table(xim, xiv, h, c, useStoredSolution = False, solver = 'gammachi')
                    else:
                        phiQueried[radInd][timeInd] = table(xim, xiv, h, c, useStoredSolution = True, solver = 'gammachi')
        print(f"Finished {phiName} a priori testing.")
        return phiQueried, phiData
    
# Get ODT data, run a priori for each case
for caseName in cases:
    print(f"################## {caseName} ##################")
    ODTpath = r'../../odt/data/'+caseName+r'/post/'

    # Auxiliary data
    ODT_xs      = np.loadtxt(ODTpath + r'fmeans_temp.dat')[:,0]
    ODT_header  = np.loadtxt(ODTpath + r'fmeans_enth.dat', comments = None, max_rows = 1, dtype = str)[2:]
    ODT_ts = np.array([float(x.split('_')[-1]) for x in ODT_header])[:-1]

    # Phi data
    ODT_hvals   = np.loadtxt(ODTpath + r'fmeans_enth.dat')[:,1:]     # Ignoring the first column (x values)
    ODT_cvals   = np.loadtxt(ODTpath + r'fmeans_progvar.dat')[:,1:]
    ODT_ximvals = np.loadtxt(ODTpath + r'fmeans_mixf.dat')[:,1:]
    ODT_xivvals = np.loadtxt(ODTpath + r'fvar_mixf.dat')[:,1:]

    ODT_Tvals   = np.loadtxt(ODTpath + r'fmeans_temp.dat')[:,1:]
    ODT_hrvals  = np.loadtxt(ODTpath + r'fmeans_hr.dat')[:,1:]
    ODT_COvals  = np.loadtxt(ODTpath + r'fmeans_y_CO.dat')[:,1:]
    ODT_OHvals  = np.loadtxt(ODTpath + r'fmeans_y_OH.dat')[:,1:]
    ODT_CO2vals = np.loadtxt(ODTpath + r'fmeans_y_CO2.dat')[:,1:]

    #################### Run a priori testing, save data
    savePath = r'./figures/ODT_aPriori/Publication/'+caseName+'/'
    # Check if the folder exists, and create it if it doesn't
    if not os.path.exists(savePath):
        os.makedirs(savePath)
    np.savetxt(savePath + 'xVals.txt', ODT_xs)
    np.savetxt(savePath + 'tVals.txt', ODT_ts)

    Tqueried, ODT_Ts = apriori(ODT_Tvals, ODT_ximvals, ODT_xivvals, ODT_hvals, ODT_cvals, 
                            T_func, phiName = 'T')
    np.savetxt(savePath + 'Tqueried.txt', Tqueried)
    np.savetxt(savePath + 'T_ODT.txt', ODT_Tvals)
    hrqueried, ODT_hrs = apriori(ODT_hrvals, ODT_ximvals, ODT_xivvals, ODT_hvals, ODT_cvals, 
                                hr_func, phiName = 'hr')
    np.savetxt(savePath + 'hrqueried.txt', hrqueried)
    np.savetxt(savePath + 'hr_ODT.txt', ODT_hrvals)
    COqueried, ODT_COs = apriori(ODT_COvals, ODT_ximvals, ODT_xivvals, ODT_hvals, ODT_cvals,
                                CO_func, phiName = 'CO')
    np.savetxt(savePath + 'COqueried.txt', COqueried)
    np.savetxt(savePath + 'CO_ODT.txt', ODT_COvals)
    OHqueried, ODT_OHs = apriori(ODT_OHvals, ODT_ximvals, ODT_xivvals, ODT_hvals, ODT_cvals,
                                OH_func, phiName = 'OH')
    np.savetxt(savePath + 'OHqueried.txt', OHqueried)
    np.savetxt(savePath + 'OH_ODT.txt', ODT_OHvals)
    CO2queried, ODT_CO2s = apriori(ODT_CO2vals, ODT_ximvals, ODT_xivvals, ODT_hvals, ODT_cvals,
                                CO2_func, phiName = 'CO2')
    np.savetxt(savePath + 'CO2queried.txt', CO2queried)
    np.savetxt(savePath + 'CO2_ODT.txt', ODT_CO2vals)

print("A priori testing complete.")
