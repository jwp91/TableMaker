# Import Packages
import matplotlib.pyplot as plt 
import codes.TableMakerMain.postGit.Archive.tableMaker as tableMaker
import numpy as np

#----- Load in data
from codes.TableMakerMain.postGit.Archive.tableMaker import *
path = r"./aPriori/TNF"
Lvals = [0.00135, 0.0014, 0.0016, 0.002, 0.004, 0.006, 0.008, 0.02, 0.04, 0.2]
tvals = np.arange(0,11,1)
data_output = get_data_files(path, Lvals, tvals)

#----- Create T Table
phi = 'T'
table, indices = tableMaker.makeLookupTable(path, Lvals, tvals, phi, \
                                    numXim = 10, numXiv = 10, get_data_files_output = data_output)

#----- Create h & c tables
phi = 'h'
h_table, h_indices = tableMaker.makeLookupTable(path, Lvals, tvals, phi, \
                                                numXim = 10, numXiv = 10, get_data_files_output = data_output)
phi = 'c'
c_table, c_indices = tableMaker.makeLookupTable(path, Lvals, tvals, phi, \
                                                numXim = 10, numXiv = 10, get_data_files_output = data_output)
print('h & c tables complete')

#----- Create interpolators
Ih = tableMaker.createInterpolator(h_table, h_indices)
Ic = tableMaker.createInterpolator(c_table, c_indices)
It = tableMaker.createInterpolator(table, indices)

#----- Define function to get T(xim, xiv, h, c) from table
Lbounds = [min(Lvals), max(Lvals)]
tbounds = [min(tvals), max(tvals)]
    
def T_table(xim, xiv, h, c):
    L,t = tableMaker.Lt_hc(h, c, xim, xiv, Ih, Ic, Lbounds, tbounds, hc_avg = 10**(-5))
    return It(xim, xiv, L, t)

#----- Import DOL-processed data
cols = "r, F, Frms, C, H, T, O2, N2, H2, H2O, CH4, CO, CO2, OH, NO" #from data file
ximcol = 1
xivcol = 2
hcol = 4
ccol = 3
tcol = 5

fileNames = [r"D075.Yall_proc", r"D30.Yall_proc", r"D60.Yall_proc", \
             r"D15.Yall_proc", r"D45.Yall_proc", r"D75.Yall_proc"]
filePath = r"./aPriori/processed/pmD.scat/"

t_table_data_all = np.empty(len(fileNames), dtype=np.ndarray)
data = np.empty((len(fileNames)), dtype=np.ndarray)

for j in range(len(fileNames)):
    netPath = filePath+fileNames[j]
    data[j] = np.loadtxt(netPath)[1:-1].T      #Indexing piece in the middle avoids xim=0
    t_data_table = np.ones(len(data[j][0]))*-1 #Initialize to store table-computed data
    
    for i in range(len(data[j][0])):
        xim = data[j][ximcol][i]
        xiv = data[j][xivcol][i]
        h   = data[j][hcol][i]
        c   = data[j][ccol][i]
        t_data_table[i] = T_table(xim, xiv, h, c)
    t_table_data_all[j] = t_data_table

colors = ['#FF0000', '#FFA500', '#00FF00', '#0000FF', '#4B0082', '#000000']

omit = 0 #number of plots to omit from the plot
for i in range(len(data)-omit):
    t_data_experiment = data[i][tcol]
    r = data[i][0]
    plt.plot(r, t_data_experiment, 'o', \
             color = colors[i])
    plt.plot(r, t_table_data_all[i], label = f"{fileNames[i]}", \
             color = colors[i])
plt.title("Table vs. Experiment")
plt.ylabel("Temperature (K)")
plt.xlabel(r"Radial Position")
plt.xlim((0,75))
plt.ylim((0,2000))
plt.legend()
plt.show();
print("""Dots = Experimental
Lines = Table""")

plt.savefig('ComparisonPlot_ExpVsTable_Res10',bbox_inches='tight')  # Save as a PNG file
