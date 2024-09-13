# Import Packages
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import ipywidgets as wgt
import tableMakerv2 as tm2
import time

# Reimport changes from editing LiuInt or tableMaker
import importlib
importlib.reload(tm2)

# Declare precision
numXim = 5
numXiv = 5

#----- OLD CODE: tableMaker
from codes.TableMakerMain.postGit.Archive import tableMaker as tableMaker
start = time.time()
path = r"./aPriori/TNF"
Lvals = [0.00135, 0.0014, 0.0016, 0.002, 0.004, 0.006, 0.008, 0.02, 0.04, 0.2]
tvals = np.arange(0,11,1)

data_output_old = tableMaker.get_data_files(path, Lvals, tvals)

#----- Create T Table
table_old, indices_old = tableMaker.makeLookupTable(path, Lvals, tvals, phi='T',
                                    numXim = numXim, numXiv = numXiv, 
                                    get_data_files_output = data_output_old)
h_table_old, h_indices_old = tableMaker.makeLookupTable(path, Lvals, tvals, phi='h',
                                    numXim = numXim, numXiv = numXiv, 
                                    get_data_files_output = data_output_old)
c_table_old, c_indices_old = tableMaker.makeLookupTable(path, Lvals, tvals, phi='c',
                                    numXim = numXim, numXiv = numXiv, 
                                    get_data_files_output = data_output_old)

# Testing the new code 
It_old = tableMaker.createInterpolator(table_old, indices_old)
Ih_old = tableMaker.createInterpolator(h_table_old, h_indices_old)
Ic_old = tableMaker.createInterpolator(c_table_old, c_indices_old)

#----- Define function to get T(xim, xiv, h, c) from table (OLD METHOD)
Lbounds = [min(Lvals), max(Lvals)]
tbounds = [min(tvals), max(tvals)]
    
def T_table_old(xim, xiv, h, c):
    L,t = tableMaker.Lt_hc(h, c, xim, xiv, Ih_old, Ic_old, Lbounds, tbounds)
    return It_old(xim, xiv, L, t)

end = time.time()
print(f"Old code time: {end-start}")

#----------- NEW CODE: tm2
start = time.time()

path = r"./aPriori/TNF"
Lvals = [0.00135, 0.0014, 0.0016, 0.002, 0.004, 0.006, 0.008, 0.02, 0.04, 0.2]
tvals = np.arange(0,11,1)

#----- Get function to get T(xim, xiv, h, c) (NEW METHOD)
T_table = tm2.phiTable(path, Lvals, tvals, phi = 'T', 
                        numXim = numXim, numXiv = numXiv)

end = time.time()
print(f"New code time: {end - start}")

#--------------- Compare T(xi) for new vs. old code
# Known grid points
hval = -424105.8324745877 
cval = 0.09807061715648245

xim = np.linspace(0.1,0.9,50)
old = np.zeros(len(xim))
new = np.zeros(len(xim))
for i, ximval in enumerate(xim):
    old[i] = T_table_old(ximval, ximval*(1-ximval)*0.5, hval, cval)
    new[i] = T_table[0](ximval, ximval*(1-ximval)*0.5, hval, cval)

plt.plot(xim, old, '.', label = 'old')
plt.plot(xim, new, '.', label = 'new (newton solve)')
plt.legend()
plt.ylabel("Temp (K)")
plt.xlabel("Xi (mixture fraction)");