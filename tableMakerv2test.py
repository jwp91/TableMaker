# Import Packages
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import tableMakerv2 as tm2
import time
from concurrent.futures import ProcessPoolExecutor
import concurrent

# Reimport changes from editing LiuInt or tableMaker
import importlib
importlib.reload(tm2)

# Declare precision
numXim = 5
numXiv = 5

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

# Known grid points
hval = -424105.8324745877 
cval = 0.09807061715648245

xim = np.linspace(0.4,0.6,50)
old = np.zeros(len(xim))
new = np.zeros(len(xim))
for i, ximval in enumerate(xim):
    new[i] = T_table[0](ximval, ximval*(1-ximval)*0.5, hval, cval)


#----- OLD CODE: tableMaker
from Archive import tableMaker as tableMaker
path = r"./aPriori/TNF"
Lvals = [0.00135, 0.0014, 0.0016, 0.002, 0.004, 0.006, 0.008, 0.02, 0.04, 0.2]
tvals = np.arange(0,11,1)

start = time.time()
data_output_old = tableMaker.get_data_files(path, Lvals, tvals)

#----- Create T Table
# Serial version
#table_old, indices_old = tableMaker.makeLookupTable(path, Lvals, tvals, phi='T',
#                                    numXim = numXim, numXiv = numXiv, 
#                                    get_data_files_output = data_output_old)
#h_table_old, h_indices_old = tableMaker.makeLookupTable(path, Lvals, tvals, phi='h',
#                                    numXim = numXim, numXiv = numXiv, 
#                                    get_data_files_output = data_output_old)
#c_table_old, c_indices_old = tableMaker.makeLookupTable(path, Lvals, tvals, phi='c',
#                                    numXim = numXim, numXiv = numXiv, 
#                                    get_data_files_output = data_output_old)
#end = time.time()
#print(f"TableGen without parallelization: {end-start}")

# Parallel version
def create_table(args):
    path, Lvals, tvals, phi, numXim, numXiv, data_output_old = args
    return tableMaker.makeLookupTable(path, Lvals, tvals, phi=phi,
                                    numXim = numXim, numXiv = numXiv, 
                                    get_data_files_output = data_output_old)
# Prepare arguments for each call
table_args = [
    (path, Lvals, tvals, 'T', numXim, numXiv, data_output_old),
    (path, Lvals, tvals, 'h', numXim, numXiv, data_output_old),
    (path, Lvals, tvals, 'c', numXim, numXiv, data_output_old),
]

# Use ProcessPoolExecutor to run table creation in parallel
with ProcessPoolExecutor() as executor:
    futures = {executor.submit(create_table, args): idx for idx, args in enumerate(table_args)}

    results = {}
    for future in concurrent.futures.as_completed(futures):
        idx = futures[future]
        try:
            results[idx] = future.result()
        except Exception as e:
            print(f"Table creation for index {idx} generated an exception: {e}")

# Unpack results into separate variables
table_old, indices_old = results[0]
h_table_old, h_indices_old = results[1]
c_table_old, c_indices_old = results[2]
end = time.time()
print(f"TableGen with parallelization: {end-start}")

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
plt.savefig("solvertest.png")