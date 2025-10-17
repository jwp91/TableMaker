# Taken from tmv3c_workspace.ipynb

##### Boilerplate
import numpy as np
import time
import tmv3_class as tmv3c

##### Table params
nxim = 3
ximLfrac = 0.2
ximGfrac = 0.5
nxiv = 3

##### Flamelet data filepath
path = r'./data/ChiGammaTablev3'
file_pattern = r'flm_.*.dat$'

##### Data parameters (change depending on how ignis was run)
tvals = np.arange(0,14,1)
Lvals = np.arange(0,26,1)
gammaValues = [0., 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.60, 0.65]
Lbounds = [min(Lvals), max(Lvals)]
tbounds = [min(tvals), max(tvals)]
path_to_hsens = './data/ChiGammaTablev3/hsens.dat' # Col1: mixf, Col2: hsens (L = 0). 

tables = tmv3c.table(path, Lvals, tvals, nxim=nxim, nxiv=nxiv, ximLfrac=ximLfrac, ximGfrac=ximGfrac,
                     gammaValues=gammaValues, flmt_file_pattern=file_pattern)

##### Generate the table
start = time.time()
tables.phi_mvhc(['T', 'hr', 'CO', 'OH', 'CO2'])
end = time.time()
seconds = int(end - start)
hours = seconds//3600
minutes = (seconds - hours*3600)//60
seconds = (seconds - hours*3600 - minutes*60)
print(f"Time elapsed creating functions: {hours} hours, {minutes} minutes, {seconds} seconds")
  
tables.save('table_demo')
