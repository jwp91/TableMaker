from tableMaker import *
path = r"../flameAICHE/run"
Lvals = [0.00135, 0.002, 0.004, 0.006, 0.008, 0.02, 0.04, 0.2]
tvals = np.arange(0,11,1)

# Get data
data_output = get_data_files(path, Lvals, tvals)
print('data retrieved')

# Make table
phi = 'T'
table, indices = makeLookupTable(None, Lvals, tvals, phi,\
    numXim = 10, numXiv = 10, get_data_files_output = data_output)



