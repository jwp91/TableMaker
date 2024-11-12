import numpy as np
import tableMakerv2 as tm2
import time
import matplotlib.pyplot as plt

# This code was taken from flameTableMaker_workspace.ipynb to test for a speed difference between VSCode and command-line python
# There is no significant difference: VSCode is then preferred for this kind of testing due to variable permanence.

try:
    path = r"./data/methaneFlame"
    Lvals = [0.0046, 0.0048, 0.005, 0.0052, 0.0056, 0.006, 0.008, 0.02, 0.04, 0.08, 0.12, 0.16, 0.2]
    for i in range(len(Lvals)-1):
        if Lvals[i+1] > Lvals[i]:
            pass
        else:
            print(f"NO: {Lvals[i+1]} < {Lvals[i]}")
    tvals = np.arange(0,11,1)

    numXim = 5
    numXiv = 5

    # Create table
    start = time.time()

    #----- Get function to get T(xim, xiv, h, c)
    T_func = tm2.phiTable(path, Lvals, tvals, phi = 'T', 
                          numXim = numXim, numXiv = numXiv, parallel = True)

    end = time.time()
    print("Time elapsed creating T_func:", end - start)

    ODThvals = np.loadtxt(r'../tjet/fmeans_enth.dat')
    ODTcvals = np.loadtxt(r'../tjet/fmeans_progvar.dat')
    ODTximvals = np.loadtxt(r'../tjet/fmeans_mixf.dat')
    ODTxivvals = np.loadtxt(r'../tjet/fvar_mixf.dat')
    ODTTvals = np.loadtxt(r'../tjet/fmeans_temp.dat')
    ODThvals[2][3]

    numRads = len(ODThvals)
    numTimes = len(ODThvals[0])
    Tqueried = np.zeros(ODThvals.shape)

    for radInd in range(numRads):
        print(f"Working on row {radInd} of {numRads} ({radInd/numRads*100:.1g}% complete)")
        for timeInd in range(numTimes):
            if timeInd == 0:
                # First row is just the radial position
                Tqueried[radInd][timeInd] = ODThvals[radInd][timeInd]
            else:
                h = ODThvals[radInd][timeInd]
                c = ODTcvals[radInd][timeInd]
                xim = ODTximvals[radInd][timeInd]
                xiv = ODTxivvals[radInd][timeInd]
                if xim<0:
                    xim = np.abs(xim)
                if xiv < 0:
                    xiv = np.abs(xiv)
                Tqueried[radInd][timeInd] = T_func[0](xim, xiv, h, c, useStoredSolution = True)
except KeyboardInterrupt:
    pass

# Check how far the code got
lastCompletedRow = len([Tqueried[i] for i in range(len(Tqueried)) if (Tqueried[i]!=0).all()])
Tqueried_completed = Tqueried[:lastCompletedRow]
len(Tqueried_completed)

# Load data to plot
t = np.loadtxt('../tjet/list_of_times.dat')
t = t[:-15]
T = ODTTvals
x = T[:lastCompletedRow,0]
T = T[:lastCompletedRow,1:-15]

# Plot the data
plt.rcParams.update({'font.size': 14})
xx,tt = np.meshgrid(x,t)
fig=plt.figure(figsize=(10,8))
ax = plt.subplot(1, 2, 1)
ax.contourf(xx,tt,T.T,100, cmap='inferno')
ax.set_xlabel('x (m)')
ax.set_ylabel('t (s)')
ax=plt.subplot(1, 2, 2)
ax.contourf(xx, tt, Tqueried_completed[:,1:-15].T, 100, cmap = 'inferno')
ax.set_xlabel('x (m)')
ax.set_ylabel('t (s)');
