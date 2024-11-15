import numpy as np
from scipy.interpolate import interp1d
from scipy.integrate import quad
from scipy.optimize import fsolve, ridder, least_squares, root
import os
import warnings
from glob import glob
from re import match, search
from statistics import variance
import LiuInt as LI # Package with functions for integrating over the BPDF, parameterized by xi_avg and xi_variance
from scipy.interpolate import RegularGridInterpolator as rgi
from datetime import datetime
import multiprocessing as mp

############################## tableMakerv2

def computeProgressVariable(data, header, c_components = ['H2', 'H2O', 'CO', 'CO2']):
    """
    Progress variable is defined as the sum of the mole fractions of a specified set of c_components.
    This function computes the flame progress variable using:
        data = Data from a flame simulation. Each row corresponds to a specific property.
            In the case of this package, this data array is "transposed_file_data" inside the function "get_file_data"
                ex. data[0] = array of temperature data.
        header = 1D array of column headers, denoting which row in "data" corresponds to which property.
            ex. If header[0] = "Temp", then data[0] should be temperature data.
        c_components = list defining whih components' mixture fractions are included in the progress variable. 
            By default, this is set to be ['H2', 'H2O', 'CO', 'CO2']
            The strings in the list should each match a string used in 'header'
    """
    #---------- Determine where the c_components are in 'data'
    indices = np.ones(len(c_components), dtype = np.int8)*-1
    for i in range(len(header)):                # For each element in the header, 
        for y in range(len(c_components)):      # Check for a match among the passed-in c_components
            if header[i]==c_components[y].replace(" ",""):
                indices[y] = int(i)             # Indices must be strictly integers (ex. 5, not 5.0)
                
    # Confirm all indices were located
    for j, ind in enumerate(indices):
        if ind == -1:
            raise ValueError(f"No match found for {c_components[j]}.")

    #---------- Compute progress variable
    c = np.zeros(len(data[0]))        # Initialize c array
    for d in range(len(data[0])):     # For each set of data points (each column),
        sum = 0
        for index in indices:         # For each of the components specified, 
            sum += data[index,d]      # Sum the mole fractions of each component
        c[d] = sum
    return c 

##############################

def get_data_files(path_to_data, Lvals, tvals, file_pattern = r'^L.*.dat$', \
                   c_components = ['H2', 'H2O', 'CO', 'CO2']):
    """
    Reads and formats data computed by a flame simulation.
    Inputs: 
        path_to_data = path to simulation data relative to the current folder. 
            NOTE: The data headers must be the last commented line before the data begins.
            The code found at https://github.com/BYUignite/flame was used in testing. 
        Each file will have been run under an array of conditions L,t:
        Lvals: values of parameter L used, formatted as a list (ex. [ 0.002, 0.02, 0.2])
        tvals: values of parameter t used, formatted as a list (ex. [ 0    , 1   , 2  ])
        file_pattern = regular expression (regex) to identify which files in the target folder are data files.  
            DEFAULT: r'^L.*.dat$'. This grabs any files that begin with "L" and end with ".dat". 
        c_components = list defining whih components' mixture fractions are included in the progress variable. 
            By default, this is set to be ['H2', 'H2O', 'CO', 'CO2']
            
    Outputs:
        all_data = an array with the data from each file, indexed using all_data[Lval][tval][column# = Property][row # = data point]
        headers  = an array with the column labels from each file, indexed using headers[Lval][tval]
            Each file should have the same columns labels for a given instance of a simulation, but all headers are redundantly included.
        extras   = an array storing any extra information included as comments at the beginning of each file, indexed using extras[Lval][tval]
            This data is not processed in any way by this code and is included only for optional accessibility
    """
    #---------- Check if the provided path is a valid directory
    if not os.path.isdir(path_to_data):
        print(f"Error: {path_to_data} is not a valid directory: no data loaded.")
        return None
    
    #---------- Use glob to list all files in the directory
    files = sorted(glob(os.path.join(path_to_data, '*')))
    
    #---------- Store data and filenames
    filenames = np.array([])
    data_files = np.array([])
    for file in files:
        if match(file_pattern, os.path.basename(file)):
            filenames = np.append(filenames,  os.path.basename(file))
            data_files= np.append(data_files, file)

    #---------- Initialize data arrays
    all_data = np.empty((len(Lvals),len(tvals)), dtype=np.ndarray)  # Initialize to grab data values
    headers  = np.empty((len(Lvals),len(tvals)), dtype=np.ndarray)  # Initialize to store headers
    extras   = np.empty((len(Lvals),len(tvals)), dtype=np.ndarray)  # Initialize to store extra info before header

    #---------- Grab and store data
    for i in range(len(data_files)):
        l = i//len(tvals)   # Row index
        t = i %len(tvals)   # Column index
        
        file = data_files[i]
        with open(file, 'r') as f:
            #---------- Make sure the assigned L and t value are in the file name:
            if str(Lvals[l]) not in f.name:
                print(f"Warning: for file name '{f.name}', mismatch: L = {Lvals[l]}")
            if str(tvals[t]) not in f.name:
                print(f"Warning: for file name '{f.name}', mismatch: t = {tvals[t]}")

            #---------- Get raw data
            lines = f.readlines()
            raw_data = np.array([line.strip() for line in lines if not line.startswith('#')])

            #---------- Grab the header and extra data (included as commented lines)
            IsHeader = True
            header = np.array([])
            extra = np.array([])
            for line in reversed(lines):               #The last of the commented lines should be the headers,
                if line.startswith('#'):               #so we grab the last of the commented lines
                    vals = np.array([val for val in line.strip().split() if val !='#'])
                    if IsHeader == True:
                        for val in vals:
                            #Remove preemtive numbers in the column labels, then store column label (assumes labels formmatted as colNum_property, e.g. 0_T)
                            #This label is used later to select which property to use when creating the table
                            header = np.append(header, val.split("_")[1])  
                        IsHeader = False               #The next line won't be the header, but should be stored in 'extras'
                    else:
                        for val in vals:
                            extra = np.append(extra, val)
        header = np.append(header, "c")
        headers[l,t] = header
        extras[l,t]  = extra
        
        #---------- Parse out numerical values
        #NOTE: the following lines could be achieved with np.loadtxt(). Because we've already read in the lines
        #      to extract the headers, we can extract the data manually with a few extra lines of code.
        
        file_data = np.empty(len(raw_data[0].split()))     # Holds the data for this file
        for row in raw_data:
            numbers = np.array([float(val) for val in row.split()])
            file_data = np.vstack((file_data,numbers))     # Adds each new row of data as a new row in file_data
        file_data = file_data[1:file_data.size]            # Get rid of first column (which is empty and only used for initialization)

        #---------- Transpose data so that each row contains data for a certain property (ex. one row is temperature data, one is density, etc.)
        transposed_file_data = file_data.T

        #---------- Add a row with progress variable (c)
        c = computeProgressVariable(transposed_file_data, header, c_components = c_components)
        transposed_file_data = np.vstack((transposed_file_data, c))   #Stacks this array of progress variable values as the last row 
        
        #---------- Arrange data by l and t indices
        all_data[l,t] = transposed_file_data
    
    #all_data is indexed using all_data[Lval][tval][column# = Property][row # = data point]
    return all_data, headers, extras

##############################

#----- Test get_data_files 
# Lvals = [0.002, 0.004, 0.006, 0.008, 0.02, 0.04, 0.2]
# tvals = np.arange(0,11,1)
# test = get_data_files(path, Lvals, tvals)[0][0][1]
# test[len(test)-1]
# test[0]

##############################

def phiFuncs(path_to_flame_data, Lvals, tvals, file_pattern = r'^L.*.dat$', c_components = ['H2', 'H2O', 'CO', 'CO2'],
             phi = 'T', Lt = False, mix_frac_name = "mixf", interpKind = 'cubic', get_data_files_output = None):
    """
    Returns an array of interpolated functions phi(ξ) where phi is any property of the flame.\n
    Inputs:\n
        path_to_data = path to simulation data relative to the current folder. \n
            NOTE: The data headers must be the last commented line before the data begins.\n
            The code found at https://github.com/BYUignite/flame was used in testing. \n
        Each file will have been run under an array of conditions L,t:\n
        Lvals: values of parameter L used, formatted as a list (ex. [ 0.002, 0.02, 0.2])\n
        tvals: values of parameter t used, formatted as a list (ex. [ 0    , 1   , 2  ])\n
        file_pattern = regular expression (regex) to identify which files in the target folder are data files. \n
            DEFAULT: r'^L.*.dat$'. This grabs any files that begin with "L" and end with ".dat". \n
        c_components = list defining which components' mixture fractions are included in the progress variable.\n 
            By default, this is set to be ['H2', 'H2O', 'CO', 'CO2']\n
        phi = desired property (ex. 'T', 'rho', etc.), case sensitive. Default = 'T'\n
            Available phi are viewable using "get_data_files(params)[1]".\n
            NOTE: c (progress variable) is available in the data. By default, c ≡ y_CO2 + y_CO + y_H2O + yH2. \n
            This definition can be changed by modifying the c_components parameter.\n
        Lt = Tuple with indices corresponding to the desired L and t. If set to False (default), the output will be an array of the functions phi(ξ) for all datafiles. \n
             Otherwise, this parameter determines which specific file should be used. \n
             Example1: phiFuncs(path, phi = 'T', Lt = (0,1)): returns the interpolated T(ξ) function ONLY from the data in the file from Lvals[0], tvals[1]. \n
             Example2: phiFuncs(path, phi = 'T'): returns an array containing the interpolated T(ξ) functions from every file in the directory\n
             Note that the values in this tuple are not values of L and t, but rather indexes of Lvals and tvals.\n
        mix_frac_name = name of the column header for mixture fraction. Default value: "mixf"\n
        interpKind = specifies the method of interpolation that should be used (uses scipy.interp1d). Default = 'cubic'. \n
        get_data_files_output = used to save time in the event that multiple tables are to be constructed. \n
            This should be the output of get_data_files, run with the relevant parameters matching those passed in to this function.\n
        
    Outputs:\n
        The output type of phiFuncs will depend on the input parameter "fileName":\n
             - If Lt is not defined (default), the output will be an array of functions.\n
             - If Lt is specified, the output will be the function for the specified file only. \n
    """
    #---------- Import data, files, and headers
    if get_data_files_output == None:
        # No processed data passed in: must generate.
        data, headers, extras = get_data_files(path_to_flame_data, Lvals, tvals, file_pattern = file_pattern, c_components = c_components)
    else:
        # Use pre-processed data
        data, headers, extras = get_data_files_output
    
    #---------- Get list of available phi (list of all data headers from original files)
    if type(Lt) == bool:
        # User did not specify a specific file.
        # This code here assumes that all datafiles have the same column labels and ordering:
        phis = headers[0][0] 
    elif Lt[0] < len(headers) and Lt[1] < len(headers[0]):
        # User specified a file and the indices were valid.
        phis = headers[Lt[0]][Lt[1]]
    else:
        # User specified a file and the indices were invalid.
        raise ValueError(f"""(L,t) indices '{Lt}' are invalid. Valid ranges for indices:
        L: (0,{len(headers)-1})
        t: (0,{len(headers[0])-1})""")
    
    #---------- Interpret user input for "phi", find relevant columns
    phi_col = -1
    xi_col = -1
    
    for i in range(len(phis)):
        if phis[i]==phi.replace(" ",""):
            # Phi column identified
            phi_col = i
        if phis[i]==mix_frac_name:
            # Mixture fraction column identified
            xi_col = i
    if phi_col == -1:
        # Phi wasn't found.
        raise ValueError("{} not recognized. Available phi are:\n {}".format(phi, phis))
        return None
    if xi_col == -1:
        # Xi wasn't found.
        raise ValueError(f"Mixture fraction ('{mix_frac_name}') was not found among data columns.")
        return None

    #---------- Interpolate phi(xi)
    phiFuncs = np.empty((len(Lvals),len(tvals)), dtype=np.ndarray)
    if Lt == False:
        #User did not specify file: must interpolate for every file
        for l in range(len(data)):
            for t in range(len(data[l])):
                xis = data[l][t][xi_col]
                phis = data[l][t][phi_col]
                phiFuncs[l][t] = interp1d(xis, phis, kind = interpKind)
        return phiFuncs
    else:
        #User specified a file
        xis = data[Lt[0]][Lt[1]][xi_col]
        phis = data[Lt[0]][Lt[1]][phi_col]
        return interp1d(xis, phis, kind = interpKind)

##############################

#----- Testing phiFuncs
# Lvals = [0.002, 0.004, 0.006, 0.008, 0.02, 0.04, 0.2]
# tvals = np.arange(0,11,1)
# phiFuncs(path, Lvals, tvals, Lt = (0,1))(0.5)
# phiFuncs("../flame/run", fileName = 'james')

##############################
    
def makeLookupTable(path_to_flame_data, Lvals, tvals, file_pattern = r'^L.*.dat$', c_components = ['H2', 'H2O', 'CO', 'CO2'],
                    phi = 'T', interpKind = 'cubic', numXim:int=5, numXiv:int = 5, get_data_files_output = None,
                    ximLfrac = 0.5, ximGfrac = 0.5):
    # NOTE: arguments later on unpack args for this function using *args. Only add parameters to the end of the
    # current list of parameters. If removing parameters, be sure to revise calls of this function later in the code.
    """
    Creates a 4D lookup table of phi_avg data. Axis are ξm, ξv, L (length scale), and t (time scale). 
    Inputs:
        path_to_flame_data = path to simulation data relative to the current folder. 
            NOTE: The data headers must be the last commented line before the data begins.
            The code found at https://github.com/BYUignite/flame was used in testing. 
        Each file will have been run under an array of conditions L,t:
        Lvals: values of parameter L used, formatted as a list (ex. [ 0.002, 0.02, 0.2])
        tvals: values of parameter t used, formatted as a list (ex. [ 0    , 1   , 2  ])
        file_pattern = regular expression (regex) to identify which files in the target folder are data files. 
            DEFAULT: r'^L.*.dat$'. This grabs any files that begin with "L" and end with ".dat". 
        c_components = list defining which components' mixture fractions are included in the progress variable. 
            By default, this is set to be ['H2', 'H2O', 'CO', 'CO2']
        phi = property for which values will be tabulated (ex. 'T', 'rho', etc.), case sensitive. Default = 'T'
            Available phi are viewable using "get_data_files(params)[1]".
            NOTE: c (progress variable) is available in the data. By default, c ≡ y_CO2 + y_CO + y_H2O + yH2. 
            This definition can be changed by modifying the c_components parameter.
        interpKind = specifies the method of interpolation that should be used (uses scipy.interp1d). Default = 'cubic'. 
        ximLfrac: fraction of the domain that should contain ximGfrac*100% of the total npoints
        numXim, numXiv: Number of data points between bounds for ξm and ξv, respectively. Default value: 5
        get_data_files_output = used to save time in the event that multiple tables are to be constructed. 
            This should be the output of get_data_files, run with the relevant parameters matching those passed in to this function.
        ximLfrac: (0 to 1), fraction of the xim domain that should contain ximGfrac of the total numXim points
        ximGfrac: (0 to 1), fraction of the total numXim points that should fall inside of ximLfrac of the total domain.
            Example: if ximLfrac = 0.2 and ximGfrac = 0.5, then 50% of the numXim points will fall in the first 20% of the domain.
    """
    if get_data_files_output == None:
        # No processed data passed in: must generate.
        funcs = phiFuncs(path_to_flame_data, Lvals, tvals, file_pattern = file_pattern, c_components = c_components, phi = phi, interpKind = interpKind, get_data_files_output = get_data_files_output)
    else:
        # Use pre-processed data
        funcs = phiFuncs(None, Lvals, tvals, file_pattern = file_pattern, c_components = c_components, phi = phi, interpKind = interpKind, get_data_files_output = get_data_files_output) 

    #---------- Create arrays of ξm and ξv
    if ximLfrac == ximGfrac:
        Xims = np.linspace(0,1,numXim)      #Xim = Mean mixture fraction.
    else:
        nsteps = numXim-1
        dx = np.ones(nsteps)/(nsteps)
        n1 = int(nsteps*ximGfrac)
        if n1 != 0:
            dx1 = ximLfrac/n1
            dx[0:n1] = dx1
            n2  = nsteps - n1
            if n2 != 0:
                dx2 = (1-ximLfrac)/n2
                dx[n1:] = dx2
        Xims = np.zeros(numXim)
        for i in range(len(Xims)-1):
            Xims[i+1] = Xims[i] + dx[i]
        Xims[-1] = 1.0
    Xivs = np.linspace(0,1,numXiv)      #Xiv = Mixture fraction variance. Maximum valid Xiv depends on Xim, so we normalize the values to the maximum
    
    #----------- Table Creation
    table = np.full((numXim, numXiv, len(Lvals), len(tvals)), -1.0)
    for m in range(len(Xims)):                                               #Loop over each value of ξm
        xim = Xims[m]
        for v in range(len(Xivs)):                                           #Loop over each value of ξv
            xivMax = xim*(1-xim)
            xiv = Xivs[v]*xivMax
            for l in range(len(Lvals)):
                for t in range(len(tvals)):
                    phiAvg = LI.IntegrateForPhiBar(xim, xiv, funcs[l][t])    #Calculates phi_Avg
                    table[m,v,l,t] = phiAvg                                  #FINAL INDEXING: table[m,v,l,t]

                            
    #Returns: table itself, then an array of the values of Xims, Xivs, Lvals, and tvals for indexing the table.
    #Ex. table[7][6][5][4] corresponds to Xim = indices[0][7], Xiv = indices[1][6], L = indices[2][5], t = indices[3][4].
    #Note: Xiv is normalized to the maximum. For table[1][2][3][4], the actual value of the variance would be indices[1][6]*Xivmax,
    #      where Xivmax = Xim*(1-Xim) =  indices[0][7]*(1-indices[0][7])
    
    indices = [Xims, Xivs, Lvals, tvals]
    return table, indices

##############################

def createInterpolator(data, inds, interpKind = 'linear'):
    """
    Creates an interpolator using RegularGridInterpolator (rgi).
    Inputs:
        data, inds =  table and indices created by makeLookupTable
        interpKind = interpolation method that RegularGridInterpolator should use. Default = 'linear'
    The returned function is called with func(xim, xiv, L, t)
    """
    xi_means = inds[0]
    xi_vars = inds[1] #Normalized to Xivmax
    Ls = inds[2]
    ts = inds[3]

    interpolator = rgi((xi_means, xi_vars, Ls, ts), data, method = interpKind)

    def func(xim, xiv, L, t):
        # Function returned to the user.
        """
        Interpolates for a value of phi given:
            Xi_mean
            Xi_variance (actual value)
            Length scale
            Time scale
        """
        xiv_max = xim*(1-xim)
        if xiv > xiv_max:
            raise ValueError(f"xiv must be less than xivMax. With xim = {xim}, xiv_max = {xiv_max}. Input xiv = {xiv}")
        if xiv_max == 0:
            if xiv != 0:
                print(f"Warning: xim = {xim}, meaning xiv_max = 0. xiv passed in was {xiv}, but has been overridden to xiv = 0.")
            xiv_norm = 0
        else:
            xiv_norm = xiv/xiv_max
        try:
            return interpolator([xim, xiv_norm, L, t])
        except Exception as e:
            print("Invalid argument passed into interpolator")
            print("Values passed into interpolator: ", xim, xiv_norm, L, t, "( xiv=", xiv, ")") #DEBUGGING
            print(f"Exception raised: {e}")

    
    return func
    
##############################

def Lt_hc(h, c, xim, xiv, hInterp, cInterp, Lbounds, tbounds, norm):
    """
    DEPRECATED: code uses Lt_hc_newton now
    Solves for L,t given:
              h: value of enthalpy
              c: value of progress variable
            xim: mean mixture fraction
            xiv: mixture fraction variance
        hInterp: interpolated function for h(xim, xiv, L, t), created using "createInterpolator"
        cInterp: interpolated function for c(xim, xiv, L, t), created using "createInterpolator"
        Lbounds: tuple containing the minimum and maximum value of L
        tbounds: tuple contianing the minimum and maximum value of L
        norm   := np.max(h_table)/np.max(c_table). Compensates for the large difference in magnitude between typical h and c values.
            
    Returns a tuple of form (L,t)
    This function is to be used for getting values of phi by phi(xim, xiv, [L,t](h,c))
    """
    def solve(Lt):
        L = Lt[0]
        t = Lt[1]

        #----------- Ensure value is within interpolator's bounds
        buffer = 1e-8
        penalty = [1e10, 1e10]
        if L < Lbounds[0]:
            return penalty
            L = Lbounds[0] + buffer
        if L > Lbounds[1]:
            return penalty
            L = Lbounds[1] - buffer
        if t < tbounds[0]:
            return penalty
            t = tbounds[0] + buffer
        if t > tbounds[1]:
            t = tbounds[1] - buffer
            return penalty
        #print("L,t = ", L,t)                    #DEBUGGING

        # Calculate residuals
        resid1 = hInterp(xim, xiv, L, t) - h
        resid2 = (cInterp(xim, xiv, L, t) - c)*norm
        return [resid1, resid2]
    
    #----------- Solve function
    Lavg = np.median(Lbounds)
    tavg = np.median(tbounds)
    ig   = (Lavg, tavg)
    lowBounds = [Lbounds[0], tbounds[0]] #DEBUG: this used to be mean: now this doesn't make sense. median of two values..?
    highBounds = [Lbounds[1], tbounds[1]]
    zero = least_squares(solve, ig, bounds = (lowBounds, highBounds)).x
    print("resids = ", solve(zero))      #DEBUGGING
    return zero

def Lt_hc_newton(hgoal, cgoal, xim, xiv, hInterp, cInterp, Lbounds, tbounds, 
                 norm, detailedWarn:bool = False, maxIter:int = 100, saveSolverStates:bool = False, 
                 useStoredSolution:bool = True):
    """
    Solves for L,t using a 2D Newton solver.
    Params:
          hgoal: value of enthalpy
          cgoal: value of progress variable
            xim: mean mixture fraction
            xiv: mixture fraction variance
        hInterp: interpolated function for h(xim, xiv, L, t), created using "createInterpolator"
        cInterp: interpolated function for c(xim, xiv, L, t), created using "createInterpolator"
        Lbounds: tuple containing the minimum and maximum value of L
        tbounds: tuple contianing the minimum and maximum value of L
        norm   := np.max(h_table)/np.max(c_table). Compensates for the large difference in magnitude between typical h and c values.
        detailedWarn: If set to true, more detailed warnings will be raised when the solver does not converge.    
        maxIter: int, sets a limit for the maximum iterations the solver should make.
        saveSolverStates: bool, if set to True, the solver states will be saved to a file in the folder "solver_data"
        useStoredSolution:bool, if set to False, the solver will not use the last solution as its initial guess. 
            Using the last initial guess (default) is generally good: CFD will solve cell-by-cell, and nearby
            cells are expected to have similar values of phi.
        
    Returns a tuple of form (L,t)
    This function is to be used for getting values of phi by phi(xim, xiv, [L,t](h,c))
    """
    # NOTE: The following functions assume constant xim and xiv. 
    # These parameters are included in F and X to allow a generic function to be used.

    def F(mvlt):
        # Computes h and c residuals from a set mvlt
        hresid = hInterp(*mvlt) - hgoal
        cresid = (cInterp(*mvlt) - cgoal)*norm # norm ensures both h and c are of similar magnitude
        return np.array([hresid, cresid])
    
    def getJac(F, X0, F0=None):
        """Computes the 2x2 Jacobian of F(X) at X
        Params:
            F = F(mvlt) = [h(mvlt) - hSet, c(mvlt)-cSet]
                Example code:
                def F(mvlt):
                    return np.array([hInterp(*mvlt)-hSet, cInterp(*mvlt)-cSet])
            X0 = [xim, xiv L, t]
            F0 = F(X0)
        Returns:
            J = [[dH/dL  dH/dt],
                [dc/dL  dc/dt]]
        """
        # Confirm X is an array
        X0 = np.array(X0)
        #print("GetJac initial point: ", X0) #DEGBUGGING

        # Compute F0 if not passed in 
        if F0 is None:
            F0 = F(X0)

        # Set deltas
        scalar = 1e-8 #square root of the machine precision
        deltaL = np.array([0, 0, X0[2]*scalar+scalar, 0]) # Adding prevents delta = 0
        deltat = np.array([0, 0, 0, X0[3]*scalar+scalar]) # Adding prevents delta = 0
        
        # Compute gradients
        # TO DO: add if-else here. Right now, adding deltas can exceed boundaries...
        if X0[2] + deltaL[2] > Lbounds[1]:
            # Avoid stepping over L boundary when adding deltaL
            J0 = (F0 - F(X0 - deltaL))/deltaL[2]  # = [dH/dL, dc/dL]
        else:
            J0 = (F(X0 + deltaL) - F0)/deltaL[2]  # = [dH/dL, dc/dL]

        if X0[3] + deltat[3] > tbounds[1]:
            # Avoid stepping over t boundary when adding deltat
            J1 = (F0 - F(X0 - deltat))/deltat[3]  # = [dH/dt, dc/dt]
        else:
            J1 = (F(X0 + deltat) - F0)/deltat[3]  # = [dH/dt, dc/dt]

        return np.array([J0, J1]).T[0] # Without this final indexing, the shape is (1, 2, 2) instead of (2, 2)
    
    def cramerSolve(F, X0):
        """
        Solves the system of equations JX=F(X0) for X using Cramer's rule.
        Params:
            F: f(mvlt) = [h(mvlt)-hSet, c(mvlt)-cSet]
            X0: [xim, xiv L, t]
        Returns:
            X = [J^(-1)][F(X0)]
        """
        #print("Cramer Solve") # DEBUGGING
        
        # Confirm X is an array
        X0 = np.array(X0)

        # Solve the system
        F0 = F(X0)
        J = getJac(F, X0, F0)

        #print("F0 = ", F0) #DEBUGGING
        #print("J  = ", J, np.array(J).shape)  #DEBUGGING
        D = (J[0][0]*J[1][1] - J[0][1]*J[1][0])
        D1 = (F0[0]*J[1][1] - J[0][1]*F0[1])
        D2 = (J[0][0]*F0[1] - F0[0]*J[1][0])
        if np.array(D) == 0:
            # Handle nan values. This will happen if the Jacobian is singular. 
            # Physically, this means that the variables are not changing in time, 
            # for example if xim = xiv = 0 (cold, non-reacting mixture). In this 
            # case, we set Lchange and tchange to 0. The remaining solver code
            # will handle the rest.
            Lchange = 0
            tchange = 0
        else:
            Lchange = D1/D
            tchange = D2/D

        # Ensure values returned are floats
        if isinstance(Lchange, np.ndarray):
            Lchange = Lchange[0]
        if isinstance(tchange, np.ndarray):
            tchange = tchange[0]

        # Relax solver: don't allow changes more than a certain fraction of the total domain
        maxFrac = 0.1*xim if xim > 0.1 else 0.01 # Maximum allowable %change relative to the domain
        Lrange = np.abs(max(Lbounds) - min(Lbounds))
        trange = np.abs(max(tbounds) - min(tbounds))
        if Lchange != 0:
            Lsign = Lchange/np.abs(Lchange)
        else:
            Lsign = 1.0
        if tchange != 0:
            tsign = tchange/np.abs(tchange)
        else:
            tsign = 1.0
        
        Lchange = np.min([np.abs(Lchange), Lrange*maxFrac])*Lsign
        tchange = np.min([np.abs(tchange), trange*maxFrac])*tsign
        
        #print("Cramer computed change: ", Lchange, tchange) #DEBUGGING
        return np.array([0, 0, Lchange, tchange])
    
    # Create initial guess
    # Get the directory of the current Python script
    current_dir = os.path.dirname(os.path.abspath(__file__))

    # Check if "file.txt" exists in the same directory
    file_path = os.path.join(current_dir, "newtonsolve_lastsolution.txt")

    Lmed = np.mean(Lbounds)
    tmed = np.mean(tbounds)
    if os.path.isfile(file_path) and useStoredSolution:
        guess = np.loadtxt("newtonsolve_lastsolution.txt")
        guess[0], guess[1] = (xim, xiv)
    else:
        guess   = [xim, xiv, Lmed, tmed]

    # Solve parameters
    tolerance = 1e-8  # Minimum SSE for solver to terminate. This was arbitrarily set to a "low" number.
    states = np.tile(guess, (maxIter, 1))
    errors = np.ones(maxIter)
    Lmin = Lbounds[0]+1e-6
    Lmax = Lbounds[1]-1e-6
    tmin = tbounds[0]+1e-6
    tmax = tbounds[1]-1e-6
    
    # Solve
    i=0 # Use this later to truncate saved data
    for i in range(maxIter):    
        #print("Iteration: ", i) # Feedback
        
        # Compute new point
        change = cramerSolve(F, guess)
        guess -= change

        # Enforce bounds
        #     If the new point is out of bounds, it will first correct the solver to a point very close to the boundary. 
        if guess[2] <= Lbounds[0]:
            guess[2] = Lmin
        elif guess[2] >= Lbounds[1]:
            guess[2] = Lmax
        if guess[3] <= tbounds[0]:
            guess[3] = tmin
        elif guess[3] >= tbounds[1]:
            guess[3] = tmax

        # If solver gets stuck, stick it somewhere random
        if i > 1 and (np.abs(states[i-1] - guess) <= tolerance).all():
            guess[2] = np.random.rand()*(Lmax-Lmin) + Lmin
            guess[3] = np.random.rand()*(tmax-tmin) + tmin
            #print("Solver got stuck: randomized guess.")# Feedback
        elif i > 2 and (np.abs(states[i-2] - guess) <= tolerance).all():
            # Looks back 2 iterations for basic periodic handling.
            guess[2] = np.random.rand()*(Lmax-Lmin) + Lmin
            guess[3] = np.random.rand()*(tmax-tmin) + tmin
            #print("Solver got stuck: randomized guess.")# Feedback
        
        # Compute SSE for this point
        errors[i] = np.sum([err**2 for err in F(guess)])
        states[i] = guess # Record point in case no solution is found
        #print("SSE: ", errors[i])                       # Feedback
        #print("State record: \n", states[i], "\n", states[i-1], "\n", states[i-2]) # Feedback
        #print()                                         # Feedback
        
        # Evaluate if change is small enough to end loop early
        if i > 0 and errors[i] < tolerance:
            break # Tolerance met: end loop

        # Throw warning if max iterations is exceeded
        if i==maxIter-1:
            # If maxIter is reached, return the case with the lowest computed SSE:
            guess = states[errors == np.min(errors)][0]
            if detailedWarn:
                warnings.warn(f"""
                            
                Maximum iterations ({maxIter}) exceeded in Lt_hc_newton solver.
                This indicates that the exact queried [xim, xiv, h, c] point was not found in the table.
                Using best-case computed result:
                    xim = {guess[0]}
                    xiv = {guess[1]}
                    L   = {guess[2]}
                    t   = {guess[3]}, for the desired point
                    h   = {hgoal}
                    c   = {cgoal}, where
                    SSE for this point in the (h,c) -> (L,t) inversion = {errors[i]:.5g}
                    Average SSE for all attepts at this inversion      = {np.mean(errors):5g}
                Result may be inaccurate.
                """)
            else:
                warnings.warn("NewtonSolve did not fully converge, using case with lowest identified SSE.")
            break

    if saveSolverStates:
        # Define the folder and file paths
        folder_name = "solver_data"
        folder_path = os.path.join(os.getcwd(), folder_name)
        subfolder_name = datetime.now().strftime("%Y%m%d")
        subfolder_path = os.path.join(folder_path, subfolder_name)
        file_name = f"Xim_{xim}_Xiv_{xiv}_h_{hgoal:.4g}_c_{cgoal:.4g}.txt"
        file_path = os.path.join(subfolder_path, file_name)

        # Check if the folder exists, and create it if it doesn't
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)
        if not os.path.exists(subfolder_path):
            os.makedirs(subfolder_path)

        # Save the data as a text file in the folder
        np.savetxt(file_path, np.hstack((states[0:i], np.array([errors[0:i]]).T)))

    # Store solution to use as initial guess next time
    np.savetxt("newtonsolve_lastsolution.txt", guess)
    return [guess[2], guess[3]]

def create_table(args):
    """This function is used in phiTable for parallelization. 
    The package used for parallelization ("concurrent") requires that the function being parallelized is defined 
    in the global scope.
    """
    # Generic table-generating function
    return makeLookupTable(*args)

def phiTable(path_to_flame_data, Lvals, tvals, file_pattern = r'^L.*.dat$', c_components = ['H2', 'H2O', 'CO', 'CO2'],
             phi = 'T', interpKind = 'cubic', numXim:int=5, numXiv:int = 5, get_data_files_output = None, 
             parallel:bool = True, detailedWarn:bool = False, ximLfrac = 0.5, ximGfrac = 0.5):
    """
    Creates a table of phi values in terms of Xim, Xiv, L, t
    Inputs:
        path_to_flame_data = path to simulation data relative to the current folder. 
            NOTE: The data headers must be the last commented line before the data begins.
            The code found at https://github.com/BYUignite/flame was used in testing.
        Lvals: values of parameter L used, formatted as a list (ex. [ 0.002, 0.02, 0.2])
        tvals: values of parameter t used, formatted as a list (ex. [ 0    , 1   , 2  ])
        file_pattern = regular expression (regex) to identify which files in the target folder are data files. 
            DEFAULT: r'^L.*.dat$'. This grabs any files that begin with "L" and end with ".dat". 
        c_components = list defining which components' mixture fractions are included in the progress variable. 
            By default, this is set to be ['H2', 'H2O', 'CO', 'CO2']
        phi = single property or list of properties for which values will be tabulated (ex. 'T', 'rho', etc.), case sensitive. Default = 'T'
            Available phi are viewable using "get_data_files(params)[1]".
            NOTE: c (progress variable) is available in the data. By default, c ≡ y_CO2 + y_CO + y_H2O + yH2.
            This definition can be changed by modifying the c_components parameter.
        interpKind = specifies the method of interpolation that should be used (uses scipy.interp1d and RegularGridInterpolator). Default = 'cubic'. 
            Note: this is the kind of interpolation that the phi functions will be created with. Once the tabulated values have been created, the rest of the table
                  will be created with a linear interpolation. This prevents excursions to beyond the system bounds due to sparse data. 
        numXim, numXiv: Number of data points between bounds for ξm and ξv, respectively. Default value: 5
        get_data_files_output = used to save time in the event that multiple tables are to be constructed. 
            This should be the output of get_data_files, run with the relevant parameters matching those passed in to this function.
        parallel:bool = if set to True (default), the code will attempt to create tables in parallel.
        detailedWarn: If set to true, more detailed warnings will be raised when the solver does not converge. 
        ximLfrac: (0 to 1), fraction of the xim domain that should contain ximGfrac of the total numXim points
        ximGfrac: (0 to 1), fraction of the total numXim points that should fall inside of ximLfrac of the total domain.
            Example: if ximLfrac = 0.2 and ximGfrac = 0.5, then 50% of the numXim points will fall in the first 20% of the domain.

    Outputs:
        Array of phi functions phi = phi(xim, xiv, h, c)
        NOTE: if only one phi is specified, if will still be returned in a single-element array.
    """
    # ------------ Pre-processing
    # Confirm h and c aren't in phi
    for p in phi:
        if p=='h' or p=='c':
            print("'h' and 'c' are used as table axis and so cannot be used as phi. Cancelling operation.")
            return None

    # Ensure array-like
    if type(phi) == type('str'):
        phi = [phi,]

    # Retrieve data
    if get_data_files_output == None:
        # No processed data passed in: must generate.
        data_output = get_data_files(path_to_flame_data, Lvals, tvals, file_pattern = file_pattern, c_components = c_components)
    else:
        # Use pre-processed data
        data_output = get_data_files_output

    Lbounds = [min(Lvals), max(Lvals)]
    tbounds = [min(tvals), max(tvals)]

    # ------------ Compute tables, parallel or serial
    if not parallel: # Serial computation
        # Create h & c tables
        h_table, h_indices = makeLookupTable(path_to_flame_data, Lvals, tvals, file_pattern, c_components,\
                                             'h', interpKind, numXim, numXiv, data_output, ximLfrac, ximGfrac)
        c_table, c_indices = makeLookupTable(path_to_flame_data, Lvals, tvals, file_pattern, c_components,\
                                             'c', interpKind, numXim, numXiv, data_output, ximLfrac, ximGfrac)
    
        # Create h & c interpolators
        Ih = createInterpolator(h_table, h_indices, interpKind = 'linear') #These should only be set to cubic with a very dense table.
        Ic = createInterpolator(c_table, c_indices, interpKind = 'linear')
    
        # Create array containing phi tables
        norm = np.max(h_table)/np.max(c_table)
        phiTables = []
        for p in phi:
            # Get base table with phi data
            table, indices = makeLookupTable(path_to_flame_data, Lvals, tvals, file_pattern, c_components,\
                                             p, interpKind, numXim, numXiv, data_output, ximLfrac, ximGfrac)
    
            # Create interpolator for phi
            InterpPhi = createInterpolator(table, indices, interpKind = 'linear')
            
            # Create function phi(xim, xiv, h, c)
            def phi_table(xim, xiv, h, c, maxIter = 100, saveSolverStates = False, useStoredSolution = True):
                # Invert from (h, c) to (L, t), then return interpolated value.
                L, t = Lt_hc_newton(h, c, xim, xiv, Ih, Ic, Lbounds, tbounds, norm, detailedWarn, 
                                    maxIter, saveSolverStates, useStoredSolution)
                return InterpPhi(xim, xiv, L, t)
    
            phiTables.append(phi_table)      
        return phiTables
        
    else: # Parallel computation
        # Import needed packages
        from concurrent.futures import ProcessPoolExecutor
        import concurrent

        phi = np.append(np.array(['h', 'c']), np.array(phi)) # Need to create h and c tables too, so add them  at the beginning. 
        table_args = [(path_to_flame_data, Lvals, tvals, file_pattern, c_components, p, interpKind, numXim, 
                       numXiv, data_output, ximLfrac, ximGfrac) for p in phi] # Arguments for each table's creation

        # Parallel table creation (should be reviewed)
        with ProcessPoolExecutor(mp_context=mp.get_context('fork')) as executor:
            futures = {executor.submit(create_table, args): idx for idx, args in enumerate(table_args)}
            results = {}
            for future in concurrent.futures.as_completed(futures):
                idx = futures[future]
                try:
                    results[idx] = future.result()
                except Exception as e:
                    print(f"Table creation for index {idx} (phi = {phi[idx]}) generated an exception: {e}")

        # Create h & c interpolators -- These should only be set to cubic interpolation with a very dense table.
        Ih = createInterpolator(results[0][0], results[0][1], interpKind = 'linear')
        Ic = createInterpolator(results[1][0], results[1][1], interpKind = 'linear')
        
        phiTables = []
        norm = np.max(results[0][0])/np.max(results[1][0])
        if np.isnan(norm):
            norm = np.average(results[0][0])/np.max(results[1][0])
        if np.isnan(norm):
            norm = 1.0
        for i in range(len(phi)-2):
            InterpPhi = createInterpolator(*results[i+2])
            # Create function phi(xim, xiv, h, c)
            def phi_table(xim, xiv, h, c, maxIter:int = 100, saveSolverStates = False, useStoredSolution = True):
                # Invert from (h, c) to (L, t)
                L, t = Lt_hc_newton(h, c, xim, xiv, Ih, Ic, Lbounds, tbounds, norm, detailedWarn, 
                                    maxIter, saveSolverStates, useStoredSolution)
                return InterpPhi(xim, xiv, L, t)
    
            phiTables.append(phi_table) 
        return phiTables