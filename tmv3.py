# TableMaker version 3
# Main Author: Jared Porter
# Contributors: Dr. David Lignell, Jansen Berryhill
# Some revisions completed with GitHub Copilot under the student license

import numpy as np
from scipy.interpolate import interp1d
from scipy.optimize import minimize, minimize_scalar
import os
import warnings
from glob import glob
from re import match, search
import LiuInt as LI # Package with functions for integrating over the BPDF, parameterized by xi_avg and xi_variance
from scipy.interpolate import RegularGridInterpolator as rgi
from datetime import datetime
import multiprocessing as mp

############################## tableMakerv2

def compute_progress_variable(data, header, c_components = ['H2', 'H2O', 'CO', 'CO2']):
    """
    Progress variable is defined as the sum of the mass fractions of a specified set of c_components.
    This function computes the flame progress variable using:
        data = Data from a flame simulation. Each row corresponds to a specific property.
            In the case of this package, this data array is "transposed_file_data" inside the function "get_file_data"
                ex. data[0] = array of temperature data.
        header = 1D array of column headers, denoting which row in "data" corresponds to which property.
            ex. If header[0] = "Temp", then data[0] should be temperature data.
        c_components = list defining which components' mass fractions are included in the progress variable. 
            By default, this is set to be ['H2', 'H2O', 'CO', 'CO2']
            The strings in the list should each match a string used in 'header'
    """
    #---------- Determine where the c_components are in 'data'
    indices = np.ones(len(c_components), dtype = np.int8)*-1
    for i in range(len(header)):                # For each element in the header, 
        for y in range(len(c_components)):      # Check for a match among the passed-in c_components
            if header[i]==c_components[y].replace(" ",""):
                indices[y] = int(i)             # Indices must be strictly integers
                
    # Confirm all indices were located
    for j, ind in enumerate(indices):
        if ind == -1:
            raise ValueError(f"No match found for {c_components[j]}.")

    #---------- Compute progress variable
    c = np.zeros(len(data[0]))        # Initialize c array
    for d in range(len(data[0])):     # For each column,
        sum = 0
        for index in indices:         # For each of the components specified, 
            sum += data[index,d]      # Sum the mass fractions of each component
        c[d] = sum
    return c 

##############################

def get_data_files(path_to_data, Lvals, tvals, file_pattern = r'^L.*.dat$', \
                   c_components = ['H2', 'H2O', 'CO', 'CO2']):
    """
    Reads and formats data computed by a square grid of flame simulations.
    Inputs: 
        path_to_data = path to simulation data relative to the current folder. 
            NOTE: The data headers must be the last commented line before the data begins.
            The code found at https://github.com/BYUignite/flame was used in testing. 
        Each file will have been run under an array of conditions L,t:
        Lvals: values of parameter L used, formatted as a list (ex. [ 0.002, 0.02, 0.2])
        tvals: values of parameter t used, formatted as a list (ex. [ 0    , 1   , 2  ])
        file_pattern = regular expression (regex) to identify which files in the target folder are data files.  
            DEFAULT: r'^L.*.dat$'. This grabs any files that begin with "L" and end with ".dat". 
        c_components = list defining whih components' mass fractions are included in the progress variable. 
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
        # This indexing assumes the same # of time scales were run for each length scale
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
                            #Remove preemtive numbers in the column labels, then store column label (assumes labels formatted as colNum_property, e.g. 0_T)
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
        c = compute_progress_variable(transposed_file_data, header, c_components = c_components)
        transposed_file_data = np.vstack((transposed_file_data, c))   #Stacks this array of progress variable values as the last row 
        
        #---------- Arrange data by l and t indices
        all_data[l,t] = transposed_file_data
    
    #all_data is indexed using all_data[Lval][tval][column# = Property][row # = data point]
    print("Completed data import ('get_data_files')")
    return all_data, headers, extras

##############################

def phi_funcs(path_to_flame_data, Lvals, tvals, file_pattern = r'^L.*.dat$', c_components = ['H2', 'H2O', 'CO', 'CO2'],
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
             Example1: phi_funcs(path, phi = 'T', Lt = (0,1)): returns the interpolated T(ξ) function ONLY from the data in the file from Lvals[0], tvals[1]. \n
             Example2: phi_funcs(path, phi = 'T'): returns an array containing the interpolated T(ξ) functions from every file in the directory\n
             Note that the values in this tuple are not values of L and t, but rather indexes of Lvals and tvals.\n
        mix_frac_name = name of the column header for mixture fraction. Default value: "mixf"\n
        interpKind = specifies the method of interpolation that should be used (uses scipy.interp1d). Default = 'cubic'. \n
        get_data_files_output = used to save time in the event that multiple tables are to be constructed. \n
            This should be the output of get_data_files, run with the relevant parameters matching those passed in to this function.\n
        
    Outputs:\n
        The output type of phi_funcs will depend on the input parameter "fileName":\n
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
    if xi_col == -1:
        # Xi wasn't found.
        raise ValueError(f"Mixture fraction ('{mix_frac_name}') was not found among data columns.")

    #---------- Interpolate phi(xi)
    phi_funcs = np.empty((len(Lvals),len(tvals)), dtype=np.ndarray)
    if Lt == False:
        #User did not specify file: must interpolate for every file
        for l in range(len(data)):
            for t in range(len(data[l])):
                xis = data[l][t][xi_col]
                phis = data[l][t][phi_col]
                phi_funcs[l][t] = interp1d(xis, phis, kind = interpKind)
        print(f"phi_funcs for {phi} created using {len(Lvals)*len(tvals)} files.")
        return phi_funcs
    else:
        #User specified a file
        xis = data[Lt[0]][Lt[1]][xi_col]
        phis = data[Lt[0]][Lt[1]][phi_col]
        print(f"phi_funcs for {phi} created using {len(Lvals)*len(tvals)} files.")
        return interp1d(xis, phis, kind = interpKind)

##############################
    
def make_lookup_table(path_to_flame_data, Lvals, tvals, file_pattern = r'^L.*.dat$', c_components = ['H2', 'H2O', 'CO', 'CO2'],
                    phi = 'T', interpKind = 'cubic', numXim:int=150, numXiv:int = 30, get_data_files_output = None,
                    ximLfrac = 0.5, ximGfrac = 0.5):
    # Note: arguments later on unpack args for this function using *args. Only add parameters to the end of the
    # current list of parameters. If removing parameters, be sure to revise calls of this function later in the code.
    """
    Creates a 4D lookup table of phi_avg data. Axis are ξm, ξv, L, and t. 
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
    # If get_data_files_output is not provided, the function will call get_data_files to generate the data.
    funcs = phi_funcs(path_to_flame_data, Lvals, tvals, file_pattern = file_pattern, c_components = c_components, \
                      phi = phi, interpKind = interpKind, get_data_files_output = get_data_files_output)

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
    markers = (len(Xims)*np.array([0.01, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9])).astype(int) # Xim indices at which to notify the user
    for m in range(len(Xims)):                                               #Loop over each value of ξm
        if m in markers:
            print(f"{phi} table {int(m/len(Xims)*100)}% complete.")
        xim = Xims[m]
        for v in range(len(Xivs)):                                           #Loop over each value of ξv
            xiv = Xivs[v]*xim*(1-xim)
            for l in range(len(Lvals)):
                for t in range(len(tvals)):
                    phiAvg = LI.IntegrateForPhiBar(xim, xiv, funcs[l][t])    #Calculates phi_Avg
                    table[m,v,l,t] = phiAvg                                  #FINAL INDEXING: table[m,v,l,t]

                            
    #Returns: table itself, then an array of the values of Xims, Xivs, Lvals, and tvals for indexing the table.
    #Ex. table[7][6][5][4] corresponds to Xim = indices[0][7], Xiv = indices[1][6], L = indices[2][5], t = indices[3][4].
    #Note: Xiv is normalized to the maximum. For table[1][2][3][4], the actual value of the variance would be indices[1][6]*Xivmax,
    #      where Xivmax = Xim*(1-Xim) =  indices[0][7]*(1-indices[0][7])
    
    indices = [Xims, Xivs, Lvals, tvals]
    print(f"Lookup table for phi = {phi} completed.")
    return table, indices

##############################

def create_interpolator_mvlt(data, inds, interpKind = 'linear', extrapolate = True):
    """
    Creates an interpolator using RegularGridInterpolator (rgi).
    Inputs:
        data, inds =  table and indices created by make_lookup_table
        interpKind = interpolation method that RegularGridInterpolator should use. Default = 'linear'
    The returned function is called with func(xim, xiv, L, t)
    """
    xi_means = inds[0]
    xi_vars = inds[1] #Normalized to Xivmax
    Ls = inds[2]
    ts = inds[3]
    
    if extrapolate:
        interpolator = rgi((xi_means, xi_vars, Ls, ts), data, method = interpKind, bounds_error = False, fill_value=None)
    else:
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
            print("Values passed into interpolator: ", xim, xiv_norm, L, t, "( xiv=", xiv, ")") # DEBUGGING
            print(f"Exception raised: {e}")
    
    return func
    
##############################
def create_hsensFunc(path_to_hsens, nxims = 150, nxivs = 30):
    # Parse needed data
    hsensdata = np.loadtxt(path_to_hsens, skiprows = 1)
    hsensFunc = interp1d(hsensdata[:,0], hsensdata[:,1], kind = 'linear') # Sensible enthalpy (J/kg) as a function of mixf
    # Make hsens table: hsens(xim, xiv)
    xims = np.linspace(0, 1, nxims)
    xivs = np.linspace(0, 1, nxivs)
    hsensTable = np.zeros((nxims, nxivs))
    for i in range(nxims):
        ximVal = xims[i]
        for j in range(nxivs):
            xivVal = xivs[j]*ximVal*(1-ximVal)
            hsensTable[i,j] = LI.IntegrateForPhiBar(ximVal, xivVal, hsensFunc)
    interpolator = rgi((xims, xivs), hsensTable, method = 'linear')  # No extrapolation
    
    def hsensFunc(xim, xiv):
        # Returns hsens for a value of xim and xiv
        xivmax = xim*(1-xim)
        xiv = max(0, min(xiv*xim*(1-xim), xivmax))
        return interpolator([xim, xiv])
    return hsensFunc

hsensFunc = None
global Lt_from_hc_GammaChi
def Lt_from_hc_GammaChi(hgoal, cgoal, xim, xiv, hInterp, cInterp, Lbounds, tbounds, 
                        norm, useStoredSolution:bool = True, path_to_hsens = './data/ChiGammaTablev3/hsens.dat', 
                        gammaValues = None, numXim=150, numXiv=30):
    """
    Solves for (L,t) given values of (h,c) in the gamma-chi formulation of the table.
    This table is constructed so that file has:
        1) an imposed heat loss parameter gamma, defined as (h_{adiabatic} - h)/h_{sensible, firstFile}
        2) a diffusive strain parameter chi, used in the opposed jet formulation of a flamelet.
    Because gamma is independent of chi, gamma may be determined first using thermodynamic data. 
    Then, chi may be determined using interpolated data from the table. 
    Note: chi:L :: gamma:t

    Function parameters:
        hgoal: value of enthalpy
        cgoal: value of progress variable
        xim: mean mixture fraction
        xiv: mixture fraction variance
        hInterp: interpolated function for h(xim, xiv, L, t), created using "create_interpolator_mvlt"
        cInterp: interpolated function for c(xim, xiv, L, t), created using "create_interpolator_mvlt"
        Lbounds: tuple containing the minimum and maximum value of L
        tbounds: tuple contianing the minimum and maximum value of L
        norm   := np.max(h_table)/np.max(c_table). Compensates for the large difference in magnitude between typical h and c values.
        useStoredSolution:bool, if set to False, the solver will not use the last solution as its initial guess. 
            Using the last initial guess (default) is generally good: CFD will solve cell-by-cell, and nearby
            cells are expected to have similar values of phi.
        path_to_hsens: path to a file containing the sensible enthalpy data (col1 = mixf, col2 = h[J/kg])
        gammaValues: array of gamma values corresponding to the index of the t values loaded in. 
            For example, if tvals = [0, 1, 2, ...], gammaValues = [0, 0.05, 0.1, ...] would be appropriate.
            
    Returns a tuple of form (L,t)
    This function is to be used for getting values of phi by phi(xim, xiv, [L,t](h,c))
    """
    gammaToIndex = interp1d(gammaValues, np.arange(len(gammaValues)), 
                            kind = 'linear', bounds_error = None, fill_value = 'extrapolate') # Converts gamma to an index

    #----- Determine gamma
    h0 = hInterp(0, 0, Lbounds[0], tbounds[0])  # Enthalpy of pure fuel
    h1 = hInterp(1, 0, Lbounds[0], tbounds[0])  # Enthalpy of pure oxidizer
    ha = h0*(1-xim) + h1*xim                    # Adiabatic enthalpy    
    global hsensFunc
    if hsensFunc is None:
        hsensFunc = create_hsensFunc(path_to_hsens, nxims = numXim, nxivs = numXiv)
    
    gamma = (ha - hgoal)/hsensFunc(xim, xiv)         # Heat loss parameter
    
    t = gammaToIndex(gamma)                     # Time scale index
    if isinstance(t, np.ndarray):
        t = t[0]

    #----- Use gamma to determine chi
    def obj(L):
        if isinstance(L, np.ndarray):
            L = L[0]
        return cInterp(xim, xiv, L, t) - cgoal
    
    # Get the directory of the current Python script
    try:
        current_dir = os.path.dirname(os.path.abspath(__file__))
    except:
        # Get the directory of the current jupyter notebook
        current_dir = os.path.dirname(os.path.abspath(''))

    # Check if previous solution was stored in the same directory
    file_path = os.path.join(current_dir, "chiGamma_lastsolution.txt")
    if os.path.isfile(file_path) and useStoredSolution:
        # Use the last solution as the initial guess
        L = np.loadtxt(file_path)
        if L < Lbounds[0] or L > Lbounds[1]:
            # If the last solution is out of bounds, use the midpoint
            guess = Lbounds[0] + (Lbounds[1]-Lbounds[0])/2
        else:
            guess = L
    else:
        guess   = Lbounds[0] + (Lbounds[1]-Lbounds[0])/2

    #L = fsolve(obj, guess)[0]
    L = minimize(lambda L: np.abs(obj(L)), guess, method = 'Nelder-Mead').x[0]
    #L = minimize_scalar(lambda L: np.abs(obj(L)), guess, method = 'bounded', bounds=Lbounds).x[0]

    np.savetxt("chiGamma_lastsolution.txt", np.array([L]))
    return [L, t]

def Lt_from_hc_newton(hgoal, cgoal, xim, xiv, hInterp, cInterp, Lbounds, tbounds, 
                 norm, detailedWarn:bool = False, maxIter:int = 100, saveSolverStates:bool = False, 
                 useStoredSolution:bool = True, LstepParams = [0.25, 0.01, 0.003], 
                 tstepParams = [0.25, 9.5, 0.02]):
    """
    DEPRECATED: Currently unused. Use Lt_from_hc_GammaChi instead.
    
    Solves for (L,t) given values of (h,c) using a 2D Newton solver.
    Params:
          hgoal: value of enthalpy
          cgoal: value of progress variable
            xim: mean mixture fraction
            xiv: mixture fraction variance
        hInterp: interpolated function for h(xim, xiv, L, t), created using "create_interpolator_mvlt"
        cInterp: interpolated function for c(xim, xiv, L, t), created using "create_interpolator_mvlt"
        Lbounds: tuple containing the minimum and maximum value of L
        tbounds: tuple contianing the minimum and maximum value of L
        norm   := np.max(h_table)/np.max(c_table). Compensates for the large difference in magnitude between typical h and c values.
        detailedWarn: If set to true, more detailed warnings will be raised when the solver does not converge.    
        maxIter: int, sets a limit for the maximum iterations the solver should make.
        saveSolverStates: bool, if set to True, the solver states will be saved to a file in the folder "solver_data"
        useStoredSolution:bool, if set to False, the solver will not use the last solution as its initial guess. 
            Using the last initial guess (default) is generally good: CFD will solve cell-by-cell, and nearby
            cells are expected to have similar values of phi.
        LstepParams: array of parameters used to relax the solver
            LstepParams[0] = 0.25; normal max step size (% of domain)
            LstepParams[1] = 0.01; threshold value of L, below which the max step size is reduced to
            LstepParams[2] = 0.003; reduced max step size (% of domain)
        tstepParams: array of parameters used to relax the solver
            tstepParams[0] = 0.25; normal max step size (% of domain)
            tstepParams[1] = 9.5; threshold value of t, above which the max step size is reduced to
            tstepParams[2] = 0.02; reduced max step size (% of domain)
        
    Returns a tuple of form (L,t)
    This function is to be used for getting values of phi by phi(xim, xiv, [L,t](h,c))
    """
    # Note: The following functions assume constant xim and xiv. 
    # These parameters are included in F and X to allow a generic function to be used.

    def F(mvlt):
        # Computes h and c residuals from a set mvlt
        hresid = hInterp(*mvlt) - hgoal
        cresid = (cInterp(*mvlt) - cgoal)*norm # norm ensures both h and c are of similar magnitude
        return np.array([hresid, cresid])
    
    def get_jac(F, X0, F0=None):
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

        # Compute F0 if not provided
        if F0 is None:
            F0 = F(X0)

        # Set deltas
        scalar = 1e-8 #square root of machine precision
        deltaL = np.array([0, 0, X0[2]*scalar+scalar, 0]) # Adding prevents delta = 0
        deltat = np.array([0, 0, 0, X0[3]*scalar+scalar]) # Adding prevents delta = 0
        
        # Compute gradients
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
    
    def cramer_solve(F, X0):
        """
        Solves the system of equations JX=F(X0) for X using Cramer's rule.
        Params:
            F: f(mvlt) = [h(mvlt)-hSet, c(mvlt)-cSet]
            X0: [xim, xiv L, t]
        Returns:
            X = [J^(-1)][F(X0)]
        """
        # Confirm X is an array
        X0 = np.array(X0)

        # Solve the system
        F0 = F(X0)
        J = get_jac(F, X0, F0)

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
        maxFrac_L = LstepParams[0] if X0[2]>LstepParams[1] else LstepParams[2] # Maximum allowable %change in L relative to the domain
        maxFrac_t = tstepParams[0] if X0[3]<tstepParams[1] else tstepParams[2] # Maximum allowable %change in L relative to the domain
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
        
        Lchange = np.min([np.abs(Lchange), Lrange*maxFrac_L])*Lsign
        tchange = np.min([np.abs(tchange), trange*maxFrac_t])*tsign
        
        return np.array([0, 0, Lchange, tchange])
    
    # Create initial guess
    # Get the directory of the current Python script
    try:
        current_dir = os.path.dirname(os.path.abspath(__file__))
    except:
        # Get the directory of the current jupyter notebook
        current_dir = os.path.dirname(os.path.abspath(''))

    # Check if "file.txt" exists in the same directory
    file_path = os.path.join(current_dir, "newtonsolve_lastsolution.txt")

    Lmin = Lbounds[0]+1e-6
    Lmax = Lbounds[1]-1e-6
    tmin = tbounds[0]+1e-6
    tmax = tbounds[1]-1e-6
    Lstart = (Lmax-Lmin)*0.25+Lmin
    tstart = (tmax-tmin)*0.9+tmin
    if os.path.isfile(file_path) and useStoredSolution:
        guess = np.loadtxt("newtonsolve_lastsolution.txt")
        guess[0], guess[1] = (xim, xiv)
    else:
        guess   = [xim, xiv, Lstart, tstart]

    # Solve parameters
    tolerance = 1e-8  # Minimum SSE for solver to terminate. This was arbitrarily set to a "low" number.
    states = np.tile(guess, (maxIter, 1))
    errors = np.ones(maxIter)
    
    # Solve
    i=0    # Store index: used later to truncate saved data
    for i in range(1, maxIter):
        
        # Compute new point
        change = cramer_solve(F, guess)
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
        if i>1 and np.abs(states[i-1][2] - guess[2]) <= tolerance:
            guess[2] = np.random.rand()*(Lmax-Lmin) + Lmin
        elif i>2 and np.abs(states[i-2][2] - guess[2]) <= tolerance:
            guess[2] = np.random.rand()*(Lmax-Lmin) + Lmin
        if i>2 and np.abs(states[i-1][3] - guess[3]) <= tolerance:
            guess[3] = np.random.rand()*(tmax-tmin) + tmin
        elif i>2 and np.abs(states[i-2][3] - guess[3]) <= tolerance:
            guess[3] = np.random.rand()*(tmax-tmin) + tmin
        
        states[i] = guess # Record point in case no solution is found
        errors[i] = np.sum([err**2 for err in F(guess)]) # SSE
        
        # Evaluate convergence
        if errors[i] <= tolerance:
            break # Tolerance met: end loop

        # Throw warning if max iterations is exceeded
        if i==maxIter:
            # If maxIter is reached, return the case with the lowest computed SSE:
            guess = states[errors == np.min(errors)][0]
            if detailedWarn:
                warnings.warn(f"""
                            
                Maximum iterations ({maxIter}) exceeded in Lt_from_hc_newton solver.
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

def create_table_aux(args):
    """Auxiliary function used in phi_mvhc for parallelization. 
    The package used for parallelization ("concurrent") requires that the function being parallelized is defined 
    in the global scope.
    """
    # Generic table-generating function
    return make_lookup_table(*args)

def phi_mvhc(path_to_flame_data, Lvals, tvals, file_pattern = r'^L.*.dat$', c_components = ['H2', 'H2O', 'CO', 'CO2'],
             phi = 'T', interpKind = 'cubic', numXim:int=5, numXiv:int = 5, get_data_files_output = None, 
             parallel:bool = True, detailedWarn:bool = False, ximLfrac = 0.5, ximGfrac = 0.5):
    """
    Creates a table of phi values in terms of Xim, Xiv, h, and c
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
        interpKind = specifies the method of interpolation that should be used for functions phi(xi) (uses scipy.interp1d). Default = 'cubic'. 
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
        phi_mvhc_arr: Array of phi functions phi = phi(xim, xiv, h, c)
            NOTE: if only one phi is specified, if will still be returned in a single-element array.
        tableArr: array of [table, indices] for each phi, beginning with h and c.

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
    # Enable solvers to be accessed when this code is imported as a package
    global Lt_from_hc_newton
    global Lt_from_hc_GammaChi
    ####### Serial computation
    if not parallel: 
        # Create h & c tables
        h_table, h_indices = make_lookup_table(path_to_flame_data, Lvals, tvals, file_pattern, c_components,\
                                             'h', interpKind, numXim, numXiv, data_output, ximLfrac, ximGfrac)
        c_table, c_indices = make_lookup_table(path_to_flame_data, Lvals, tvals, file_pattern, c_components,\
                                             'c', interpKind, numXim, numXiv, data_output, ximLfrac, ximGfrac)
    
        # Create h & c interpolators
        Ih = create_interpolator_mvlt(h_table, h_indices, interpKind = 'linear') # These can only be set to cubic with a very dense table.
        Ic = create_interpolator_mvlt(c_table, c_indices, interpKind = 'linear') # Otherwise, the values may be nonsensical.
    
        # Create array containing phi tables
        norm = np.max(np.abs(h_table))/np.max(c_table)
        phi_mvhc_arr = []
        tableArr = [[h_table, h_indices], [c_table, c_indices]]
        for p in phi:
            # Get base table with phi data
            table, indices = make_lookup_table(path_to_flame_data, Lvals, tvals, file_pattern, c_components,\
                                             p, interpKind, numXim, numXiv, data_output, ximLfrac, ximGfrac)
    
            # Create interpolator for phi
            InterpPhi = create_interpolator_mvlt(table, indices, interpKind = 'linear')
            
            # Create function phi(xim, xiv, h, c)
            def phi_table(xim, xiv, h, c, maxIter = 100, saveSolverStates = False, useStoredSolution = True, 
                          LstepParams = [0.25, 0.01, 0.003], tstepParams = [0.25, 9.5, 0.02], solver = 'newton', 
                          path_to_hsens = None, gammaValues = None):
                # Invert from (h, c) to (L, t), then return interpolated value.
                if solver == 'newton':
                    L, t = Lt_from_hc_newton(h, c, xim, xiv, Ih, Ic, Lbounds, tbounds, norm, detailedWarn, 
                                             maxIter, saveSolverStates, useStoredSolution, LstepParams, tstepParams)
                elif solver == 'gammaChi':
                    L, t = Lt_from_hc_GammaChi(h, c, xim, xiv, Ih, Ic, Lbounds, tbounds, norm, useStoredSolution, path_to_hsens = path_to_hsens,
                                               gammaValues = gammaValues, numXim=numXim, numXiv=numXiv)
                else:
                    raise ValueError("Invalid solver type. Must be 'newton' or 'gammaChi'.")
                
                return InterpPhi(xim, xiv, L, t)
    
            phi_mvhc_arr.append(phi_table)
            tableArr.append([table, indices])
        return phi_mvhc_arr, tableArr
        
    ####### Parallel computation
    else: 
        # Import needed packages
        from concurrent.futures import ProcessPoolExecutor
        import concurrent

        phi = np.append(np.array(['h', 'c']), np.array(phi)) # Need to create h and c tables too, so add them at the beginning. 
        table_args = [(path_to_flame_data, Lvals, tvals, file_pattern, c_components, p, interpKind, numXim, 
                       numXiv, data_output, ximLfrac, ximGfrac) for p in phi] # Arguments for each table's creation

        # Parallel table creation (should be reviewed)
        with ProcessPoolExecutor(mp_context=mp.get_context('fork')) as executor:
            futures = {executor.submit(create_table_aux, args): idx for idx, args in enumerate(table_args)}
            results = {}
            for future in concurrent.futures.as_completed(futures):
                idx = futures[future]
                try:
                    results[idx] = future.result()
                except Exception as e:
                    print(f"Table creation for index {idx} (phi = {phi[idx]}) generated an exception: {e}")

        # Create h & c interpolators -- These should only be set to cubic interpolation with a very dense table.
        Ih = create_interpolator_mvlt(results[0][0], results[0][1], interpKind = 'linear')
        Ic = create_interpolator_mvlt(results[1][0], results[1][1], interpKind = 'linear')
        
        phi_mvhc_arr = []
        tableArr = [[results[0][0], results[0][1]], [results[1][0], results[1][1]]] # [[h_table, h_indices], [c_table, c_indices]]
        norm = np.max(np.abs(results[0][0]))/np.max(results[1][0])
        if np.isnan(norm):
            norm = np.average(np.abs(results[0][0]))/np.max(results[1][0])
        if np.isnan(norm):
            norm = 1.0

        # Create functions for phi(xim, xiv, h, c)
        for i in range(len(phi)-2):
            tableI, indsI = results[i+2]
            InterpPhi = create_interpolator_mvlt(tableI, indsI, interpKind = 'linear')
            
            # Create function phi(xim, xiv, h, c)
            def create_phi_table(interp_phi):
                def phi_table(xim, xiv, h, c, maxIter:int = 100, saveSolverStates = False, useStoredSolution = True,
                        LstepParams = [0.25, 0.01, 0.003], tstepParams = [0.25, 9.5, 0.02], solver = 'gammaChi',
                        path_to_hsens = None, gammaValues = None):
                    # Invert from (h, c) to (L, t), then return interpolated value.
                    if solver == 'newton':
                        L, t = Lt_from_hc_newton(h, c, xim, xiv, Ih, Ic, Lbounds, tbounds, norm, detailedWarn, 
                                    maxIter, saveSolverStates, useStoredSolution, LstepParams, tstepParams)
                    elif solver == 'gammaChi':
                        L, t = Lt_from_hc_GammaChi(h, c, xim, xiv, Ih, Ic, Lbounds, tbounds, norm, useStoredSolution, path_to_hsens = path_to_hsens,
                                    gammaValues = gammaValues, numXim=numXim, numXiv=numXiv)
                    else:
                        raise ValueError("Invalid solver type. Must be 'newton' or 'gammaChi'.")
                    
                    return interp_phi(xim, xiv, L, t)
                return phi_table

            phi_mvhc_arr.append(create_phi_table(InterpPhi)) 
            tableArr.append([tableI, indsI])
        return phi_mvhc_arr, tableArr