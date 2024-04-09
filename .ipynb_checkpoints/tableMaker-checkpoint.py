import numpy as np
from scipy.interpolate import interp1d
from scipy.integrate import quad
from scipy.optimize import fsolve, ridder, least_squares
import os
from glob import glob
from re import match, search
from statistics import variance
import LiuInt as LI #Package with functions for integrating over the BPDF, parameterized by xi_avg and xi_variance

##############################

def computeProgressVariable(data, header, components = ['H2', 'H2O', 'CO', 'CO2']):
    """
    Progress variable is defined as the sum of the mole fractions of a specified set of components.
    This function computes the flame progress variable using:
        data = Data from a flame simulation. Each row corresponds to a specific property.
            In the case of this package, this data array is "transposed_file_data" inside the function "get_file_data"
                ex. data[0] = array of temperature data.
        header = 1D array of column headers, denoting which row in "data" corresponds to which property.
            ex. If header[0] = "Temp", then data[0] should be temperature data.
        
    """
    indices = np.empty(len(components), dtype = np.int8)
    
    #---------- Determine where the components are in 'data'
    for i in range(len(header)):
        for y in range(len(components)):
            if header[i].lower()==components[y].replace(" ","").lower():
                indices[y] = int(i)           #Indices must be strictly integers (ex. 5, not 5.0)

    #---------- Compute progress variable
    c = np.zeros(len(data[0]))                #Initialize c array
    for d in range(len(data[0])):             #For each set of data points (each column),
        sum = 0
        for index in indices:
            #print(indices)
            #print(d)
            sum += data[index,d]              #Sum the mole fractions of each component
        c[d] = sum
    return c 

##############################

def get_data_files(path_to_data, Lvals, tvals, file_pattern = r'^L.*.dat$'):
    """
    Reads and formats data computed by a flame simulation.
    Inputs: 
        path_to_data = path to simulation data relative to the current folder. The data headers should be the last commented line before the data begins.
            The code found at https://github.com/BYUignite/flame was used in testing. 
        Each file will have been run under an array of conditions L,t. The following input parameters:
        Lvals: values of parameter L used, in array format (ex. [ 0.002, 0.02, 0.2])
        tvals: values of parameter t used, in array format (ex. [ 0    , 1   , 2  ])
        file_pattern = regular expression to identify which files in the target folder are data files. 
            - DEFAULT: r'^L.*.dat$'. This grabs any files that begin with "L" and end with ".dat". 
            
    Outputs:
        all_data = an array with the data from each file, indexed using all_data[Lval][tval][column# = Property][row # = data point]
        headers  = an array with the column labels from each file. 
            - Each file should have the same columns labels for a given instance of a simulation, but all headers are redundantly included.
        extras   = an array storing any extra information included at the beginning of each file.
            - This data is not processed in any way by this code and is included only for optional accessibility
    """
    #---------- Check if the provided path is a valid directory
    if not os.path.isdir(path_to_data):
        print(f"Error: {path_to_data} is not a valid directory.")
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
    all_data = np.empty((len(Lvals),len(tvals)), dtype=np.ndarray)    #initialize to grab data values
    headers  = np.empty((len(Lvals),len(tvals)), dtype=np.ndarray)    #Initialize to store headers
    extras   = np.empty((len(Lvals),len(tvals)), dtype=np.ndarray)    #initialize to store extra info before header

    #---------- Grab and store data
    for i in range(len(data_files)):
        l = i//len(tvals)
        t = i %len(tvals)
        
        file = data_files[i]
        with open(file, 'r') as f:
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
                            #Remove preemtive numbers in the column labels, then store column label.
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
        #NOTE: the following lines could be achieved with np.loadtxt(). This would require writing a modified file that only
        #      had the data and headers, then reading it back in. Because we've already read in the lines to extract the headers, 
        #      we can extract the data manually with a few extra lines of code.
        file_data = np.empty(len(raw_data[0].split()))     # will hold the data for this file
        
        for row in raw_data:
            numbers = np.array([float(val) for val in row.split()])
            file_data = np.vstack((file_data,numbers)) #Adds each new row of data as a new row in file_data
        file_data = file_data[1:file_data.size]        #Get rid of first column (which is empty and only used for initialization)

        #---------- Transpose data so that each row is data for a certain property (ex. one row is temperature data, one is density, etc.)
        transposed_file_data = file_data.T

        #---------- Add a row with progress variable (c)
        c = computeProgressVariable(transposed_file_data, header)     #Gets an array of values of progress variable across the domain
        transposed_file_data = np.vstack((transposed_file_data, c))   #Stacks this array of progress variable values as the last row 
        
        #---------- Arrange data by l and t indices
        all_data[l,t] = transposed_file_data
    
    #all_data is indexed using all_data[Lval][tval][column# = Property][row # = data point]
    return all_data, headers, extras

##############################

#----- Testing get_data_files
# Lvals = [0.002, 0.004, 0.006, 0.008, 0.02, 0.04, 0.2]
# tvals = np.arange(0,11,1)
# test = get_data_files(path, Lvals, tvals)[0][0][1]
# test[len(test)-1]
# test[0]

##############################

def phiFuncs(path_to_flame_data, Lvals, tvals, phi = 'T', Lt = False, mix_frac_name = "mixf", get_data_files_output = None):
    """
    Returns an array of interpolated functions phi(ξ) where phi is any property of the flame.
    Inputs:
        path_to_flame_data = the path on the local machine pointing to the flame simulation code's data file.
            Data generated from the code found at https://github.com/BYUignite/flame was used in testing. 
        Lvals = values of parameter L used, in array format (ex. [ 0.002, 0.02, 0.2])
        tvals = values of parameter t used, in array format (ex. [ 0    , 1   , 2  ])
        phi = desired property (ex. 'T', 'rho', etc.) Available phi are viewable using "get_data_files(params)[1]".
            NOTE: c (progress variable) is available in the data. Currently, c ≡ y_CO2 + y_CO + y_H2O + yH2. This definition can be changed
                  by modifying the "computeProgressVariable" function. 
        Lt = Tuple with values of L and t. If set to false (default), the output will be an array of the functions phi(ξ) for all datafiles. 
             Otherwise, this parameter determines which specific file should be used. 
             Example1: phiFuncs(path, phi = 'T', fileName = (0,1)): returns the interpolated T(ξ) function ONLY from the data in the file from Lvals[0], tvals[1]. 
             Example2: phiFuncs(path, phi = 'T'): returns an array containing the interpolated T(ξ) functions from every file in the directory
        mix_frac_name = data column header for mixture fraction. Default value: "mixf"
        get_data_files_output = used to save time in the event that multiple tables are to be constructed. This should be the output of:
            >>> get_data_files(path_to_data, Lvals, tvals, file_pattern = r'^L.*.dat$', col_names = None)
        
    Outputs:
        The output type of phiFuncs will depend on the input parameter "fileName":
             - If fileName is not defined (default: False), the output will be an array of functions.
             - If fileName is specified, the output will be the function for that specific file only. 
    """
    #---------- Import data, files, and headers
    if get_data_files_output == None:
        data, headers, extras = get_data_files(path_to_flame_data, Lvals, tvals)
    else:
        data, headers, extras = get_data_files_output
    
    #---------- Get list of available phi (list of all data headers from original files)
    if type(Lt) == bool:
        #This assumes all datafiles have the same column labels and ordering
        phis = headers[0][0] 
    elif Lt[0] < len(headers) and Lt[1] < len(headers[0]):
        phis = headers[Lt[0]][Lt[1]]
    else:
        raise ValueError(f"""(L,t) indices '{Lt}' are invalid. Valid ranges for indices:
        L: (0,{len(headers)-1})
        t: (0,{len(headers[0])-1})""")
    
    #---------- Interpret user input for "phi", find relevant columns
    phi_col = -1
    xi_col = -1
    
    for i in range(len(phis)):
        if phis[i].lower()==phi.replace(" ","").lower():
            phi_col = i
        if phis[i].lower()==mix_frac_name:
            xi_col = i
    if phi_col == -1:
        raise ValueError("{} not recognized. Available phi are:\n {}".format(phi, phis))
        return None
    if xi_col == -1:
        raise ValueError(f"Mixture fraction ('{mix_frac_name}') was not found among data columns.")
        return None

    #---------- Interpolate phi(xi)
    phiFuncs = np.empty((len(Lvals),len(tvals)), dtype=np.ndarray)
    if Lt == False:
        #Have to interpolate for every file
        for l in range(len(data)):
            for t in range(len(data[l])):
                xis = data[l][t][xi_col]
                phis = data[l][t][phi_col]
                func = interp1d(xis, phis, kind = 'cubic')
                phiFuncs[l][t] = func
        return phiFuncs
    elif Lt[0] < len(headers) and Lt[1] < len(headers[0]):
        xis = data[Lt[0]][Lt[1]][xi_col]
        phis = data[Lt[0]][Lt[1]][phi_col]
        func = interp1d(xis, phis, kind = 'cubic')
        return func

    raise ValueError("Error in code execution: no functions were returned.")
    return None #Code should never reach here

##############################

#----- Testing phiFuncs
# Lvals = [0.002, 0.004, 0.006, 0.008, 0.02, 0.04, 0.2]
# tvals = np.arange(0,11,1)
# phiFuncs(path, Lvals, tvals, Lt = (0,1))(0.5)
# phiFuncs("../flame/run", fileName = 'james')

##############################

def makeLookupTable(path_to_flame_data, Lvals, tvals, phi, numXim=5, numXiv = 5, get_data_files_output = None):
    """
    Creates a 4D lookup table of phi_avg data. Axis are ξm, ξv, L (length scale), and t (time scale). 
    Inputs:
        path_to_flame_data = = path to simulation data relative to the current folder. The data headers should be the last commented line before the data begins.
            The code found at https://github.com/BYUignite/flame was used in testing. 
        phi = property for which values will be tabulated. List of available phi for each file can be obtained using the following:
            get_data_files(path_to_flame_data)[1][fileName]
        Lvals = values of parameter L used, in array format (ex. [ 0.002, 0.02, 0.2])
        tvals = values of parameter t used, in array format (ex. [ 0    , 1   , 2  ])
        numXim, numXiv: Number of data points between bounds for ξm and ξv, respectively. Default value: 5
        get_data_files_output = used to save time in the event that multiple tables are to be constructed. This should be the output of:
            >>> get_data_files(path_to_data, Lvals, tvals, file_pattern = r'^L.*.dat$', col_names = None)
    """
    if get_data_files_output == None:
        funcs = phiFuncs(path_to_flame_data, Lvals, tvals, phi)
    else:
        funcs = phiFuncs(None, Lvals, tvals, phi, \
                         get_data_files_output = get_data_files_output) 

    #---------- Create arrays of ξm and ξv
    Xims = np.linspace(0,1,numXim)      #Xim = Mean mixture fraction. Values 0 and 1 will be adjusted slightly inside LiuInt package to ensure integration is possible.
    Xivs = np.zeros((len(Xims),numXiv)) #Xiv = Mixture fraction variance. Maximum valid Xiv depends on Xim, so this must be a 2D array.
    for i in range(len(Xivs)):
        Xivs[i] = np.linspace(0, Xims[i]*(1-Xims[i]), numXiv)
    
    #----------- Table Creation
    table = np.full((numXim, numXiv, len(Lvals), len(tvals)), -1.0)
    for m in range(len(Xims)):                                               #Loop over each value of ξm
        xim = Xims[m]
        for v in range(len(Xivs[m])):                                        #Loop over each value of ξv
            xiv = Xivs[m][v]
            for l in range(len(Lvals)):
                for t in range(len(tvals)):
                    phiAvg = LI.IntegrateForPhiBar(xim, xiv, funcs[l][t])    #Calculates phi__Avg
                    table[m,v,l,t] = phiAvg                                  #FINAL INDEXING: table[m,v,l,t]

                            
    #Returns: table itself, then an array of the values of Xims, Xivs, Lvals, and tvals for indexing the table.
    #Ex. table[1][2][3][4] corresponds to Xim = Xims[1], Xiv = Xivs[1][2], L = Lvals[3], t = tvals[4].
    #Note that because each Xim has a different set of corresponding Xivs, Xivs is 2D
    indices = [Xims, Xivs, Lvals, tvals]
    return table, indices

##############################

#Because Xiv vs. Xim is not square, we have to interpolate using indices.
def createInterpolator(data, inds):
    from scipy.interpolate import RegularGridInterpolator as rgi
    """
    Accepts a table and indices created by makeLookupTable to create an 
    interpolator using RegularGridInterpolator
    The returned function is called with func(xim, xiv, L, t)
    """
    xi_means = inds[0]
    xi_vars = inds[1] #2D array. xi_vars[i] has xiv values for xi_means[i]
    
    xi_mean_indices = range(len(xi_means))
    xi_var_indices = range(len(xi_vars[0])) #each row has the same # of xivs
    # NOTE: Because each value of ximean has a different set of xivars, 
    # we must interpolate by index. We do the same with xi_mean itself.
    Ls = inds[2]
    ts = inds[3]
    Ls_indices = range(len(Ls))
    ts_indices = range(len(ts))
    
    interpolator = rgi((xi_mean_indices, xi_var_indices, Ls_indices, ts_indices), data, method = 'cubic')

    def translate(xim, xiv, L, t):
        """
        Translates xim, xiv, L, and t values to their respective indices, 
        which are then used in the interpolator. 
        """
        xim_ind = interp1d(xi_means, xi_mean_indices, kind = 'linear')(xim)

        #xi_vars is 2D. xi_means[i] has the corresponding 
        #variances xi_vars[i]. Thus:
        interp = rgi((xi_mean_indices, xi_var_indices), xi_vars)
        def solve(index):
            xiIndMin = min(xi_var_indices)
            xiIndMax = max(xi_var_indices)
            buffer = 1e-8
            if index < xiIndMin:
                index = xiIndMin + buffer
            if index > xiIndMax:
                index = xiIndMax - buffer
            return interp((xim_ind, index)) - xiv
        ig = 0.01 #initial guess for xiv index
        xiv_ind = np.array(least_squares(solve, ig, bounds = [min(xi_var_indices), max(xi_var_indices)]).x)[0]
        L_ind = interp1d(Ls, Ls_indices, kind = 'linear')(L)
        t_ind = interp1d(ts, ts_indices, kind = 'linear')(t)
            #Using cubic interpolators here might be better if the length and time scales had more regular spacing. 
            #If there are large gaps, a cubic interpolator can return out-of-bounds values when interpolating
        
        return (xim_ind, xiv_ind, L_ind, t_ind)

    def func(xim, xiv, L, t):
        """
        Function returned to the user. 
        Accepts values of Xi_mean, Xi_variance, length, and time scale. 
        """
        xim_ind, xiv_ind, L_ind, t_ind = translate(xim, xiv, L, t)
        return interpolator((xim_ind, xiv_ind, L_ind, t_ind))
    
    return func

##############################

def Lt_hc(h, c, xim, xiv, hInterp, cInterp, Lbounds, tbounds, hc_avg = 10**5):
    """
    Solves for L,t given:
        h:   value of enthalpy
        c:   value of progress variable
        xim: mean mixture fraction
        xiv: mixture fraction variance
        hInterp: interpolated function for h(xim, xiv, L, t)
        cInterp: interpolated function for c(xim, xiv, L, t)
            - Both of these functions can be created using "createInterpolator"
        Lbounds: tuple with minimum and maximum value of L
        tbounds: tuple with minimum and maximum value of L
        hc_avg: used to correct error differences. Because values of h are so low, 
                the residual will be multiplied by this scalar to bring the residual 
                for h and c to similar scales. In theory, the ideal value for this 
                parameter would be h(avg)/c(avg) for the given domain.
            
    Returns a tuple of form (L,t)
    This function is to be used for getting values of phi by phi(xim, xiv, [L,t](h,c))
    """
    def solve(Lt):
        L = Lt[0]
        t = Lt[1]

        #----------- Ensure value is within interpolator's bounds
        buffer = 1e-8
        if L < Lbounds[0]:
            L = Lbounds[0] + buffer
        if L > Lbounds[1]:
            L = Lbounds[1] - buffer
        if t < tbounds[0]:
            t = tbounds[0] + buffer
        if t > tbounds[1]:
            t = tbounds[1] - buffer
        #print("L,t = ", L,t)                    DEBUGGING
        
        resid1 = (hInterp(xim, xiv, L, t) - h)*hc_avg #h values are typically low, we have to inflate this error
        resid2 = cInterp(xim, xiv, L, t) - c
        #print("resids = ", resid1, resid2)      DEBUGGING
        return [resid1, resid2]
    
    #----------- Solve function
    Lavg = np.median(Lbounds)
    tavg = np.median(tbounds)
    ig   = (Lavg, tavg)
    lowBounds = [Lbounds[0], tbounds[0]]
    highBounds = [Lbounds[1], tbounds[1]]
    leastSq = least_squares(solve, ig, bounds = (lowBounds, highBounds))
    #ridder(solve, lowBounds, highBounds) DOESN'T WORK: ridder can't accept vector functions
    #fsol = fsolve(solve, ig). Doesn't respect the bounds for some reason. 
    return leastSq.x