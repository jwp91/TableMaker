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

class table:
    """
    Class for creating flamelet-progress variable tables of flame properties.
    The table is created using data from a rectangular set of flame simulations of dimension (L,t) where L and t are some parameters. 
    """
    
    def __init__(self, path_to_flame_data, Lvals, tvals, Lbounds, tbounds, file_pattern = r'^L.*.dat$', 
                 c_components = ['H2', 'H2O', 'CO', 'CO2'], phiFunc_interpKind = 'cubic', 
                 mix_frac_name = "mixf", mvlt_interpKind = 'linear', nxim:int=5, nxiv:int = 5,
                ximLfrac = 0.5, ximGfrac = 0.5, path_to_hsens = './data/ChiGammaTablev3/hsens.dat', 
                gammaValues = None):
        """
        Initializes the TableMaker class with the path to the flame data and the L and t values.
        Inputs:
            path_to_data = path to flamelet data relative to the current folder. 
                NOTE: The data headers must be the last commented line before the data begins.
                The code found at https://github.com/BYUignite/flame was used in testing. 
            Each file will have been run under an array of conditions L,t:
            Lvals: values of parameter L used, formatted as a list (ex. [ 0.002, 0.02, 0.2])
            tvals: values of parameter t used, formatted as a list (ex. [ 0    , 1   , 2  ])
            Lbounds: tuple containing the minimum and maximum value of L
            tbounds: tuple contianing the minimum and maximum value of t
            file_pattern = regular expression (regex) to identify which files in the target folder are data files.  
                DEFAULT: r'^L.*.dat$'. This grabs any files that begin with "L" and end with ".dat". 
            c_components = list defining whih components' mass fractions are included in the progress variable. 
                By default, this is set to be ['H2', 'H2O', 'CO', 'CO2']
                The strings in the list should each match a string used in 'header'
            phiFunc_interpKind = specifies the method of interpolation that should be used for phi(xi) functions (uses scipy.interp1d). 
                Default = 'cubic'.
            mix_frac_name = name of the column header for mixture fraction in flamelet data files. Default value: "mixf"\n
            mvlt_interpKind = interpolation method that RegularGridInterpolator should use in the create_interpolator_mvlt method. 
                Default = 'linear'
            numXim, numXiv: Number of data points between bounds for ξm and ξv, respectively. Default value: 5
            ximLfrac: (0 to 1), fraction of the xim domain that should contain ximGfrac of the total numXim points
            ximGfrac: (0 to 1), fraction of the total numXim points that should fall inside of ximLfrac of the total domain.
                Example: if ximLfrac = 0.2 and ximGfrac = 0.5, then 50% of the numXim points will fall in the first 20% of the domain.
            path_to_hsens: path to a file containing the sensible enthalpy data (col1 = mixf, col2 = h[J/kg])
            gammaValues: array of gamma values corresponding to the index of the t values loaded in. 
                For example, if tvals = [0, 1, 2, ...], gammaValues = [0, 0.05, 0.1, ...] would be appropriate.
        """
        self.path_to_flame_data = path_to_flame_data
        self.path_to_data = self.path_to_flame_data   # Alias
        self.Lvals = Lvals
        self.tvals = tvals
        self.Lbounds = Lbounds
        self.tbounds = tbounds
        self.file_pattern = file_pattern
        self.c_components = c_components
        self.phiFunc_interpKind = phiFunc_interpKind
        self.mix_frac_name = mix_frac_name
        self.mvlt_interpKind = mvlt_interpKind
        self.nxim = nxim
        self.nxiv = nxiv
        self.ximLfrac = ximLfrac
        self.ximGfrac = ximGfrac
        self.path_to_hsens = path_to_hsens
        self.gammaValues = gammaValues

        #---------- Create arrays of ξm and ξv
        if ximLfrac == ximGfrac:
            xims = np.linspace(0,1,nxim)      #Xim = Mean mixture fraction.
        else:
            nsteps = nxim-1
            dx = np.ones(nsteps)/(nsteps)
            n1 = int(nsteps*ximGfrac)
            if n1 != 0:
                dx1 = ximLfrac/n1
                dx[0:n1] = dx1
                n2  = nsteps - n1
                if n2 != 0:
                    dx2 = (1-ximLfrac)/n2
                    dx[n1:] = dx2
            xims = np.zeros(nxim)
            for i in range(len(xims)-1):
                xims[i+1] = xims[i] + dx[i]
            xims[-1] = 1.0
        xivs = np.linspace(0,1,nxiv)      #Xiv = Mixture fraction variance. Maximum valid Xiv depends on Xim, so we normalize the values to the maximum
        self.xims = xims
        self.xivs = xivs

        # Values created later
        self.flmt_data = None
        self.headers = None
        self.extras = None
        self.phi_funcs = {}
        self.table_storage = {}
        self.indices_storage = {}
        self.mvlt_interp_storage = {}
        self.hsensFunc = None
        self.phi_mvhc_funcs = {}

    
    def compute_progress_variable(self, data, header):
        """
        Progress variable is defined as the sum of the mass fractions of a specified set of c_components.
        This function computes the flame progress variable using:
            data = Data from a flame simulation. Each row corresponds to a specific property.
                In the case of this package, this data array is "transposed_file_data" inside the function "get_file_data"
                    ex. data[0] = array of temperature data.
            header = 1D array of column headers, denoting which row in "data" corresponds to which property.
                ex. If header[0] = "Temp", then data[0] should be temperature data.
        """
        #---------- Determine where the c_components are in 'data'
        indices = np.ones(len(self.c_components), dtype = np.int8)*-1
        for i in range(len(header)):                # For each element in the header, 
            for y in range(len(self.c_components)):      # Check for a match among the passed-in c_components
                if header[i]==self.c_components[y].replace(" ",""):
                    indices[y] = int(i)             # Indices must be strictly integers
                    
        # Confirm all indices were located
        for j, ind in enumerate(indices):
            if ind == -1:
                raise ValueError(f"No match found for {self.c_components[j]}.")

        #---------- Compute progress variable
        c = np.zeros(len(data[0]))        # Initialize c array
        for d in range(len(data[0])):     # For each column,
            sum = 0
            for index in indices:         # For each of the components specified, 
                sum += data[index,d]      # Sum the mass fractions of each component
            c[d] = sum
        return c 

    ##############################

    def parse_data(self):
        """
        Reads and formats data resulting from a grid of flamelet simulations.
    
        Outputs:
            flmt_data = an array with the data from each file, indexed using flmt_data[Lval][tval][column# = Property][row # = data point]
            headers  = an array with the column labels from each file, indexed using headers[Lval][tval]
                Each file should have the same columns labels for a given instance of a simulation, but all headers are redundantly included.
            extras   = an array storing any extra information included as comments at the beginning of each file, indexed using extras[Lval][tval]
                This data is not processed in any way by this code and is included only for optional accessibility
        """
        s = self
        #---------- Check if the provided path is a valid directory
        if not os.path.isdir(s.path_to_data):
            print(f"Error: {s.path_to_data} is not a valid directory: no data loaded.")
            return None
        
        #---------- Use glob to list all files in the directory
        files = sorted(glob(os.path.join(s.path_to_data, '*')))
        
        #---------- Store data and filenames
        filenames = np.array([])
        data_files = np.array([])
        for file in files:
            if match(s.file_pattern, os.path.basename(file)):
                filenames = np.append(filenames,  os.path.basename(file))
                data_files= np.append(data_files, file)

        #---------- Initialize data arrays
        flmt_data = np.empty((len(s.Lvals),len(s.tvals)), dtype=np.ndarray)  # Initialize to grab data values
        headers  = np.empty((len(s.Lvals),len(s.tvals)), dtype=np.ndarray)  # Initialize to store headers
        extras   = np.empty((len(s.Lvals),len(s.tvals)), dtype=np.ndarray)  # Initialize to store extra info before header

        #---------- Grab and store data
        for i in range(len(data_files)):
            # This indexing assumes the same # of time scales were run for each length scale
            l = i//len(s.tvals)   # Row index
            t = i %len(s.tvals)   # Column index
            
            file = data_files[i]
            with open(file, 'r') as f:
                #---------- Make sure the assigned L and t value are in the file name:
                if str(s.Lvals[l]) not in f.name:
                    print(f"Warning: for file name '{f.name}', mismatch: L = {s.Lvals[l]}")
                if str(s.tvals[t]) not in f.name:
                    print(f"Warning: for file name '{f.name}', mismatch: t = {s.tvals[t]}")

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
            c = s.compute_progress_variable(transposed_file_data, header)
            transposed_file_data = np.vstack((transposed_file_data, c))   #Stacks this array of progress variable values as the last row 
            
            #---------- Arrange data by l and t indices
            flmt_data[l,t] = transposed_file_data
        
        #flmt_data is indexed using flmt_data[Lval][tval][column# = Property][row # = data point]
        print("Completed data import ('parse_data')")
        self.flmt_data = flmt_data
        self.headers = headers
        self.extras = extras
        return flmt_data, headers, extras

    ##############################

    def make_phi_funcs(self, phi = 'T', Lt = False):
        """
        Returns an array of interpolated functions phi(ξ) where phi is any property of the flame.\n
        Inputs:\n
            phi = desired property (ex. 'T', 'rho', etc.), case sensitive. Default = 'T'\n
                Available phi are viewable using "parse_data(params)[1]".\n
                NOTE: c (progress variable) is available in the data. By default, c ≡ y_CO2 + y_CO + y_H2O + yH2. \n
                This definition can be changed by modifying the c_components parameter.\n
            Lt = Tuple with indices corresponding to the desired L and t. If set to False (default), the output will be an array of the functions phi(ξ) for all datafiles. \n
                Otherwise, this parameter determines which specific file should be used. \n
                Example1: make_phi_funcs(path, phi = 'T', Lt = (0,1)): returns the interpolated T(ξ) function ONLY from the data in the file from Lvals[0], tvals[1]. \n
                Example2: make_phi_funcs(path, phi = 'T'): returns an array containing the interpolated T(ξ) functions from every file in the directory\n
                Note that the values in this tuple are not values of L and t, but rather indexes of Lvals and tvals.\n
            
        Outputs:\n
            The output type of phi_funcs will depend on the input parameter "fileName":\n
                - If Lt is not defined (default), the output will be an array of functions.\n
                - If Lt is specified, the output will be the function for the specified file only. \n
        """
        s = self
        #---------- Import data, files, and headers
        if s.flmt_data == None:
            # No processed data passed in: must generate.
            s.flmt_data, s.headers, s.extras = s.parse_data()
        
        #---------- Get list of available phi (list of all data headers from original files)
        if type(Lt) == bool:
            # User did not specify a specific file.
            # This code here assumes that all datafiles have the same column labels and ordering:
            phis = s.headers[0][0] 
        elif Lt[0] < len(s.headers) and Lt[1] < len(s.headers[0]):
            # User specified a file and the indices were valid.
            phis = s.headers[Lt[0]][Lt[1]]
        else:
            # User specified a file and the indices were invalid.
            raise ValueError(f"""(L,t) indices '{Lt}' are invalid. Valid ranges for indices:
            L: (0,{len(s.headers)-1})
            t: (0,{len(s.headers[0])-1})""")
        
        #---------- Interpret user input for "phi", find relevant columns
        phi_col = -1
        xi_col = -1
        
        for i in range(len(phis)):
            if phis[i]==phi.replace(" ",""):
                # Phi column identified
                phi_col = i
            if phis[i]==s.mix_frac_name:
                # Mixture fraction column identified
                xi_col = i
        if phi_col == -1:
            # Phi wasn't found.
            raise ValueError("{} not recognized. Available phi are:\n {}".format(phi, phis))
        if xi_col == -1:
            # Xi wasn't found.
            raise ValueError(f"Mixture fraction ('{s.mix_frac_name}') was not found among data columns.")

        #---------- Interpolate phi(xi)
        phi_funcs = np.empty((len(s.Lvals),len(s.tvals)), dtype=np.ndarray)
        if Lt == False:
            #User did not specify file: must interpolate for every file
            for l in range(len(s.flmt_data)):
                for t in range(len(s.flmt_data[l])):
                    xis = s.flmt_data[l][t][xi_col]
                    phis = s.flmt_data[l][t][phi_col]
                    phi_funcs[l][t] = interp1d(xis, phis, kind = s.phiFunc_interpKind)
            print(f"phi_funcs for {phi} created using {len(s.Lvals)*len(s.tvals)} files.")
            s.phi_funcs[phi] = phi_funcs
            return phi_funcs
        else:
            #User specified a file
            xis = s.flmt_data[Lt[0]][Lt[1]][xi_col]
            phis = s.flmt_data[Lt[0]][Lt[1]][phi_col]
            print(f"phi_funcs for {phi} created using {len(s.Lvals)*len(s.tvals)} files.")
            phi_funcs[Lt[0]][Lt[1]] = interp1d(xis, phis, kind = s.phiFunc_interpKind)
            s.phi_funcs[phi] = phi_funcs
            return phi_funcs[Lt[0]][Lt[1]]  # Returns the function for the specified file only.

    ##############################
        
    def make_lookup_table(self, phi = 'T'):
        """
        Creates a 4D lookup table of phi_avg data. Axis are ξm, ξv, L, and t. 
        Inputs:
            phi = property for which values will be tabulated (ex. 'T', 'rho', etc.), case sensitive. Default = 'T'
                Available phi are viewable using "parse_data(params)[1]".
                NOTE: c (progress variable) is available in the data. By default, c ≡ y_CO2 + y_CO + y_H2O + yH2. 
                This definition can be changed by modifying the c_components parameter.
        """
        s = self
        # If parse_data_output is not provided, the function will call parse_data to generate the data.
        if phi not in s.phi_funcs:
            s.make_phi_funcs(phi = phi)
        
        #----------- Table Creation
        table = np.full((s.nxim, s.nxiv, len(s.Lvals), len(s.tvals)), -1.0)
        markers = (len(s.xims)*np.array([0.01, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9])).astype(int) # Xim indices at which to notify the user
        for m in range(len(s.xims)):                                               #Loop over each value of ξm
            if m in markers:
                print(f"{phi} table {int(m/len(s.xims)*100)}% complete.")
            xim = s.xims[m]
            for v in range(len(s.xivs)):                                           #Loop over each value of ξv
                xiv = s.xivs[v]*xim*(1-xim)
                for l in range(len(s.Lvals)):
                    for t in range(len(s.tvals)):
                        phiAvg = LI.IntegrateForPhiBar(xim, xiv, s.phi_funcs[phi][l][t])    #Calculates phi_Avg
                        table[m,v,l,t] = phiAvg                                  # FINAL INDEXING: table[m,v,l,t]

                                
        #Returns: table itself, then an array of the values of Xims, Xivs, Lvals, and tvals for indexing the table.
        #Ex. table[7][6][5][4] corresponds to Xim = indices[0][7], Xiv = indices[1][6], L = indices[2][5], t = indices[3][4].
        #Note: Xiv is normalized to the maximum. For table[1][2][3][4], the actual value of the variance would be indices[1][6]*Xivmax,
        #      where Xivmax = Xim*(1-Xim) =  indices[0][7]*(1-indices[0][7])
        
        indices = [s.xims, s.xivs, s.Lvals, s.tvals]
        print(f"Lookup table for phi = {phi} completed.")
        s.table_storage[phi] = table
        s.indices_storage[phi] = indices
        return table, indices

    ##############################

    def create_interpolator_mvlt(self, phi, data, inds, extrapolate = True):
        """
        Creates an interpolator using RegularGridInterpolator (rgi).
        Inputs:
            phi = property for which values will be tabulated (ex. 'T', 'rho', etc.), case sensitive.
            data, inds =  table and indices created by make_lookup_table

        The returned function is called with func(xim, xiv, L, t)
        """
        s = self
        xi_means = inds[0]
        xi_vars = inds[1] # Normalized to Xivmax
        Ls = inds[2]
        ts = inds[3]
        
        if extrapolate:
            interpolator = rgi((xi_means, xi_vars, Ls, ts), data, method = s.mvlt_interpKind, bounds_error = False, fill_value=None)
        else:
            interpolator = rgi((xi_means, xi_vars, Ls, ts), data, method = s.mvlt_interpKind)

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
        
        if phi not in s.mvlt_interp_storage:
            s.mvlt_interp_storage[phi] = func
        return func
    
    ##############################
    def create_hsensFunc(self, path_to_hsens):
        s = self
        # Parse needed data
        hsensdata = np.loadtxt(path_to_hsens, skiprows = 1)
        hsensFunc = interp1d(hsensdata[:,0], hsensdata[:,1], kind = 'linear') # Sensible enthalpy (J/kg) as a function of mixf
        # Make hsens table: hsens(xim, xiv)
        xims = np.linspace(0, 1, s.nxims)
        xivs = np.linspace(0, 1, s.nxivs)
        hsensTable = np.zeros((s.nxims, s.nxivs))
        for i in range(s.nxims):
            ximVal = xims[i]
            for j in range(s.nxivs):
                xivVal = xivs[j]*ximVal*(1-ximVal)
                hsensTable[i,j] = LI.IntegrateForPhiBar(ximVal, xivVal, hsensFunc)
        interpolator = rgi((xims, xivs), hsensTable, method = 'linear')  # No extrapolation
        
        def hsensFunc(xim, xiv):
            # Returns hsens for a value of xim and xiv
            xivmax = xim*(1-xim)
            xiv = max(0, min(xiv*xim*(1-xim), xivmax))
            return interpolator([xim, xiv])
        return hsensFunc

    def Lt_from_hc_GammaChi(self, hgoal, cgoal, xim, xiv, hInterp, cInterp,
                            useStoredSolution:bool = True):
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
            useStoredSolution:bool, if set to False, the solver will not use the last solution as its initial guess. 
                Using the last initial guess (default) is generally good: CFD will solve cell-by-cell, and nearby
                cells are expected to have similar values of phi.
                
        Returns a tuple of form (L,t)
        This function is to be used for getting values of phi by phi(xim, xiv, [L,t](h,c))
        """
        s = self
        gammaToIndex = interp1d(s.gammaValues, np.arange(len(s.gammaValues)), 
                                kind = 'linear', bounds_error = None, fill_value = 'extrapolate') # Converts gamma to an index

        #----- Determine gamma
        h0 = hInterp(0, 0, s.Lbounds[0], s.tbounds[0])  # Enthalpy of pure fuel
        h1 = hInterp(1, 0, s.Lbounds[0], s.tbounds[0])  # Enthalpy of pure oxidizer
        ha = h0*(1-xim) + h1*xim                        # Adiabatic enthalpy    
        global hsensFunc
        if hsensFunc is None:
            hsensFunc = s.create_hsensFunc(s.path_to_hsens, nxims = s.nxim, nxivs = s.nxiv)
        
        gamma = (ha - hgoal)/hsensFunc(xim, xiv)        # Heat loss parameter
        
        t = gammaToIndex(gamma)                         # Time scale index
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
            if L < s.Lbounds[0] or L > s.Lbounds[1]:
                # If the last solution is out of bounds, use the midpoint
                guess = s.Lbounds[0] + (s.Lbounds[1]-s.Lbounds[0])/2
            else:
                guess = L
        else:
            guess   = s.Lbounds[0] + (s.Lbounds[1]-s.Lbounds[0])/2

        #L = fsolve(obj, guess)[0]
        L = minimize(lambda L: np.abs(obj(L)), guess, method = 'Nelder-Mead').x[0]
        #L = minimize_scalar(lambda L: np.abs(obj(L)), guess, method = 'bounded', bounds=Lbounds).x[0]

        np.savetxt("chiGamma_lastsolution.txt", np.array([L]))
        return [L, t]

    def create_table_aux(self, args):
        """Auxiliary function used in phi_mvhc for parallelization. 
        The package used for parallelization ("concurrent") requires that the function being parallelized is defined 
        in the global scope.
        """
        # Generic table-generating function
        return self.make_lookup_table(*args)

    def phi_mvhc(self, phi = 'T', parallel:bool = True, detailedWarn:bool = False):
        """
        Creates a table of phi values in terms of xim, xiv, h, and c
        Inputs:
            phi = single property or list of properties for which values will be tabulated (ex. 'T', 'rho', etc.), case sensitive. Default = 'T'
                Available phi are viewable using "parse_data(params)[1]".
                NOTE: c (progress variable) is available in the data. By default, c ≡ y_CO2 + y_CO + y_H2O + yH2.
                This definition can be changed by modifying the c_components parameter.
            parallel:bool = if set to True (default), the code will attempt to create tables in parallel.
            detailedWarn: If set to true, more detailed warnings will be raised when the solver does not converge. 
            
        Outputs:
            phi_mvhc_arr: Array of phi functions phi = phi(xim, xiv, h, c)
                NOTE: if only one phi is specified, if will still be returned in a single-element array.
            tableArr: array of [table, indices] for each phi, beginning with h and c.

        """
        s = self
        # ------------ Pre-processing
        # Confirm h and c aren't in phi
        for p in phi:
            if p=='h' or p=='c':
                print("'h' and 'c' are used as table axis and so cannot be used as phi. Cancelling operation.")
                return None

        # Ensure array-like
        if type(phi) == type('str'):
            phi = [phi,]

        # Import data, files, and headers
        if s.flmt_data == None:
            # No processed data passed in: must generate.
            s.flmt_data, s.headers, s.extras = s.parse_data()

        # ------------ Compute tables, parallel or serial
        # Enable solvers to be accessed when this code is imported as a package

        ####### Serial computation
        if not parallel: 
            # Create h & c tables
            if 'h' not in self.table_storage or 'h' not in self.indices_storage:
                h_table, h_indices = s.make_lookup_table('h')
            else:
                h_table = self.table_storage['h']
                h_indices = self.indices_storage['h']
            if 'c' not in self.table_storage or 'c' not in self.indices_storage:
                c_table, c_indices = s.make_lookup_table('c')
            else:
                c_table = self.table_storage['c']
                c_indices = self.indices_storage['c']
        
            # Create h & c interpolators
            if 'h' not in s.mvlt_interp_storage:
                Ih = s.create_interpolator_mvlt('h', h_table, h_indices)
            else:
                Ih = s.mvlt_interp_storage['h']
            if 'c' not in s.mvlt_interp_storage:
                Ic = s.create_interpolator_mvlt('c', c_table, c_indices)
            else:
                Ic = s.mvlt_interp_storage['c']
        
            # Create array containing phi tables
            for p in phi:
                # Get base table with phi data
                if p not in s.table_storage or p not in s.indices_storage:
                    table, indices = s.make_lookup_table(p)
                else:
                    table = s.table_storage[p]
                    indices = s.indices_storage[p]
        
                # Create interpolator for phi
                if p not in s.mvlt_interp_storage:
                    InterpPhi = s.create_interpolator_mvlt(p, table, indices)
                else:
                    InterpPhi = s.mvlt_interp_storage[p]
                
                # Create function phi(xim, xiv, h, c)
                def create_phi_table(interp_phi):
                    def phi_table(xim, xiv, h, c, useStoredSolution = True):
                        # Invert from (h, c) to (L, t), then return interpolated value.
                        L, t = s.Lt_from_hc_GammaChi(h, c, xim, xiv, Ih, Ic, useStoredSolution)
                        
                        return interp_phi(xim, xiv, L, t)
                    return phi_table

                s.phi_mvhc_funcs[p] = create_phi_table(InterpPhi)  # Store function in the class
            return s.phi_mvhc_funcs, s.table_storage
            
        ####### Parallel computation
        else: 
            # Import needed packages
            from concurrent.futures import ProcessPoolExecutor
            import concurrent

            phi = np.append(np.array(['h', 'c']), np.array(phi)) # Need to create h and c tables too, so add them at the beginning. 
            table_args = [(p,) for p in phi] # Arguments for each table's creation

            # Parallel table creation (should be reviewed)
            with ProcessPoolExecutor(mp_context=mp.get_context('fork')) as executor:
                futures = {executor.submit(s.create_table_aux, args): idx for idx, args in enumerate(table_args)}
                results = {}
                for future in concurrent.futures.as_completed(futures):
                    idx = futures[future]
                    try:
                        results[idx] = future.result()
                    except Exception as e:
                        print(f"Table creation for index {idx} (phi = {phi[idx]}) generated an exception: {e}")

            # Create h & c interpolators -- These should only be set to cubic interpolation with a very dense table.
            if 'h' not in s.mvlt_interp_storage:
                h_table, h_indices = results[0]
                Ih = s.create_interpolator_mvlt('h', h_table, h_indices)
            else:
                Ih = s.mvlt_interp_storage['h']
            if 'c' not in s.mvlt_interp_storage:
                c_table, c_indices = results[1]
                Ic = s.create_interpolator_mvlt('c', c_table, c_indices)
            
            # Create functions for phi(xim, xiv, h, c)
            for p in phi:
                i = np.where(phi = p)
                if p not in s.mvlt_interp_storage:
                    InterpPhi = s.create_interpolator_mvlt(p, results[i][0], results[i][1])
                else:
                    InterpPhi = s.mvlt_interp_storage[p]
                
                # Create function phi(xim, xiv, h, c)
                def create_phi_table(interp_phi):
                    def phi_table(xim, xiv, h, c, useStoredSolution = True):
                        # Invert from (h, c) to (L, t), then return interpolated value.
                        L, t = s.Lt_from_hc_GammaChi(h, c, xim, xiv, Ih, Ic, useStoredSolution)
                        
                        return interp_phi(xim, xiv, L, t)
                    return phi_table

                s.phi_mvhc_funcs[p] = create_phi_table(InterpPhi)  # Store function in the class
            return s.phi_mvhc_funcs, s.table_storage