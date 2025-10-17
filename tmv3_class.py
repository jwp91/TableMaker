# TableMaker version 3
# Main Author: Jared Porter
# Contributors: Dr. David Lignell, Jansen Berryhill
# Some revisions and refactors completed with GitHub Copilot under the student license

import numpy as np
from scipy.interpolate import interp1d
from scipy.optimize import minimize
import os
import warnings
from glob import glob
from re import match
import LiuInt as LI # Package with functions for integrating over the BPDF, parameterized by the average and variance of the mixture fraction.
from scipy.interpolate import RegularGridInterpolator as rgi
from datetime import datetime
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor
import concurrent
import dill
import time
import sys

############################## tableMakerv3 - object-oriented

class table:
    """
    Object for maintaining flamelet-progress variable tables of arbitrary flame properties, phi.
    Tables are created using data from a set of flamelet simulations of dimensions (L,t),
        where L and t are some indexed parameters (e.g., integer values only)
    Example case:
        L is a strain parameter and t is a heat loss parameter. 
        A flamelet simulation is run for each combination of L and t. 
        The results of each simulation are stored in a data file, 
            where each column contains property data (e.g., mixf, T, rho, Yi, etc.)
            for each row (spatial location).
        Each results files' name contains (indexed!) values of L, t.
            - For the first file, this would be (0,0), for the second file (0,1), etc.
            - The code that parses this data checks a user-provided list of expected L and t indices 
              against the file names to ensure there are no mismatches.
        These files are stored in a certain directory, the path to which is passed as an argument to the 'table' object. 
        'table' can then:
            - Parse the data files in the directory, including the creation of progress variable data based on
              the mass fractions of specified chemical components (c_components)
            - Create functions phi(ξ) for any specified property, phi
            - Create a lookup table of phi_avg(ξm, ξv, L, t) for any specified property, phi
                - This assumes a subgrid beta-PDF distribution for the mixture fraction, ξ, 
                  parameterized by ξm (mean mixture fraction) and ξv (mixture fraction variance).
                - The subgrid distributions of L and t are assumed to be delta functions about their means, independent of mixture fraction.
            - Create an interpolator for phi_avg(ξm, ξv, L, t) using RegularGridInterpolator
            - Create a function phi_avg(ξm, ξv, h, c)
                - Solves internally for (L,t) given (h,c). This can done using either:
                    1. The gamma-chi formulation, detailed by Porter, Lignell, and Berryhill (2025?), or
                    2. A 2D Newton solver.
                - The latter is substantially slower and includes tuning parameters, but is more general. 

    'table' is initialized with:
        - path to the directory containing the simulated flamelet data files (created using e.g., Ignis [Lignell])
        - list of L indices used in the file names
        - list of t indices used in the file names
        - regular expression to identify which files in the aforementioned path are data files
            - Default: r'^L.*.dat$' (grabs any files that begin with "L" and end with ".dat")
        - list of named chemical components, the sum of whose mass fractions will be used to compute the progress variable
            - Default: ['H2', 'H2O', 'CO', 'CO2']
        - interpolation method for the functions phi(xi)
            - Default: 'cubic'
        - name of the column header identifying the mixture fraction in the data files
            - Default: 'mixf'
        - interpolation method for the intermediate function phi(xim, xiv, L, t)
            - Default: 'linear'
        - number of data points between bounds for both xim and xiv (mean and variance of mixture fraction, respectively)
            - Default values: nxim = 5, nxiv = 5 (much too course, but low computational load for initial testing)
        - fraction of the xim domain that should contain ximGfrac (another parameter) of the total xim points
            - Default values: ximLfrac = 0.5, ximGfrac = 0.5 (even spacing)
            - Example: if ximLfrac = 0.2 and ximGfrac = 0.5, then 50% of the nxim points will fall in the first 20% of the domain.
        - path to a file containing columns of mixf and sensible enthalpy data (J/kg)
            - Default: './data/ChiGammaTablev3/hsens.dat'
            - This is needed because the flamelet data files do not contain sensible enthalpy data, and this parameter is nonlinearly
              related to mixture fraction.
        - array of gamma values, implying the functional relationship between t (index) and gamma (a continuous parameter)
            For example, if tvals = [0, 1, 2, ...], gammaValues = [0, 0.05, 0.1, ...] defines the function gamma(t).
    """
    
    def __init__(self, path_to_data, Lvals, tvals, flmt_file_pattern = r'^flm.*.dat$', 
                 c_components = ['H2', 'H2O', 'CO', 'CO2'], phiFunc_interpKind = 'cubic', 
                 mixf_col_name = 'mixf', mvlt_interpKind = 'linear', nxim:int=5, nxiv:int = 5,
                 ximLfrac = 0.5, ximGfrac = 0.5, path_to_hsens = './data/ChiGammaTablev3/hsens.dat', 
                 gammaValues = None):
        """
        Initializes the TableMaker class with necessary variables for table creation.
        Inputs:
            path_to_data = path to flamelet data relative to the current folder. 
                NOTE: The data headers must be the last commented line before the data begins.
                The code found at https://github.com/BYUignite/flame was used in testing. 
            flmt_file_pattern = regular expression (regex) to identify which files in the target folder are data files.  
                Default = r'^L.*.dat$'. This grabs any files that begin with "L" and end with ".dat". 
            Lvals = values of parameter L used, formatted as a list (e.g., [ 0, 1, 2, ...])
            tvals = values of parameter t used, formatted as a list (e.g., [ 0, 1, 2, ...])
            c_components = list defining whih components' mass fractions are included in the progress variable. 
                Default =  ['H2', 'H2O', 'CO', 'CO2']
                The strings in the list should each match the strings used in the header of the flamelet data files.
            phiFunc_interpKind = specifies the method of interpolation that should be used for phi(xi) functions (scipy.interp1d). 
                Default = 'cubic'.
            mixf_col_name = name of the column header for mixture fraction in flamelet data files. 
                Default value = 'mixf'
            mvlt_interpKind = interpolation method that RegularGridInterpolator should use in the create_interpolator_mvlt method. 
                Default = 'linear'
            nxim, nxiv: Number of data points between bounds for ξm and ξv, respectively. Default value: 5
            ximLfrac = (0 to 1), fraction of the xim domain that should contain ximGfrac of the total nxim points
            ximGfrac = (0 to 1), fraction of the total nxim points that should fall inside of ximLfrac of the total domain.
                Example: if ximLfrac = 0.2 and ximGfrac = 0.5, then 50% of the nxim points will fall in the first 20% of the domain.
            path_to_hsens = path to a file containing the sensible enthalpy data (col1 = mixf, col2 = h[J/kg])
            gammaValues = array of gamma values (continuous) corresponding to the t values (indexed) passed in. 
                For example, if tvals = [0, 1, 2, ...], gammaValues = [0, 0.05, 0.1, ...] would be appropriate.
        """
        self.path_to_data = path_to_data
        self.path_to_flame_data = self.path_to_data   # Alias
        self.Lvals = Lvals
        self.tvals = tvals
        self.Lbounds = [min(Lvals), max(Lvals)]
        self.tbounds = [min(tvals), max(tvals)]
        self.flmt_file_pattern = flmt_file_pattern
        self.c_components = c_components
        self.phiFunc_interpKind = phiFunc_interpKind
        self.mixf_col_name = mixf_col_name
        self.mvlt_interpKind = mvlt_interpKind
        self.nxim = nxim
        self.nxiv = nxiv
        self.ximLfrac = ximLfrac
        self.ximGfrac = ximGfrac
        self.path_to_hsens = path_to_hsens
        self.gammaValues = gammaValues

        # Get the directory of the current Python script
        try:
            self.current_dir = os.path.dirname(os.path.abspath(__file__))
        except:
            # Get the directory of the current jupyter notebook
            self.current_dir = os.path.dirname(os.path.abspath(''))

        # Create a directory for auxiliary data (solver files, etc.)
        self.result_dir = os.path.join(self.current_dir, 'results')
        if not os.path.exists(self.result_dir):
            os.makedirs(self.result_dir)

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

        # Data populated later
        self.flmt_data = None
        self.headers = None
        self.extras = None
        self.phi_funcs = {}
        self.table_storage = {}
        self.indices_storage = {}
        self.mvlt_interp_storage = {}
        self.hsensFunc = None
        self.phi_mvhc_funcs = {}
        self.norm = False # Used in newton solve

    def force_warning(self, queue=None):
        message = """Notice: setting force=True in a function only forces that particular function to re-run.
              For example, calling create_phi_funcs('T', force=True) will force the phi_funcs for T to be re-created,
              but will not force the source data to be re-parsed."""
        if queue == None:
            print(message)
        else:
            queue.put(message)

    def compute_progress_variable(self, data, header):
        """
        Progress variable is defined as the sum of the mass fractions of a specified set of c_components.
        This function computes the flame progress variable using:
            data: Data from a flame simulation. Each row corresponds to a specific property.
                In the case of this package, this data array is "transposed_file_data" inside the function "get_file_data"
                    ex. data[0] = array of temperature data.
            header: 1D array of column headers, denoting which row in "data" corresponds to which property.
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
                raise ValueError(f"No match found for {self.c_components[j]} when computing progress variable.")

        #---------- Compute progress variable
        c = np.zeros(len(data[0]))        # Initialize c array
        for d in range(len(data[0])):     # For each column,
            sum = 0
            for index in indices:         # For each of the components specified, 
                sum += data[index,d]      # Sum the mass fractions of each component
            c[d] = sum
        return c 

    ##############################

    def parse_data(self, force = False):
        """
        Reads and formats data resulting from a grid of flamelet simulations.
    
        Outputs:
            flmt_data = an array with the data from each file, indexed using flmt_data[Lval][tval][column# = Property][row # = data point]
            headers = an array with the column labels from each file, indexed using headers[Lval][tval]
                Each file should have the same columns labels for a given instance of a simulation, but all headers are redundantly included.
            extras = an array storing any extra information included as comments at the beginning of each file, indexed using extras[Lval][tval]
                This data is not processed in any way by this code and is included only for optional accessibility
        """
        s = self
        if force:
            s.force_warning()
        elif s.flmt_data is not None and s.headers is not None and s.extras is not None:
            # Data already parsed and stored: no need to re-parse.
            print("Data already parsed. Use 'force = True' to re-parse data.")
            return s.flmt_data, s.headers, s.extras
            
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
            if match(s.flmt_file_pattern, os.path.basename(file)):
                filenames = np.append(filenames,  os.path.basename(file))
                data_files= np.append(data_files, file)

        #---------- Initialize data arrays
        flmt_data = np.empty((len(s.Lvals),len(s.tvals)), dtype=np.ndarray)  # Initialize to grab data values
        headers   = np.empty((len(s.Lvals),len(s.tvals)), dtype=np.ndarray)  # Initialize to store headers
        extras    = np.empty((len(s.Lvals),len(s.tvals)), dtype=np.ndarray)  # Initialize to store extra info before header

        #---------- Grab and store data
        print("Parsing data files...")
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
                for line in reversed(lines):               # The last of the commented lines should be the headers,
                    if line.startswith('#'):               # so we grab the last of the commented lines
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
            #NOTE: the following lines could also be achieved with np.loadtxt(). Because we've already read in the lines
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
        print("Completed data import (parse_data)")
        self.flmt_data = flmt_data
        self.headers = headers
        self.extras = extras
        return flmt_data, headers, extras

    ##############################

    def create_phi_funcs(self, phi, Lt = False, force = False):
        """
        Returns an array of interpolated functions phi(ξ) where phi is any property of the flame.\n
        Inputs:\n
            phi = desired property (ex. 'T', 'rho', etc.), case sensitive. Default = 'T'\n
                Available phi are viewable using "parse_data(params)[1]".\n
                NOTE: c (progress variable) is available in the data. By default, c ≡ y_CO2 + y_CO + y_H2O + yH2. \n
                This definition can be changed by modifying the c_components parameter.\n
            Lt = Tuple with indices corresponding to the desired L and t. If set to False (default), the output will be an array of the functions phi(ξ) for all datafiles. \n
                Otherwise, this parameter determines which specific file should be used. \n
                Example1: create_phi_funcs(path, phi = 'T', Lt = (0,1)): returns the interpolated T(ξ) function ONLY from the data in the file from Lvals[0], tvals[1]. \n
                Example2: create_phi_funcs(path, phi = 'T'): returns an array containing the interpolated T(ξ) functions from every file in the directory\n
                Note that the values in this tuple are not values of L and t, but rather indexes of Lvals and tvals.\n
            
        Outputs:\n
            The output type of phi_funcs will depend on the input parameter "fileName":\n
                - If Lt is not defined (default), the output will be an array of functions.\n
                - If Lt is specified, the output will be the function for the specified file only. \n
        """
        s = self
        if force:
            s.force_warning()
        elif phi in s.phi_funcs:
            # If the phi function already exists, return it.
            print(f"phi_funcs for {phi} already exists. Use 'force = True' to re-create it.")
            if Lt == False:
                return s.phi_funcs[phi]
            else:
                return s.phi_funcs[phi][Lt[0]][Lt[1]]

        #---------- Import data, files, and headers
        if s.flmt_data is None:
            # No processed data yet generated: must generate.
            s.parse_data()
        
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
            if phis[i]==s.mixf_col_name:
                # Mixture fraction column identified
                xi_col = i
        if phi_col == -1:
            # Phi wasn't found.
            raise ValueError("{} not recognized. Available phi are:\n {}".format(phi, phis))
        if xi_col == -1:
            # Xi wasn't found.
            raise ValueError(f"Mixture fraction ('{s.mixf_col_name}') was not found among data columns.")

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
        
    def create_lookup_table(self, phi, force = False, queue = None):
        """
        Creates a 4D lookup table of phi_avg data. Axis are ξm, ξv, L, and t. 
        Inputs:
            phi = property for which values will be tabulated (ex. 'T', 'rho', etc.), case sensitive. Default = 'T'
                Available phi are viewable using "parse_data(params)[1]".
                NOTE: c (progress variable) is available in the data. By default, c ≡ y_CO2 + y_CO + y_H2O + yH2. 
                This definition can be changed by modifying the c_components parameter.
            queue = multiprocessing queue object, used to handle parallel objects printing statements. Used internally.
        """
        s = self
        if force:
            s.force_warning(queue=queue)
        elif phi in s.table_storage:
            # If the table already exists, return it.
            if queue == None:
                print(f"Lookup table for {phi} already exists. Use 'force = True' to re-create it.")
            else:
                queue.put(f"Lookup table for {phi} already exists. Use 'force = True' to re-create it.")
                queue.put("SENTINEL")
            return s.table_storage[phi], s.indices_storage[phi]
        
        # If needed phi_funcs don't exist, create them
        if phi not in s.phi_funcs:
            if queue == None:
                print(f"Creating phi_funcs for {phi}...")
            else:
                queue.put(f"Creating phi_funcs for {phi}...")
            s.create_phi_funcs(phi = phi)
        
        #----------- Table Creation
        table = np.full((s.nxim, s.nxiv, len(s.Lvals), len(s.tvals)), -1.0)
        markers = (len(s.xims)*np.array([0.01, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9])).astype(int) # Xim indices at which to notify the user
        for m in range(len(s.xims)):                                               #Loop over each value of ξm
            if m in markers:
                if queue == None:
                    print(f"{phi} table {int(m/len(s.xims)*100)}% complete.")
                else:
                    queue.put(f"{phi} table {int(m/len(s.xims)*100)}% complete.")
            xim = s.xims[m]
            for v in range(len(s.xivs)):                                           #Loop over each value of ξv
                xiv = s.xivs[v]*xim*(1-xim)
                for l in range(len(s.Lvals)):
                    for t in range(len(s.tvals)):
                        phiAvg = LI.IntegrateForPhiBar(xim, xiv, s.phi_funcs[phi][l][t])    #Calculates phi_Avg
                        table[m,v,l,t] = phiAvg                                  # FINAL INDEXING: table[m,v,l,t]

                                
        # Returns: table itself, then an array of the values of Xims, Xivs, Lvals, and tvals for indexing the table.
        # Ex. table[7][6][5][4] corresponds to Xim = indices[0][7], Xiv = indices[1][6], L = indices[2][5], t = indices[3][4].
        # Note: Xiv is normalized to the maximum. For table[1][2][3][4], the actual value of the variance would be indices[1][6]*Xivmax,
        #       where Xivmax = Xim*(1-Xim) =  indices[0][7]*(1-indices[0][7])
        
        indices = [s.xims, s.xivs, s.Lvals, s.tvals]
        if queue == None:
            print(f"Lookup table for phi = {phi} completed.")
        else:
            queue.put(f"Lookup table for phi = {phi} completed.")
            queue.put("SENTINEL") # Signal to the main process that one of the tables completed

        s.table_storage[phi] = table
        s.indices_storage[phi] = indices
        return table, indices

    ##############################

    def create_interpolator_mvlt(self, phi, data, inds, force = False):
        """
        Creates an interpolator using RegularGridInterpolator (rgi).
        Inputs:
            phi = property for which values will be tabulated (ex. 'T', 'rho', etc.), case sensitive.
            data, inds = table and indices created by create_lookup_table

        The returned function is called with func(xim, xiv, L, t)
        """
        s = self
        if force:
            s.force_warning()
        elif phi in s.mvlt_interp_storage:
            # If the interpolator already exists, return it.
            print(f"Interpolator for {phi} already exists. Use 'force = True' to re-create it.")
            return s.mvlt_interp_storage[phi]
        
        xi_means = inds[0]
        xi_vars = inds[1] # Normalized to Xivmax
        Ls = inds[2]
        ts = inds[3]

        interpolator = rgi((xi_means, xi_vars, Ls, ts), data, method = s.mvlt_interpKind, \
                           bounds_error = False, fill_value=None)

        def func(xim, xiv, L, t, extrapolate=True, bound = False):
            # Function returned to the user.
            """
            Interpolates for a value of phi given:
                Xi_mean
                Xi_variance (actual value)
                Length scale
                Time scale
            Parameters:
                extrapolate: if True (default), the fucntion allows extrapolation outside of the table. 
                    If False, the function behaves according to the 'bound' parameter.
                bound: if False (default), the function errors when input values are outside of the table bounds.
                    If True, the function will force the input values to be within the table bounds.
                NOTE: xiv is never allowed to exceed [0, Xivmax], where Xivmax = Xim*(1-Xim)
            """
            # Check xiv value
            xiv_max = xim*(1-xim)
            if xiv > xiv_max:
                raise ValueError(f"xiv must be less than xivMax. With xim = {xim}, xiv_max = {xiv_max}. Input xiv = {xiv}")
            if xiv_max == 0:
                if xiv != 0:
                    print(f"Warning: xim = {xim}, meaning xiv_max = 0. xiv passed in was {xiv}, \
but has been overridden to xiv = 0.")
                xiv_norm = 0
            else:
                xiv_norm = xiv/xiv_max
                
            # Check bounds, if relevant
            if not extrapolate and bound:
                # Force bounding
                L = max(s.Lbounds[0], min(L, s.Lbounds[1]))
                t = max(s.tbounds[0], min(t, s.tbounds[1]))

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
    def create_hsensFunc(self, path_to_hsens, force = False):
        s = self
        if force:
            s.force_warning()
        elif s.hsensFunc is not None:
            # If the hsens function already exists, return it.
            print("hsensFunc already exists. Use 'force = True' to re-create it.")
            return s.hsensFunc

        # Parse needed data
        hsensdata = np.loadtxt(path_to_hsens, skiprows = 1)
        hsensFunc = interp1d(hsensdata[:,0], hsensdata[:,1], kind = 'linear') # Sensible enthalpy (J/kg) as a function of mixf
        
        # Make hsens table: hsens(xim, xiv)
        hsensTable = np.zeros((s.nxim, s.nxiv))
        for i in range(s.nxim):
            ximVal = s.xims[i]
            for j in range(s.nxiv):
                xivVal = s.xivs[j]*ximVal*(1-ximVal)
                hsensTable[i,j] = LI.IntegrateForPhiBar(ximVal, xivVal, hsensFunc)
        interpolator = rgi((s.xims, s.xivs), hsensTable, method = 'linear')  # No extrapolation
        
        def hsensFunc(xim, xiv):
            # Returns hsens for a value of xim and xiv
            xivmax = xim*(1-xim)
            xiv = max(0, min(xiv*xim*(1-xim), xivmax)) # Forced bounding
            return interpolator([xim, xiv])
        s.hsensFunc = hsensFunc
        return s.hsensFunc

    def Lt_from_hc_GammaChi(self, hgoal, cgoal, xim, xiv, hInterp, cInterp,
                            useStoredSolution:bool = True):
        """
        Solves for (L,t) given values of (h,c) in the gamma-chi formulation of the table.
        This table is constructed so that file has:
            1) an imposed heat loss parameter gamma, defined as (h_{adiabatic} - h)/h_{sensible, firstFile}
            2) a diffusive strain parameter chi, used in the opposed jet formulation of a flamelet.
        Because gamma is independent of chi, gamma may be determined first using thermodynamic data. 
            - This data is passed in and processed using the create_hsensFunc method.
        Then, chi may be determined using interpolated data from the table. 
        Note: chi:L:strain :: gamma:t:heat loss

        Function parameters:
            hgoal = value of enthalpy
            cgoal = value of progress variable
            xim = mean mixture fraction
            xiv = mixture fraction variance
            hInterp = interpolated function for h(xim, xiv, L, t), created using "create_interpolator_mvlt"
            cInterp = interpolated function for c(xim, xiv, L, t), created using "create_interpolator_mvlt"
            useStoredSolution:bool = if set to False, the solver will not use the last solution as its initial guess. 
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

        if s.hsensFunc is None:
            print("hsensFunc not yet created. Creating now...")
            s.hsensFunc = s.create_hsensFunc(s.path_to_hsens)
        
        gamma = (ha - hgoal)/s.hsensFunc(xim, xiv)        # Heat loss parameter
        
        t = gammaToIndex(gamma)                         # Time scale index
        if isinstance(t, np.ndarray):
            t = t[0]

        #----- Use gamma to determine chi
        def obj(L):
            if isinstance(L, np.ndarray):
                L = L[0]
            return cInterp(xim, xiv, L, t) - cgoal

        # Check if previous solution was stored
        file_path = os.path.join(s.result_dir, "chiGamma_lastsolution.txt")
        if os.path.isfile(file_path) and useStoredSolution:
            # Use the last solution as the initial guess
            L = np.loadtxt(file_path)
            if L < s.Lbounds[0] or L > s.Lbounds[1]:
                # If the last solution is out of bounds, use the midpoint
                guess = s.Lbounds[0] + (s.Lbounds[1]-s.Lbounds[0])/2
            else:
                guess = L
        else:
            guess     = s.Lbounds[0] + (s.Lbounds[1]-s.Lbounds[0])/2

        #L = fsolve(obj, guess)[0]
        L = minimize(lambda L: np.abs(obj(L)), guess, method = 'Nelder-Mead').x[0]
        #L = minimize_scalar(lambda L: np.abs(obj(L)), guess, method = 'bounded', bounds=Lbounds).x[0]

        np.savetxt(file_path, np.array([L]))
        return [L, t]

    def Lt_from_hc_newton(self, hgoal, cgoal, xim, xiv, hInterp, cInterp, norm, detailedWarn:bool = False, 
                          maxIter:int = 100, saveSolverStates:bool = False, useStoredSolution:bool = True, 
                          LstepParams = [0.25, 0.01, 0.003], tstepParams = [0.25, 9.5, 0.02]):
        """
        For the gamma-chi formlation of the table, this function is inefficient. Use Lt_from_hc_GammaChi instead.

        Solves for (L,t) given values of (h,c) using a 2D Newton solver.
        Params:
            hgoal = value of enthalpy
            cgoal = value of progress variable
                xim = mean mixture fraction
                xiv = mixture fraction variance
            hInterp = interpolated function for h(xim, xiv, L, t), created using "create_interpolator_mvlt"
            cInterp = interpolated function for c(xim, xiv, L, t), created using "create_interpolator_mvlt"
            norm   := np.max(h_table)/np.max(c_table). Compensates for the large difference in magnitude between typical h and c values.
            detailedWarn = If set to true, more detailed warnings will be raised when the solver does not converge.    
            maxIter:int = sets a limit for the maximum iterations the solver should make.
            saveSolverStates: bool = if set to True, the solver states will be saved to a file in the folder "solver_data"
            useStoredSolution:bool = if set to False, the solver will not use the last solution as its initial guess. 
                Using the last initial guess (default) is generally good: CFD will solve cell-by-cell, and nearby
                cells are expected to have similar values of phi.
            LstepParams = array of parameters used to relax the solver
                LstepParams[0] = 0.25; normal max step size (% of domain)
                LstepParams[1] = 0.01; threshold value of L, below which the max step size is reduced to
                LstepParams[2] = 0.003; reduced max step size (% of domain)
            tstepParams = array of parameters used to relax the solver
                tstepParams[0] = 0.25; normal max step size (% of domain)
                tstepParams[1] = 9.5; threshold value of t, above which the max step size is reduced to
                tstepParams[2] = 0.02; reduced max step size (% of domain)
            
        Returns a tuple of form (L,t)
        This function is to be used for getting values of phi by phi(xim, xiv, [L,t](h,c))
        """
        s = self
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
                F  = F(mvlt) = [h(mvlt) - hSet, c(mvlt)-cSet]
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
            if X0[2] + deltaL[2] > s.Lbounds[1]:
                # Avoid stepping over L boundary when adding deltaL
                J0 = (F0 - F(X0 - deltaL))/deltaL[2]  # = [dH/dL, dc/dL]
            else:
                J0 = (F(X0 + deltaL) - F0)/deltaL[2]  # = [dH/dL, dc/dL]

            if X0[3] + deltat[3] > s.tbounds[1]:
                # Avoid stepping over t boundary when adding deltat
                J1 = (F0 - F(X0 - deltat))/deltat[3]  # = [dH/dt, dc/dt]
            else:
                J1 = (F(X0 + deltat) - F0)/deltat[3]  # = [dH/dt, dc/dt]

            return np.array([J0, J1]).T[0] # Without this final indexing, the shape is (1, 2, 2) instead of (2, 2)
        
        def cramer_solve(F, X0):
            """
            Solves the system of equations JX=F(X0) for X using Cramer's rule.
            Params:
                F  = f(mvlt) = [h(mvlt)-hSet, c(mvlt)-cSet]
                X0 =  [xim, xiv L, t]
            Returns:
                X  = [J^(-1)][F(X0)]
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
            Lrange = np.abs(max(s.Lbounds) - min(s.Lbounds))
            trange = np.abs(max(s.tbounds) - min(s.tbounds))
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
        Lmin = s.Lbounds[0]+1e-6
        Lmax = s.Lbounds[1]-1e-6
        tmin = s.tbounds[0]+1e-6
        tmax = s.tbounds[1]-1e-6
        Lstart = (Lmax-Lmin)*0.25+Lmin
        tstart = (tmax-tmin)*0.9+tmin

        file_path = os.path.join(s.result_dir, "newtonsolve_lastsolution.txt")
        if os.path.isfile(file_path) and useStoredSolution:
            guess = np.loadtxt(file_path)
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
            if guess[2] <= s.Lbounds[0]:
                guess[2] = Lmin
            elif guess[2] >= s.Lbounds[1]:
                guess[2] = Lmax
            if guess[3] <= s.tbounds[0]:
                guess[3] = tmin
            elif guess[3] >= s.tbounds[1]:
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
        np.savetxt(file_path, guess)
        return [guess[2], guess[3]]

    def create_table_aux(self, args):
        """Auxiliary function used in phi_mvhc for parallelization. 
        The package used for parallelization ("concurrent") requires that the function being parallelized is defined 
        in the global scope.
        """
        # Generic table-generating function
        return self.create_lookup_table(*args)

    def phi_mvhc(self, phi, parallel:bool = True, recreate_all = False):
        """
        Creates a table of phi values in terms of xim, xiv, h, and c
        Inputs:
            phi = single property or list of properties for which values will be tabulated (ex. 'T', 'rho', etc.), case sensitive. Default = 'T'
                Available phi are viewable using "parse_data(params)[1]".
                NOTE: c (progress variable) is available in the data. By default, c ≡ y_CO2 + y_CO + y_H2O + yH2.
                This definition can be changed by modifying the c_components parameter.
            parallel:bool = if set to True (default), the code will attempt to create tables in parallel.
            recreate_all:bool = if set to True, the code will re-create all tables, even if they already exist.
            
        Outputs:
            phi_mvhc_arr = Array of phi functions phi = phi(xim, xiv, h, c)
                NOTE: if only one phi is specified, if will still be returned in a single-element array.
            tableArr = array of [table, indices] for each phi, beginning with h and c.

        """
        s = self
        # ------------ Pre-processing
        # Remove h and c from phi (they are table axis, and so don't need to be tabulated)
        phi = [p for p in phi if p != 'h' and p != 'c']

        # Ensure array-like
        if type(phi) == type('str'):
            phi = [phi,]

        # ------------ Compute tables, parallel or serial
        ####### Serial computation
        if not parallel: 
            # Create h & c tables
            if recreate_all or 'h' not in self.table_storage or 'h' not in self.indices_storage:
                h_table, h_indices = s.create_lookup_table('h')
            else:
                h_table = self.table_storage['h']
                h_indices = self.indices_storage['h']
            if recreate_all or 'c' not in self.table_storage or 'c' not in self.indices_storage:
                c_table, c_indices = s.create_lookup_table('c')
            else:
                c_table = self.table_storage['c']
                c_indices = self.indices_storage['c']

            # Set normalization factor
            s.norm = np.max(np.abs(h_table))/np.max(c_table)

            # Create h & c interpolators
            if recreate_all or 'h' not in s.mvlt_interp_storage:
                Ih = s.create_interpolator_mvlt('h', h_table, h_indices)
            else:
                Ih = s.mvlt_interp_storage['h']
            if recreate_all or 'c' not in s.mvlt_interp_storage:
                Ic = s.create_interpolator_mvlt('c', c_table, c_indices)
            else:
                Ic = s.mvlt_interp_storage['c']
        
            # Create array containing phi tables
            for p in phi:
                # Get base table with phi data
                if recreate_all or p not in s.table_storage or p not in s.indices_storage:
                    table, indices = s.create_lookup_table(p)
                else:
                    table = s.table_storage[p]
                    indices = s.indices_storage[p]
        
                # Create interpolator for phi
                if recreate_all or p not in s.mvlt_interp_storage:
                    InterpPhi = s.create_interpolator_mvlt(p, table, indices)
                else:
                    InterpPhi = s.mvlt_interp_storage[p]
                
                # Create function phi(xim, xiv, h, c)
                def create_phi_table(interp_phi):
                    # Auxiliary function to create phi_table with correct interp_phi binding
                    def phi_table(xim, xiv, h, c, useStoredSolution = True, solver='gammachi', 
                                extrapolate:bool = True, bound:bool = False, minVal = None, 
                                maxIter = 100, saveSolverStates = False,
                                LstepParams = [0.25, 0.01, 0.003], tstepParams = [0.25, 9.5, 0.02], detailedWarn = False,):
                        """Returns phi for given xim, xiv, h, c by inverting (h,c) -> (L,t) then interpolating.
                        Inputs:
                            xim = mean mixture fraction
                            xiv = mixture fraction variance
                            h   = enthalpy
                            c   = progress variable
                            useStoredSolution = if set to False, the solver will not use the last solution as its initial guess. 
                                Using the last initial guess (default) is generally good: CFD will solve cell-by-cell, and nearby
                                cells are expected to have similar values of phi.
                            solver = 'gammachi' (default) or 'newton'. Selects which solver to use for (h,c) -> (L,t) inversion.
                            extrapolate: if True (default), the fucntion allows extrapolation outside of the table. 
                                If False, the function behaves according to the 'bound' parameter.
                            bound: if False (default), the function errors when input values are outside of the table bounds.
                                If True, the function will force the input values to be within the table bounds.
                            minVal: if set to a float value, the returned phi value will be no less than minVal.
                                Useful for ensuring properties like mass fraction remain non-negative.
                        Inputs only used for solver = 'newton':
                            maxIter:int = sets a limit for the maximum iterations the solver should make. Default = 100
                            saveSolverStates: bool = if set to True (not default), the solver states will be saved to a file in the folder "solver_data".
                                Useful for analyzing solver convergence.
                            LstepParams = array of parameters used to relax the solver
                                LstepParams[0] = 0.25; normal max step size (% of domain)
                                LstepParams[1] = 0.01; threshold value of L, below which the max step size is reduced to
                                LstepParams[2] = 0.003; reduced max step size (% of domain)
                            tstepParams = array of parameters used to relax the solver
                                tstepParams[0] = 0.25; normal max step size (% of domain)
                                tstepParams[1] = 9.5; threshold value of t, above which the max step size is reduced to
                                tstepParams[2] = 0.02; reduced max step size (% of domain)
                            detailedWarn = Only used for solver = 'newton'.
                                If set to true, more detailed warnings will be raised when the solver does not converge.    
                        """
                        # Invert from (h, c) to (L, t), then return interpolated value.
                        if solver == 'gammachi':
                            L, t = s.Lt_from_hc_GammaChi(h, c, xim, xiv, Ih, Ic, useStoredSolution)
                        elif solver == 'newton':
                            L, t = s.Lt_from_hc_newton(h, c, xim, xiv, Ih, Ic, s.norm, detailedWarn, maxIter, 
                                                    saveSolverStates, useStoredSolution, LstepParams, tstepParams)
                        else:
                            raise ValueError("Invalid solver specified. Use 'gammachi' or 'newton'.")
                        if minVal is not None:
                            return max(minVal, interp_phi(xim, xiv, L, t, extrapolate=extrapolate, bound=bound))
                        return interp_phi(xim, xiv, L, t, extrapolate=extrapolate, bound=bound) 
                    
                    return phi_table

                s.phi_mvhc_funcs[p] = create_phi_table(InterpPhi)  # Store function in the class
            return s.phi_mvhc_funcs, s.table_storage
            
        ####### Parallel computation
        else:
            # Parallel table creation (should be reviewed)
            # Try using 'fork', fall back to 'spawn'
            try:
                # 'fork' is only available for Unix-like systems and is faster
                ctx = mp.get_context('fork')
            except ValueError:
                # 'fork' is not available, use 'spawn'. Slower, but more compatible.
                ctx = mp.get_context('spawn')
            
            phi = ['h', 'c'] + list(phi) # Need to create h and c tables too, so add them at the beginning. 
            phi_tables = phi.copy()
            if recreate_all:
                pass
            else:
                # Remove phi that already exist
                phi_tables = [p for p in phi_tables if p not in s.table_storage or p not in s.indices_storage]
            force_arr = np.full(len(phi_tables), recreate_all)
            queue = mp.Manager().Queue()
            table_args = [(phi_tables[i],force_arr[i], queue) for i in range(len(phi_tables))] # Arguments for each table's creation
            
            sentinels_received = 0
            sentinels_expected = len(table_args)

            if s.flmt_data is None:
                # No processed data yet generated: must generate.
                print("Parsing flamelet data...")
                s.parse_data()
            print()
            print(f"Beginning parallel table creation for phis {phi_tables}.")
            futures = []
            try:
                with ProcessPoolExecutor(mp_context=ctx) as executor:
                    futures = {executor.submit(s.create_table_aux, args): idx for idx, args in enumerate(table_args)}

                    while sentinels_received < sentinels_expected:
                        try:
                            message = queue.get_nowait()
                            if 'SENTINEL' in message:
                                sentinels_received += 1
                            else:
                                print(message, flush=True)
                        except mp.queues.Empty:
                            time.sleep(0.5)
                    
                    results = {}
                    for future in concurrent.futures.as_completed(futures):
                        idx = futures[future]
                        try:
                            results[idx] = future.result()
                        except Exception as e:
                            print(f"Table creation for index {idx} (phi = {phi_tables[idx]}) generated an exception: {e}")

                # Redundantly store tables and indices (parallel process cannot modify class variables, apparently).
                for i, p in enumerate(phi_tables):
                    s.table_storage[p] = results[i][0]
                    s.indices_storage[p] = results[i][1]
                print("All tables created. Creating interpolators...")
                # Create h & c interpolators -- These should only be set to cubic interpolation with a very dense table.
                h_table, h_indices = s.table_storage['h'], s.indices_storage['h']
                c_table, c_indices = s.table_storage['c'], s.indices_storage['c']
                if recreate_all or 'h' not in s.mvlt_interp_storage:
                    Ih = s.create_interpolator_mvlt('h', h_table, h_indices)
                else:
                    Ih = s.mvlt_interp_storage['h']
                if recreate_all or 'c' not in s.mvlt_interp_storage:
                    Ic = s.create_interpolator_mvlt('c', c_table, c_indices)
                else:
                    Ic = s.mvlt_interp_storage['c']
                
                # Set normalization factor
                s.norm = np.max(np.abs(h_table))/np.max(c_table)
                
                print("Creating phi(xim, xiv, h, c) functions...")
                # Create functions for phi(xim, xiv, h, c)
                for i, p in enumerate(phi):
                    if recreate_all or p not in s.mvlt_interp_storage:
                        print(f"Creating interpolator for phi = {p}")
                        InterpPhi = s.create_interpolator_mvlt(p, s.table_storage[p], s.indices_storage[p])
                    else:
                        InterpPhi = s.mvlt_interp_storage[p]
                    
                    # Create function phi(xim, xiv, h, c)
                    def create_phi_table(interp_phi):
                        # Auxiliary function to create phi_table with correct interp_phi binding
                        def phi_table(xim, xiv, h, c, useStoredSolution = True, solver='gammachi', 
                                    extrapolate:bool = True, bound:bool = False, minVal = None, 
                                    maxIter = 100, saveSolverStates = False,
                                    LstepParams = [0.25, 0.01, 0.003], tstepParams = [0.25, 9.5, 0.02], detailedWarn = False,):
                            """Returns phi for given xim, xiv, h, c by inverting (h,c) -> (L,t) then interpolating.
                            Inputs:
                                xim = mean mixture fraction
                                xiv = mixture fraction variance
                                h   = enthalpy
                                c   = progress variable
                                useStoredSolution = if set to False, the solver will not use the last solution as its initial guess. 
                                    Using the last initial guess (default) is generally good: CFD will solve cell-by-cell, and nearby
                                    cells are expected to have similar values of phi.
                                solver = 'gammachi' (default) or 'newton'. Selects which solver to use for (h,c) -> (L,t) inversion.
                                extrapolate: if True (default), the fucntion allows extrapolation outside of the table. 
                                    If False, the function behaves according to the 'bound' parameter.
                                bound: if False (default), the function errors when input values are outside of the table bounds.
                                    If True, the function will force the input values to be within the table bounds.
                                minVal: if set to a float value, the returned phi value will be no less than minVal.
                                    Useful for ensuring properties like mass fraction remain non-negative.
                            Inputs only used for solver = 'newton':
                                maxIter:int = sets a limit for the maximum iterations the solver should make. Default = 100
                                saveSolverStates: bool = if set to True (not default), the solver states will be saved to a file in the folder "solver_data".
                                    Useful for analyzing solver convergence.
                                LstepParams = array of parameters used to relax the solver
                                    LstepParams[0] = 0.25; normal max step size (% of domain)
                                    LstepParams[1] = 0.01; threshold value of L, below which the max step size is reduced to
                                    LstepParams[2] = 0.003; reduced max step size (% of domain)
                                tstepParams = array of parameters used to relax the solver
                                    tstepParams[0] = 0.25; normal max step size (% of domain)
                                    tstepParams[1] = 9.5; threshold value of t, above which the max step size is reduced to
                                    tstepParams[2] = 0.02; reduced max step size (% of domain)
                                detailedWarn = Only used for solver = 'newton'.
                                    If set to true, more detailed warnings will be raised when the solver does not converge.    
                            """
                            # Invert from (h, c) to (L, t), then return interpolated value.
                            if solver == 'gammachi':
                                L, t = s.Lt_from_hc_GammaChi(h, c, xim, xiv, Ih, Ic, useStoredSolution)
                            elif solver == 'newton':
                                L, t = s.Lt_from_hc_newton(h, c, xim, xiv, Ih, Ic, s.norm, detailedWarn, maxIter, 
                                                        saveSolverStates, useStoredSolution, LstepParams, tstepParams)
                            else:
                                raise ValueError("Invalid solver specified. Use 'gammachi' or 'newton'.")
                            if minVal is not None:
                                return max(minVal, interp_phi(xim, xiv, L, t, extrapolate=extrapolate, bound=bound))
                            return interp_phi(xim, xiv, L, t, extrapolate=extrapolate, bound=bound) 
                        
                        return phi_table

                    s.phi_mvhc_funcs[p] = create_phi_table(InterpPhi)  # Store function in the class

                return s.phi_mvhc_funcs, s.table_storage
            
            except KeyboardInterrupt:
                for future in futures:
                    if not future.done():
                        future.cancel()
                print("Shutdown complete.", flush=True)
                sys.exit(1)

    def reset_funcs(self, reset_interps:bool = False, parallel:bool = True):
        """
        Resets phi_mvhc_funcs and, optionally, mvlt_interp_storage.
        This is useful if a set of tables has been pickled using the save() method, 
        then loaded using the load() method into a different context (for example, a different
        machine or environment)."""
        s = self
        phis = list(s.phi_mvhc_funcs.keys())
        s.phi_mvhc_funcs = {}
        if reset_interps:
            s.mvlt_interp_storage = {}
        
        # Recreate funcs and interpolators
        s.phi_mvhc(phis, parallel=parallel, recreate_all=False)

        print("Recreated functions and interpolators.")
        return None

    def save(self, name = 'table'):
        """
        Saves this instance of table to a file.
        Args:
            name = Name of the file to save the table as. Default = 'table'.
        """
        s = self
        path = os.path.join(self.result_dir, name+'.pkl')
        with open(path, 'wb') as f:
            dill.dump(self, f)
        
        date = time.strftime("%Y%m%d")
        np.savetxt(os.path.join(self.result_dir, name+'_metadata.txt'), 
                   [s.nxim, s.ximLfrac, s.ximGfrac, s.nxiv, len(s.tvals), len(s.Lvals), len(s.gammaValues), date],
                header="nxim, ximLfrac, ximGfrac, nxiv, nt, nL, ngamma, date", fmt="%s")

        print(f"Table saved to {path}")

def load(name = 'table'):
    """
    Loads a table from a file.
    Args:
        name = Name of the file to load the table from. Default = 'table'.
    """
    # Get current directory
    try:
        current_dir = os.path.dirname(os.path.abspath(__file__))
    except:
        # Get the directory of the current jupyter notebook
        current_dir = os.path.dirname(os.path.abspath(''))
    result_dir = os.path.join(current_dir, 'results') 
    
    # Warn if table was computed more than a month ago
    metadata = np.loadtxt(os.path.join(result_dir, name+'_metadata.txt'), skiprows=1)
    print(f"Table {name} was created on {metadata[7]:.0f}")
   
    # Load the table
    path = os.path.join(result_dir, name+'.pkl')
    with open(path, 'rb') as f:
        print(f"Table loaded from {path}")
        loaded = dill.load(f)

    loaded.result_dir = result_dir # Update the result_dir in case the table is being used on a different machine
    loaded.current_dir = current_dir # Update the current_dir in case the table is being used on a different machine

    loaded.reset_funcs(reset_interps=True) # Reset functions and interpolators to ensure they work in the new context

    return loaded
