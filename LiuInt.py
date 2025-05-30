import numpy as np
from scipy.special import gamma, gammaln
from scipy.integrate import quad

def normFunc(a, b):
    """Computes the normalization factor for the beta-PDF"""
    q = gammaln(a+b) - gammaln(a) - gammaln(b)
    if np.abs(q) > 709:
        # Avoid overflow or zeroing out
        return np.inf
    else:
        return np.exp(q)

def IntegrateForPhiBar(ξm, ξv, ϕ, ϵ = 1e-6, low = 0, upp = 1, silence:bool = False):
    """
    Function for calculating ϕ_avg for a given ξ_avg and ξ_variance. 
    Parameters:
        ξm: Mixture fraction mean
        ξv: Mixture fraction variance
         ϕ: Property as a function of mixture fraction [Ex. Temp(ξ) or ρ(ξ)]. Can be set as a constant.
             * NOTE: Function must be valid on the domain [0,1]
         ϵ: Small parameter to solve βPDF boundary singularity issue
            * Recommended value of ϵ=1e-6
            * Per the literature, this approximation is valid for B<40000.
                * The lowest variance this allows is a ξv of 3.7e-6 (occurs at ξm = 0.33337)
            *Reference: Liu et. al, July 2002, "A robust and..." [ https://www.sciencedirect.com/science/article/abs/pii/S1290072902013704 ]
        low: lower bound for integration. Will be zero in most applications
        upp: upper bound for integration. Will be one in most applications
        silence: bool; if set to True, no warnings will be printed
    """
    # Avoids casting error if ϕ(ξ) is a constant
    if(type(ϕ)==int):
        return ϕ
    
    # If ξm is numerically zero or one, the variance must be zero:
    if ξm == 0 and ξv != 0:
        if not silence:
            print(f"LiuInt Error: ξv must be zero because ξm==0. ξv inputted = {ξv}, corrected to 0")
        return ϕ(ξm)
    elif ξm == 1 and ξv != 0:
        if not silence:
            print(f"LiuInt Error: ξv must be zero because ξm==1. ξv inputted = {ξv}, corrected to 0")
        return ϕ(ξm)        

    # At max variance, the PDF becomes 2 delta functions, the heights of which are proportional to the mean mixture fraction. 
    ξv_max = ξm*(1-ξm)
    if ξv == ξv_max:
        return (1-ξm)*ϕ(0) + ξm*ϕ(1)
    
    # If variance is zero, return the property at the mean mixture fraction
    if ξv == 0:
        return ϕ(ξm)
    

    # Non-normalized βPDF
    def P(ξ, a, b):
        P = ξ**(a-1) * (1-ξ)**(b-1)
        return P
    

    # Function to be integrated: ϕ(ξ)*P(ξ; ξm, ξv)
    def ϕP(ξ, a, b):
        return ϕ(ξ)*P(ξ, a, b)
        
    # Calculate parameters
    a = ( ξv_max/ξv - 1 )*ξm
    b = ( ξv_max/ξv - 1 )*(1 - ξm)

    # Avoid βPDF singularities.
    zero = 1e-8
    if a <= zero:
        if not silence:
            print(f"LiuInt Warning: 'a' corrected from {a} to {zero}")
        a = zero
    if b <= zero:
        if not silence:
            print(f"LiuInt Warning: 'b' corrected from {b} to {zero}")
        b = zero
    
    # Calculate the normalization constant
    norm = normFunc(a, b)
    if norm == np.inf:
        # a or b is too large to compute norm
        if a < 1 or b < 1:
            # Approaching a delta function on one or both sides
            # Handle boundary singularity from a or b < 1 (Liu 767)
            normDen = ϵ**a/a + quad(P, ϵ, 1-ϵ, args = (a, b), points = [ϵ*1.01,1-ϵ*1.01])[0] + ϵ**b/b
            if normDen == 0 or normDen == np.inf:
                # The probability density function has gotten as close to a delta function as 
                # these approximations can tolerate. Return a value
                if ξv <= 0.5*ξv_max:
                    # For low variances, return the property at the mean:
                    return ϕ(ξm) 
                else:
                    # For high variances, return a weighted average of the endpoints:
                    return (1-ξm)*ϕ(0) + ξm*ϕ(1)
            else:
                norm = 1/normDen
        else:
            # Simply returning the mean value works more consistently than Liu's method below.
            return ϕ(ξm)   # Comment this line out to use Liu's method. 
            # Approaching a delta function in the middle
            # Handle very large a or b (Liu 767, Eqs. 25 & 26)
            fmax = 1/(1 + (b - 1)/(a - 1))
            if a > 1000 and a>= b:
                # Limit value of a
                a = 1000
                b = (a - 1 - fmax*(a - 2))/fmax
            elif b > 1000 and b >= a:
                # Limit value of b
                b = 1000
                a = (1 + fmax*(b - 2))/(1 - fmax)
            else:
                # The code shouldn't get to here
                raise ValueError(f"""LiuInt Err1: normalization factor could not be computed.
                                a  = {a}
                                b  = {b}
                                ξm = {ξm}
                                ξv = {ξv}""")
            norm = normFunc(a, b)
            if norm == np.inf:
                # This means that the probability density function has gotten as close to a delta function as 
                # these approximations can tolerate. At this point, ignore the variance and return the property
                # evaluated at the mean.
                return ϕ(ξm)
    
    #--------- Correction for boundary singularity (Liu 767). Utilizes the fact that ϕ(0) and ϕ(1) are known at the endpoints.
    ϕ0 = ϕ(0)
    ϕ1 = ϕ(1)
    
    #--------- BASE CODE
    p1 = ϕ0*(ϵ**a)/a                          # 0   < ξ < ϵ
    p2 = quad(ϕP, ϵ, 1-ϵ, args = (a, b), points = [ξm,])[0]   # ϵ   < ξ < 1-ϵ
    p3 = ϕ1*(ϵ**b)/b                          # 1-ϵ < ξ < ϵ
    
    #--------- Conditionals to handle instances where bounds are not (0,1)
    if low == 0:
        if upp == 1:
            pass
        elif 0 <= upp <= ϵ:
            p1 = p1*(upp/ϵ)
            p2, p3 = 0, 0
        elif ϵ < upp < 1-ϵ:
            p2 = quad(ϕP, ϵ, upp, args = (a, b))[0]
            p3 = 0
        elif upp < 1:
            p3 = p3*(upp-(1-ϵ))/ϵ
            #If upp == 1, p3 is already accurate
    elif 0 < low <= ϵ:
        if 0 <= upp <= ϵ:
            p1 = p1*(upp-low)/ϵ
            p2, p3 = 0, 0
        else:
            p1 = p1*(ϵ-low)/ϵ
            if ϵ < upp < 1-ϵ:
                p2 = quad(ϕP, ϵ, upp, args = (a, b))[0]
                p3 = 0
            elif upp < 1:
                p3 = p3*(upp-(1-ϵ))/ϵ
                #If upp == 1, p3 is already accurate
    elif ϵ < low < 1-ϵ:
        p1 = 0
        if ϵ < upp < 1-ϵ:
            p2 = quad(ϕP, low, upp, args = (a, b))[0]
            p3 = 0
        else:
            p2 = quad(ϕP, low, 1-ϵ, args = (a, b))[0]
            if upp < 1:
                p3 = p3*(upp-(1-ϵ))/ϵ
                #If upp == 1, p3 is already accurate
    else:
        p1, p2 = 0,0
        p3 = p3*(upp-low)/ϵ

    # Error detection
    if np.isnan((p1+p2+p3)*norm):
        print("ERROR: returned value is nan. Details:")
        print(f"p1 = {p1}, p2 = {p2}, p3 = {p3}")
        print(f"xim = {ξm}, xiv = {ξv}, a = {a}, b = {b}")
        print(f"norm = {norm}")

    return (p1+p2+p3)*norm        # Normalizes the βPDF integration before returning.

def bPdf(ξ, ξm, ξv, ϵ = 1e-6, silence = True):
    """
    Calculates P(ξ) according to the Beta PDF
    Parameters:
         ξ = Mixture fraction. Can be a single value or array
        ξm = Mean mixture fraction
        ξv = Mixture fraction variance
        ϵ: Small parameter to solve βPDF boundary singularity issue
            * Recommended value of ϵ=1e-6
            * Per the literature, this approximation is valid for B<40000.
                * The lowest variance this allows is a ξv of 3.7e-6 (occurs at ξm = 0.33337)
            *Reference: Liu et. al, July 2002, "A robust and..." [ https://www.sciencedirect.com/science/article/abs/pii/S1290072902013704 ]
    """
    def P(ξ, a, b):
        P = ξ**(a-1) * (1-ξ)**(b-1)       # βPDF, non-normalized
        return P
    

    np.seterr(divide='ignore')           # Disables ZeroDivisionError for when ξ = 0 or 1

    # Calculate parameters
    ξv_max = ξm*(1-ξm)
    a = ( ξv_max/ξv - 1 )*ξm
    b = ( ξv_max/ξv - 1 )*(1-ξm)

    # Avoid βPDF singularities.
    zero = 1e-8
    if a <= zero:
        if not silence:
            print(f"LiuInt Warning: 'a' corrected from {a} to {zero}")
        a = zero
    if b <= zero:
        if not silence:
            print(f"LiuInt Warning: 'b' corrected from {b} to {zero}")
        b = zero
    
    norm = normFunc(a, b)
    if norm == np.inf:
        # a or b is too large
        if a < 1 or b < 1:
            # Approaching a delta function on one side
            # Handle boundary singularity from a or b < 1 (Liu 767)
            normDen = ϵ**a/a + quad(P, ϵ, 1-ϵ, args = (a, b), points = [ϵ*1.01,1-ϵ*1.01])[0] + ϵ**b/b
            if normDen == 0 or normDen == np.inf:
                # This means that the probability density function has gotten as close to two delta functions as 
                # these approximations can tolerate.
                return (1-ξm)*ϕ(0) + ξm*ϕ(1)
            else:
                norm = 1/normDen
        else:
            # NOTE: this needs further evaluation.
            # Handle very large a or b (Liu 767)
            fmax = 1/(1 + (b - 1)/(a - 1))
            if a > 500 and a>= b:
                # Limit value of a
                a = 500
                b = (a - 1 - fmax*(a - 2))/fmax
            elif b > 500 and b >= a:
                # Limit value of b
                b = 500
                a = (1 + fmax*(b - 2))/(1 - fmax)
            else:
                # The code shouldn't get to here
                raise ValueError(f"""LiuInt Err1: normalization factor could not be computed.
                                a  = {a}
                                b  = {b}
                                ξm = {ξm}
                                ξv = {ξv}""")
            norm = normFunc(a, b)
            if norm == np.inf:
                # This means that the probability density function has gotten as close to a delta function as 
                # these approximations can tolerate. At this point, ignore the variance and return the property
                # evaluated at the mean.
                return 1.0
        
    P = ξ**(a-1) * (1-ξ)**(b-1) * norm
        
    return P

def example():
    """
    Full demonstration of ϕAvg and supporting functions.
    Displays plots of data for visual reference.
    """
    from scipy.interpolate import interp1d
    import numpy as np
    import matplotlib.pyplot as plt
    import cantera as ct
    
    gas = ct.Solution('gri30.yaml')

    P   = 101325
    T0  = 300.
    x0  = "O2:2, N2:3.76"     # --> 21% O2, 79% N2
    T1  = 300.
    x1  = "CH4:1 N2:1"

    #--------- Set state for ξ=0 and ξ=1
    gas.TPX = T0, P, x0       #Sets T, P, x
    h0 = gas.enthalpy_mass    
    y0 = gas.Y                #Gets mass fractions
    gas.TPX = T1, P, x1       #Sets T, P, x
    h1 = gas.enthalpy_mass    
    y1 = gas.Y                #Gets mass fractions

    def T(ξ):
        y = y0*(1-ξ) + y1*ξ
        h = h0*(1-ξ) + h1*ξ
        gas.HPY = h, P, y
        gas.equilibrate("HP")   #Equilibrate, keeping specific enthalpy and pressure constant.
        return gas.T
    
    #--------- Specify parameters for the βPDF
    ξm = 0.25
    ξv = 0.1
    
    print("""
      ------- Data -------
             ξ0 = Air
             ξ1 = 1:1 mix of CH4 and N2
             ξm = 0.25
             ξv = 0.1
       Pressure = 101.325 kPa
    Temperature = 300 K""")
    
    #--------- Create tabulated data for Temperature(ξ)
    ξ = np.linspace(0,1,50)
    Ts = np.empty(len(ξ))
    for i in range(len(ξ)):
        Ts[i] = T(ξ[i])
        
    Tinterp = interp1d(ξ, Ts, kind = 'cubic')
    
    #--------- Plot BetaPDF and Temp data for visual reference
    fig, ax1 = plt.subplots()
    ax2 = ax1.twinx()
    
    plt.title("βPdf and Temperature")

    pts = np.linspace(0,1, 40)
    ax1.plot(pts, Tinterp(pts), color = 'royalblue', label = "Interpolated Temp. Data")
    ax1.plot(ξ, Ts, '.', color = 'r', label = "Simulated Temp. Data")
    ax1.set_xlabel("ξ")
    ax1.set_ylabel("T (K)")
    
    ax2.plot(ξ, bPdf(ξ, ξm, ξv), '--', color = 'darkviolet', label = f"βPdf(ξm={ξm}, ξv={ξv})")
    ax2.set_ylabel("P(ξ)")
    fig.legend(loc="upper right", bbox_to_anchor=(1,1), bbox_transform=ax1.transAxes);
    
    #--------- Compute T_avg using ϕAvg function
    Tavg = IntegrateForPhiBar(ξm, ξv, ϕ = Tinterp)
    print()
    print(f"""
    Average Temperature is calculated by calling
    >> ϕAvg(ξm, ξv, ϕ = Tinterp)
    ...where 'Tinterp' is an interpolated function T(ξ) from tabulated data.
    For this example, Average Temperature = {round(Tavg,3)} K
    
    Charted data for visual reference:
    """)