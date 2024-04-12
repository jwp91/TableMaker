import numpy as np
from scipy.special import gamma
from scipy.integrate import quad

def IntegrateForPhiBar(ξm, ξv, ϕ, ϵ = 1e-6, low = 0, upp = 1):
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
    """
    if(type(ϕ)==int):
        return ϕ              # Avoids casting error if ϕ(ξ) is a constant
    #--------- Function to be integrated: ϕ(ξ)*P(ξ; ξm, ξv)
    def P(ξ, a, b):
        P = ξ**(a-1) * (1-ξ)**(b-1)       # βPDF, non-normalized
        return P
        
    def ϕP(ξ, a, b):
        return ϕ(ξ)*P(ξ, a, b)
    
    #--------- βPDF parameters:
    #Note: if ξm is numerically zero or one, the variance must be zero:
    if ξm == 0 and ξv != 0:
        ξv = 0
        print(f"LiuInt Error: ξv must be zero because ξm==0. ξv inputted = {ξv}")
    if ξm == 1 and ξv != 0:
        ξv = 0
        print(f"LiuInt Error: ξv must be zero because ξm==1. ξv inputted = {ξv}")

    #Treating ξv boundary conditions:
    if ξv == 0:
        return ϕ(ξm)

    ξv_max = ξm*(1-ξm)
    if ξv == ξv_max:
        return (1-ξm)*ϕ(0) + ξm*ϕ(1) 
        #This can be derived knowing that with max variance, the PDF becomes
        #2 delta functions whose height are proportional to the mean mixture fraction. 
        
    # Calculate parameters
    a = ( ξm*(1-ξm)/ξv - 1 )*ξm
    b = ( ξm*(1-ξm)/ξv - 1 )*(1-ξm)

    # Avoid βPDF singularities. This block shouldn't execute, hence the print statements for debugging. 
    zero = 1e-8
    if a <= zero:
        a = zero
        print(f"LiuInt Warning: 'a' computed to be zero. Corrected to {zero}")
    if b <= zero:
        b = zero
        print(f"LiuInt Warning: 'b' computed to be zero. Corrected to {zero}")

    # Handle very large a and b (Liu 767)
    if a > 500 and a >= b:
        # Limit value of a
        fmax = 1/(1 + (b - 1)/(a - 1))
        a = 500
        b = (a - 1 - fmax*(a - 2))/fmax
        norm = ϵ**a/a + quad(P, ϵ, 1-ϵ, args = (a, b))[0] + ϵ**b/b
    elif b > 500 and b >= a:
        # Limit value of b
        fmax = 1/(1 + (b - 1)/(a - 1))
        b = 500
        a = (1 + fmax*(b - 2))/(1 - fmax)
        norm = ϵ**a/a + quad(P, ϵ, 1-ϵ, args = (a, b))[0] + ϵ**b/b
    else:
        norm = gamma(a+b)/gamma(a)/gamma(b)   # Normalizes PDF to integrate to 1
    

    #--------- Correction for boundary singularity (Liu 767). Utilizes the fact that ϕ(0) and ϕ(1) are known at the endpoints.
    ϕ0 = ϕ(0)
    ϕ1 = ϕ(1)
    
    #--------- BASE CODE
    p1 = ϕ0*(ϵ**a)/a                          # 0   < ξ < ϵ
    p2 = quad(ϕP, ϵ, 1-ϵ, args = (a, b))[0]   # ϵ   < ξ < 1-ϵ
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

    # DEBUGGING
    if np.isnan(p1) or np.isnan(p2) or np.isnan(p3) or np.isnan(norm):
        print("ERROR: returned value is nan. Details:")
        print(f"p1 = {p1}, p2 = {p2}, p3 = {p3}")
        print(f"xim = {ξm}, xiv = {ξv}, a = {a}, b = {b}")
        print(f"norm = {norm}")
    if np.isnan((p1+p2+p3)*norm):
        print("ERROR: returned value is nan. Details:")
        print(f"p1 = {p1}, p2 = {p2}, p3 = {p3}")
        print(f"xim = {ξm}, xiv = {ξv}, a = {a}, b = {b}")
        print(f"norm = {norm}")
    return (p1+p2+p3)*norm        # Normalizes the βPDF integration before returning.

def βPdf(ξ, ξm, ξv, ϵ = 1e-6):
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
    zero = ϵ #Parameter to avoid divide by zero errors
    if ξm == 0:
        ξm = ϵ
    if ξv == 0:
        ξv = ϵ
    a = ξm*( ξm*(1-ξm)/ξv - 1 )
    b = ( ξm*(1-ξm)/ξv - 1 )*(1-ξm)
    
    np.seterr(divide='ignore')           # Disables ZeroDivisionError for when ξ = 0 or 1
    
    P = ξ**(a-1) * (1-ξ)**(b-1) * gamma(a+b)/gamma(a)/gamma(b)
        
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
    
    ax2.plot(ξ, βPdf(ξ, ξm, ξv), '--', color = 'darkviolet', label = f"βPdf(ξm={ξm}, ξv={ξv})")
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