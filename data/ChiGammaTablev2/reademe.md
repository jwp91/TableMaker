Directories
- FMGlobal_2024: 
    - presentation Jared gave: comparison of TNF flames with progress variable, adiabatic
- AIChE_2024: 
    - presentation Jared gave
- diffusion_flame_radiative
    - DOL simulations to address table inversion issues and results
    - These are simulations with the "diffusion_table" mode of ignis, run in the physical coordinate. Previous simulations were with differential diffusion, these here had unity Le.
    - Radiation is on.
    - Parameterized in terms of mixture fraction, progress variable, and enthalpy.
    - Simulations at FMGlobal and AIChE had L and t. At blowout, L wrapped into the t direction (see slides).
    - Simulations here don't do that. At blowout, we continue unsteady in the same direction, but adiabatic. Then to fill in the table for those values in the radiation direction we just copy the adiabatic unsteady profiles into the enthalpy direction. This may or may not be an improvement.
    - Problems:
        - The inversion is not easy (invert from h,c to L, t); it works, but is finicky; fsolve sometimes complains, and the results are sensitive to the initial guess (better to put it in the valley, low L and low t, see plot.ipynb).
        - Digging in, doing the inversion with multilinear interpolation is not great. My mental image was a plane inside a given cell with four corners. But a plane is defined by three points, not four, and with four you can get weird shapes, like a valley. A Delaunay triangulation makes sense.
        - But this suggests using a higher resolution table. So directory "table" is coarse, but "table3" is finer and much better. With that, we can match c and h pretty well. But then then T is very poor. This is because, while c and h match, that is done by extrapolating the table, which goes way outside of the table, and then T is extrapolated, so it isn't good.
        - Looked into different interpolators, etc. 
        - Comparing the ODT points to the table shows that the ODT is often outside the table. This isn't good because we are doing heat loss through radiation, and so there is not much we can do about this. I ran with very large domains too. No help.
- flamelets_with_heat_loss
    - I built a model to do a flamelet table (like the diffusion table), but with a heat loss parameter $\gamma$ instead of radiation. 
    - See the readme file in this folder for details.
    - This works great!
    - I'm doing the tabulation and lookup as for the diffusion_table above. This works fine.
    - However, the way this is formulated allows for a simpler, cheaper inversion.
    - Basically, each heat loss has a full range of progress variable. So, for a given mixture fraction and enthalpy we can know the heat loss parameter. Then for that given heat loss we can find the corresponding progress variable. That is, a one-D inversion instead of a two-D inversion.
    - When plotting all the ODT data, there are heat loss values for which the ODT progress variable is outside of the table. This is not a problem because these progress variable values don't correspond to those heat loss values. To see this, plot progress variable versus $\gamma$ for all the ODT data. You'll see that low progress variable corresponds with low heat loss, and everything is inside the table, so the inversion works. Have a look at table/outc and table/outh to see the coverage.
    - The ignis code in this directory is the revised (for flamelet_table) that was used for the previous diffusion_table cases above. 


