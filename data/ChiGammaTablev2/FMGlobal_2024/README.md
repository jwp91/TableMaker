# FM Global Meeting 2024

Code driver is setup to run the TNF piloted jet flames: C, D, E, F.
* See driver.cc
* See also flame.cc setGrid --> using uniform grid since the stoichiometric mixture fraction is 0.353

See the flame/run folder for data and processing scripts.

See the results.pptx file for plots that were made, along with notes about those plots.

The flame code outputs files L_00.dat, L_01.dat ... L_18.dat.
* Files L_00.dat to L_09.dat are for adiabatic, unsteady extinction of L=0.00134 m.
* Files L_10.dat to L_00.dat are for steady state, adiabatic, burning flames of L=0.0014 m to L=0.2 m.

The main analysis code is in tableMaker.ipynb.
* The folder jared_porter has files tableMaker.py and flameTableMaker_workspace.ipynb that are Jared's codes.
* I modified his python notebook slightly for some output.
* My code doesn't do any unsteady heat loss, so there is just the progress variable space.
* Jared's code does both both L and t (progress variable and enthalpy).
* My code is simpler and makes more assumptions about the file names and such.
* There is also betaPdf.py, which I modified from Jared's LiuInt.py.

Folders pmC.scat, pmD.scat, pmE.scat, and pmF.scat contain processed TNF data. 
* pmC.scat/C075.Yall is flame C at downstream location x/d = 7.5. It contains individual laser data shots for various radial locations (the last column in the file). 
* pmC.scat/C075.Yall_proc is my processed data (see associated process_data.ipynb file) at each radius the Favre values of mixture fraction (F), its rms, progress variable C, enthalpy (H: J/kg), temperature, and species mass fractions are given. 
    * These are the experimental data that is plotted, and also used as a-priori input to the table for table/experimental data comparison.
    * I've stripped off the redundant radial values (which cross the centerline).
    * Note, rms is not variance, need to square rms to get variance (this is obvious, but I made the dumb mistake of using rms in the file where variance was wanted and couldn't get good comparisons with that simple but significant error.)
    * Also, the table doesn't like looking up a zero mixture fraction (so, for a couple downstream locations in flame D I commented out radial points); need to fix this.

Files D80_20_19.dat, and similar, were for comparing table resolution, and refer to flame D, with 80 mixture fraction points, 20 variances, and 19 progress variable values. Other plots (pdf files) can be referred to the results.pptx presentation.
* pdf_D30_r12.pdf and similar are comparisons of the beta pdf (Favre, parameterized by the Favre mean mixture fraction and its Favre variance), and the experimental pdf.
* The folder pmCDEFarchives is the original TNF data with my original processing. I then copied the relevent folders to pmC.scat, etc., and edited them as noted above.
    * (So there is some redundancy there).
    * Earlier, I had compared my Favre averaging with the TNF provided data in the pmC.stat (etc.) folders, to make sure I was doing things correctly, or at least as a verification of my processing codes.

*(Be careful of some of the plots as I did all the analysis really quickly, adding plots to the results.pptx presentation. It is possible that a plot got overwritten with a different case by not commenting out the plt.savefig command. So, if you do a detailed comparison and something looks off, that might be why. You'd need to rerun to compare.)*

Using mixture average transport properties (nonunity Le).

The older_run_folder has output from an earlier set of runs. They should be correct, and consistent with the later runs in the parent run folder, but I moved to the latter for convenience in processing when I started using my tableMaker.ipynb to get stuff done quickly for the presentation. This older_run_folder has folder TableMaker, which is from Jared Porter (the original that I downloaded from his github.). 
* See plots.ipynb for some initial plots I made for the presentation. 
* These include processing the flame data, like convolving it over the beta pdf to show the behavior of the table in the first 10 slides of the results.pptx presentation (since I didn't need to do the a-priori query of the table for that.). 
    * Obviously, it would be convenient to not have the redundancy and move the plots.ipynb back to the parent run and have it operate on the L_00.dat (etc.) files in there. (But I have other things to do...)
* Note: case L=0.00135 is there instead of case L=0.00134. 
    * The 0.00135 doesn't blow out under strain, but it does blow out as soon as radiation is turned on. That is interesting and is included in Slide 8 of the presentation.
        * That slide actually refers to L=1.37, which I think is right, but that was probably using 80 mixture fraction points (and likely before I was using a uniform grid), so the blowout size was a little different. (I think my memory is right on this...)
    * The 0.00134 case does blow out under strain.

## Notes from discussion at the meeting, and subsequent ideas
* Arnaud Trouve doesn't like flamelet models. Says they cannot capture heat release.
    * John Hewson has seen similar things
    * This seems strange to me for several reasons.
    * John notes this might be because heat release happens in very narrow regions of mixture fraction.
    * So, it might be worth looking at heat release rate profiles, rather than simply temperature.
        * I'm not sure if we can do this apriori with the TNF data or not. They give species profiles, so maybe we can compute key reaction rates, which are needed for heat release. In any case, if species and temperature profiles are resolved, then we should be able to get heat release right. But we might need to pay attention to the mixture fraction grid. And also try different fuels.
* Todo: table validation with heat loss, using HiPS and/or ODT
* Todo: redo TNF comparisons with the Le=1 assumption
* Todo: TNF comparisons with scalars besides temperature
* Todo: "apparent" heat loss due to differential diffusion (consider for flame F especially).
* Todo: See how much heat loss there actually is in the TNF flames (by comparing enthalpy to adiabatic enthalpy)
* Todo: Joint PDFs of mixture fraction and progress variable.
* Todo: parameterized heat loss versus radiative heat loss (this can force radiative extinction for any strain, which doesn't happen for low enough strain when using radiation and unsteady heat loss.)


