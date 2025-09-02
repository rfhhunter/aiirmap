##**AIIRPower Specific** uses nk_path and a lot of other assumptions on the type of structure to simulate in the mDBL code


"""
Interfacing functions to link aiirmap and Daisy's matlab mDBL (modified detailed balance limit) code.


Get (m)DB(L) optimized thicknesses for a list of inp(uts lists).
    laserWL         : float, laser wavelength [um]
    Eg_in           : float, bandgap [eV]
    numSegs         : int, number of segments
    nk_path         : string, path to nk data
    BLabsTarget     : float, absorption target in fraction
    BLthTarget      : float, thickness constraint
    homoEmFactor    : float or 0, factor size between emitter and base layers for bottom homojunction architecture OR if 0, all heterojnc
    backReflector   : bool: run with perfect mirror back-reflector (True) or run with substrate at bottom (False)
    maxSegTh        : float, maximum segment thickness allowed
    IRE             : float, internal radiative limit in fractional form
    Pin             : float, input power [W/cm2]
    lc              : bool:, include luminescent coupling?
    moleFrac        : float or None, the InGaAsP Ga mole fraction, if None; calculate from the bandgap




~23.07.31
"""


import io
import time
import array
from tqdm import tqdm
from multiprocessing import Pool

#import matlab.engine

from aiirpower_specific import *
from config import *
from . import BeerLambertTools as bl
from . import databasing as dbh
import numpy as np
import pandas as pd
import scipy.optimize as opt



def main():
    #test/exercise functions in this file

    #DB_extract test
    # inputs = [[1.480, 0.8, 10, nk_path, 0.98, 7, moonshotHomoEmFactor, False, 4.0, 1.0, 10, False, 0.142518]]
    # inputs = [[1.550, 0.7322, 10, nk_path, 0.98, 7, moonshotHomoEmFactor, False, 4.0, 1.0, 10, False, 0.0]]
    # [[1.480, 0.8, 4, nk_path, 0.98, 7, moonshotHomoEmFactor, False, 2.5, 0.9, 10, False, 0.142518],
    #           [1.480, 0.8, 4, nk_path, 0.98, 7, moonshotHomoEmFactor, False, 2.5, 0.9, 10, True, 0.142518],
    #           [1.480, 0.8, 4, nk_path, 0.98, 7, 0, False, 2.5, 0.9, 10, False, 0.142518],
    #           [1.480, 0.8, 4, nk_path, 0.98, 7, 0, False, 2.5, 0.9, 10, True, 0.142518]]
    # run_DB_matlab(inputs, printRes=True, runType=0)


    #sendDB_toMatlOpt test
    # indb = dbh.loadDBs(os.path.join(dbdir, "230821_*", "230821_*input.csv"))
    # indb = dbh.loadDBs(os.path.join(dbdir, "230810_paper_on-subs_4umBJ_bl", "*input_1.csv"))
    # indb.dataframe =indb.dataframe.iloc[0:5,:]
    # indb.dbSaveFile(saveName=os.path.join(dbdir, "230913_matlabOpt_smallBatchTest", "random5_from_230810_paper_on-subs_4umBJ_bl.csv"))
    # outdb0 = sendDB_toMatlOpt(indb, justGetEff=True)
    # outdb = sendDB_toMatlOpt(indb, justGetEff=False)
    # outdb.dbSaveFile(saveName=os.path.join(dbdir, "230913_matlabOpt_smallBatchTest", "matlabOpt_IRE-1.csv"))

    indb = dbh.loadDBs(os.path.join(dbdir, "230913_matlabOpt_smallBatchTest", "random5_from_230810*.csv"))
    outdb = sendDB_toMatlOpt(indb, IRE=0.93, justGetEff=False, includeARCinOpt=True)
    # outdb.dbSaveFile(saveName=os.path.join(dbdir, "230913_matlabOpt_smallBatchTest", "230922_matlabOpt_IRE-0p93_ARCincl.csv"))
    outdb = sendDB_toMatlOpt(indb, IRE=1, justGetEff=False, includeARCinOpt=True)
    # outdb.dbSaveFile(saveName=os.path.join(dbdir, "230913_matlabOpt_smallBatchTest", "230922_matlabOpt_IRE-1_ARCincl.csv"))
    # compdb =dbh.loadDBs(os.path.join(dbdir, "230810_paper_on-subs_4umBJ_bl", "*part_1","*001_of_200.csv"))
    # compdb.dataframe = compdb.dataframe.iloc[0:5, :]
    # compdb.dbSaveFile(saveName=os.path.join(dbdir, "230913_matlabOpt_smallBatchTest", "jphOpt.csv"))

    # outdb2 = sendDB_toMatlOpt(indb, justGetEff=True)

    print()
    return


####################################################################################
## parallelization functions

def map_DB_extract(x):
    # Parallelized run call's call-to-run Daisy's detailed balance code to get thicknesses.
    #runType: (1 = use matl_optTh function , ELSE = use DB_extract fnc)
    return DB_Extract(*x)

def map_run_matlOpt(x):
    return run_matlOpt(*x)


def run_DB_matlab(inp, numPool=None, printRes=False, runType=0):
    #Parallized run call
    #input:     list of input params lists needed for (modified) DB(L) calculations. See DB_Extract for input details/order.
    #numPool:   int, number of parallel processes
    #printRes:  bool, print results?
    #runType: (1 = use matl_optTh function , ELSE = use DB_extract fnc)

    if runType == 1 : runDescr = "run_matlOpt"
    else: runDescr = "DB_extract"
    print(f"run matlab mDBL: Running {len(inp)} modified DBL experiments with max. {numPool} in parallel w/ runType={runDescr}...")
    if not isinstance(inp[0], list): inp = [inp] #ensure list of lists

    # run experiments in batches
    with Pool(numPool) as pool:
        if runType == 1: #run_matlOpt
            results = list(tqdm(pool.imap(map_run_matlOpt, inp), total=len(inp)))
        else: #DB_extract
            results = list(tqdm(pool.imap(map_DB_extract, inp), total=len(inp)))
    print("run matlab mDBL: Success :Completed mDBL experiments.")
    if printRes:
        print(f"run matlab mDBL: Printing results..." )
        for i in range(len(results)): print(f"{i} ;\ninp ;  {inp[i]}\nres ; {results[i]}")
    return results



####################################################################################
## matlab engine run functions

def DB_Extract(laserWL,Eg_in,numSegs,nk_path,BLabsTarget,BLthTarget,homoEmFactor, backReflector, maxSegTh, IRE, Pin, lc, moleFrac=None):
    """
    Get the modified detailed balance (DB) thicknesses from Daisy's matlab code assuming some constraints
    (DB = modified DBL)

    Parameters
    ----------
    laserWL         : float, laser wavelength [um]
    Eg_in           : float, bandgap [eV], overwritten by the moleFrac associated Eg if it is not None
    numSegs         : int, number of segments
    nk_path         : string, path to nk data
    BLabsTarget     : float, absorption target in fraction
    BLthTarget      : float, thickness constraint
    homoEmFactor    : float or 0, factor size between emitter and base layers for bottom homojunction architecture OR if 0, all heterojnc
    backReflector   : bool: run with perfect mirror back-reflector (True) or run with substrate at bottom (False)
    maxSegTh        : float, maximum segment thickness allowed
    IRE             : float, internal radiative limit in fractional form
    Pin             : float, input power [W/cm2]
    lc              : bool:, include luminescent coupling?
    moleFrac        : float or None, the InGaAsP Ga (or is it P?!) mole fraction, if None; calculate from the bandgap

    Returns
    -------
    segThsTtoB      : list, (m)DB(L) optimized thicknesses from top to bottom [um]
    moleFrac        : float, molefraction used in calculations
    mask            : bool, True = good results, False = Bad results

    """

    if backReflector: brSwitch = 2
    else: brSwitch = 3
    if lc: lc_off = 0
    else: lc_off = 1

    segThsTtoBs = [] # segment thickness list
    # fill list of k's, by shifting k in energy from closest molefraction to the second decimal place.
    if moleFrac is None:
        res = opt.minimize(lambda x: abs(helpF.Eg_Func(x) - Eg_in), 0.5, method='Nelder-Mead')
        moleFrac = res.x[0]
        if moleFrac < 0.0:
            moleFrac = 0.0
        Eg_calc = 'Eg_in'
    else:
        Eg_calc = helpF.Eg_Func(moleFrac)
    yMole = np.round(0.47*(1.0 - moleFrac),2)
    Eg_round = helpF.Eg_Func(1.0-yMole/0.47)
    lnk = np.loadtxt(nk_path + "/GaInAsP_x" + "{:0.2f}".format(yMole) + "_nk.txt",delimiter=" ", unpack=True)
    nAtWl = np.interp(laserWL*Eg_in/Eg_round,lnk[0],lnk[1])
    kappaAtWl = np.interp(laserWL*Eg_in/Eg_round,lnk[0],lnk[2])
    print(f"n={nAtWl}, k={kappaAtWl,4}, Eg_in={Eg_in}, Eg_calc={Eg_calc}, Eg_round={Eg_round}")
    Eg_in = Eg_calc



    """Get BL thicknesses for each mole fraction (from top to bottom), account for homo junction, and max segment thickness"""

    mask = True
    # use BL thickness or contrained total thickness
    BLabsTarget_constrained = 1.0 - np.exp(-BLthTarget*4.0*np.pi*kappaAtWl/laserWL)
    if BLabsTarget > BLabsTarget_constrained:
        BLabsTarget = BLabsTarget_constrained


    # get BL th
    if BLabsTarget < 0.8: # dont run DB optimizer if absorption is below 0.8
        mask = False
        segThsTtoB_noHomo = np.zeros(numSegs)
        print(f"ERROR :modified DBL th extraction: Laser Wavelength corresponds with k=0. No thickness solution.")
    else: # run DB through matlab
        eng = matlab.engine.start_matlab()
        eng.addpath(os.path.join(__file__[0:__file__.find('aiirmap')+7],"MJDB2023.0"), nargout=0)  # addpath to matlab folder with code
        out = io.StringIO()
        err = io.StringIO()
        #print(f"Initiated matlab. (Abs,nJ,IRE,Pin,laser,Eg) {BLabsTarget,numSegs,IRE,Pin,laserWL,Eg_in}")
        #### RUN MATLAB ####
        try:
            DB_res = eng.matlab_LC(float(BLabsTarget), float(maxSegTh), float(numSegs), float(IRE), float(Pin), float(brSwitch), float(laserWL), float(Eg_in), float(nAtWl), float(kappaAtWl), float(lc_off), nargout=2,stdout=out,stderr=err)
        except Exception as exception:
            print(f"ERROR :modified DBL th extraction: Could not run matlab_LC.\n"
                  f"ERROR :modified DBL th extraction: Matlab error; {exception}")
            DB_res = [[np.nan]*numSegs, np.nan]
        try:
            eng.quit()
        except SystemError:
            pass  # probably already closed ... ... ... probably
        #print(f"Matlab worker finished. (Abs,nJ,IRE,Pin,laser,Eg) {BLabsTarget,numSegs,IRE,Pin,laserWL,Eg_in}")
        segThsTtoB_noHomo = list(np.flip(np.array(DB_res[0][0])))
        # print(f"!!!!--\n{DB_res}")
        dblEff = float(DB_res[1])
        print(f"modified DBL th extraction: mDBL obtained efficiency of {dblEff}... for th stack {segThsTtoB_noHomo}.")


    # divide bottom jnc into homojnc if homoEmFactor !=
    if homoEmFactor != 0:
        segThsTtoB = segThsTtoB_noHomo[0:(numSegs-1)] + [homoEmFactor*segThsTtoB_noHomo[numSegs-1]] + [(1-homoEmFactor)*segThsTtoB_noHomo[numSegs-1]]
    else: segThsTtoB = segThsTtoB_noHomo

    # convert to np.array and run max seg th check
    segThsTtoB = np.array(segThsTtoB)
    if segThsTtoB.max() > maxSegTh:
        mask = False
        print(f"ERROR :modified DBL th extraction: The max segment thickness is greater than the allowed value ({segThsTtoB.max()} > {maxSegTh}). Cannot continue...")
        #exit()
    else:
        pass
       # print(f"opt from BL then run elec: The max segment thickness for all designs is {segThsTtoB.max()}. Continuing...")

    #save results into list for given laser wavelength
    return segThsTtoB, moleFrac, mask





def retrieve_BL(laserWL=1.480, Eg_in=0.8, numSegs=4, nAtWl=None, kappaAtWl=None, nk_path=nk_path, moleFrac=None, BLabsTarget=0.99, backReflector=False):

    """
    Retrieve 1 or 2 pass BL thicknesses from the matlab code.
        matlab fncs; retrieve_BL < runParams_v2 and thickguess

    if either nAtWl or kappaAtWl are None uses nk_path at moleFrac to extract nk data
    backReflector sets 1 vs 2 pass (if backReflector => 2pass)

    laserWL         : float, laser wavelength [um]
    Eg_in           : float, bandgap [eV]
    numSegs         : int, number of segments
    nAtWl           : float or None, real refractive index value at wl
    kappaAtWl       : float or None, imaginary component of refractive index value at wl
    nk_path         : string, path to nk data
    moleFrac        : float or None, the P mole fraction for the InGaAs(P) (if None uses Eg_in and helperFuncs)
    BLabsTarget     : float, absorption target in fraction
    BLthTarget      : float, thickness constraint
    backReflector   : bool: run with perfect mirror back-reflector (True) or run with substrate at bottom (False)

    @return:
    """

    if backReflector:
        brSwitch = 2 #convert input to matlab setting
        nPass = 2
    else:
        brSwitch = 3; nPass = 1


    print(f"matlab BL thicknesses: Extracting {nPass}-pass {BLabsTarget}% BL thicknesses from matlab code for {numSegs}J {Eg_in}eV material at {laserWL}um from Daisy's matlab code.")



    if nAtWl is None or kappaAtWl is None:
        #warn if it is only of the two params which is None
        if not (nAtWl is None and kappaAtWl is None):
            print(f"WARNING :matlab BL thicknesses: One of nAtWl or kappaAtWl is None, using nk_path data with mole fraction of {moleFrac} (if None uses Eg_in).")

        #retrieve from nk_path
        if moleFrac is None:
            res = opt.minimize(lambda x: abs(helpF.Eg_Func(x) - Eg_in), 0.5, method='Nelder-Mead')
            moleFrac = res.x[0]
            if moleFrac < 0.0:
                moleFrac = 0.0
            Eg_calc = 'Eg_in'
        else:
            Eg_calc = helpF.Eg_Func(moleFrac)
        yMole = np.round(0.47*(1.0 - moleFrac),2)
        Eg_round = helpF.Eg_Func(1.0-yMole/0.47)
        lnk = np.loadtxt(nk_path + "/GaInAsP_x" + "{:0.2f}".format(yMole) + "_nk.txt",delimiter=" ", unpack=True)
        nAtWl = np.interp(laserWL*Eg_in/Eg_round,lnk[0],lnk[1])
        kappaAtWl = np.interp(laserWL*Eg_in/Eg_round,lnk[0],lnk[2])
        print(f"matlab BL thicknesses: nk_path retrieval; n={nAtWl}, k={kappaAtWl,4}, Eg_in={Eg_in}, Eg_calc={Eg_calc}, Eg_round={Eg_round}")
        Eg_in = Eg_calc


    eng = matlab.engine.start_matlab()

    eng.addpath(os.path.join(__file__[0:__file__.find('aiirmap')+7],"MJDB2023.0"), nargout=0)  # addpath to matlab folder with code
    out = io.StringIO()
    err = io.StringIO()
    #print(f"Initiated matlab. (Abs,nJ,IRE,Pin,laser,Eg) {BLabsTarget,numSegs,IRE,Pin,laserWL,Eg_in}")
    #### RUN MATLAB ####
    try:
        DB_res = eng.retrieve_BL(float(BLabsTarget), float(numSegs), float(brSwitch), float(laserWL), float(Eg_in), float(nAtWl), float(kappaAtWl), nargout=2,stdout=out,stderr=err)
    except Exception as exception:
        print(f"ERROR :matlab BL thicknesses: Could not retrieve_BL for BLabsTarget={BLabsTarget}, brSwitch={brSwitch}.\n"
              f"ERROR :matlab BL thicknesses: Matlab error; {exception}")
        DB_res = [[np.nan] * numSegs, np.nan]
    try:
        eng.quit()
    except SystemError:
        pass #probably already closed ... ... ... probably

    #print(f"Matlab worker finished. (Abs,nJ,IRE,Pin,laser,Eg) {BLabsTarget,numSegs,IRE,Pin,laserWL,Eg_in}")
    segThsTtoB_noHomo = list(np.flip(np.array(DB_res[0][0])))
    # print(f"!!!!--\n{DB_res}")
    dblEff = float(DB_res[1])
    print(f"matlab BL thicknesses: Success :Matlab obtained efficiency of {dblEff}... for {numSegs}J {nPass}-pass thickness stack;\n{segThsTtoB_noHomo}.")

    return segThsTtoB_noHomo






def run_matlOpt(TtoB_absTh, arc1th, arc2th, maxSegThick, IRE, Ps, backReflector, WL, Eg, nAtWl, kappaAtWl, lc, homoEmFactor=moonshotHomoEmFactor, includeARCinOpt=True, justGetEff=False):

    """
    Optimize thicknesses within matlab using the input thicknesses for the optimization start point.
    Uses MJDB's optAbs_fromTh function

    Automatically detects and handles (input and output of) bottom homojnc designs
        Emitter/base separation not included in MJDB opt; add together before, split using supplied homoEmFactor after
        Note; if homoEmFactor == 0; all rear-hetero arch.

    Can also be used to grab the matlab efficiency using the justGetEff flag.

    @param TtoB_absTh: list of float, the top to bottom absorber layer thicknesses in um
    @param arc1th: float, top arc thickness in um
    @param arc2th: float, bottom arc th in um
    @param maxSegThick: float, a constraint in the optimization; the maximum single layer thickness in um
    @param IRE: float, The internal radiative efficiency (0<IRE<1]
    @param Ps: float, Power in [W/cm2]
    @param backReflector : bool, run with (2-pass) or without (1-pass) back-reflector
    @param WL: float, the input laser wavelength [um]
    @param Eg: float, the bandgap of the material
    @param n: flaot, the real component of the refractive index at the WL
    @param k: float, the imaginary component of the refractive index at the WL
    @param lc: bool, run with LC (True) or without (False)
    @param homoEmFactor    : float or 0, factor size between emitter and base layers for bottom homojunction architecture OR if 0, all heterojnc
    @param includeARCinOpt: bool, include the ARC (True) or only optimize ABS layers (False)
    @param justGetEff: bool, If true, run get_maxEff instead of running optAbs_fromTh (just grab efficiency, don't run the optimization)
    @return: segThsTtoB: list of float, the optimized top to bottom segment layer thicknesses
    """

    #prep parameters and matlab engine
    if backReflector: brSwitch = 2
    else: brSwitch = 3
    if lc: lc_off = 0
    else: lc_off = 1
    if includeARCinOpt: includeARCinOpt = 1
    else: includeARCinOpt = 0

    if homoEmFactor != 0: #bottom homojnc so need to add the emitter and base th
        TtoB_absTh = list(TtoB_absTh)
        TtoB_absTh = TtoB_absTh[0:-2] + [TtoB_absTh[-2]+TtoB_absTh[-1]]
        TtoB_absTh = np.array(TtoB_absTh)
    numSegs = len(TtoB_absTh)
    BtoT_absTh = list(np.flip(TtoB_absTh))
    BtoT_absTh = [float(th) for th in BtoT_absTh]


    # BtoT_absTh = np.array(BtoT_absTh)


    eng = matlab.engine.start_matlab()
    eng.addpath(os.path.join(__file__[0:__file__.find('aiirmap')+7],"MJDB2023.0"), nargout=0)  # addpath to matlab folder with code
    out = io.StringIO()
    err = io.StringIO()

    #### RUN MATLAB ####
    if justGetEff:
        try:
            DB_res = eng.get_maxEff(matlab.double(BtoT_absTh), float(arc1th), float(arc2th), float(IRE), float(Ps), float(brSwitch), float(WL), float(Eg), float(nAtWl), float(kappaAtWl), float(lc_off), float(includeARCinOpt), nargout=1,stdout=out,stderr=err)
            print(f"run matlab opt (get eff mode): Success : mDBL efficiency of {float(DB_res[0])} obtained.")
        except Exception as exception:
            print(f"ERROR :run matlab opt (get eff mode): Could not get_maxEff for BtoTAbs={BtoT_absTh} and BtoTARC=[{arc1th},{arc2th}]\n"
                  f"ERROR :run matlab opt (get eff mode): Matlab error; {exception}")
            DB_res=[np.nan]
        try:
            eng.quit()
        except SystemError:
            pass  # probably already closed ... ... ... probably
        matlEff = float(DB_res[0])
        return [matlEff] #exit without optimization

    #else: optimize
    try:
        DB_res = eng.optAbs_fromTh(matlab.double(BtoT_absTh), float(arc1th), float(arc2th), float(maxSegThick), float(IRE), float(Ps), float(brSwitch), float(WL), float(Eg), float(nAtWl), float(kappaAtWl), float(lc_off), float(includeARCinOpt), nargout=4,stdout=out,stderr=err)
        # print(DB_res)
    except Exception as exception:
        print(
            f"ERROR :run matlab opt: Could not find optAbs_fromTh for BtoTAbs={BtoT_absTh} and BtoTARC=[{arc1th},{arc2th}]\n"
            f"ERROR :run matlab opt: Matlab error; {exception}")
        DB_res = [[[np.nan, np.nan]], [[np.nan]*len(BtoT_absTh)], np.nan, np.nan]
    try:
        eng.quit()
    except SystemError:
        pass  # probably already closed ... ... ... probably
    arcThTtoB = list(np.array(DB_res[0][0]))
    segThsTtoB_noHomo = list(np.flip(np.array(DB_res[1][0])))
    eff_opt = float(DB_res[2])
    try: eff_in = float(DB_res[3])
    except: eff_in = np.nan
    if isinstance(segThsTtoB_noHomo[0], int):
        print(f"run matlab opt: Success :mDBL obtained efficiency increase from {eff_in:.4f} to {eff_opt:.4f}. Optimized th stack; {arcThTtoB} + {segThsTtoB_noHomo}.")

    # divide bottom jnc into homojnc if homoEmFactor != 0
    if homoEmFactor != 0:
        segThsTtoB = segThsTtoB_noHomo[0:(numSegs-1)] + [homoEmFactor*segThsTtoB_noHomo[numSegs-1]] + [(1-homoEmFactor)*segThsTtoB_noHomo[numSegs-1]]
    else: segThsTtoB = segThsTtoB_noHomo

    return [arcThTtoB, segThsTtoB, eff_opt, eff_in]






####################################################################################
## aiirmap database wrappers

def sendDB_toMatlOpt(indb, parallelizationFactor=25, inputThLabelsTtoB=arcThLabelsTtoB+absThLabelsTtoB_bottomHomo, outputThLabelsTtoB=arcThLabelsTtoB_S4+absThLabelsTtoB_bottomHomo_S4, maxSegTh=4.0, IRE=1.0, Pin=10, lc=True, Eg=None, nAtWl=None, kappaAtWl=None, nk_path=nk_path, includeARCinOpt=True, justGetEff=False):
    """
    Optimize all experiments in an aiirmap DataBase object using the matlab code (ie. run matlab function optAbs_fromTh for each design in the DB)
    (Or if justGetEff; get matlab efficiency for each design (ie. run get_maxEff matlab function)

    This function obtains the list of input lists out of the db dataframe and sends them to run_matlOpt (using run_DB_matlab and map_run_matlOpt for parallelization)
    It also takes the outputs from matlab and constructs an output db (also reusing input data)

    For nk data;
    Use the nk_path and the mole fraction (em_xMole col) to pull the data UNLESS Eg, nAtWl, and kappaAtWl are all supplied

    @param indb: DataBase, the input grid database with required columns present
    @param parallelizationFactor: int, the number of matlab threads to run at a time
    @param inputThLabelsTtoB: list of float, the ARC and ABSorber thickness labels, top to bottom [um], to be used as input to the matlab functions
    @param outputThLabelsTtoB: list of float, the ARC and ABSorber thickness labels, top to bottom [um], to be used to label the thickness output of the matlab optimization
    @param maxSegTh: float, the max single layer thickness optimization constraint [um]
    @param IRE: float, The internal radiative efficiency (0<IRE<1]
    @param Ps: float, Power in [W/cm2]
    @param lc: bool, run with LC (True) or without (False)
    @param Eg: float or None, the bandgap of the material [eV]
    @param n: float or None, the real component of the refractive index at the WL
    @param k: float or NOne, the imaginary component of the refractive index at the WL
    @param nk_path: string, path to the InGaAs(P) nk data folder
    @param homoEmFactor    : float or 0, factor size between emitter and base layers for bottom homojunction architecture OR if 0, all heterojnc
    @param includeARCinOpt: bool, include the ARC (True) or only optimize ABS layers (False)
    @param justGetEff: bool, If true, run get_maxEff instead of running optAbs_fromTh (just grab efficiency, don't run the optimization)
    @return: outdb: [DataBase] the aiirmap database with matlab absorber thickness, post-opt mDBL efficiency, and pre-opt mDBL eff

    """


    """
    ----------
    run_matlOpt parameters (ie the input list for the matlab code)
        param TtoB_absTh: list of float, the top to bottom absorber layer thicknesses in um
        param arc1th: float, TOP arc thickness in um
        param arc2th: float, BOTTOM arc th in um
        param maxSegThick: float, a constraint in the optimization; the maximum single layer thickness in um
        param IRE: float, The internal radiative efficiency (0<IRE<1]
        param Ps: float, Power in [W/cm2]
        param backReflector : bool, run with (2-pass) or without (1-pass) back-reflector
        param WL: float, the input laser wavelength [um]
        param Eg: float, the bandgap of the material
        param n: flaot, the real component of the refractive index at the WL
        param k: float, the imaginary component of the refractive index at the WL
        param lc: bool, run with LC (True) or without (False)
        param includeARCinOpt: bool, run with ARC in optimization (True) or without (False)
        param homoEmFactor    : float or 0, factor size between emitter and base layers for bottom homojunction architecture OR if 0, all heterojnc
        param justGetEff: bool, If true, run get_maxEff instead of running optAbs_fromTh (just grab efficiency, don't run the optimization)
    """
    # create list of input lists for each expt in indb for run_DB_matlab
    if justGetEff:
        print(f"send to matlab opt: just get eff mode - Getting efficiency for all {len(indb.dataframe)} exp'ts in DB {indb.dbfilename}")
    else:
        if indb.dbfilename is None:
            print(f"send to matlab opt: Running matlab mDBL optimization for DB... (length={len(indb.dataframe)})")

        else:
            print(f"send to matlab opt: Running matlab mDBL optimization for DB ({indb.dbfilename})... (length={len(indb.dataframe)})")

    inputs = []
    for eidx in range(len(indb.dataframe)):
        inputs.append([])
        inputs[-1].append(indb.dataframe.loc[eidx, inputThLabelsTtoB[2:]].values)
        inputs[-1].append(indb.dataframe.loc[eidx, inputThLabelsTtoB[0]])
        inputs[-1].append(indb.dataframe.loc[eidx, inputThLabelsTtoB[1]])
        inputs[-1].append(maxSegTh)
        inputs[-1].append(IRE)
        inputs[-1].append(Pin)
        if indb.dataframe.loc[eidx, 'substrate_t'] == 0:
            inputs[-1].append(True)
        else:
            inputs[-1].append(False)
        inputs[-1].append(indb.dataframe.loc[eidx, 'spectrum'])
        if Eg is None or nAtWl is None or kappaAtWl is None: #use input values if both supplied, otherwise pull nk from the moleFrac in the indb
            Eg, nAtWl, kappaAtWl = bl.getEgNKfromMF(indb.dataframe.loc[eidx, 'em_xMole' ], indb.dataframe.loc[eidx, 'spectrum'] , nk_path=nk_path)
        inputs[-1].append(Eg) #Eg (will be picked from moleFrac)
        inputs[-1].append(nAtWl)
        inputs[-1].append(kappaAtWl)
        inputs[-1].append(lc)
        if indb.dataframe.loc[eidx, 'seg1_em2_t'] == 0:
            homoEmFactor = 0 #all rear-hetero
        else:
            homoEmFactor = indb.dataframe.loc[eidx, 'seg1_em_t'] / indb.dataframe.loc[eidx, 'seg1_em2_t']
        inputs[-1].append(homoEmFactor)
        inputs[-1].append(includeARCinOpt)
        inputs[-1].append(justGetEff)

    #run matlab with parallelization
    results = run_DB_matlab(inputs, numPool=parallelizationFactor, printRes=False, runType=1)
    #results in form of [[segThsTtoB_expt1, effOut_expt1, effIn_expt1], [_expt2, ], ...]


    #create output db merging the input db with the output cols
    #note; use the same column names for segments optimized by S4 (these now distinguish opt in general)
    outdb = indb.dbCopy()
    outdb.grid=False
    if indb.dbfilename != None:
        outdb.dbfilename = indb.dbfilename[:-4] + "_matlOpt.csv"
    else: outdb.dbfilename = "matlOpt.csv"
    if indb.dbFile != None:
        outdb.dbFile = indb.dbFile[:-4] + "_matlOpt.csv"
    outdb.lineage.append([time.strftime(timeStr), "Thicknesses optimized using MJDB opt_fromTh and aiirmap sendDB_toMatlOpt functions"])
    try: x = outdb.dataframe['experiment-inputs']
    except KeyError: outdb.dataframe.insert(loc=0, column='experiment-inputs', value=range(1,len(outdb.dataframe)+1))
    try: x = outdb.dataframe['experiment-outputs']
    except KeyError: outdb.dataframe['experiment-outputs'] = outdb.dataframe['experiment-inputs']

    if justGetEff:
        #if just get Eff than the results are just a list of matlab efficiencies for each experiment, add it to the output as a column
        try: x = outdb.dataframe['matl_eff']
        except KeyError:
            outdb.dataframe["matl_eff"] = [results[i][0] for i in range(len(results))]
        else:
            print("WARNING :send to matlab opt: just get eff mode - Overwriting existing matl_eff column...")
            outdb.dataframe["matl_eff"] = [results[i][0] for i in range(len(results))]
    else:
        resultsArray = [results[i][0] + results[i][1] + [results[i][2]] + [results[i][3]] for i in range(len(results))]
        resultsArray = np.matrix(resultsArray)
        outdb.dataframe = pd.concat([outdb.dataframe, pd.DataFrame(resultsArray, columns=outputThLabelsTtoB+["matlEff1", "matlEff0"])], axis=1)

    outdb.dbCleanIdxsAndGridness()

    if justGetEff:
        print(f"send to matlab opt: Success :just get eff mode - Efficiency for {len(outdb.dataframe)}/{len(indb.dataframe)} exp'ts in DB {indb.dbfilename}")
    else:
        print(f"send to matlab opt: Success :Matlab mDBL optimization complete for {len(outdb.dataframe)}/{len(indb.dataframe)} exp'ts in DB {indb.dbfilename}")

    return outdb










if __name__ == '__main__':
    main()
