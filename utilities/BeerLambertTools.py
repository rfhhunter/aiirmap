"""
Functions in this utility employ the Beer-Lambert absorption law to calculate absorption fractions, thicknesses, etc.
for given/defined/sourced absorption coefficients or complex refractive index etc.

Multi-absorbing segment structures are obtained with the numSeg(ment)s param
"""


from aiirmapCommon import *


def main():
    """
    Test/use the functions in this script
    """

    bl99 = gen_1passBLTh(wl=1.55, kappaAtWl=3.0, numSegs=10, targetType='abs', absTarget=0.99, thTarget=7.0)
    print(bl99)

    return







def calcAbs(alpha, th):
    """
    Calculate the fraction of photons which are absorbed for a given layer thickness and absorption coefficient (at a single wl)
    :param alpha: [float] the absorption coefficient of the material (at wl) [um^(-1)]
    :param th: [float] the thickness of the layer [um]
    :return: the fraction of photons (of wl) absorbed in the layer
    """
    return 1-np.exp(-1*alpha*th)



def calcTh(alpha, abs):
    """
    Calculate a layer thickness based upon an absorption fraction and absorption coefficient (for a single wl)
    :param abs: [float] fraction of photons (of wl) absorbed over the layer
    :param alpha: [float] absorption coefficient of the material (at wl) [um^(-1)]
    :return: the layer thickness to absorb the provided fraction of light [um]
    """
    return - np.log(1-abs) / alpha



######################
##   1 PASS
######################


def calc_1passTh(absFrac, alphaAtWl=None, wl=None, kappaAtWl=None):
    """
    Calculation of 1 pass thickness from absorption fraction. (Wrapper)
    :param absFrac: [float] fraction of photons absorbed in layer
    :param alphaAtWl: [float or None] the absorption coefficient at the wavelength of interest [um^(-1)]
    :param wl: [float or None] the wavelength of interest, to calculate the absorption coefficient [um]
    :param kappaAtWl: [float or None] the imaginary refractive index at the wl of interest, to calculate the abs coeff
    :return: [float] thickness of the layer in um
    """

    if (wl==None or kappaAtWl==None) and alphaAtWl==None:
        print("ERROR :1-pass BL th-calc: Cannot compute thicknesses when no alpha, or kappa and wl, is provided. Returning None...")
        return None

    if (wl!=None and kappaAtWl!=None) and alphaAtWl!=None:
        print("WARNING :1-pass BL th-calc: Alpha, and kappa and wl, are all provided. Using provided alpha.")

    if alphaAtWl == None:
        alphaAtWl = 4 * np.pi * kappaAtWl / wl  # [um^-1] Abs coefficient at illum wavelength

    return calcTh(alphaAtWl, absFrac)



def gen_1passBLTh(alphaAtWl=None, wl=None, kappaAtWl=None, numSegs=10, targetType='abs', absTarget=0.98, thTarget=7.0, plotTh=False):
    """
    Calculate the 1-pass Beer Lambert thicknesses from top to bottom
        for a given absorption coefficient or imaginary refractive index and wavelength
        setting either a target full device absorption or full device thickness.

    :param alphaAtWl: [float or None] the absorption coefficient at the wavelength of interest [um^(-1)]
    :param wl: [float or None] the wavelength of interest, to calculate the absorption coefficient [um]
    :param kappaAtWl: [float or None] the imaginary refractive index at the wl of interest, to calculate the abs coeff
    :param numSegs: [int] the number of segments in the device
    :param targetType: ['abs' or 'th'] the type of target to use
    :param absTarget: [float] the target full device absorption fraction [0,1]
    :param thTarget: [float] the target full device absorber thickness [um]
    :param plotTh: [bool] whether to generate a plot of the segment thicknesses vs segment. STOPS CODE RUNNING USING A PLT.SHOW()
    :return: segThs [list of floats] the segment thicknesses from TOP to BOTTOM [um]
    """


    if (wl==None or kappaAtWl==None) and alphaAtWl==None:
        print("ERROR :1-pass BL th-calc: Cannot compute thicknesses when no alpha, or kappa and wl, is provided. Returning None...")
        return None

    if (wl!=None and kappaAtWl!=None) and alphaAtWl!=None:
        print("WARNING :1-pass BL th-calc: Alpha, and kappa and wl, are all provided. Using provided alpha.")

    if alphaAtWl == None:
        alphaAtWl = 4 * np.pi * kappaAtWl / wl  # [um^-1] Abs coefficient at illum wavelength

    # hc = 1.98644586e-25 #[J m]
    # numPhIn = Pin / (hc / (wl*1e-6)) #[#ph / (m^2 s)] Photon flux into cell

    if targetType.lower() == 'abs' or targetType.lower()[0:1] == 'a':
        #set thicknesses for target abs fraction over full device
        segPhFrac = absTarget/numSegs
        infostr = f"target full device absorption fraction of {absTarget}"
    else:
        #set thicknesses for target full device absorber th
        absPh = calcAbs(alphaAtWl, thTarget)
        segPhFrac = absPh/numSegs
        infostr = f"target full device absorber thickness of {thTarget}"

    print(f"1-pass BL th-calc: Calculating thicknesses for {numSegs} segment device with alpha of {alphaAtWl} at {wl} um with {infostr}")


    segThs= []
    segThsSums=[]
    for sidx in range(numSegs):
        segPhTarget = segPhFrac*(sidx+1)
        segTh = calcTh(alphaAtWl, segPhTarget)
        segThsSums.append(segTh)
        if sidx > 0:
            segThs.append(segTh-segThsSums[-2])
        else:
            segThs.append(segTh)

    print(f"1-pass BL th-calc: Success :Total thickness: {sum(segThs)} um.  Total absorption: {segPhTarget*100} %")
    print(f"1-pass BL th-calc: Segment Th from top to bottom : {segThs} um")

    if plotTh:
        plt.plot(segThs)
        plt.title(f"1 Pass BL Thicknesses for {infostr}")
        plt.xlabel("Seg Index (Top-Down)")
        plt.ylabel("Segment Thickness [um]")
        plt.show()

    return segThs



def gen_many1passBLTh(alphaAtWl=None, kappaAtWl=None, wl=None, numSegs=10, targetType='abs', absTarget=0.98, thTarget=10):
    """
    Wrapper for gen_1passBLTh which handles lists of alpha / kappa and wl.

    provide a list of alpha,
    OR a list of kappa and a single wl
    OR equal length lists of kappa and wl (correspondent to each other)

    returns a list of lists of segment thicknesses FROM TOP TO BOTTOM

    -------FROM gen_1passBLTh----------
    Calculate the 1-pass Beer Lambert thicknesses from top to bottom
        for a given absorption coefficient or imaginary refractive index and wavelength
        setting either a target full device absorption or full device thickness.

    :param alphaAtWl: [float or None] the absorption coefficient at the wavelength of interest [um^(-1)]
    :param wl: [float or None] the wavelength of interest, to calculate the absorption coefficient [um]
    :param kappaAtWl: [float or None] the imaginary refractive index at the wl of interest, to calculate the abs coeff
    :param numSegs: [int] the number of segments in the device
    :param targetType: ['abs' or 'th'] the type of target to use
    :param absTarget: [float] the target full device absorption fraction [0,1]
    :param thTarget: [float] the target full device absorber thickness [um]
    :param plotTh: [bool] whether to generate a plot of the segment thicknesses vs segment. STOPS CODE RUNNING USING A PLT.SHOW()
    :return: segThs [list of floats] the segment thicknesses from TOP to BOTTOM [um]
    -----------------------------------

    """

    if (wl == None or kappaAtWl == None) and alphaAtWl == None:
        print(
            "ERROR :many 1-pass BL th-calc: Cannot compute thicknesses when no alpha, or kappa and wl, is provided. Returning None...")
        return None

    if (wl != None and kappaAtWl != None) and alphaAtWl != None:
        print("WARNING :many 1-pass BL th-calc: Alpha, and kappa and wl, are all provided. Using provided alpha.")


    #handle lists and list lengths
    if alphaAtWl == None:
        #using kappa and wl to calculate alpha
        if not isinstance(kappaAtWl, list):
            kappaAtWl = [kappaAtWl]

        if isinstance(wl, list):
            if len(wl) != len(kappaAtWl):
                print(f"ERROR :many 1-pass BL th-calc: The number of provided wls ({len(wl)}) does not match the number of provided kappa ({len(kappaAtWl)}). Cannot continue, returning None...")
                return None
        else:
            wl = [wl]*len(kappaAtWl)

        alphaAtWl = [4*np.pi*kappaAtWl[i]/wl[i] for i in range(len(kappaAtWl))]  # [um^-1] Abs coefficient at illum wavelength

    else:
        #using alpha
        if not isinstance(alphaAtWl, list):
            alphaAtWl = [alphaAtWl]


    print(f"many 1-pass BL th-calc: Obtaining 1-pass BL thicknesses for {len(alphaAtWl)} data points.")

    segThsPerInput = []
    for i in range(len(alphaAtWl)):
        segThsPerInput.append(gen_1passBLTh(alphaAtWl=alphaAtWl[i], numSegs=numSegs, targetType=targetType, absTarget=absTarget, thTarget=thTarget, plotTh=False))

    print(f"many 1-pass BL th-calc: Success :1-pass BL thicknesses obtained for {len(alphaAtWl)} data points.")

    return segThsPerInput




def reparamTh_1Pass_absFrac(thDataframeTtoB, alphaAtWl=None, wl=None, kappaAtWl=None):
    """
    Calculate 1-pass BL absorption fractions for given layer thickness and addend to the provided dataframe with columns <th-label>_absFrac1p
        Provide the absorption coefficient, or, imaginary refractive index and wavelength so it can be calculated
        Also calculates the tot_em_t_absFrac1p (sum of the layer absFrac)

    :param thDataframeTtoB: [pd.DataFrame] the dataframe with the layer thicknesses to calculate the abs for, thicknesses must be ordered top to bottom of the stack (top is light-side)
    :param alphaAtWl: [float or None] the absorption coefficient at the wavelength of interest [um^(-1)]
    :param wl: [float or None] the wavelength of interest, to calculate the absorption coefficient [um]
    :param kappaAtWl: [float or None] the imaginary refractive index at the wl of interest, to calculate the abs coeff
    :return: thDataframe: [pd.DataFrame] the input dataframe but with the <th-label>_absFrac1p columns added
    """

    if (wl==None or kappaAtWl==None) and alphaAtWl==None:
        print("ERROR :1-pass th reparam: Cannot compute reparameterized thicknesses when no alpha, or kappa and wl, is provided. Returning None...")
        return None

    if (wl!=None and kappaAtWl!=None) and alphaAtWl!=None:
        print("WARNING :1-pass th reparam: Alpha, and kappa and wl, are all provided. Using provided alpha.")

    if alphaAtWl == None:
        alphaAtWl = 4 * np.pi * kappaAtWl / wl  # [um^-1] Abs coefficient at illum wavelength


    print(f"1-pass th reparam: Generating _absFrac1p columns using alphaAtWl of '{alphaAtWl}' um^(-1) for the top to bottom thicknesses; '{thDataframeTtoB.columns.values}'.")

    outDF = thDataframeTtoB.copy(deep=True)


    transmissionFracTotals = [1] * thDataframeTtoB.shape[0] #track transmission fraction for each design as we move down the stack
    for sidx in range(len(thDataframeTtoB.columns.values)): #segment/layer idx
        thLabel = thDataframeTtoB.columns.values[sidx]
        outDF[thLabel+"_absFrac1p"] = [transmissionFracTotals[i]*calcAbs(alphaAtWl, thDataframeTtoB.loc[i, (thLabel)]) for i in range(thDataframeTtoB.shape[0])]
        transmissionFracTotals = [transmissionFracTotals[i] - outDF.loc[i,(thLabel+"_absFrac1p")] for i in range(thDataframeTtoB.shape[0])]
        if sidx==0: totLabel = f"tot_{thLabel[-4:]}_absFrac1p" #to avoid creating from a em2 layer

    outDF[totLabel] = [1-tf for tf in transmissionFracTotals]

    print(f"1-pass th reparam: Success :Generation of _absFrac1p columns complete.")

    return outDF







################################################################################################################################################################################
################################################################################################################################################################################
################################################################################################################################################################################

######################
##   2 PASS
######################


def calc_2passThStackFromAbs(absFrac, alphaAtWl=None, wl=None, kappaAtWl=None):
    ##TODO? very low priority
    print(f"ERROR\n! 2pass thickness calculation from absorption fraction is not implemented !\nERROR")
    return absFrac


def calc_2PassAbsFromThStack(alphaAtWl, thStackTtoB, inputAbsFrac=1):
    """
    Calculate the 2 pass BL absorption fraction for a given thickness stack
    :param alphaAtWl: [float or None] the absorption coefficient at the wavelength of interest [um^(-1)]
    :param thStackTtoB: [list of float] the absorber layer thicknesses FROM TOP TO BOTTOM in um
    :param inputAbsFrac: [0<=float<=1] fraction of light to start with (incoming to the stack)
    :return: absFracPerLayer [list of float] the absorption fraction for each layer
    """

    transmissionFrac = inputAbsFrac
    absFracPerTh = []
    absFracPerLayer = []

    # first pass
    for layerTh in thStackTtoB:
        absFracPerTh.append(calcAbs(alpha=alphaAtWl, th=layerTh))
        absFracPerLayer.append(transmissionFrac * absFracPerTh[-1])
        transmissionFrac = transmissionFrac - absFracPerLayer[-1]

    # second pass (upwards)
    absFracPerTh.reverse()
    absFracPerLayer.reverse()
    for lidx in range(len(thStackTtoB)):
        secondPassAbsFrac = transmissionFrac * absFracPerTh[lidx]
        absFracPerLayer[lidx] += secondPassAbsFrac
        transmissionFrac = transmissionFrac - secondPassAbsFrac
    absFracPerLayer.reverse()

    return absFracPerLayer






def reparamTh_2Pass_absFrac(thDataframeTtoB, alphaAtWl=None, wl=None, kappaAtWl=None):
    """
    Calculate 2-pass BL absorption fractions for given layer thickness and append to the provided dataframe with columns <th-label>_absFrac2p
        Provide the absorption coefficient, or, imaginary refractive index and wavelength so it can be calculated
        Also calculates the tot_em_t_absFrac2p (sum of the layer absFrac)

    :param thDataframeTtoB: [pd.DataFrame] the dataframe with the layer thicknesses to calculate the abs for, thicknesses must be ordered top to bottom of the stack (top is light-side)
    :param alphaAtWl: [float or None] the absorption coefficient at the wavelength of interest [um^(-1)]
    :param wl: [float or None] the wavelength of interest, to calculate the absorption coefficient [um]
    :param kappaAtWl: [float or None] the imaginary refractive index at the wl of interest, to calculate the abs coeff
    :return: thDataframe: [pd.DataFrame] the input dataframe but with the <th-label>_absFrac2p columns added
    """

    if (wl==None or kappaAtWl==None) and alphaAtWl==None:
        print("ERROR :2-pass th reparam: Cannot compute reparameterized thicknesses when no alpha, or kappa and wl, is provided. Returning None...")
        return None

    if (wl!=None and kappaAtWl!=None) and alphaAtWl!=None:
        print("WARNING :2-pass th reparam: Alpha, and kappa and wl, are all provided. Using provided alpha.")

    if alphaAtWl == None:
        alphaAtWl = 4 * np.pi * kappaAtWl / wl  # [um^-1] Abs coefficient at illum wavelength


    print(f"2-pass th reparam: Generating _absFrac2p columns using alphaAtWl of '{alphaAtWl}' um^(-1) for the top to bottom thicknesses; '{thDataframeTtoB.columns.values}'.")

    #prep
    thLabelList = thDataframeTtoB.columns.values.tolist()
    absFracLabelList = [label+"_absFrac2p" for label in thLabelList]
    absFracDF = pd.DataFrame(columns=absFracLabelList)

    #run for each design
    for ptidx in range(thDataframeTtoB.shape[0]):

        thStackForPtIdx = list(thDataframeTtoB.loc[ptidx, thLabelList])
        absFracPerLayer = calc_2PassAbsFromThStack(alphaAtWl=alphaAtWl, thStackTtoB=thStackForPtIdx)
        absFracPerLayer.append(sum(absFracPerLayer))  # add sum data

        absFracSeries = pd.Series(absFracPerLayer, index=absFracLabelList+["tot_"+thLabelList[0][-4:]+"_absFrac2p"])
        absFracSeries = absFracSeries.to_frame().T
        absFracDF = pd.concat([absFracDF, absFracSeries], ignore_index=True)

    outDF = pd.concat([thDataframeTtoB, absFracDF], axis=1)#, ignore_index=True, join='outer', sort=False)

    print(f"2-pass th reparam: Success :Generation of _absFrac2p columns complete.")

    return outDF











if __name__ == '__main__':
    main()







