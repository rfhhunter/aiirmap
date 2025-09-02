import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import shutil
import pickle
import os
import time
import sys
import itertools
import random
from functools import reduce
from pathlib import Path
from sklearn.metrics import pairwise_distances

sys.path.append(str(Path(__file__).resolve().parents[1]))#append utilities to path
sys.path.append(str(Path(__file__).resolve().parents[2])) #append aiirmap home to path
from config import *
import databasing as dbh
import BeerLambertTools as bl
import plotting as pl
from pca import DimensionalityReductionPCA


# ------------------------------------------------------------------------------------------------------------------
# ------------------------------------------------------------------------------------------------------------------
# !    AIIRMAP CLASS AND OPERATION(S)
# ------------------------------------------------------------------------------------------------------------------
# ------------------------------------------------------------------------------------------------------------------
class aiirMapping():
    """
    Data class containing objects relevant to use of ML algorithms to map between data spaces.

        Attributes
        ----------
        mapType: {DR} [string]
            ML algorithm type
        mapName: [string]
            Descriptor of the mapping for file and folder names
        mapFolder: [string]
            String path to the mapping's ref folder
        settings: [dictionary]
            Dict of input variables for the mapping (includes algorithm hyperparameters)
            See below for valid settings for each algorithm
        ml: {drObj} [DimensionalityReduction]
            The machine learning object matching mapType
        filteredInputDB: [DataBase]
            DataBase after filtering - ML training DB, with all columns (inputs and outputs)
        filteredInputGrid: [DataBase]
            DataBase after filtering, grid extraction, and column cleaning - The actual ML training DB (only the columns incorporated into the DR)
        results: [dict]
            Results of the machine learning application (will vary for each algorithm)
            See below for valid results variables for each algorithm
    """

    #Todo: make sure comment below is up-to-date after implementing AE
    """
    Algorithm specific settings:
    
    PCA: 
    a two level dictionary
        'dr': {
            'type': 'pca',  # One of ['pca']; The type of DR algorithm to run
            'n_components': 2,  # (int) Number of reduced dimensions (for PCA)
            'scale': True,  # (bool) scale training data before DR
        }
        'cols': { #colummns to include/cut
            'include': None,  # (list of col labels, None) Supercedes cols.drop if not None! The input cols to include in the DR. All others are ignored. Pass only inputs.
            'drop': [], #(list of col labels) columns to drop before the DR; reference, outputs, non-number, and columns with only one unique value are already excluded, combined with dr_colsToDrop_baseline,
        },
        'filter': { #figure of merit filter
            'fom': 'Jph_norm0',  # (str, None) figure of merit column label
            'lo': 0.99,  # (float, None) the lower bound cutoff for the figure of merit (expts w fom < val are cut)
            'hi': None,  # (float,None) the upper bound cutoff for the fom (expts with fom > val are cut)
            'stage': None,  # (str,None) include only expts with 'sim_type' == val
        },
        'sweep': { #reduced space sweep grid
            'distance': None, #(float, array, None) dr.pca.subspace_mesh distance; Manhattan distance, in original space, between mesh pts (array is original component-wise)
            'n_points': None, #(int,array, None) dr.pca.subspace_mesh; Number of sampling points. (array is reduced space component-wise)
            'boundaries': None #(array (2,n_redSpaceComps)) dr.pca.subspace_mesh; The lower and upper bounnds of the region to be sampled in the reduced space. [reduced space units] (default; +-5 times the std dev of the projected training data)
        }
  
    """


    """
    Algorithm specific results:
    
    PCA:
        projectedGrid: [list]
            List of the sampled points, in the reduced space
        projectedGridNormalized: [list]
            List of the sampled points, in the reduced space but normalized
            such that the distance between two consecutive points in each
            dimension is the same (see subspace_mesh function in pca.py)
        outputGridDB: [DataBase]
            DataBase Grid object for subspace parameter search (in the original dimensional space)

    """


    def __init__(self, mapType, mapName, mapFolder, settings, ml, filteredInputDB, filteredInputGrid, results):

        if mapFolder == None:
            mapFolder = os.path.join(dbdir, f"map_{mapType}_{time.strftime(timeStr)}")
            if not os.path.exists(mapFolder):
                os.mkdir(mapFolder)

        self.mapType = mapType
        self.mapName = mapName
        self.mapFolder = mapFolder
        self.settings = settings
        self.ml = ml
        self.filteredInputDB = filteredInputDB
        self.filteredInputGrid = filteredInputGrid
        self.results = results
        # self.projectedGrid = projectedGrid
        # self.projectedGridNormalized = projectedGridNormalized
        # self.outputGridDB = outputGridDB

    # ------------------------------------------------------------------------------------------------------------------

    def mapSavePkl(self, saveName):
        """#Saves a map pkl file"""
        with open(f'{saveName}.pkl', 'wb') as file:
            pickle.dump(self, file)

    # ------------------------------------------------------------------------------------------------------------------

    def getGridCenterPtInRS(self):
        """
        finds and returns the center point of the projectedGrid
        @return: [pca1,pca2,...] list of coordinate values for the center of the grid in the RS
        """
        mins = []
        maxs = []
        for rsidx in range(len(self.projectedGrid[0])):
            mins.append(reduce(lambda x, y: min(x, y[rsidx]), self.projectedGrid, float('inf')))
            maxs.append(reduce(lambda x, y: max(x, y[rsidx]), self.projectedGrid, float('-inf')))
        ctr = []
        for rsidx in range(len(mins)):
            ctr.append((maxs[rsidx] - mins[rsidx]) / 2 + mins[rsidx])
        return ctr


# ------------------------------------------------------------------------------------------------------------------


def loadMapPkl(pklPath):
    # TODO allow it to take a glob/wildcard input
    """#opens a pkl map object"""
    with open(pklPath, 'rb') as file:
        aiirmap = pickle.load(file)
    return aiirmap



#------------------------------------------------------------------------------------------------------------------


##TODO make this general for both ML methods
def convertDictToMapSettings(dict):
    """
    Convert a flat dictonary containing the mapSetting keys to a mapSettings levelled dictionary.
    :param dict: [dict] Flat dictonary (single level) with the map setting keys (assumes all keys are present)
    :return: ms: [dict (of dicts)] 2-level dictionary in the MapSettings style
    """

    ms = {
        'dr': {
            'type': dict['type'],
            'n_components': dict['n_components'],
            'scale': dict['scale'],
        },
        'cols':{
            'include': dict['include'],
            'drop': dict['drop'],
        },
        'filter': {
            'fom': dict['fom'],
            'lo': dict['lo'],
            'hi': dict['hi'],
            'stage': dict['stage'],
        },
        'sweep': {
            'distance': dict['distance'],
            'n_points': dict['n_points'],
            'boundaries': dict['boundaries'],
        }
    }

    return ms




# ------------------------------------------------------------------------------------------------------------------
# ------------------------------------------------------------------------------------------------------------------
# ------------------------------------------------------------------------------------------------------------------

class DimensionalityReduction:
    """
        Basic class defining the dimensionality reduction

    """
    
    @staticmethod
    def get_dm(drSettings):
        """ 
        Return the proper dimensionality reduction object
        
        Parameters
        ----------
        settings: dict (formerly Parameters)
            List of settings appropriate for the selected dimensionality reduction method
        
        Returns
        ----------
        dm: object
            The dimensionality reduction object
        
        """
        
        if drSettings['type'] == 'pca':
            return DimensionalityReductionPCA(settings=drSettings)









# ------------------------------------------------------------------------------------------------------------------
# ------------------------------------------------------------------------------------------------------------------
# !    WORKER FUNCTIONS
# ------------------------------------------------------------------------------------------------------------------
# ------------------------------------------------------------------------------------------------------------------


def applyDR(inputDB, mapDescriptor, mapFolder=None, mapSettings=mapSettings_default, alphaAtWl=None, wl=None,
            kappaAtWl=None, returnNewRef=True, overwrite=True):
    """
    Prepare an input database for a DR run (performance filter, input extraction, column cleaning)
    Train DR algorithm (with scaling)
    Generate reduced space mesh to run on and a database from that
    Save everything to the folder and to an aiirmapping.

    :param inputDB: [DataBase] DataBase to apply filter, DR and to generate sweep for
    :param mapFolder: [str] path to the folder for the mapping
    :param mapDescriptor: [str] run descriptor string to use in folder and filenames
    :param mapSettings: [dict] the settings for the dimensionality reduction mapping (default values given in config)
                #algorithm
                'dr.type': 'pca', #One of ['pca']; The type of DR algorithm to run ##TODO should this be here? (in the settings or separated out as in aiirMapping.mapType)
                'dr.no_components': 2, #(int) Number of reduced dimensions (for PCA)
                'dr.scale': True, #(bool) scale trainig daat before DR
                #columns to include/cut
                'cols.include': None, #(list of col labels) Supercedes cols.drop if not None! The input cols to include in the DR. All others are ignored. Send inputs only.
                'cols.drop': [], #(list of col labels) columns to drop before the DR; reference, outputs, non-number, and columns with only one unique value are already excluded (see db clean), combined with dr_colsToDrop_baseline,
                #figure of merit filter
                'filter.fom': 'Jsc_norm_min', #(str) figure of merit column label
                'filter.lo': None, #(float, None) the lower bound cutoff for the figure of merit (expts w fom < val are cut)
                'filter.hi': 0.1, #(float,None) the upper bound cutoff for the fom (expts with fom > val are cut)
                'filter.stage': None, #(str,None) include only expts with 'sim_type' == val
                #reduced space sweep grid
                'sweep.distance': None, #(float, array, None) dr.pca.subspace_mesh distance; Manhattan distance, in original space, between mesh pts (array is original component-wise)
                'sweep.n_points': None, #(int,array, None) dr.pca.subspace_mesh; Number of sampling points. (array is reduced space component-wise)
                'sweep.boundaries': None #(array (2,n_redSpaceComps)) dr.pca.subspace_mesh; The lower and upper bounnds of the region to be sampled in the reduced space. [reduced space units] (default; +-5 times the std dev of the projected training data)
    :param alphaAtWl: [float or None] the absorption coefficient at the wavelength of interest [um^(-1)] (used to get th from reparameterizations)
    :param wl: [float or None] the wavelength of interest, to calculate the absorption coefficient [um] (used to get th from reparameterizations)
    :param kappaAtWl: [float or None] the imaginary refractive index at the wl of interest, to calculate the abs coeff (used to get th from reparameterizations)
    :param returnNewRef: [boolean] return the db reference for the new database file or return the input db file
    :param overwrite: [boolean] whether to overwrite the data if the mapFolder exists or to create a time-appended new folder
    :return: inputDB or sweepDB depending upon returnNewRef
    :return: map: [AiirMapping] output aiirmap
    """

    if mapFolder is None:
        mapFolder = os.path.join(dbdir, mapDescriptor)

    print(f"\nDR: RUNNING DR mapping '{mapDescriptor}'.\n"
          f"DR: Map Settings: {mapSettings}"
          f"DR: Results will be saved to '{mapFolder}'")

    if os.path.isdir(mapFolder):
        if overwrite:
            print(f"WARNING :DR: mapFolder '{mapFolder}' already exists, overwriting")
        else:
            print(
                f"WARNING :DR: mapFolder '{mapFolder}' already exists, saving to '{mapFolder}_{time.strftime(timeStr)}'")
            mapFolder = f'{mapFolder}_{time.strftime(timeStr)}'
            os.makedirs(mapFolder)
    else:
        os.makedirs(mapFolder)

    # PREP DB for DR
    # performance row filtering
    if mapSettings['filter']['fom'] is not None:
        filteredInputDB = inputDB.dbFilter(mapSettings['filter'], fileDescriptor=mapDescriptor)
    else:
        filteredInputDB = inputDB
    filteredInputDB.dbSaveFile(os.path.join(mapFolder, filteredInputDB.dbfilename))

    # Get grid with all inputs (will be used in reconstruction)
    filteredInputGrid_AllInputs = filteredInputDB.dbExtractGrid()

    # Clean columns; ie. get only the columns wanted for DR (process cols.include/drop)
    if mapSettings['cols']['include'] is not None:
        # feed the filtered input DB rather than the grid; allows DR to be run on output and calculated cols (such as S4 optimized thicknesses, or absFrac reparam th)
        filteredInputGrid = filteredInputDB.dbDRCleanCols(drCols=mapSettings['cols']['include'],
                                                            fileDescriptor=mapDescriptor)
    else:
        # columns.drop option is active, use filtered grid
        filteredInputGrid = filteredInputGrid_AllInputs.dbDRCleanCols(drCols=mapSettings['cols']['include'],
                                                                          cleanOption=0,
                                                                          colsToDrop=mapSettings['cols']['drop'],
                                                                          fileDescriptor=mapDescriptor)  # clean incompatible and excess columns for algorithm
    filteredInputGrid.dbSaveFile(os.path.join(mapFolder, filteredInputGrid.dbfilename))

    # Scale; is accomplished within the DR object
    # filterDRInputGrid.dbScale() #scale data for algorithm

    # RUN DR
    # load and train DR, save stats to file, run subspace meshing
    print(f"\nDIMENSIONAL REDUCTION:")
    drobj = DimensionalityReduction.get_dm(mapSettings['dr'])  # type, n_components, scale
    drobj.train_model(training_data=filteredInputGrid.dataframe)
    drobj.save_stats(os.path.join(mapFolder, mapDescriptor + "_DR-performance"),
                     f"fileCreationTime,{time.strftime(timeStr)}\nmapFolder,{mapFolder}\nmapDescriptor,{mapDescriptor}\nmapSettings,{mapSettings}\n")
    if mapSettings['sweep']['distance'] != None or mapSettings['sweep']['n_points'] != None:
        sweepData, projected_grid, normalized_projected_grid = drobj.subspace_mesh(mapSettings['sweep']['distance'],
                                                                                   mapSettings['sweep']['n_points'],
                                                                                   mapSettings['sweep']['boundaries'])

        # Create DB from sweep data (th reparameterization is handled within)
        sweepInputsCols = list(filteredInputGrid.dataframe.columns)
        sweepDB = buildSweepDB(mapDescriptor, sweepData, sweepInputsCols, filteredInputGrid_AllInputs,
                               alphaAtWl=alphaAtWl, wl=wl,
                               kappaAtWl=kappaAtWl)  # build full sweep db using input information
        sweepDB.dbSaveFile(os.path.join(mapFolder, sweepDB.dbfilename))
        if returnNewRef:
            print(f"DR: New db reference, '{sweepDB.dbfilename}', returned.")

    else:
        sweepDB = dbh.DataBase()
        projected_grid = None
        normalized_projected_grid = None

    if returnNewRef:
        outdb = sweepDB
    else:
        outdb = inputDB

    # SAVE

    # create map object
    map = aiirMapping("DR", mapDescriptor, mapFolder, mapSettings, drobj, filteredInputDB, filteredInputGrid,
                      projected_grid, normalized_projected_grid, sweepDB)
    map.mapSavePkl(os.path.join(mapFolder, mapDescriptor))

    print(f"DR: Success :Mapping '{mapDescriptor}' complete.")

    return outdb, map, drobj


# ------------------------------------------------------------------------------------------------------------------


def buildSweepDB(mapDescriptor, rawSweepData, sweepInputsCols, fullInputData, alphaAtWl=None, wl=None, kappaAtWl=None):
    """
    !!!This function may need some work to be general!!!
    Build a DataBase from a DR subspace sweep data array
        Use the rawSweepData as the points to sweep over
        Get the column heading names for the rawSweepData from sweepInputCols
        Fill remaining model inputs using the first entry from each of the columns in fullInputData that are missing in the sweepData

    :param mapDescriptor: [str] descriptor string for filename
    :param rawSweepData: [ndarray or list] the sweep points in the original design space
    :param sweepInputsCols: [list of str] the column labels for the rawSweepData (ordered)
    :param fullInputData: [DataBase] the database containing all model inputs (to fill inputs which were not incl in the DR)
    :param alphaAtWl: [float or None] the absorption coefficient at the wavelength of interest [um^(-1)] (used for th reparameterizations)
    :param wl: [float or None] the wavelength of interest, to calculate the absorption coefficient [um] (used for th reparameterizations)
    :param kappaAtWl: [float or None] the imaginary refractive index at the wl of interest, to calculate the abs coeff (used for th reparameterizations)
    :return: sweepDB [DataBase] A database for the stage 3 sweep
    """

    # sweep db init and non-data params
    sweepDB = dbh.DataBase()
    sweepDB.dbFile = None
    sweepDB.dbfilename = f"{mapDescriptor}_SWEEP-GRID.csv"
    sweepDB.lineage = []
    sweepDB.lineage.append([time.strftime(timeStr),
                            f"created as output sweep grid from mapping run '{mapDescriptor}' with input grid '{fullInputData.dbfilename}'"])
    sweepDB.grid = True

    # Create DF

    # sweepInputsCols = list(sweepInputsDataBase.dataframe.columns) #do not change the original data
    sweepData = rawSweepData.copy()

    # account for th reparameterizations if present in columns to which DR was applied
    # find how many th columns are present in the original input DB
    thColInModel = []
    for col in fullInputData.dataframe.columns:
        if col[0:3] == "ARC" and (col[-1:] == "t" or col[-1:] == "d"):
            if not (fullInputData.dataframe[col] == 0).all():
                thColInModel.append(col)
        elif (col[0:3] == "seg" or col[0:2] == 'sc') and (col[-1:] == "t" or col[-1:] == "d") and "em" in col:
            if not (fullInputData.dataframe[col] == 0).all():
                thColInModel.append(col)

    # handle reparameterizations
    thColInDR = []
    for cidx in range(len(sweepInputsCols)):
        # handle absFrac
        if sweepInputsCols[cidx][-6:-2] == "Frac":  # absFrac column
            if sweepInputsCols[cidx][-2:] == "2p":  # 2pass absfrac column
                # calc thickness from 2pass absFrac for the corresponding column in sweepData
                for pt in range(sweepData.shape[0]):
                    sweepData[pt][cidx] = bl.calc_2passThStackFromAbs(sweepData[pt][cidx], alphaAtWl=alphaAtWl, wl=wl,
                                                                      kappaAtWl=kappaAtWl)
                # rename column in the list to reflect change (ie. remove _absFracNp)
                sweepInputsCols[cidx] = sweepInputsCols[cidx][:-10]
            else:  # 1pass absFrac column
                # calc thickness from 1pass absFrac for the corresponding column in sweepData
                for pt in range(sweepData.shape[0]):
                    sweepData[pt][cidx] = bl.calc_1passTh(sweepData[pt][cidx], alphaAtWl=alphaAtWl, wl=wl,
                                                          kappaAtWl=kappaAtWl)
                # rename column to reflect change
                sweepInputsCols[cidx] = sweepInputsCols[cidx][:-10]

        # handle S4 output names
        if sweepInputsCols[cidx][0:2] == "sc":
            # sc<n>_em<(2)>_d   -->   seg<n>_em<(2)>_t
            sweepInputsCols[cidx] = sweepInputsCols[cidx].replace('sc', 'seg').replace('d', 't')
            thColInDR.append(sweepInputsCols[cidx])

        if sweepInputsCols[cidx][0:3] == "ARC" and sweepInputsCols[cidx][-1:] == "d":
            # ARC<n>_d  -->  ARC<n>_t
            sweepInputsCols[cidx] = sweepInputsCols[cidx].replace('d', 't')
            thColInDR.append(sweepInputsCols[cidx])

    sweepDF = pd.DataFrame(sweepData, columns=sweepInputsCols)

    # handle totals columns if needed
    # !ASSUMES! only one tot column will be present and that the missing column is an emitter thickness
    if set(thColInDR) != set(thColInModel):
        # determine which layer thickness is missing
        missingCol = list(set(thColInModel) - set(thColInDR))

        if len(missingCol) > 1:
            print(
                f"ERROR :build sweep db: More than one missing th layer column ({missingCol}). Cannot continue with reparameterization. Returning empty sweepDB.dataframe...")
            sweepDB.dataframe = pd.DataFrame()
            return sweepDB
        missingCol = missingCol[0]

        # find the total column
        totColMask = ["tot" in colLabel for colLabel in sweepInputsCols]
        totCol = [sweepInputsCols[i] for i in range(len(sweepInputsCols)) if totColMask[i]]
        if len(totCol) > 1:
            print(
                f"WARNING :build sweep db: More than one column containing 'tot' substring are present ({totCol}). Using first entry to extract '{missingCol}' value.")
        elif len(totCol) == 0:
            print(
                f"ERROR :build sweep db: Missing reparam columns found ({missingCol}) but no 'tot' columns to use to calculate missing data. Cannot continue. Returning empty sweepDB.dataframe... ")
            sweepDB.dataframe = pd.DataFrame()
            return sweepDB

        # pull and remove tot columns from the data
        totData = sweepDF.loc[:, totColMask]
        totData = totData.loc[:, totData.columns[0]]
        notTotColMask = [not coli for coli in totColMask]
        sweepDF = sweepDF.loc[:, notTotColMask]

        # remove tot columns from header list
        sweepInputsCols = [sweepInputsCols[i] for i in range(len(sweepInputsCols)) if not totColMask[i]]

        # get the emitter th columns
        emThColMask = [("seg" in colLabel and "t" in colLabel) for colLabel in sweepInputsCols]
        missingData = totData - sweepDF.loc[:, emThColMask].sum(axis=1)
        sweepDF[missingCol] = missingData

    # create df
    sweepDB.dataframe = sweepDF  # pd.DataFrame(rawSweepData, columns=sweepInputsData.dataframe.columns)
    # print(f":::::::::\nFull Input: {fullInputData.dataframe.shape}\nSweep Output: {sweepDB.dataframe.shape}\n::::::::")

    # add parameter (model input) columns which are in the DataBase which is input into applyDR but not included in the DR
    # use first value from each of these columns as the default
    for column in fullInputData.dataframe.columns:
        if column not in sweepDB.dataframe.columns:
            sweepDB.dataframe.insert(loc=len(sweepDB.dataframe.columns), column=column,
                                     value=fullInputData.dataframe[column].iloc[0])

    # OLD IMPLEMENTATION FOR MISSING COLUMN FILL (<230310) ... UNINTUITIVE, BUGGY? what if columns in sweep are not in same order in fullInputData
    # cidx=0
    # while cidx < fullInputData.dataframe.shape[1]:
    #     #print(f"{cidx}
    #     if cidx >= sweepDB.dataframe.shape[1]:  # reached end of rawSweepData, remaining cols go at end
    #         sweepDB.dataframe.insert(loc=cidx, column=f"{fullInputData.dataframe.columns[cidx]}", value=fullInputData.dataframe.iloc[0,cidx])
    #     elif sweepDB.dataframe.columns[cidx] != fullInputData.dataframe.columns[cidx]: #
    #         #print(f"{sweepDB.dataframe.columns[cidx]} != {fullInputData.dataframe.columns[cidx]}")
    #
    #         sweepDB.dataframe.insert(loc = cidx, column = f"{fullInputData.dataframe.columns[cidx]}", value = fullInputData.dataframe.iloc[0,cidx])
    #         cidx = cidx - 1
    #
    #     cidx= cidx + 1

    # print(f":::::::::\nFull Input: {fullInputData.dataframe.shape}\nSweep Output: {sweepDB.dataframe.shape}\n::::::::")

    return sweepDB






#------------------------------------------------------------------------------------------------------------------

def investigateDR(DBs, invMapSettings, invDescriptor=f"{time.strftime(timeStr)}_DR_investigation", invFolder=None, overwrite=True, alphaAtWl=None, wl=None, kappaAtWl=None):
    """
    Inv(estigate) the effect of varying map and DR settings.
        Take invMapSettings where all desired variants are defined as lists in the dict values
        Convert to pandas dataframe
        Run applyDR, get results and fill to dataframe
        Output investigation dataframe
        Included columns:
            mapSettings.dr,cols, and filter settings
            outputgrid, map, drobj
            drobj.model.explained_variance_ratio_, its sum,

    Results are saved to dbdir/invDescriptor/ (or invFolder if it is not None)
    Within this folder are subfolders for each DB (if more than one), the mapping folders, the investigation dataframe

    ##to add. here?.. No, not here. Could be added as a new function for which this is a subfnc.
    ##If runSentaurus: run the sentaurus project for the reduced design spaces (!computationally expensive! consider the PCA investigation and reduced space sweep params carefully )
    ##Could also write a compare subfnc

    :param DBs: [DataBase or list of DataBase] input DB(s) upon which to run the DB innvestigation (run on each separately)
    :param invMapSettings: [dict of dicts] map settings with the desired variant settings in lists
    :param invDescriptor: [str] investigation descriptor for files and folder names
    :param invFolder: [str,None] folder to save results (will use dbdir/invDescriptor if None)
    :param overwrite: [bool] overwrite the dbdir save folder if it exists (appends __timeStr to folder name if False)
    :param alphaAtWl: [float or None] the absorption coefficient at the wavelength of interest [um^(-1)] (used to get th from reparameterizations)
    :param wl: [float or None] the wavelength of interest, to calculate the absorption coefficient [um] (used to get th from reparameterizations)
    :param kappaAtWl: [float or None] the imaginary refractive index at the wl of interest, to calculate the abs coeff (used to get th from reparameterizations)
    :return: investigation_dataframes_perDB [(list of )pd.DataFrames] the investigation dataframes for each DB (list if multiple DBs provided)
    """

    if invFolder is None:
        invFolder = os.path.join(dbdir,invDescriptor)
    if os.path.exists(invFolder):
        if overwrite:
            shutil.rmtree(invFolder)
        else:
            invFolder = invFolder+f"__{time.strftime(timeStr)}"
    os.makedirs(invFolder)

    print(f"investigate dr: Running DR investigation '{invDescriptor}'...\n" \
            f"investigate dr: Results will be saved to '{invFolder}")

    #dr.scale needs to be int representation of bool for itertools
    # if not isinstance(invMapSettings['dr']['scale'], list):
    #     if invMapSettings['dr']['scale']:
    #         invMapSettings['dr']['scale'] = 1
    #     else:
    #         invMapSettings['dr']['scale'] = 0
    # else:
    #     for sidx in range(len(invMapSettings['dr']['scale'])):
    #         if invMapSettings['dr']['scale'][sidx]:
    #             invMapSettings['dr']['scale'][sidx] = 1
    #         else:
    #             invMapSettings['dr']['scale'][sidx] = 0

    #cols needs to be lists of lists if present (a set is datapt)
    #check and convert
    if invMapSettings['cols']['include'] is not None:
        if not isinstance(invMapSettings['cols']['include'][0], list):
            invMapSettings['cols']['include'] = [ invMapSettings['cols']['include'] ]
    if invMapSettings['cols']['drop'] is not None:
        if len(invMapSettings['cols']['drop']) > 0:
            if not isinstance(invMapSettings['cols']['drop'][0], list):
                invMapSettings['cols']['drop'] = [ invMapSettings['cols']['drop'] ]
        else:
            invMapSettings['cols']['drop'] = [invMapSettings['cols']['drop']]


    #generate list of dicts with permutations, convert to dataframe
    flatdict = {**invMapSettings['dr'], **invMapSettings['cols'], **invMapSettings['filter'], **invMapSettings['sweep']}
    for key in flatdict: #put unique values in lists for itertools
        if not isinstance(flatdict[key], list):
            flatdict[key]= [flatdict[key]]
    keys,vals = zip(*flatdict.items())
    permutationdicts = [dict(zip(keys,v)) for v in itertools.product(*vals)]
    invDF = pd.DataFrame.from_records(permutationdicts)
    #prepare for per-db results
    investigation_dataframes_perDB = []

    #For each DB in the list, run separately, save to own folder if more than one
    if not isinstance(DBs, list):
        DBs = [DBs]

    for dbidx in range(len(DBs)):
        print(f"investigate dr: Running DR investigation on DataBase '{dbidx+1}' of '{len(DBs)}' (filename: {DBs[dbidx].dbfilename})...")
        invDFdb = invDF.copy()
        sweepdbs, maps, drobjs = [], [], []
        var, var_sum = [], [] #variance explained
        var_ratios, var_ratio_sums = [], [] #variance explained [%]
        data_var = [] #variance in the data (calculated from var_sum / var_sum_ratio)
        n_expts_train = [] #number of experiments in training data
        mapFolderi = ""

        #run each setting
        for eidx in range(invDF.shape[0]):
            n = len(str(invDF.shape[0])) #length of left padding for map-nn folder names
            #set map descriptor and folder
            if len(DBs) > 1:
                mapDescriptori = f"map-{eidx:0{n}}__{invDescriptor}__DB-{dbidx}"
                mapFolderi = os.path.join(invFolder, DBs[dbidx].dbfilename[0:-4], mapDescriptori)
            else:
                mapDescriptori = f"map-{eidx:0{n}}__{invDescriptor}"
                mapFolderi = os.path.join(invFolder, mapDescriptori)
            # os.makedirs(mapFolderi) #made by applyDR

            # get expt map settings
            dicti = permutationdicts[eidx]
            mapseti = convertDictToMapSettings(dicti)

            #apply dr
            sweepdbi, mapi, drobji = dbh.applyDR(DBs[dbidx], mapDescriptori, mapFolder=mapFolderi, mapSettings=mapseti, alphaAtWl=alphaAtWl, wl=wl, kappaAtWl=kappaAtWl,)

            sweepdbs.append(sweepdbi.dbFile) ; maps.append(mapi.mapFolder) ; drobjs.append(drobji)
            var.append(drobji.model.explained_variance_)
            var_sum.append(np.sum(var[-1]))
            var_ratios.append(drobji.model.explained_variance_ratio_)
            var_ratio_sums.append(np.sum(var_ratios[-1]))
            data_var.append(var_sum[-1]/var_ratio_sums[-1])
            n_expts_train.append(mapi.filteredDRInputGrid.dataframe.shape[0])


        #add result lists to db dataframe, save, pass to results
        invDFdb['sweep_db'] = sweepdbs
        invDFdb['map'] = maps
        invDFdb['drobj'] = drobjs
        invDFdb['var'] = var
        invDFdb['var_tot'] = var_sum
        invDFdb['var_pc'] = var_ratios
        invDFdb['var_tot_pc'] = var_ratio_sums
        invDFdb['data_var'] = data_var
        invDFdb['n_expts_train'] = n_expts_train

        invDFdb.to_pickle(os.path.join(os.path.split(mapFolderi)[0], f"{invDescriptor}__{DBs[dbidx].dbfilename[0:-4]}__investigation-results.pkl"))
        invDFdb.to_csv(os.path.join(os.path.split(mapFolderi)[0], f"{invDescriptor}__{DBs[dbidx].dbfilename[0:-4]}__investigation-results.csv"))

        investigation_dataframes_perDB.append(invDFdb)


    print(f"investigate dr: Success :Completed DR investigation '{invDescriptor}'. Results are stored in '{invFolder}'.")

    if len(investigation_dataframes_perDB) == 1:
        return investigation_dataframes_perDB[0]
    else:
        return investigation_dataframes_perDB

# ------------------------------------------------------------------------------------------------------------------

def load_DRinvestigation_pkl(pklpath):
    """
    Simple load pickle wrapper. For loading investigation dataframes 
    :param pklpath: [str or list of str] The paths to the investigation dataframe pickles
    :return: invDF: [DataFrame or list of DataFrame] The investigation DataFrames from each path
    """
    if not isinstance(pklpath,list): pklpath = [pklpath]

    invDF = []
    for path in pklpath:
        print(f"load invDF: Loading existing investigation dataframe pickle; '{path}'")
        with open(path, 'rb') as file:
            idf = pickle.load(file)
        invDF.append(idf)

    if len(invDF) == 1:
        return invDF[0]
    else:
        return invDF
    

# ------------------------------------------------------------------------------------------------------------------

def runNewDR(indb, mapSettings, drFileDescriptor, alphaAtWl=None, wl=None, kappaAtWl=None):
    """
    Check if the input mapSettings indicate a DR investigation or a single DR run and run the right one

    :param mapSettings: [2-level map settings dictionary] The map settings for the dr run/investigation.
    :param indb: [DataBase or list of DataBases] The DataBase(s) to run the DR on, each DB is run separately
    :param drFileDescriptor: [str or list of str] folder and filename descriptor for the dr run(s)/investigation(s)
    :param alphaAtWl: [float or None] the absorption coefficient at the wavelength of interest [um^(-1)] (used to get th from reparameterizations - pass alpha or both wl and kappa)
    :param wl: [float or None] the wavelength of interest, to calculate the absorption coefficient [um] (used to get th from reparameterizations)
    :param kappaAtWl: [float or None] the imaginary refractive index at the wl of interest, to calculate the abs coeff (used to get th from reparameterizations)
    :return: outdb: [DataBase or list of DataBase or None] The subspace sweep grid DataBase for each input DataBase. For a single DR run, None if an investigation.
    :return: map: [aiirMapping or list of aiirMapping or None] The aiirMapping objects for each input DataBase. For a single DR run, None if an investigation.
    :return: drobj: [DR Object or list of DR Object or None] The dimensionality reduction objects for each input DataBase. For a single DR run, None if an investigation.
    :return: invDF: [pd.DataFrame or list of DataFrames or None] The DR investigation DataFrames for each input DataBase. For an investigation,, None if a single DR run.
    """

    #DR investigation
    if isinstance(mapSettings['dr']['type'], list) \
            or isinstance(mapSettings['dr']['n_components'], list) \
            or isinstance(mapSettings['filter']['lo'], list) \
            or isinstance(mapSettings['filter']['hi'], list)\
            or isinstance(mapSettings['cols']['include'][0], list):
        if not isinstance(indb, list): #single db
            print(f"run DR: Running DR investigation for '{indb.dbfilename}'")
            invDF = investigateDR(indb, mapSettings, invDescriptor=drFileDescriptor,  alphaAtWl=alphaAtWl, wl=wl, kappaAtWl=kappaAtWl)
        else: #multiple dbs
            print(f"run DR: Running DR application for '{len(indb)}' DBs...")
            invDF = []
            for dbidx in range(len(indb)):
                invDFi = investigateDR(indb[dbidx], mapSettings, invDescriptor=drFileDescriptor[dbidx], alphaAtWl=alphaAtWl, wl=wl, kappaAtWl=kappaAtWl)
                invDF.append(invDFi)
        outdb, map, drobj = None, None, None

    #single DR run
    else:
        if not isinstance(indb,list): #single db
            print(f"run DR: Running DR application for DB '{indb.dbfilename}'...")
            outdb, map, drobj = dbh.applyDR(indb, drFileDescriptor, mapSettings=mapSettings, alphaAtWl=alphaAtWl, wl=wl, kappaAtWl=kappaAtWl)
        else: #multiple dbs
            print(f"run DR: Running DR application for '{len(indb)}' DBs...")
            outdb, map, drobj = [], [], []
            for db in indb:
                odb,m,dro = dbh.applyDR(db, drFileDescriptor+"__"+db.dbfilename, mapSettings=mapSettings, alphaAtWl=alphaAtWl, wl=wl, kappaAtWl=kappaAtWl)
                outdb.append(odb)
                map.append(m)
                drobj.append(dro)
        invDF = None

    print(f"run DR: Success :Completed DR run.")

    return outdb, map, drobj, invDF



#------------------------------------------------------------------------------------------------------------------

def runStage3MeshGen(indb, colsToIncl, drDescriptor=None,
                     dim=6, fom='Jph_norm1', fomLo=0.986, fomHi=None,
                     mesh_distance=None, mesh_n_points=None, mesh_boundaries=None,
                     trainingDataBoundaries=True, meshSizeDistanceCheck=True, negativeValReplacement='rand',
                     alphaAtWl=None, wl=None, kappaAtWl=None, saveGrid=True):
    """
    Create a stage 3 reduced space grid database.

    :param indb: [DataBase] The database to run DR on.
    :param colsToIncl: [list of str] The labels of the columns to be included in the DR (must already be included in indb)
    :param drDescriptor: [str or None] Descriptor for the DR run and Stage 3 grid folders and filenames. If None uses ~{indb.dbfilename}_dim-{dim}_{fom}-{fomCutStr}
    :param dim: [int] The dimensionality of the reduced space
    :param fom: [str] The label for the FOM to filter the input db on for the DR run.
    :param fomLo: [float or None] The FOM lo cut-off value to filter with (Filter is >= fomLo. Only one of fomLo or fomHi can be non-None)
    :param fomHi: [float or None] The FOM hi cut-off value to filter with (Filter is <= fomHi. Only one of fomLo or fomHi can be non-None)
    :param mesh_distance: [float, list of float, or None] The subspace mesh distance(s) to use for all RS dimensions, or for each RS dimension, or...
                                        if meshSizeDistanceCheck the distances to check the size of the subspace for. USER INPUT DURING FNC RUN IS REQUIRED TO SELECT WHICH DISTANCE TO USE
                                        (only one of mesh_distance and mesh_n_points can be set, if setting for each RS dimension make sure that len(mesh_distance)==dim)
    :param mesh_n_points: [int, list of int, or None] The number of points to use in all RS dimensions, or for each RS dimension
                                        (only one of mesh_distance and mesh_n_points can be set, if setting for each RS dimension make sure that len(mesh_n_points)==dim)
    :param mesh_boundaries: [2d np.array; with dims (2,n_components) or None] the lower (first row) and upper (second row) bounds
                            for each dimension of the reduced space, in PCA order, in units of the reduced space dimensions
                            If None and trainingDataBoundaries is False; Uses pca.subspace_mesh default (+- 5 standard deviations)
                            If trainingDataBoundaries is True; will use the extrema of the training data in the RS (overwrites any hard-setting of this variable)
    :param trainingDataBoundaries: [bool or float or None] Whether of not to use the extrema of the training data in the RS for the RS boundaries (overwrites any hard-setting of mesh_boundaries)
                                        if False or None; use mesh_boundaries
                                        if True or 1.0; use the training data bounds as the boundaries
                                        if float and != 1.0; use the training data bounds ranges extended (or reduced) by this multiplier, keep centered. See getMinMaxInReducedSpace for more info
    :param meshSizeDistanceCheck: [bool] Whether to run a check of how many points are in each reduced space for the given list of mesh_distance. REQUIRES USER INPUT DURING FNC OPERATION
    :param negativeValReplacement: ['rand','abs', or float] How to replace negative thicknesses in the grid
                                        rand; Use a random value in the range of the training data (default, used if input is unrecognized)
                                        abs; Use the absolute value of the negative thickness
                                        float; Thickness to replace negative layer thicknesses with. In um. Applied to all neg th.
    :param alphaAtWl: [float or None] Absorption coefficient at wl of interest for the absorber material (used to compute thicknesses if absFrac are included in the DR instead of th, supercedes wl and kappa)
    :param wl: [float or None] The wavelength of interest (used to compute thicknesses if absFrac are included in the DR instead of th, superseded by alphaAtWl, require kappaAtWl defn)
    :param kappaAtWl: [float or None] The imaginary refractive index for the absorber at the wl of interest (used to compute thicknesses if absFrac are included in the DR instead of th, superseded by alphaAtWl, require wl defn)
    :param saveGrid: [bool] Save the stage 3 grid? (will be saved to dbdir/drDescriptor/drDescriptor_full-grid.csv
    :return: outdb: [DataBase] The stage 3 grid DataBase
    :return: mapobj: [AiirMapping] The aiirmap object for the Stage 3 grid generation DR run.
    :return: drobj: [DimensionalityReduction] The DR object for teh Stage 3 grid generation DR run.
    """

    # prep inputs
    if (fomLo != None and fomHi != None) or (fomLo == None and fomHi == None):
        print(f"ERROR :stage 3 mesh gen: Cannot define or not-define both fomLo and fomHi. Exiting...")
        return
    else:
        if fomLo is not None:
            fomCutStr = f"{fom}-gt{str(fomLo).replace('.', 'p')}"
        else:
            fomCutStr = f"{fom}-lt{str(fomHi).replace('.', 'p')}"

    if (mesh_distance == None and mesh_n_points == None) or (mesh_distance != None and mesh_n_points != None):
        print(f"ERROR :stage 3 mesh gen: Cannot define or not-define both mesh_distance and mesh_n_points. Exiting...")
        return

    if mesh_distance is None and meshSizeDistanceCheck:
        print(
            f"WARNING :stage 3 mesh gen: Cannot check mesh size for different distances when mesh_distance input is None. Using mesh_n_points to generate subspace mesh...")
        meshSizeDistanceCheck = False

    if mesh_boundaries is not None and trainingDataBoundaries:
        print(
            f"WARNING :stage 3 mesh gen: trainingDataBoundaries is True and mesh_boundaries is not None. mesh_boundaries will be overwritten with the training data extrema...")

    if drDescriptor is None:
        drDescriptor = indb.dbfilename[:-4] + f"_St3_dim-{dim}_{fomCutStr}"

    print(
        f"stage 3 mesh gen: Generating Stage 3 reduced space mesh '{drDescriptor}' using dimensionality '{dim}', FOM cutoff {fomCutStr}, and columns '{colsToIncl}'.")
    print(f"stage 3 mesh gen: Mesh size distance check is selected. !THIS WILL REQUIRE USER INPUT!... ")

    # DR settings
    mapSettings = {
        # algorithm
        'dr': {
            'type': 'pca',  # One of ['pca']; The type of DR algorithm to run
            'n_components': dim,  # [2,3,4,5,6],  # (int) Number of reduced dimensions (for PCA)
            'scale': True,  # (bool) scale training data before DR
        },
        # colummns to include/cut
        'cols': {
            'include': colsToIncl,  # To be overwritten below
            # (list of col labels, None) Supercedes cols.drop if not None! The input cols to include in the DR. All others are ignored. Pass only inputs.
            'drop': [],
            # (list of col labels) columns to drop before the DR; reference, outputs, non-number, and columns with only one unique value are already excluded, combined with dr_colsToDrop_baseline,
        },
        # figure of merit filter
        'filter': {
            'fom': fom,  # (str, None) figure of merit column label
            'lo': fomLo,  # (float, None) the lower bound cutoff for the figure of merit (expts w fom < val are cut)
            'hi': fomHi,  # (float,None) the upper bound cutoff for the fom (expts with fom > val are cut)
            'stage': None,  # (str,None) include only expts with 'sim_type' == val
        },
        # reduced space sweep grid, to be overwritten below
        'sweep': {
            'distance': None,
            # (float, array, None) dr.pca.subspace_mesh distance; Manhattan distance, in original space, between mesh pts (array is original component-wise)
            'n_points': None,
            # (int,array, None) dr.pca.subspace_mesh; Number of sampling points. (array is reduced space component-wise)
            'boundaries': None,
            # (array (2,n_redSpaceComps),None) dr.pca.subspace_mesh; The lower and upper bounnds of the region to be sampled in the reduced space. [reduced space units] (default; +-5 times the std dev of the projected training data)
        }
    }

    # get subspace mesh settings
    if trainingDataBoundaries != None:
        if trainingDataBoundaries != False:
            if trainingDataBoundaries == True: trainingDataBoundaries = 1.0
            print(f"stage 3 mesh gen: Running DR to obtain training data bounds in the reduced space...")
            # run with small subspace mesh to get the bounds for the big mesh
            outdb, mapobj, drobj, _ = runNewDR(indb, mapSettings=mapSettings,
                                               drFileDescriptor=drDescriptor + "_forBounds", wl=wl,
                                               kappaAtWl=kappaAtWl, alphaAtWl=alphaAtWl)
            mesh_boundaries = getMinMaxInReducedSpace(drobj, trainingDataBoundaries)
            print(f"stage 3 mesh gen: Training data bounds in reduced space obtained.")
            print(f"minmax\n{mesh_boundaries}")

    if meshSizeDistanceCheck:
        print(f"stage 3 mesh gen: Running mesh size distance check...")
        checkPCASubspaceSizeGivenScalarDistances(drobj, mesh_distance, boundsInRedSp=mesh_boundaries)
        mesh_distance = input(f"stage 3 mesh gen: WHAT DISTANCE WOULD YOU LIKE TO USE?")
        while not isinstance(mesh_distance, float) and not isinstance(mesh_distance, int):
            try:
                mesh_distance = float(mesh_distance)
            except TypeError:
                mesh_distance = input(
                    f"ERROR: stage 3 mesh gen: Entered mesh_distance is not an int or a float. Please reenter.")

    # set mapSettings for the mesh generation
    mapSettings['sweep']['distance'] = mesh_distance
    mapSettings['sweep']['n_points'] = mesh_n_points
    mapSettings['sweep']['boundaries'] = mesh_boundaries

    print(
        f"stage 3 mesh gen: Running DR to generate Stage 3 mesh using: mesh_distance={mesh_distance}, mesh_n_points={mesh_n_points}, mesh_boundaries={mesh_boundaries}...")

    outdb, mapobj, drobj, _ = runNewDR(indb, mapSettings=mapSettings, drFileDescriptor=drDescriptor, wl=wl,
                                       kappaAtWl=kappaAtWl, alphaAtWl=alphaAtWl)

    print(f"stage 3 mesh gen: Reduced space mesh database has been generated. Checking for negative thicknesses...")

    # make checks for negative thicknesses
    subdf = outdb.dataframe[colsToIncl]  # check all possible th
    print(
        f"WARNING: stage 3 mesh gen: There are {(subdf.lt(0).sum().sum())} negative values in the generated colsToIncl in the dataframe. These will be replaced with {negativeValReplacement} values...")
    if (subdf.values < 0).any():
        plt.figure("Negative Value Replacement Histo")
        # negVals = [int(i) for i in list(subdf[subdf <= 0.0].count().values[0])]
        cols = list(subdf.columns.values)
        negVals = subdf[subdf <= 0.0].count().values
        negDFData = np.array([negVals])  # ,cols])
        cols = np.array([cols])
        negDF = pd.DataFrame(data=negDFData.transpose(), columns=['vals'], dtype=int)  # ,'cols'])
        negDF = pd.concat([negDF, pd.DataFrame(data=cols.transpose(), columns=['cols'])], axis=1)
        # negDF.T.hist()#negVals, bins=len(subdf.columns.values))#x  , bins=subdf.columns.values)
        # plt.xticks((list(subdf.columns.values)))
        negDF.plot.bar(x='cols', y='vals')
        plt.legend().remove()
        plt.ylabel('# of Negative Values in RS')

        if isinstance(negativeValReplacement, float):
            ##OLD METHOD = Use a set value for negative thicknesses (was default set to 50nm)
            subdf[subdf < 0] = negativeValReplacement
            outdb.dataframe.loc[:, colsToIncl] = subdf
            # print(f"NEGATIVE CHECK #2:\n{(subdf.values < 0).any()}")

        elif negativeValReplacement.lower() in ['abs', 'a', 'absolute']:
            ## NEW; Use the absolute value of the negative thickness
            cols = cols[0]
            for colidx in range(len(cols)):
                if negVals[colidx] > 0:
                    x = (outdb.dataframe[cols[colidx]] <= 0.0)  # list of true false
                    outdb.dataframe.loc[x, cols[colidx]] = np.abs(outdb.dataframe.loc[x, cols[colidx]])
                    pvals = np.abs(outdb.dataframe.loc[x, cols[colidx]])
                    pvals = pvals.reset_index()
                    print(
                        f"stage 3 mesh gen: Replacing {negVals[colidx]} negative values in column {cols[colidx]} using absolute values")
                    plt.figure()
                    plt.title(f"Replacement values for {cols[colidx]}")
                    plt.plot(pvals.loc[:, cols[colidx]])
                    plt.xlabel('Replacement Idx')
                    plt.ylabel('Value [$\mu m$]')
                    plt.figure()
                    plt.title(f"Replacement indicies for {cols[colidx]}")
                    plt.plot(pvals.loc[:, 'index'], marker='.', linestyle='')
                    plt.xlabel('Replacement Idx')
                    plt.ylabel('Actual Database Idx')
            # subdf[subdf < 0] = np.abs(subdf[subdf < 0])
            # outdb.dataframe.loc[:, arcThLabelsTtoB+absThLabelsTtoB_bottomHomo] = subdf
            print(f"stage 3 mesh gen: Replacing negative values using their absolute values")
            # print(f"NEGATIVE CHECK #2:\n{(subdf.values < 0).any()}")

        else:
            ##(NEW METHOD = Use a random value in the training data range for replacement)
            cols = cols[0]
            for colidx in range(len(cols)):
                if negVals[colidx] > 0:
                    minval = mapobj.filteredDRInputGrid.dataframe[colsToIncl[colidx]].min()
                    maxval = mapobj.filteredDRInputGrid.dataframe[colsToIncl[colidx]].max()
                    randNums = [random.uniform(minval, maxval) for i in range(negVals[colidx])]
                    x = (outdb.dataframe[cols[colidx]] <= 0.0)
                    outdb.dataframe.loc[x, cols[colidx]] = randNums
                    pvals = outdb.dataframe.loc[x, cols[colidx]]
                    pvals = pvals.reset_index()
                    print(
                        f"stage 3 mesh gen: Replacing {negVals[colidx]} negative values in column {cols[colidx]} using values between {minval} and {maxval}")
                    plt.figure()
                    plt.title(f"Replacement values for {cols[colidx]}")
                    plt.plot(randNums)
                    plt.xlabel('Replacement Idx')
                    plt.ylabel('Value [$\mu m$]')
                    plt.figure()
                    plt.title(f"Replacement indicies for {cols[colidx]}")
                    plt.plot(pvals.loc[:, 'index'], marker='.', linestyle='')
                    plt.xlabel('Replacement Idx')
                    plt.ylabel('Actual Database Index')

        if (outdb.dataframe[colsToIncl].values <= 0).any():
            print(f"ERROR\nERROR\nERROR\nUNEXPECTED; STILL HAVE NEGATIVE VALUES IN colsToIncl")

    if saveGrid:
        outdb.dbSaveFile(saveName=os.path.join(dbdir, drDescriptor, f"{drDescriptor}_full-grid.csv"))

    print(f"stage 3 mesh gen: Success :Reduced space mesh database has been generated.")

    return outdb, mapobj, drobj


# ------------------------------------------------------------------------------------------------------------------

def checkPCASubspaceSizeGivenScalarDistances(drobj, distances, boundsInRedSp='minmax', verbose=False):
    """
    Obtain the number of points in the reduced space mesh given one or many distances/resolutions
        Uses pca.subspace_mesh

    :param drobj: [DR Object] The dimensionality reduction object
    :param distances: [scalar, or list of scalars] The subspace_mesh distance scalars to use to create the mesh
                                            NOTE: if input is a list; this function interprets each item the list as separate grid resolution (not as a resolution per red. dim. as is possible in subspace_mesh)
    :param boundsInRedSp: [None, 'minmax' or 2d np.array; with dims (2,n_components)] the lower (first row) and upper (second row) bounds
                            for each dimension of the reduced space, in PCA order, in units of the reduced space dimensions]
                            If 'minmax': use the min and max of the training data
                            If None: uses subspace_mesh default of +-5 * training data std deviation
    :return: [int or list of ints] The number of points in the subspace mesh for each input distance
    """

    if not isinstance(distances, list):
        distances = [distances]

    if boundsInRedSp == 'minmax':
        boundsInRedSp = getMinMaxInReducedSpace(drobj)


    print(f"check subspace size: Checking number of points in reduced space mesh for scalar distances of '{distances}'...")
    print(f'check subspace size: Number of training data points; {len(drobj.training_data)} ')
    sizes = []
    for didx in range(len(distances)):
        mapping_grid, mapping_projected_grid, mapping_normalized_projected_grid = drobj.subspace_mesh(
            distance=distances[didx], boundaries=boundsInRedSp)
        #get unique values which will be investigated in the RS
        rsdimlists = []
        rsdimsizes = []
        for dim in range(len(mapping_projected_grid[0])):
            rsdimlists.append([])
            for nidx in range(len(mapping_projected_grid)):
                rsdimlists[-1].append(mapping_projected_grid[nidx][dim])
            rsdimlists[-1] = np.unique(np.array(rsdimlists[-1]))
            rsdimsizes.append(len(rsdimlists[-1]))

        print(f"check subspace size: {didx};\tDistance={distances[didx]}\tMesh Size={mapping_grid.shape}\tN-pts in RS={rsdimsizes}")
        if verbose: print(f"RS pts={rsdimlists}\n")
        sizes.append(mapping_grid.shape)

    print(f"check subspace size: Success :Check complete.")

    return sizes


# ------------------------------------------------------------------------------------------------------------------

def getMinMaxInReducedSpace(drobj, boundsExtenderMultiplier=1.0):
    """
    Get the minimum and maximum value for each reduced dimension that is present in the training data.
    Extend these ranges by the multiplier, keeping it centered.
    eg    if multiplier is 1; return the min and max as they are in the data
          if multiplier is 2; return the min-(max-min)/2 and max+(max-min)/2
          in general; return the min - (multiplier-1)*(max-min)/2  and  max + (multiplier-1)*(max-min)/2

    :param drobj: [Dimensionality Reduction Object] The DR object to analyze
    :param boundsExtenderMultiplier: [float] greater than 0, the range extender multiplier, see above for more info
    :return: minMaxInRedSp [2d np.array; with dims (2,n_components)] the lower (first row) and upper (second row) bounds
                            for each dimension of the reduced space, in PCA order, in units of the reduced space dimensions
    """

    print(f"get min max in RS: Obtaining min and max values of training data in the RS...")

    minMaxOfTrainingInRedSp = []
    for didx in range(drobj.projected_training_data.shape[1]):
        mini = min(drobj.projected_training_data[:,didx])
        maxi = max(drobj.projected_training_data[:,didx])
        toExtend = (boundsExtenderMultiplier-1.0)*(maxi-mini)/2
        mini = mini - toExtend
        maxi = maxi + toExtend
        minMaxOfTrainingInRedSp.append([mini,maxi])

    minMaxInRedSpArray = np.array(minMaxOfTrainingInRedSp)
    minMaxInRedSpArray = np.transpose(minMaxInRedSpArray)

    print(f"get min max in RS: Success : Min and max values of training data in the RS obtained. Returned with bounds extender multiplier of {boundsExtenderMultiplier}.")

    return minMaxInRedSpArray


def runDRInv_calcAvgRecoError_createMatrix(invDF, inputDB=None, pwdmetrics=['cityblock', 'euclidean'], inclPlot=True,
                                           savePlot=False, saveNamePlot=os.path.join(pldir,
                                                                                     f"DRinv_reconstruction_error_matrix_{time.strftime(timeStr)}.png"),
                                           **kwargs):
    """
    Obtains the average pwd between all input data and its reconstructed pairs and saves this data to the DRinv dataframe.
    Creates a invDF 2d heat matrix with average pwd as the performance metric (if inclPlot)

    Note: to use the training data for each RS in the DR investigation as the input data, set inputDB to None

    :param invDF: [pd.DataFrame or str] The investigation dataframe to run or the path to it's pickle file
    :param inputDB: [DataBase or None] the DataBase with the points to send through each dr in the investigation
                                            if None; uses training data
    :param pwdmetrics: [str or list of str] the sklearn pairwise distance metrics to use
    :param inclPlot: [bool] Create the 2d DR inv matrix plot of avg reconstruction error?
    :param savePlot: [bool] Save the 2d plot? (inclPlot must be True)
    :param saveNamePlot: [str] Path to save the 2d matrix plot (includes filename)
    :param kwargs: [dict] keyword arguments for pl.createDRInv_matrixPlot (uses defaults set below for any kwargs not supplied)
    :return: invDF: [dataframe] The DR investigation DF with the average reconstruction error column(s) added. Column labels; 'avgRecoErr_{pwdmetrics[midx]}_{dbName}'
    """

    # prep inputs
    if not isinstance(pwdmetrics, list):
        pwdmetrics = [pwdmetrics]

    if inputDB is None:
        dbName = "training"
    else:
        dbName = inputDB.dbfilename[:-4]

    print(
        f"average reco error for DR inv.: Calculating average reconstruction error for DR investigation using '{dbName}' data and '{pwdmetrics}' pwd metrics...")

    # load invDF if not directly supplied and get the useful data labels
    if isinstance(invDF, str):
        invDF = pd.read_pickle(invDF)

    map0 = invDF.loc[0, 'map']
    try:
        originalDimLabels = map0.filteredDRInputGrid.dataframe.columns.values  # if pkl includes map objects
    except AttributeError:  # if pkl includes map strings
        pathToTry = os.path.join(invDF.loc[0, 'map'], f"{os.path.split(invDF.loc[0, 'map'])[1]}.pkl")
        try:
            map0 = dbh.loadMapPkl(pathToTry)
        except FileNotFoundError:
            pathToTry = pathToTry[pathToTry.find('databases') + 10: len(pathToTry)]
            pathToTry = os.path.join(dbdir, pathToTry)
            map0 = dbh.loadMapPkl(pathToTry)
        originalDimLabels = map0.filteredDRInputGrid.dataframe.columns.values

    print(f"average reco error for DR inv.: Columns included in computation; '{originalDimLabels}'")

    # pull data if db is supplied
    if inputDB is not None:
        inputData = inputDB.dataframe[originalDimLabels]
        inputDataArray = inputData.to_numpy()

    # prep for output
    avgRecoErrorPwdPerMetric = []
    for midx in range(len(pwdmetrics)):
        avgRecoErrorPwdPerMetric.append([])

    # run reconstruction error computation sequentially for each DR in the inv DF
    for mapidx in range(invDF.shape[0]):
        aiirmap = invDF['map'].values[mapidx]
        if isinstance(aiirmap, str):
            # aiirmap = dbh.loadMapPkl(os.path.join(aiirmap, f"{os.path.split(aiirmap)[1]}.pkl"))
            pathToTry = os.path.join(aiirmap, f"{os.path.split(aiirmap)[1]}.pkl")
            try:
                aiirmap = dbh.loadMapPkl(pathToTry)
            except FileNotFoundError:
                pathToTry = pathToTry[pathToTry.find('databases') + 10: len(pathToTry)]
                pathToTry = os.path.join(dbdir, pathToTry)
                aiirmap = dbh.loadMapPkl(pathToTry)

        if inputDB is None:  # use training data
            inputData = aiirmap.filteredDRInputGrid.dataframe[originalDimLabels]
            inputDataArray = inputData.to_numpy()

        # project into RD and then back to OD
        projectedDataArray = aiirmap.ml.project(inputDataArray)
        reconstDataArray = aiirmap.ml.invert_model(projectedDataArray)

        # calc avg pwd (by calc'ing pwd for each point and finding avg at the end)
        for midx in range(len(pwdmetrics)):
            pwdsPerMetricPerPt = []
            for ptidx in range(len(inputData)):
                pwdmatrix = pairwise_distances([inputDataArray[ptidx, :], reconstDataArray[ptidx, :]],
                                               metric=pwdmetrics[midx])
                pwdsPerMetricPerPt.append(pwdmatrix[0, 1])
            avgRecoErrorPwdPerMetric[midx].append(np.array(pwdsPerMetricPerPt).mean())

    # set kwargs; use input kwargs to overwrite defaults (use defaults for the rest)
    if inclPlot:
        inkwargs = kwargs
        kwargs = {'xlabel': 'n_components', 'ylabel': 'lo', 'zlabel': f'avgRecoErr_{pwdmetrics[midx]}',
                  'xPlotLabel': 'Number of Reduced Dimensions', 'yPlotLabel': 'Jph FOM Cut Off',
                  'zPlotLabel': f'Average {pwdmetrics[midx]} PWD [um]',  # 'title': f'{pwdmetrics[midx]}',
                  'show': False, 'save': savePlot, 'saveName': saveNamePlot, 'cmap': 'YlOrRd'}
        for w in inkwargs:
            kwargs[w] = inkwargs[w]

    # save results back to dataframe and plot if desired
    for midx in range(len(pwdmetrics)):
        invDF[f'avgRecoErr_{pwdmetrics[midx]}'] = avgRecoErrorPwdPerMetric[midx]
        print(
            f"average reco error for DR inv.: Column 'avgRecoErr_{pwdmetrics[midx]}' added to the investigation dataframe.")
        if inclPlot:
            print(f"average reco error for DR inv.: Creating average reconstruction error DR investigation matrix.")
            pl.createDRInv_matrixPlot(invDF, **kwargs)

    print(f"average reco error for DR inv.: Success :Average reco error calculation complete.")

    return invDF



# ------------------------------------------------------------------------------------------------------------------

def runMap_recoErrorVsExpt_createPlot(aiirmap, inputDB=None, pwdmetrics=['cityblock', 'euclidean'], xlabel="Expt Idx",
                                      pwdUnitStr=" [um]", inclPlots=True, savePlots=False, saveFolderPlots=None):
    """
    Take a db and a DR object, project the relevant dimensions from the DB into the reduced dimensions and then back to the original dimensions
    Compute and return the PWD between the input and reconstructed data.

    Plot the PWD vs expt idx and the input and reconstructed data values in the original dimensions (if inclPlots)

    Note: Relevant dimensions are automatically detected from the aiirmap (they are the ones included in the DR)

    :param aiirmap: [dbh.aiirMapping] The aiirmap to run the reconstruction projection through.
    :param inputDB: [DataBase or None] The database with the data to run. If None uses the aiirmap's training data.
    :param pwdmetrics: [str or list of str] The pairwise distance metrics to use. (https://scikit-learn.org/stable/modules/generated/sklearn.metrics.pairwise_distances.html)
    :param xlabel: [str] Label for the x-axis of the plots.
    :param pwdUnitStr: [str] The string to append to the end of the y-axis label. (eg. " [um]", *include the space)
    :param inclPlots: [bool] Create the plots?
    :param savePlots: [bool] Save the plots (and info text file)?
    :param saveFolderPlots: [str] Path to the folder to save the plots and info text file
    :return: 
    """

    # prep inputs
    if not isinstance(pwdmetrics, list):
        pwdmetrics = [pwdmetrics]

    if savePlots and not os.path.exists(saveFolderPlots):
        os.makedirs(saveFolderPlots)

    # get the useful data labels
    originalDimLabels = aiirmap.filteredDRInputGrid.dataframe.columns.values
    # print(f"original space dimension labels: {originalDimLabels}")

    # pull input data
    if inputDB is None:  # use the training data
        inputDB = aiirmap.filteredDRInputGrid
        inputData = aiirmap.filteredDRInputGrid.dataframe[originalDimLabels]
        dbName = 'training'
    else:  # use inputDB
        inputData = inputDB.dataframe[originalDimLabels]
        dbName = inputDB.dbfilename[:-4]
    inputDataArray = inputData.to_numpy()

    print(
        f"reconstruction error vs expt: Computing the reconstruction error for '{dbName}' data through the '{aiirmap.mapName}' aiirmap...")
    print(f"reconstruction error vs expt: Columns included in computation; '{originalDimLabels}'.")

    # project into RD and then back to OD
    projectedDataArray = aiirmap.ml.project(inputDataArray)
    reconstDataArray = aiirmap.ml.invert_model(projectedDataArray)
    # reconstrData = pd.DataFrame(reconstDataArray, columns=originalDimLabels)
    # print(f"Original:{inputDataArray.shape}\tReconstructed:{reconstDataArray.shape}\tProjected:{projectedDataArray.shape}")

    # compute the pwd for each pt for each metric
    recoPwds = []
    for midx in range(len(pwdmetrics)):
        recoPwds.append([])
    for ptidx in range(len(inputData)):
        for midx in range(len(pwdmetrics)):
            pwdmatrix = pairwise_distances([inputDataArray[ptidx, :], reconstDataArray[ptidx, :]],
                                           metric=pwdmetrics[midx])
            # print(f"pwdmatrix:{pwdmatrix.shape}")
            recoPwds[midx].append(pwdmatrix[0, 1])

    # print stats to screen
    print(
        f"reconstruction error vs expt: Mapping information and reconstruction stats ---------------------------------")
    info = f"DB name: {dbName}\nNumber of Expts:{inputDB.dataframe.shape[0]}" \
           f"aiirmap name: {aiirmap.mapName}\nDR algorithm: {aiirmap.ml._type}\nRS dimensionality: {aiirmap.ml.n_components}\n" \
           f"OS dimensions incl: {originalDimLabels}\n\n" \
           f"PWD metric\t| Reco Error Avg.\t| Reco Error S.D.\n"
    for midx in range(len(pwdmetrics)):
        info += f"{pwdmetrics[midx]}\t| {np.mean(recoPwds[midx])}\t| {np.std(recoPwds[midx])}\n"
    print(info)
    print(
        f"------------------------------------------------------------------------------------------------------------")

    # Plotting (and save info to txt if savePlots)
    if inclPlots:
        # stats text file
        if savePlots:
            text_file = open(
                os.path.join(saveFolderPlots, f"reconstruction_error_vs_expt_INFO_{time.strftime(timeStr)}.txt"), "w")
            text_file.write(f"Reconstruction Error vs Expt\n{time.strftime(timeStr)}\n")
            text_file.write(info)
            text_file.close()

        # reco error vs expt idx
        plt.figure(f"reconstruction error vs expt for {dbName} through {aiirmap.mapName} ({time.strftime(timeStr)})")
        for midx in range(len(pwdmetrics)):
            plt.plot(recoPwds[midx], label=pwdmetrics[midx].capitalize(), alpha=0.5)
        plt.xlabel(xlabel)
        if len(pwdmetrics) == 1:
            plt.ylabel(f"{pwdmetrics[0].capitalize()} Reconstruction Error{pwdUnitStr}")
            plt.title(f"{pwdmetrics[0].capitalize()} Reconstruction Error{pwdUnitStr}")
        else:
            plt.ylabel(f"Reconstruction Error{pwdUnitStr}")
            plt.title(f"Reconstruction Error{pwdUnitStr}")
            plt.legend()
        if savePlots: plt.savefig(
            os.path.join(saveFolderPlots, f"reconstruction_error_vs_expt_{time.strftime(timeStr)}.png"))

        # input and reco data vs expt plot
        fig, axes = plt.subplots(nrows=3, ncols=int(np.ceil(len(originalDimLabels) / 3)))
        fig.canvas.manager.set_window_title(f"reconstruction error in original dimensions ; {time.strftime(timeStr)}")
        ax = axes.ravel()
        for didx in range(len(originalDimLabels)):
            ax[didx].plot(inputDataArray[:, didx], label='Input', color='k', alpha=0.8)
            ax[didx].plot(reconstDataArray[:, didx], label='Reconstr', color='r', alpha=0.5)
            ax[didx].set_xlabel(xlabel)
            ax[didx].set_ylabel(originalDimLabels[didx])
            ax[didx].legend()
            ax[didx].set_title(originalDimLabels[didx])
        # fig.tight_layout()
        if savePlots: plt.savefig(os.path.join(saveFolderPlots,
                                               f"reconstruction_error_vs_expt_input-vs-recons-values_{time.strftime(timeStr)}.png"))

    print(
        f"reconstruction error vs expt: Success :Completed computation of reco error vs expt (inclPlots={inclPlots}).")

    return recoPwds