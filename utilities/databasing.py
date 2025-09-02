"""
Database and AiirMapping Functions

Data classes and functions for interacting with them.
Includes (#!):
- Database Class (csv file based)
- Database Operations
- Functions for creating custom experiments/databases
-Helper Functions

22.06.29
"""


import csv
import pickle
import os
import time
import sys
import glob
import random
import numpy as np
import pandas as pd
from datetime import datetime
from functools import reduce

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
import BeerLambertTools as bl
sys.path.append("..")
from config import *


#------------------------------------------------------------------------------------------------------------------
#------------------------------------------------------------------------------------------------------------------
#!    DATABASE CLASS
#------------------------------------------------------------------------------------------------------------------
#------------------------------------------------------------------------------------------------------------------

class DataBase:
    """
        Class to operationally store the information about the database

        Attributes
        ----------
        grid:
            Boolean indicator if DataBase contains only inputs (ie. is a Grid)
        dataframe:
            Pandas Dataframe of the database
        dbFile: string
            path to the db file
        dbfilename: string
            DB and save filename (dbName + dbFileType)
        lineage: list of format [[<datetime>,db action], ... ]
            the past manipulations of the database
        sample_num: int
            Number of entries
        current_sample: obj
            Points to last pandas DataSeries
    """

    #todo?: Should functions such as dbFilter operate on self? Via a flag?

    #TODO; inherited lineage. keep it or remove it?


    def __init__(self, dbFile=None, dbfilename=None, verbose=amverbose):

        self.grid = None
        self.dbFile = dbFile
        self.dataframe = None
        self.dbfilename = dbfilename
        self.lineage = [] #list of format [[<datetime>,db action], ... ]
        self.sample_num = 0
        self.current_sample = None
        if dbFile == None:  # create a new db
            if dbfilename is not None: dbfnstr = f"'{dbfilename}' "
            else: dbfnstr=''
            if verbose: print(f"\ndb init: Empty db {dbfnstr}initialized.")
            pass
        else: #load db from db file
            #note that dbfilename will be pulled from the file in this case (so dbfn input here is ignored)
            if verbose: print(f"\ndb init: Initializing from file; '{dbFile}'")
            self.dbReadFile(dbFile)

    def __iter__(self):
        self._n_iteration = 0
        return self

    def __next__(self):
        if self._n_iteration < self.sample_num:
            self._n_iteration += 1
            return self.sample[self._n_iteration - 1]
        else:
            raise StopIteration

# ------------------------------------------------------------------------------------------------------------------

    def dbReadFile(self, dbFile, verbose=amverbose):

        if dbFile == None:
            print(f"ERROR :db read: dbFile cannot be None. Exiting.")
            return self

        self.dbFile = dbFile
        self.dbfilename = os.path.split(dbFile)[1]

        if verbose: print(f"db read: Reading from file '{self.dbFile}'")

        if not os.path.exists(dbFile):
            print(f"ERROR :db read: dbFile path '{dbFile}' does not exist. Returning empty DB.")
            return self

        # database file read on file type case (support csv, hdf5, pkl dbs and {sFileNameBody}_Info.csv sentaurus project info

        # csv db file or sentaurus project collection
        if dbFile.endswith('.csv'):

            if dbFile.endswith(f'{sFileNameBody}_Info.csv'):
                # new db reqd for sentaurus input
                if verbose: print(f"db read: Processing Sentaurus project collection file...")

                #extract the header's project info
                headerLen = csvFindHeader(dbFile)
                with open(dbFile, 'r') as inputFile:
                    lines = inputFile.readlines()
                #check if there are outputs (is it a GRID)
                if verbose: print(f"File Header:\n"+lines[headerLen])
                if "experiment-outputs" in lines[headerLen]: #check if its a grid
                    self.grid = False
                else:
                    self.grid = True
                headerData = [] #list of lists with each inner list a tab-separated header line
                skipNext = False
                for lidx in range(headerLen):
                    if lines[lidx] == "" or lines[lidx] == "\n" or lines[lidx].startswith("#") or skipNext:
                        skipNext = False
                        continue
                    linelist = lines[lidx].split(",")
                    if linelist[0].strip() ==  "" or linelist[0].startswith("File Creation Time"):
                        continue
                    elif linelist[0].startswith("Tool Flow"):
                        skipNext = True #skip next line
                        continue
                    elif len(linelist) < 3:
                        print(f"WARNING :db read: Could not load sentaurus project info header line beginning with "
                              f"'{linelist[0]}' into db. Ensure format is 'Variable Name,varLabel,value'. SKIPPING")
                        continue
                    else:
                        headerData.append(linelist[:3])
                #read per experiment data
                self.dataframe = pd.read_csv(dbFile, skiprows=headerLen)
                #add header/project info as DataSeries (new columns) to the start of the dataframe
                for hdidx in range(len(headerData)):
                    #print(f"hdidx-{hdidx}:  '{headerData[hdidx][1]}'")
                    self.dataframe.insert(loc=hdidx, column=headerData[hdidx][1], value=headerData[hdidx][2])
                #add filing cabinet origin
                self.dataframe.insert(loc=0, column="file_dir", value=os.path.split(dbFile)[0])
                #add lineage
                now = time.strftime(timeStr)
                self.lineage.append([now,f"created from {self.dbFile}"])
                self.dataframe.insert(loc=0, column="time_db", value=now)
                self.dbCleanIdxsAndGridness()
                if verbose: print(f"db read: Success :Loaded NEW db object from Sentaurus Project Info. UNSAVED.")



            else:  # csv db file
                if verbose: print(f"db read: Processing csv db file...")
                headerLen = csvFindHeader(dbFile)
                #load meta info
                lineageLine, keyResLine = False, False
                self.grid = False
                with open(dbFile, 'r') as inputFile:
                    lines = inputFile.readlines()
                for lidx in range(headerLen):
                    if lines[lidx] == "" or lines[lidx] == "\n" or lines[lidx].startswith('#') or lines[lidx].strip().startswith(","):
                        continue
                    else:
                        lines[lidx] = lines[lidx][:-1] #remove the \n
                    if lines[lidx].startswith('GRID FILE'):
                        self.grid = True
                    elif lines[lidx].startswith("dbfilename"):
                        self.dbfilename = lines[lidx].split(",")[1]
                    elif lines[lidx].startswith("dbFile"):
                        self.dbFile = lines[lidx].split(",")[1]
                    elif lines[lidx].startswith("Lineage:"):
                        lineageLine = True
                        continue
                    elif lines[lidx].startswith("key results:"): #was never implemented elsewhere
                        lineageLine = False
                        keyResLine = True
                        continue
                    else:
                        if lineageLine:
                            self.lineage.append(lines[lidx].split(",")[:2]) ##TODO; what about inherited lineage?
                        elif keyResLine:
                            continue #do not load key results
                        else:
                            print(f"WARNING :db read: Loading from csv db file, did not recognize header line:"
                                  f"'{lines[lidx]}'\nSKIPPING LINE")
                #load dataframe
                self.dataframe = pd.read_csv(dbFile, skiprows=headerLen, index_col=0, float_precision='round_trip')
                self.dbCleanIdxsAndGridness()
                if verbose: print(f"db read: Loaded db file '{dbFile}'")

        elif dbFile.endswith('hdf5'):  # hdf db file
            print(f"ERROR :db read: Load from hdf5 db file is not supported.")
        elif dbFile.endswith("pkl"):  # pickle db file
            print(f"ERROR :db read: Load from pkl db file is not supported.")
        else:
            print(f"ERROR :db read: File type not supported.")

        if verbose and self.dataframe != None:
            print(f"VALUES:\n{self.dataframe.to_string()}")
            print(f"DTYPES: {self.dataframe.dtypes}")
            print(f"DESCRIBE: {self.dataframe.describe()}")


        return self

# ------------------------------------------------------------------------------------------------------------------

    def dbSaveFile(self, saveName=None, returnNewRef=True, verbose=amverbose):
        """
        Saves database to file. Returns either a DataBase with unmodified dbFile pointer or a new DataBase pointing
        to the new database file at saveName.
        If saveName == self.dbFile return self. dbFile reference does not change (default, if saveName not supplied)
        If saveName == {sFileNameBody}_Info.csv return database pointing to <date_time>_db.csv.
        If saveName is a different name then return a database pointing to the new dbFile pointer if
        returnNewRef is True, otherwise return database pointing to same dbFile as input self.

        :param saveName: string: Path to the save file
        :param returnNewRef: boolean: return database object pointing to the (new) saveName
        :return: self: DataBase: (unmodified or modified dbFile pointer based upon above)
        """

        if saveName is None:
            if self.dbFile is not None:
                saveName = self.dbFile
            elif self.dbfilename is not None:
                saveName = os.path.join(dbdir, self.dbfilename)
            else:
                saveName = os.path.join(dbdir, f"{time.strftime(timeStr)}_DB.csv")

        print(f"\ndb save: Saving to file '{saveName}'. (returnNewRef={returnNewRef})")

        if self.dataframe is None:
            print(f"ERROR :db save: Empty dataframe! Cannot save. Exiting.")
            return self

        if saveName.endswith('.csv'):  # csv db file or sentaurus pyCollection_Info.csv file
            switchStr = ""

            #new file; switch self if desired
            if saveName != self.dbFile:
                if returnNewRef:
                    self.dbFile = saveName
                    self.dbfilename = os.path.split(saveName)[1]
                    switchStr = f"\ndb save: New database reference, '{self.dbfilename}', returned"
            else:
                if os.path.isfile(saveName):
                    print(f"WARNING :db save: saveName, '{saveName}', matches existing dbFile. Overwriting DB!")
                    if returnNewRef:
                        print(f"WARNING :db save: Cannot return'New'Ref when saveName is existing dbFile. Same DB reference is returned (DB is overwritten).")

            if not os.path.exists(os.path.split(saveName)[0]):
                os.makedirs(os.path.split(saveName)[0])


            #savename is sentaurus project info file; create new file
            if saveName.endswith(f'{sFileNameBody}_Info.csv'):
                print(f"WARNING :db save: Cannot overwrite pyCollection_Info.csv, saving instead file="
                      f"'/databaseFolder/<time>_db.csv'")
                saveName = os.path.join(dbdir, f"{time.strftime(timeStr)}_db.csv")
                self.dbFile = saveName
                self.dbfilename=os.path.split(saveName)[1]
                switchStr = f"\ndb save: New database reference, '{self.dbfilename}', returned"

            #Record save to lineage
            self.lineage.append([time.strftime(timeStr), f"Saved to file {saveName}."])

            #Prep and then write to file
            formattedLineage, grid = "", ""
            if self.grid == True: grid = "GRID FILE"
            for lidx in range(len(self.lineage)):
                formattedLineage += f"{self.lineage[lidx][0]},{self.lineage[lidx][1]}\n"
            with open(saveName,'w', newline='',encoding='UTF8') as file:
                file.write(f"#AiirMap DataBase\n{grid}\n"
                           f"dbfilename,{os.path.split(saveName)[1]}\ndbFile,{saveName}\n"
                           f"Lineage:\n{formattedLineage}")
            self.dataframe.to_csv(saveName, sep=",", na_rep="nan", index_label="db_idx", mode='a',encoding='UTF8')
            print(f"db save: Success : Saved csv db file {saveName}{switchStr}.")


        elif saveName.endswith('hdf5'):  # hdf db file
            print(f"ERROR :db save: Saving to hdf5 is not supported.")

        else:  # pickle db file
            print(f"ERROR :db save: Saving to pkl format is not supported.")

        return self

# ------------------------------------------------------------------------------------------------------------------

    def dbCherryPick(self, columnsToPick, pickWhere, numberToPick, fileDescriptor=None):
        """
        Pick out a few simulations (rows) from the database. Return a new database with just the selected results.
        Uses pandas.dataframe nsmallest and nlargest to pick out the first numberToPick smallest or largest entries ordered by columnsToPick
        https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.nlargest.html
        :param columnsToPick: [string or list] column name(s) upon which to pick/order by (column datatype should be a number)
        :param pickWhere: [string = "top", "bottom"] to pick from the top (largest) or bottom (smallest)
        :param numberToPick: [int] how many results to select
        :return: pickedDB: [DataBase] the database with the picked results

        """

        print(f"db cherry pick: Picking {pickWhere} {numberToPick} {columnsToPick} entries from {self.dbfilename}.")

        if isinstance(columnsToPick, str):
            if not columnsToPick in self.dataframe.columns:
                print(f"ERROR :db cherry pick: Columns '{columnsToPick} are not in the {self.dbfilename} dataframe. Returning self.'")
                return self
        else:
            if not any(columnsToPick in self.dataframe.columns):
                print(f"ERROR :db cherry pick: Columns '{columnsToPick} are not in the {self.dbfilename} dataframe. Returning self.'")
                return self

        # prep picked db
        pickedDB = DataBase()
        pickedDB.dbFile = None
        if fileDescriptor is None:
            pickedDB.dbfilename = f"{self.dbfilename}_PICKED-DB.csv"
        elif fileDescriptor[-4:] == ".csv":
            pickedDB.dbfilename = fileDescriptor
        else:
            pickedDB.dbfilename = fileDescriptor + "_PICKED-DB.csv"
        pickedDB.lineage = self.lineage.copy()
        pickedDB.lineage.append([time.strftime(timeStr), f"cherry picked db for '{pickWhere}' {numberToPick} of the '{columnsToPick}' entries"])

        # generate picked dataframe
        if pickWhere == 'top' or pickWhere == 't':
            pickedDB.dataframe = self.dataframe.nlargest(numberToPick, columnsToPick)
        else:  # pickWhere.lower == "bottom" or pickWhere.lower == "b"
            pickedDB.dataframe = self.dataframe.nsmallest(numberToPick, columnsToPick)

        pickedDB.dbCleanIdxsAndGridness()

        print(f"db cherry pick: Success : Picked {pickWhere} {numberToPick} {columnsToPick} entries from {self.dbfilename}.")
        return pickedDB



# ------------------------------------------------------------------------------------------------------------------

    def dbFilter(self, filterSettings=mapSettings_default['filter'], fileDescriptor=None):
        """
        Filter a DB's experiments according to one of the dataframe columns.
        :param filterSettings: (dict) unsupplied keys are sourced from mapSettings_default in config
                    'fom': 'Jsc_norm_min', #(str) figure of merit column label
                    'lo': None, #(float, None) the lower bound cutoff for the figure of merit (expts w fom < val are cut)
                    'hi': 0.1, #(float,None) the upper bound cutoff for the fom (expts with fom > val are cut)
                    'stage': None, #(str,None) include only expts with 'sim_type' == val

        :param fileDescriptor: (str) filename descriptor
        :return: filteredDB: (DataBase) the database w expts meeting the filter
        """

        #resolve missing filter settings from default
        keys = ['fom','hi','lo','stage']
        for kidx in range(len(keys)):
            if not keys[kidx] in filterSettings:
                filterSettings[keys[kidx]] = mapSettings_default['filter'][keys[kidx]]


        print(f"db filter: Filtering {self.dbfilename} for {filterSettings}")

        #check if valid figure of merit
        #construct if need be
        #apply bounds filter
        filteredDB = DataBase()
        filteredDB.dbFile = None


        if fileDescriptor is None:
            filteredDB.dbfilename = f"{self.dbfilename}_{filterSettings['fom']}_FILTERED-DB.csv"
        elif fileDescriptor[-4:] == ".csv":
            filteredDB.dbfilename = fileDescriptor
        else:
            filteredDB.dbfilename = fileDescriptor+f"_{filterSettings['fom']}_FILTERED-DB.csv"
        filteredDB.lineage = self.lineage.copy()
        filteredDB.lineage.append([time.strftime(timeStr),f"filtered db w settings: {filterSettings}"])

        #construct dataframe query
        query = ""
        if filterSettings["lo"] is not None and filterSettings["hi"] is not None:
            query = f'{filterSettings["lo"]} <= {filterSettings["fom"]} <= {filterSettings["hi"]}'
        elif filterSettings["lo"] is not None:
            query = f'{filterSettings["lo"]} <= {filterSettings["fom"]}'
        elif filterSettings["hi"] is not None:
            query = f'{filterSettings["fom"]} <= {filterSettings["hi"]}'
        else:
            print(f"WARNING :db filter: Cannot filter, upper or lower bound must be defined ({filterSettings})")
            return self
        if filterSettings["stage"] is not None:
            query += f' and sim_type = {filterSettings["stage"]}'

        #run dataframe query and check if successful
        filteredDB.dataframe = self.dataframe.query(query)
        if filteredDB.dataframe.empty:
            print (f"WARNING :db filter: Query failed.\nquery: '{query}'")
            return self

        filteredDB.dbCleanIdxsAndGridness()

        print(f"db filter: Success :UNSAVED filtered db '{filteredDB.dbfilename}' created.")
        return filteredDB




# ------------------------------------------------------------------------------------------------------------------


    def dbDRCleanCols(self, drCols=None, cleanOption=0, colsToDrop = mapSettings_default['cols']['drop'] + dr_colsToDrop_baseline, fileDescriptor = None):
        """
        Clean columns from the database dataframe.
        If drCols is not None; returns db with a datafram containing only drCols.
        Else, runs a cleanOption to drop a number of columns naturally and defined

        :param drCols: [list] the columns to include in the DR, Supercedes colsToDrop and cleanOption!!
        :param cleanOption: [int = 0,1] flag for different cleaning/cutting option sets
            0 : remove reference, object, and constant columns, remove columns {colsToDrop}
            1 : remove reference, object, and constant columns
        :param colsToDrop: [list] the names of the cleaned dataframe columns which you would like to drop
            (columns to drop After the dataframe is cleaned according to cleanOption)
        :param fileDescriptor: [string] descriptor string for the filename
        :return: cleanDB: [DataBase] cleaned database


        #CLEANING OPERATIONS
        #remove reference columns using inputStartStrs
        #remove any dtype = object columns
        #remove any dtype = <number-type> colummns where all rows have the same value
        #remove the columns given by colsToDrop AND colsToCleanDR(the config.py default colsToDrop)
        """

        #resolve clean method
        if drCols is not None:
            # include only
            cleanStr = f"include only {drCols}"
        else:
            # cut cols
            # prep cols to drop list from input and aiirmap's config.py
            if colsToDrop is None:
                colsToDrop = mapSettings_default.cols.drop + dr_colsToDrop_baseline
            if colsToDrop is not dr_colsToDrop_baseline:
                colsToDrop = list(set(dr_colsToDrop_baseline + colsToDrop))

            #resolve clean option
            if cleanOption == 0:
                cleanStr = f"remove reference, output, object, and constant columns, remove columns {colsToDrop}"
            elif cleanOption == 1:
                cleanStr = "remove reference, output, object, and constant columns"



        print(f"db dr clean: Cleaning {self.dbfilename} for DR mapping run {fileDescriptor}. Cleaning; {cleanStr}")

        #CLEAN
        #begin preparing cleaned DB
        #extract grid / remove outputs
        if not self.grid:
            tmpDB = self.dbExtractGrid()
        else:
            tmpDB = self.dbCopy()

        #process db info
        cleanDB = DataBase()
        cleanDB.dbFile = None
        if fileDescriptor is None:
            cleanDB.dbfilename = f"{self.dbfilename}_CLEAN-GRID.csv"
        elif fileDescriptor[-4:] == ".csv":
            cleanDB.dbfilename = fileDescriptor
        else:
            cleanDB.dbfilename = fileDescriptor+f"_CLEAN-GRID.csv"
        cleanDB.lineage = self.lineage.copy()
        cleanDB.lineage.append([time.strftime(timeStr), f"cleaned for dr ({cleanStr})"])
        #print(f"DTYPES:\n{self.dataframe.dtypes}")
        #print(f"DESCRIBE:\n{self.dataframe.describe}")

        if drCols is not None:
            cleanDB.dataframe = self.dataframe[drCols]

        else:
            #remove reference info (ie. the first columns of meta-parameters)
            inputColIndex,_ = findDFColIdx(tmpDB.dataframe, inputStartStrs)
            cleanDB.dataframe = tmpDB.dataframe.iloc[:, inputColIndex + 1:]

            #remove columns with no unique values
            cleanDB.dataframe = cleanDB.dataframe.loc[:, cleanDB.dataframe.apply(pd.Series.nunique) != 1]

            #TODO:handle object columns with differing data using one hot encoding
            #remove object columns
            cleanDB.dataframe = cleanDB.dataframe.select_dtypes(exclude=['object'])

            #remove cols to drop
            if cleanOption == 0:
                try: cleanDB.dataframe.drop(colsToDrop, axis=1, inplace=True)
                except KeyError:
                    print(f"ERROR :db dr clean: User selected columns to drop has an invalid column key. DR clean cancelled for {self.dbfilename}.")

        cleanDB.dbCleanIdxsAndGridness()

        print(f"db dr clean: Success :Cleaned db '{self.dbfilename}' with cleaning; {cleanStr}.")
        return cleanDB



# ------------------------------------------------------------------------------------------------------------------


    def dbExtractGrid(self, justInputs=True):
        """
        Extracts the inputs (grid) from the database.
        Returns the grid Database.

        :param: justInputs (bool), return just experiment-input columns (True) or reference and expt-input (False)
        :return: grid : (Database) the input grid DataBase object
        """

        print(f"db grid extract: Extracting input grid database out of '{self.dbfilename}'")
        #Pull all inputs
        grid = DataBase()
        grid.dbFile = None
        if self.lineage is not None:
            grid.lineage = self.lineage.copy()
        else: grid.lineage = []
        if self.dbfilename is not None:
            grid.dbfilename = f"{self.dbfilename[:-4]}_GRID{self.dbfilename[-4:]}"
            grid.lineage.append([time.strftime(timeStr), f"grid extracted from '{self.dbfilename}'"])
        grid.grid = True

        inputColIdx = 0
        if justInputs:
            for issidx in range(len(inputStartStrs)):
                try:
                    inputColIdx = self.dataframe.columns.get_loc(inputStartStrs[issidx])
                except KeyError:
                    if issidx == len(inputStartStrs)-1:
                        print(f"WARNING :db extract grid: Could not find any inputStartStrings. Input-only grid extraction failed. Is it already a grid? Returning self.")
                        return self
                    else: pass
                else:
                    break

        outputColIdx = 0
        for ossidx in range(len(outputStartStrs)):
            try:
                outputColIdx = self.dataframe.columns.get_loc(outputStartStrs[ossidx])
            except KeyError:
                if ossidx == len(outputStartStrs)-1:
                    print(f"WARNING :db extract grid: Could not find any outputStartStrings. Grid extraction failed. Is it already a grid? Returning self.")
                    return self
                else: pass
            else:
                break

        grid.dataframe = self.dataframe.iloc[:, inputColIdx+2:outputColIdx] #inputColIdx+2 removes inputStartString and expt_num columns

        print(f"db grid extract: Success :UNSAVED grid '{grid.dbfilename}' created.")

        return grid



 # ------------------------------------------------------------------------------------------------------------------


    def dbCleanIdxsAndGridness(self, verbose=amverbose):
        """
        Resets the index columns in the dataframe
        Resets grid based upon the presence of outputStartStrs in dataframe.columns
        :return:
        """
        if verbose: print(f"db clean idxs and gridness: Running for DB '{self.dbfilename}'.")
        self.dataframe = cleanDFIdxs(self.dataframe, verbose=verbose)
        grid = checkDBGridness(self, verbose=verbose)
        if self.grid != grid and self.grid != None:
            print(f"WARNING :db clean idx and gridness: The detected gridness, '{grid}', does not match the input DB's previous setting ({self.grid}). Overwriting...'")
        self.grid = grid
        if verbose: print(f"db clean idxs and gridness: Success :Indexs and gridness cleaned for DB '{self.dbfilename}'.")
        return self



# ------------------------------------------------------------------------------------------------------------------


    def dbCopy(self, new_dbFile=None):
        """
        Copy the DataBase to a new object (no references to old db objects)
        (use pandas.dataframe.copy())
        :return:
        """

        if new_dbFile is None:
            if self.dbFile is not None:
                new_dbFile = self.dbFile[0:-4] + f"_COPY-{time.strftime(timeStr)}.csv"
                new_dbfilename = os.path.split(new_dbFile)[1]
                dbfstr = new_dbFile
            else:
                new_dbFile = None
                if self.dbfilename is None:
                    new_dbfilename = None
                else:
                    new_dbfilename = self.dbfilename
                dbfstr = "UNSAVED"
        else:
            new_dbfilename = os.path.split(new_dbFile)[1]
            dbfstr = new_dbFile

        print(f"db copy: Copying DB '{self.dbfilename}' to '{new_dbFile}'...")
        newDB = DataBase()
        newDB.grid = self.grid
        newDB.dbFile = new_dbFile
        newDB.dbfilename = new_dbfilename
        newDB.lineage = self.lineage.copy()
        newDB.lineage.append([time.strftime(timeStr), f"Copied from {self.dbfilename} to '{dbfstr}'"])
        newDB.dataframe = self.dataframe.copy(deep=True)

        print(f"db copy: Success :Returning copy '{dbfstr}'.")

        return newDB






#------------------------------------------------------------------------------------------------------------------
#------------------------------------------------------------------------------------------------------------------
#!    DATABASE OPERATIONS
#------------------------------------------------------------------------------------------------------------------
#------------------------------------------------------------------------------------------------------------------

def mergeDBs(DBs, fileDescriptor=None, join='outer', **kwargs):
    """
        Concatenates DB dataframes along the simulation runs axis (Uses pandas.concat).
        Columns merge method is specified using join.
            If join=outer splits inputs and outputs to retain I/O separability
        Checks for duplicate entries and added nans at the end.

        :param DBs: [list of DataBases] to concatenate/merge
        :param fileDescriptor: [str] Descriptor string for merged db filenam (if ends in .csv; is filename)
        :param join: ['inner','outer','right','left'] method for dataframe join (see the link below)
        :param kwargs: pandas dataframe concat keywords
        https://pandas.pydata.org/docs/reference/api/pandas.concat.html
        :return: mergedDB: [DataBase] Database of the merged data.
    """

    #get number of nans currently in the databases' dataframes for later check
    nansAtStart = []
    dbfilenames = []
    for dbidx in range(len(DBs)):
        nansAtStart.append(DBs[dbidx].dataframe.isna().sum().sum())
        dbfilenames.append(DBs[dbidx].dbfilename)

    print(f"\ndb merge: Merging {dbfilenames} using join '{join}' and  kwargs '{kwargs}'.")

    # prep merged DB
    mergedDB = DataBase()
    mergedDB.dbFile = None

    #get file name
    if fileDescriptor is None:
        mergedDB.dbfilename = f"{time.strftime(timeStr)}__MERGED-DB" + DBs[0].dbfilename[-4:]
    else:
        if fileDescriptor[-4:] == ".csv":
            mergedDB.dbfilename = fileDescriptor
        else:
            mergedDB.dbfilename = fileDescriptor + ".csv"

    #set lineage
    mergedDB.lineage = []
    mergedDB.lineage.append([time.strftime(timeStr), f"merged from dbs '{DBs}' w settings: join;{join} and kwargs;{kwargs}"])
    if inheritLineage:
        for dbidx in range(len(DBs)):
            mergedDB.lineage.append([">>>>>",f"Inherited lineage: {DBs[dbidx].dbfilename}"])
            mergedDB.lineage + DBs[dbidx].lineage[-inheritLineageLength:]
            mergedDB.lineage.append(["<<<<<",""])
        mergedDB.lineage.append(["",""])


    #Run merge
    #if join is outer; separate (references/)inputs and outputs to maintain DB's IO separability
    if join == 'outer':
        inputDFs = []
        outputDFs = []
        for dbidx in range(len(DBs)):
            inputs, outputs = splitDBDataframeIO(DBs[dbidx])
            inputDFs.append(inputs)
            outputDFs.append(outputs)

        #merge Inputs/Outputs
        mergedIn = pd.concat(inputDFs, axis=0, join=join, ignore_index=False, **kwargs)
        mergedIn = cleanDFIdxs(mergedIn)
        mergedOut = pd.concat(outputDFs, axis=0, join=join, ignore_index=False, **kwargs)
        mergedOut = cleanDFIdxs(mergedOut)
        #recombine IO
        mergedDB.dataframe = pd.concat([mergedIn, mergedOut], axis=1, join=join, ignore_index=False)#, left_on=mergedIn.columns[0], right_on=mergedOut.columns[0] ) #recombine

    #inner, right, and left joins should maintain column ordering
    else:
        mergedDB.dataframe = pd.concat(DBs, axis=0, join=join, ignore_index=False, **kwargs)

    mergedDB.dbCleanIdxsAndGridness()


    #Make Checks
    #Check for duplicate entries
    #For duplicate entries all input and output values must be identical but reference variables can vary
    #split off pure inputs and outputs
    ioCheckDF = mergedDB.dataframe

    #print(f"\n\n{mergedDB.dataframe.columns}\n")

    colIdx, _ = findDFColIdx(ioCheckDF, inputStartStrs)
    if colIdx is not None:
        ioCheckDF = ioCheckDF.iloc[:, (colIdx+1):]
    _, ssidx = findDFColIdx(ioCheckDF, outputStartStrs)
    try: ioCheckDF.drop(labels=outputStartStrs[ssidx], axis=1, inplace=True)
    except KeyError: pass
    duplicates = ioCheckDF.duplicated()

    if any(duplicates):
        print(f"WARNING :db merge: Found {duplicates.value_counts().loc[True]} duplicate rows after merging.")
        for dridx in range(duplicates.value_counts().loc[True]):
            mergedDB.dataframe.iloc[ duplicates[duplicates].index[dridx], :]
        mergedDB.dataframe.drop(duplicates[duplicates].index, inplace=True)
        print(f"WARNING :db merge: Duplicate rows have been dropped from the merged database '{mergedDB.dbfilename}'")
    else:
        print(f"db merge: No duplicate rows in the merged database '{mergedDB.dbfilename}'")


    #Check for new nans
    nansAtEnd = 0
    nansAtEnd += (mergedDB.dataframe.isna().sum().sum())

    #nan notify
    if sum(nansAtStart) > 0:
        print(f"WARNING :db merge: There were {sum(nansAtStart)} nans in the databases before merge (per db: {nansAtStart})")
    if sum(nansAtStart) < nansAtEnd:
        print(f"!WARNING! :db merge: The number of nans increased during merge! Before:{sum(nansAtStart)}, After:{nansAtEnd}")
    elif nansAtEnd > 0:
        print(f"WARNING :db merge: There are {nansAtEnd} nans in the merged database.")
    elif nansAtEnd == 0:
        print(f"db merge: No nans in merged database '{mergedDB.dbfilename}'.")


    print(f"db merge: Success :Merge completed, created unsaved database '{mergedDB.dbfilename}'.")

    return mergedDB



#------------------------------------------------------------------------------------------------------------------


def saveDBs(dbs, saveNames=None, returnNewRef=True):
    """
    Save multiple dbs. (exercise dbSaveFile)
    :param dbs: [list of DataBase] to be saved
    :param saveNames: location for db file saving, can be:
        [string] directory path; dbs saved with names dbfilename in given directory
        [string] filePathandDescriptor; dbs saved with names {saveNames}_{dbidx+1}
        [list] save names; list of save names (must be same length as dbs or uses default)
        [default] dbdir; dbs saved with names dbfilename in dbdir
    :param returnNewRef: return the db references for the new database files or return the input db files
    :return: returnedDBs: [list of DataBase] dbs with new or old reference files
    """

    #prep save names
    saveNamesList = []
    if os.path.isdir(saveNames):
        saveStr = f"by dbfilename in '{saveNames}'"
        for dbidx in range(len(dbs)):
            saveNamesList.append(os.path.join(saveNames,dbs[dbidx].dbfilename))

    elif isinstance(saveNames, str):
        saveStr = f"with names '{saveNames}_(dbidx+1)'"
        for dbidx in range(len(dbs)):
            saveNamesList.append(f"{saveNames}_{dbidx+1}")

    elif isinstance(saveNames, list):
        if len(saveNames) == len(dbs):
            saveStr = f"with saveNames, '{saveNames}' , "
            saveNamesList = saveNames
        else:
            print(f"save dbs: WARNING :Length of saveNames does not match dbs. Using default saveNames.")
            saveStr = f"by dbfilename in '{dbdir}'"
            for dbidx in range(len(dbs)):
                saveNamesList.append(os.path.join(dbdir, dbs[dbidx].dbfilename))
    else:
        saveStr = f"by dbfilename in '{dbdir}'"
        for dbidx in range(len(dbs)):
            saveNamesList.append(os.path.join(dbdir, dbs[dbidx].dbfilename))


    #save
    print(f"save dbs: Saving dbs {saveStr}, for dbs in list: {dbs}")
    returnedDBs = len(dbs)*[None]
    for dbidx in range(len(dbs)):
        returnedDBs[dbidx] = dbs[dbidx].dbSaveFile(saveNamesList[dbidx], returnNewRef=returnNewRef)
    print(f"save dbs: Success :Saved dbs {saveStr}, for dbs in list. returnNewRef={returnNewRef}. Return list: {returnedDBs}")
    return returnedDBs



#------------------------------------------------------------------------------------------------------------------


def loadDBs(DBfilespath, excludeFilenames=[], verbose=amverbose):
    """
    Load multiple DBs (exercise dbLoadFile)
    :param DBfilespath: [str, list] a glob path string, a path to a directory or a path to single or multiple files
                                    or, a list of paths (assumes user has pre-prepared the glob)
    :param excludeFilenames: [list] if DBfilespath is a list; this is the list of csv filenames to ignore
    :return:
    """
    #DBfilespath can be a glob string; path to a directory or path to single or multiple files
        #DBfilespath can be a list; of paths; assumes user has pre-prepared the glob
    #if the DBfilespath is a strinng; excludeFilenames is a list of .csv filenames in the glob path which are not DBs and should be ignored
    #returns a list of DBs


    if isinstance(DBfilespath,list):
        if verbose: print(f"load dbs: Loading .csv files from '{DBfilespath}'. ")
        globo = DBfilespath
    else:
        if verbose: print(f"load dbs: Loading .csv files in '{DBfilespath}' glob. Filename exclusion list: {excludeFilenames}")
        globo = glob.glob(DBfilespath)

    if len(globo) == 0:
        print(f"ERROR :load dbs: Glob search path is empty. Returning empty DB list.")
        return []

    glob_csvs = []
    for gidx in range(len(globo)):
        if globo[gidx][-4:] == ".csv" and os.path.split(globo[gidx])[1] is not any(excludeFilenames):
            glob_csvs.append(globo[gidx])

    if len(glob_csvs) == 0:
        print(f"WARNING :load dbs: Glob search path has no csvs. Returning empty DB list.")
        return glob_csvs

    dbs = []
    for dbidx in range(len(glob_csvs)):
        db = DataBase(dbFile=glob_csvs[dbidx])
        dbs.append(db)

    if verbose: print(f"load dbs: Success :Loaded {dbidx+1} DataBases from '{DBfilespath}'.")

    if len(dbs) == 1:
        if verbose: print(f"load dbs: Returning single DB '{dbs[0].dbfilename}'.")
        return dbs[0]
    else:
        if verbose: print(f"load dbs: Returning list of DBs.")
        return dbs



#------------------------------------------------------------------------------------------------------------------


def filterDBs(dbs, filterSettings=mapSettings_default['filter'], fileDescriptor=None):
    """
    Apply performance filter to multiple DBs (exercise dbFilter)
    :param dbs: [list of DBs] DataBases to filter
    :param filterSettings: [dict] settings for the performance filter
                #figure of merit filter
                'filter.fom': 'Jsc_norm_min', #(str) figure of merit column label
                'filter.lo': None, #(float, None) the lower bound cutoff for the figure of merit (expts w fom < val are cut)
                'filter.hi': 0.1, #(float,None) the upper bound cutoff for the fom (expts with fom > val are cut)
                'filter.stage': None, #(str,None) include only expts with 'sim_type' == val
    :param fileDescriptor: [str] descriptor to include in the filename of the filtered dbs
    :return: filteredDBs [list] the filtered DBs, ordered as the input list
    """
    print(f"filter dbs: Filtering with settings, {filterSettings}, for dbs in list: {dbs}")
    filteredDBs = []
    for dbidx in range(len(dbs)):
        if fileDescriptor is not None: dbFileDescriptor = fileDescriptor + f"_{dbs[dbidx].dbfilename}"
        else: dbFileDescriptor = None
        filteredDBs.append(dbs[dbidx].dbFilter(filterSettings, fileDescriptor=dbFileDescriptor))
    print(f"filter dbs: Success :Filtered with settings, {filterSettings}, for dbs in list. Return list: {filteredDBs}")
    return filteredDBs



#------------------------------------------------------------------------------------------------------------------


def cherryPickDBs(dbs, colsToPick, pickwhere, numberResults, fileDescriptor=None):
    """
    Cherry pick multiple DBs (exercise dbCherryPick)
    :param dbs: [list] Dbs from which to cherry pick
    :param colsToPick: [string or list] column name(s) upon which to pick/order by (column datatype should be a number)
    :param pickwhere: [string = "top", "bottom"] to pick from the top (largest) or bottom (smallest)
    :param numberResults: [int] how many results to select
    :param fileDescriptor: [str] decriptor to include in filename of the cherry picked dbs
    :return: pickedDBs: [list] the dbs with teh cherry picked results, ordered as input list
    """

    print(f"pick dbs: Cherry picking {pickwhere} {numberResults} {colsToPick} entries for dbs in list: {dbs}")
    pickedDBs = []
    for dbidx in range(len(dbs)):
        if fileDescriptor is not None: dbFileDescriptor = fileDescriptor + f"_{dbs[dbidx].dbfilename}"
        else: dbFileDescriptor = None
        pickedDBs.append(dbs[dbidx].dbCherryPick(colsToPick, pickwhere, numberResults, fileDescriptor=dbFileDescriptor))
    print(f"pick dbs: Success : Picked {pickwhere} {numberResults} {colsToPick} entries for dbs in list. Return list: {pickedDBs}")
    return pickedDBs



#------------------------------------------------------------------------------------------------------------------


def drCleanDBs(dbs, drCols=None, cleanOption=0, colsToDrop=mapSettings_default['cols']['drop'] + dr_colsToDrop_baseline, fileDescriptor = None):
    """
    Clean, for DR, multiple DBs. (exercise dbDRCleanCols)
    :param dbs:
    :param drCols: [list] the columns to include in the DR, Supercedes colsToDrop and cleanOption!
    :param cleanOption: [int = 0,1] flag for different cleaning/cutting option sets
                0 : remove reference, object, and constant columns, remove columns {colsToDrop}
                1 : remove reference, object, and constant columns
    :param colsToDrop: [list] the names of the cleaned dataframe columns which you would like to drop
            (columns to drop After the dataframe is cleaned according to cleanOption)
    :param fileDescriptor: [string] descriptor string for the filename
    :return: cleanDBs [list] the cleaned DBs
    """

    if drCols is not None:
        cleanStr = f"include only {drCols}"

    else:
        #resolve clean option
        if cleanOption == 0:
            cleanStr = f"remove reference, object, and constant columns, remove columns {colsToDrop}"
        elif cleanOption == 1:
            cleanStr = "remove reference, object, and constant columns"

    print(f"clean dbs for dr: Cleaning '{len(dbs)}' DBs; {cleanStr}")
    cleanedDBs = []
    for dbidx in range(len(dbs)):
        if fileDescriptor is not None: dbFileDescriptor = fileDescriptor + f"_{dbs[dbidx].dbfilename}"
        else: dbFileDescriptor = None
        cleanedDBs.append(dbs[dbidx].dbDRCleanCols(drCols=drCols, cleanOption=cleanOption, colsToDrop=colsToDrop, fileDescriptor=dbFileDescriptor))
    print(f"clean dbs for dr: Success : Cleaned '{len(dbs)}'. Cleaning; {cleanStr}")
    return cleanedDBs



#------------------------------------------------------------------------------------------------------------------


def extractDBGrids(dbs):
    """
    Extract grids from multiple DBs (exercise dbExtractGrid)
    :param dbs: [list] DataBbases from which to extract the grids
    :return: grids [list] the grids (input parameter mesh) extracted from each DB
    """

    print(f"get grids: Extracting grids for dbs in list: {dbs}")
    grids = []
    for dbidx in range(len(dbs)):
        grids.append(dbs[dbidx].dbExtractGrid())
    print(f"get grids: Success : Extracted grids for dbs in list. Return grids: {grids}")
    return grids



#------------------------------------------------------------------------------------------------------------------


def checkDBGridness(db, verbose=amverbose):
    """
    Check whether the DB is a grid or not (contains only input (and reference) columns)
        (Checks;  any outputStartStrs present? >> if so, outputs are present and it is not a grid)
    :param db: [DataBase] DB to check
    :return: grid [bool] is the db a grid?
    """

    if verbose: print(f"check gridness: Checking if db '{db.dbfilename}' is a grid.")
    idxCol, ssidx = findDFColIdx(db.dataframe, outputStartStrs)
    grid = False
    if idxCol is None:
        if verbose: print(f"check gridness: DB '{db.dbfilename}' is a GRID (contains inputs only).")
        grid = True
    else:
        if verbose: print(f"check gridness: DB '{db.dbfilename}' is not a grid (contains inputs and outputs).")
    return grid



#------------------------------------------------------------------------------------------------------------------


def cleanDFIdxs(dataframe, verbose=amverbose):
    """
    Resets the index column numbering for the DB's dataframe.
    Determines whether the database is a grid or not
        (if contains an outputStartStrs column then it contains outputs and is not a grid)
    Needed after most operations.
    :param: db: [DataBase] database to clean indexs and detect grid
    :return: db: [DataBase] the database with edited dataframe and grid attributes
    """

    if verbose: print(f"clean df idxs: Cleaning idxs of dataframe.")

    # New index data
    nIncrData = np.array(range(1, dataframe.shape[0] + 1, 1))

    # find indexes and replace data
    indexFound = False
    # # db_idx
    dataframe.reset_index(inplace=True, drop=True)
    # idxCol, _ = findDFColIdx(dataframe, 'db_idx')
    # if idxCol is None:
    #     if amverbose: print(f"clean df idxs: No db_idx column in dataframe.")
    # else:
    #     dataframe['db_idx'] = nIncrData
    #     indexFound = True

    # inputStartStrs
    idxCol, ssidx = findDFColIdx(dataframe, inputStartStrs)
    if idxCol is None:
        if verbose: print(f"clean df idxs: No inputStartStrs column in dataframe.")
    else:
        # dataframe[inputStartStrs[ssidx]] = nIncrData
        dataframe.loc[:,f'{inputStartStrs[ssidx]}'] = nIncrData
        indexFound = True

    # outputStartStrs
    idxCol, ssidx = findDFColIdx(dataframe, outputStartStrs)
    if idxCol is None:
        if verbose: print(f"clean df idxs: No outputStartStrs column in dataframe.")
    else:
        # dataframe[outputStartStrs[ssidx]] = nIncrData
        dataframe.loc[:,f'{outputStartStrs[ssidx]}'] = nIncrData
        indexFound = True

    if not indexFound:
        print(f"WARNING :clean df idxs: No index columns in dataframe.")
        return dataframe



    if verbose: print(f"clean df idxs: Successfully cleaned dataframe index columns.")

    return dataframe



#------------------------------------------------------------------------------------------------------------------


def splitDBDataframeIO(inputDB):
    """
    Split the DB's dataframe into the input&reference columns and the output columns
    :param inputDB: [DataBase] to split
    :return: inDF: [pd.dataframe] the inputs and reference columns of inputDB.dataframe
    :return: outDF: [pd.dataframe] the outputs columns of inputDB.dataframe
    """

    outColIdx, _ = findDFColIdx(inputDB.dataframe, outputStartStrs)
    inDF = inputDB.dataframe.iloc[:, :outColIdx]
    outDF = inputDB.dataframe.iloc[:, outColIdx:]

    return inDF, outDF



#------------------------------------------------------------------------------------------------------------------


def findDFColIdx(dataframe, strsToSearch):
    """
    Tries to find a column labelled with one of the strsToSearch. Returns the index of the first one found.
    eg usage: set strsToSearch= in/outputSearchStrs to locate model inputs and outputs
    :param dataframe: the dataframe in which to find the labelled column
    :return: colIndex: (int) the index of the found col (None if not found)
    """
    if isinstance(strsToSearch, str):
        strsToSearch=[strsToSearch] #single col label to search for, convert to list for search

    colIndex = None
    for ssidx in range(len(strsToSearch)):
        try:
            colIndex = dataframe.columns.get_loc(strsToSearch[ssidx])
        except KeyError:
            if ssidx == len(strsToSearch):
                print(f"WARNING :find df col idx: Could not find any columns with any of the names {strsToSearch}.")
                colIndex = None
        else:
            break

    return colIndex, ssidx






# ------------------------------------------------------------------------------------------------------------------
# ------------------------------------------------------------------------------------------------------------------
#!    FUNCTIONS FOR CREATING CUSTOM EXPERIMENTS AND DATABASES
# ------------------------------------------------------------------------------------------------------------------
# ------------------------------------------------------------------------------------------------------------------


# ------------------------------------------------------------------------------------------------------------------

def create_randomPts_grid(numExpts, setCols, setBoundsVals, outGridPath=None, saveGrid=False):
    """
    Create a grid DB with given columns either set to a specific value or randomized within bounds (with uniform probability).
    :param numExpts: int, Number of experiments to include in the DB
    :param setCols: list of strings, The parameter labels that are to be set
    :param setBoundsVals: list of values and/or lists, The set values or bounds to randomize within, matched element-wise with the setCols
                                if element is a list, => bounds to randomize within (should be a list of length 2)
                                if element is not a list, => a set value for a constant data column
    :param outGridPath: string, path to the location to save the output DB
    :param saveGrid: boolean, whether or not to save the DB (True) or just return it (False)
    :return: outGrid: DataBase, a grid database containing (only) the set columns
    """

    #check inputs.
    if len(setCols) != len(setBoundsVals):
        print(f"ERROR :create random pt grid/gtree: Length of the columns to set and supplied boundaries/values do not match. Cannot create grid/gtree. Exiting...")
        return None
    if outGridPath != None:
        outGridPath = os.path.abspath(outGridPath)

    print(f"create random pt grid: Creating random pt grid with '{numExpts}' expts and '{len(setCols)}' columns.")

    #create data
    setDataDict = {'experiment-inputs': list(range(1,numExpts+1))} #dict for the filled data
    for cidx in range(len(setCols)):
        if not isinstance(setBoundsVals[cidx], list):
            #if setBoundsVals is a single object / not a list then create a dict entry with constant data
            setDataDict[setCols[cidx]] = [setBoundsVals[cidx]] * numExpts
            print(f"create random pt grid: Constant data column '{setCols[cidx]}' created (val='{setBoundsVals[cidx]}').")
        else:
            #else setBoundsVals is boundaries, create a dict entry with random data between the bounds
            setDataDict[setCols[cidx]] = create_randomPts_data(numPts=numExpts, boundaries=setBoundsVals[cidx])
            print(f"create random pt grid: Random data column '{setCols[cidx]}' created (boundaries={setBoundsVals[cidx]}).")

    #create dataframe
    df = pd.DataFrame(setDataDict)
    df.index.name = 'db_idx'

    #create grid database
    outGrid = DataBase()
    outGrid.grid = True
    outGrid.lineage.append([time.strftime(timeStr),"Created using create_randomPts_grid"])
    outGrid.dataframe = df
    outGrid.dbCleanIdxsAndGridness()
    if outGridPath != None:
        if saveGrid:
            outGrid.dbSaveFile(saveName=outGridPath)
        else:
            outGrid.dbFile = outGridPath
            outGrid.dbfilename = os.path.split(outGridPath)[1]
            print(f"create random pt grid: Returning UNSAVED database...")
    else:
        if saveGrid:
            print(f"WARNING :create random pt grid: Cannot save grid without a outGridPath. Not saving, returning UNSAVED grid...")

    print(f"create random pt grid: Success :Random pt grid created.")
    return outGrid


# ------------------------------------------------------------------------------------------------------------------

def create_randomPts_data(numPts, boundaries):
    """
    Create random data points uniformly distributed between the lower and upper boundary (inclusive).
    :param numPts: int, number of random points to create
    :param boundaries: list of floats length 2, the lower and upper bounds over which to create random data pts
    :return: outList, list, the random points
    """

    #boundaries check
    if not isinstance(boundaries, list):
        if len(boundaries) != 2:
            print(f"ERROR :create random pt data: Boundaries input is not a list of length two. Exiting...")
    else:
        if boundaries[0] > boundaries[1]:
            boundaries.reverse()
        elif boundaries[0] == boundaries[1]:
            print(f"WARNING :create random pt data: Upper and lower bound are identical. Returning constant data (value={boundaries[0]}).")
            return [boundaries[0]]*numPts

    #create random data points list
    outList = []
    for n in range(numPts):
        outList.append(random.uniform(0,1)*(boundaries[1]-boundaries[0]) + boundaries[0])

    # print(f"create random pt data: Success : {numPts} random data points between {boundaries[0]} and {boundaries[1]} created.")

    return outList


 # ------------------------------------------------------------------------------------------------------------------


def extend_gridForNewColVals(inputGrid, col, vals, fullSetsForAllVals=True, separateGridsForEachVal=False):
    """
    Extend a grid DB dataframe to include the full set of experiments for each value of column <col> given in <vals>

    Will create a single grid with all expts with all values of col if separateGridsForEachVal is False (ie. an actual extension)
    Otherwise, creates a grid for each value of col  ... ie. extends into separate grids

    Handles different cases of whether col or vals are present in the input dataframe. See code for more info.
        eg. The input dataframe does not need to include the column col. It will be added in this case.

        eg2. If there are values of col in the input dataframe but not in the supplied list;
            If fullSetsForAllVals is True: Creates full sets of experiments for the union of the supplied list and existent values
            Else: Creates full sets for all values in supplied list and retains the partial expt sets which are existent in the input DB for the other vals

    :param inputGrid: DataBase, The input grid database to extend
    :param col: string, Label of the column which will serve as the basis for the extension
    :param vals: value or list of values, The value(s) of col to extend the grid to
    :param fullSetsForAllVals: boolean, See above.
    :param separateGridsForEachVal: boolean, See above.
    :return: outGrids: DataBase or list of DataBases, the extended or separated grid DBs
    :return: valKeys: string/value or list of values, the value of col in each of the corresponding outGrids ("all" if separateGridsForEachVal==False)
    """

    print(f"extend grid for new col vals: Extending grid '{inputGrid.dbfilename}' for column '{col}' values '{vals}'. (fullSetsForAllVals={fullSetsForAllVals} separateGridsForEachVal={separateGridsForEachVal})")

    #check inputs
    try:
        col = str(col)
    except ValueError:
        print(f"ERROR :extend grid for new col vals: Input col should be the column label but it cannot be converted to a string. Returning input grid...")
        return inputGrid, "input"
    if not isinstance(vals,list): #just a single value
        vals = [vals]

    inDF = inputGrid.dataframe
    outDFs = [] #list of the output dataframes each with different values for col


    #create the outDFs;
    # handle each case; does the col exist already, if so unique or multiple vals, do vals overlap with supplied ones
    if col not in inDF.columns:
        #column does not exist in dataframe
        print(f"extend grid for new col vals: Column '{col}' does not exist in the input grid. Creating new column...")
        #create a copy of the inDF for each value and add the col with the relevant val
        for vidx in range(len(vals)):
            outDF = inDF.copy(deep=True)
            outDF[col] = vals[vidx]
            outDFs.append(outDF)
    else:
        #column exists in dataframe already
        if len(inDF[col].unique()) == 1:
            #column exists with only one value
            for vidx in range(len(vals)):
                if vals[vidx] == inDF[col].unique()[0]:
                    #the val is already in the df, no need to create copy
                    outDFs.append(inDF)
                else:
                    #the val is a new one, create a copy overwrite val
                    outDF = inDF.copy(deep=True)
                    outDF[col] = vals[vidx]
                    outDFs.append(outDF)
        else:
            #column exists in input with multiple values
            dfColVals = inDF[col].unique().tolist()
            valsSet = set(vals)
            dfColValsSet = set(dfColVals) #must do this silly construction for the compiler
            if (dfColValsSet == valsSet):
                #the input DF has some expts with one of the vals and some with another, etc...
                for vidx in range(len(vals)):
                    #create outDFs with full sets for each val
                    outDF = inDF.drop(labels=col, axis=1).copy(deep=True)
                    outDF[col] = vals[vidx]
                    outDFs.append(outDF)
            else:
                #there exists values of the column which are different from any value supplied
                #Or, there are supplied values which are not in the dataframe
                if fullSetsForAllVals:
                    #create outDFs with full sets for each val in the union of the supplied and existent val sets
                    vals = list(dfColValsSet.union(valsSet))
                    for vidx in range(len(vals)):
                        # create outDFs with full sets for each val
                        outDF = inDF.drop(labels=col, axis=1).copy(deep=True)
                        outDF[col] = vals[vidx]
                        outDFs.append(outDF)
                else:
                    #create outDFs with full sets for supplied vals and retain partial sets for vals in the dataframe and not in the supplied
                    partialSetVals = list(dfColValsSet.difference(valsSet))
                    if len(partialSetVals) != 0:
                        #there are partials to retain
                        for vidx in range(len(partialSetVals)):
                            outDFs.append(inDF[inDF[col] == partialSetVals[vidx]])
                    #full sets for all supplied vals
                    for vidx in range(len(vals)):
                        outDF = inDF.drop(labels=col, axis=1).copy(deep=True)
                        outDF[col] = vals[vidx]
                        outDFs.append(outDF)

    print(f"extend grid for new col vals: Extended dataframes created. Now creating grid(s)...")

    #create grid DB(s)
    outGrids = [] #list of output grids
    valKeys = [] #list of the val in each of the output grids
    outDFShapes = []
    if separateGridsForEachVal:
        #create a grid for each outDF
        for gidx in range(len(outDFs)):
            valKeys.append(outDFs[gidx][col][0])
            outGrid = DataBase()
            outGrid.grid=True
            outGrid.dataframe = outDFs[gidx]
            outGrid.dbFile = inputGrid.dbFile[:-4]+f"_{col}-{valKeys[-1]}"
            outGrid.dbfilename = os.path.split(outGrid.dbFile)[1]
            outGrid.lineage = inputGrid.lineage
            outGrid.lineage.append([time.strftime(timeStr),f"Extended using extend_gridForNewColVal - separated grid with {col} value of {valKeys[-1]}"])
            outGrid.dbCleanIdxsAndGridness()
            outGrids.append(outGrid)
            outDFShapes.append(outDFs[gidx].shape)
    else:
        #create a single grid with all data (ie. extend the input grid)
        fullDF = pd.concat(outDFs, axis=0, ignore_index=True)
        outGrid = DataBase()
        outGrid.grid = True
        outGrid.dataframe = fullDF
        outGrid.dbFile = inputGrid.dbFile
        outGrid.dbfilename = inputGrid.dbfilename
        outGrid.lineage = inputGrid.lineage
        outGrid.lineage.append([time.strftime(timeStr),f"Extended using extend_gridForNewColVal - all values for column '{col} included'"])
        outGrid.dbCleanIdxsAndGridness()
        outGrids.append(outGrid)
        valKeys.append("all")
        outDFShapes.append(fullDF.shape)

    print(f"extend grid for new col vals: Success :Created '{len(outGrids)}' output grid(s), with value {valKeys} for column '{col}', respectively.")
    print(f"extend grid for new col vals: Shape of input grid dataframe = '{inDF.shape}', output grid dataframe shape(s) = {outDFShapes}")

    #don't return a list if just one grid is output
    if len(outGrids) == 1:
        outGrids = outGrids[0]
        valKeys = valKeys[0]

    return outGrids, valKeys


 # ------------------------------------------------------------------------------------------------------------------

def sensitivityAnalysis_varySeparately_forAll(inDB, colsToVary=['spectrum'], baselineVals=None, variations=[20,40], variationType='value', variationDirection='both',  retainAllInputs=True, save=True, saveLoc=None, saveName=None):
    """
    Create a grid where each of the colsToVary is varied separately according to the given variation
    Does this for each pt in the inputDB

     If retainAllInputs==False; Output includes only the varied columns and a column nom_exptidx which links each row to its nominal (unvaried) row
            Else; Output includes all input columns in the inputDB plus the nom_exptidx column. The non-colsToVary values are obtained from the nominal expt in the inputDB
    Unvaried expt rows are included in the db. They are first.
    If the nom_exptIdx column exists then it's values are inherited by varied rows from the unvaried rows.

    Separately, for all = Each unvaried row recieves a set of varied rows where each column in colsToVary is varied by each variation
    ie. for each input row there will be len(colsToVary) * len(variations) [* 2 if variationDirection=both] varied rows

    This is essentially a wrapper for dbh.extend_gridForNewColVals which handles the typical sensitivity analysis variation input format and the nom_exptIdx column

    :param inDB: [DataBase] the DataBase with the pts to run the sensitivity analysis on
    :param colsToVary: [str or list of strings] the label(s) for the columns to vary
    :param baselineVals: [float or list of float or None] the baseline value for each column, if None then the value from the first expt in the inputDB is used
    :param variations: [list of float] the variations to apply
    :param variationType: [one of 'fraction', 'f', 'value', 'v'] the type of variation
                                            fraction => varVals = nomVals * (1 +- var)   (ie. fractions are with respect to 1 (0.05 = 105%, -0.05 = 95%))
                                            value => varVals = nomVals +- var (value should be correct units, eg. should be supplied in um if colsToVary are the th cols)
    :param variationDirection: [one of 'b(oth)', 'i(ncrease)', 'd(ecrease)', 'u(p)', 'd(own)'] which direction to vary in
    :param retainAllInputs: [bool] (see above) Retain the inputDB's input values/columns?
    :param save: [bool] save the grid db? (Or just return it, if False)
    :param saveLoc: [str or None] the folder to save the grid in, if None uses the folder of the inputDB
    :param saveName: [str or None] the filename to save the grid, if None uses <inputDB.dbfilename>_unifSepSensAna.csv
    :return: griddb [DataBase] the generated grid db (no output values are retained)
                                expts are in order of; decreases (if present) then increases (if present) both in order of variations
    """

    inputDB = inDB.dbCopy()

    # make input checks
    if not isinstance(colsToVary, list):
        colsToVary = [colsToVary]

    if baselineVals is None:
        baselineVals = list(inputDB.dataframe.loc[0, colsToVary].values)

    if not isinstance(baselineVals, list):
        baselineVals = [baselineVals]
    if not len(baselineVals) == len(colsToVary):
        print(f"ERROR :sensitivity analysis vary separately: Length of baselineVals ('{len(baselineVals)}') does not match length of colsToVary ('{len(colsToVary)}'). Using baselineVals obtained from the first expt in the inputDB...")
        baselineVals = list(inputDB.dataframe.loc[0, colsToVary].values)

    if not isinstance(variations, list):
        variations = [variations]

    if variationType.lower() not in ['fraction', 'f', 'value', 'v']:
        if variations[-1] < 1:
            print(
                f"ERROR :sensitivity analysis vary separately: variationType '{variationType}' is not recognized. It must be one of 'f(raction)' or 'v(alue)'. Using fraction...")
            variationType = 'fraction'
        else:
            print(
                f"ERROR :sensitivity analysis vary separately: variationType '{variationType}' is not recognized. It must be one of 'f(raction)' or 'v(alue)'. Using value...")
            variationType = 'value'

    if variationDirection.lower() not in ['increase', 'i', 'decrease', 'd', 'up', 'u', 'down', 'both', 'b']:
        print(
            f"ERROR :sensitivity analysis vary separately: variationDirection '{variationDirection}' is not recognized. It must be one of ['increase', 'i', 'decrease', 'd', 'up', 'u', 'down', 'd', 'both', 'b']. Setting to both...")
        variationDirection = 'both'

    print(f"sensitivity analysis vary separately: Creating sensitivity analysis grid by varying columns '{colsToVary}' using '{variationType}' variations of '{variations}' on baseline values of '{baselineVals}' in '{variationDirection}' direction.")


    #prep grid for extension, add nom_exptIdx if not present

    inputgrid = inputDB.dbExtractGrid()
    griddb = inputgrid.dbCopy()

    try: x = griddb.dataframe['nom_exptIdx']
    except KeyError:
        griddb.dataframe = cleanDFIdxs(griddb.dataframe)
        griddb.dataframe['nom_exptIdx'] = griddb.dataframe.index


    #run extension for each column
    for cidx in range(len(colsToVary)):

        #get variation vals
        varVals = [baselineVals[cidx]]

        #decreased variations
        if variationDirection.lower().startswith('b') or variationDirection.lower().startswith('d'):
            if variationType.lower().startswith('f'): #fractional
                for var in variations:
                    varVals.append(baselineVals[cidx]*(1-var))
            else:
                for var in variations: # value
                    varVals.append(baselineVals[cidx]-var)
        #increased variations
        if variationDirection.lower().startswith('b') or variationDirection.lower().startswith('i') or variationDirection.lower().startswith('u'):
            if variationType.lower().startswith('f'):  # fractional
                for var in variations:
                    varVals.append(baselineVals[cidx] * (1 + var))
            else:
                for var in variations:  # value
                    varVals.append(baselineVals[cidx] + var)

        #run extension
        griddb,_ = extend_gridForNewColVals(griddb, colsToVary[cidx], varVals, fullSetsForAllVals=False, separateGridsForEachVal=False)


    #truncate if retainAllInputs is False
    if not retainAllInputs:
        griddb.dataframe = griddb.dataframe[colsToVary+['nom_exptIdx']]

    #create the output grid db
    griddb.grid = True
    griddb.dbSaveFile
    griddb.dbFile = inDB.dbFile
    griddb.dbfilename = inDB.dbfilename
    griddb.lineage.append([time.strftime(timeStr), "Created from vary separately sensitivity analysis grid creator"])
    if save:
        if saveName is not None:
            griddb.dbfilename = saveName
        else:
            griddb.dbfilename = inputDB.dbfilename[:-4]+"_unifSepSensAna.csv"
        if saveLoc is not None:
            griddb.dbFile = os.path.join(os.path.abspath(saveLoc), griddb.dbfilename)
        else:
            griddb.dbFile = os.path.join(os.path.abspath(os.path.split(inputDB.dbFile)[0]), griddb.dbfilename)
        griddb.dbSaveFile(saveName=griddb.dbFile)

    print(f"sensitivity analysis vary separately: Success :Created new grid db (input dataframe shape: '{inputDB.dataframe.shape}', input dataframe inputs shape: '{inputgrid.dataframe.shape}', output grid shape: '{griddb.dataframe.shape}')")

    return griddb


 # ------------------------------------------------------------------------------------------------------------------



def sensitivityAnalysis_varyMultipleUniformly(inDB, colsToVary, retainAllInputs=True, variations=[0.05,0.1], variationType='fraction', variationDirection='both', save=True, saveLoc=None, saveName=None):
    """
    Create a grid db where the values for the specified columns are varied together, uniformly, by the given variation (creates as many varied sets as there are variations)
    Does this for each pt in the input DB
    
    Uniformly together = all columns are varied by the same amount at the same time

    If retainAllInputs==False; Output includes only the varied columns and a column nom_exptidx which links each row to its nominal (unvaried) row
        Else; Output includes all input columns in the inputDB plus the nom_exptidx column. The non-colsToVary values are obtained from the nominal expt in the inputDB
    If the nom_exptIdx column exists then it's values are inherited by varied rows from the unvaried rows.
    
    Output DB contains both the unvaried and varied sets;
        Unvaried expt rows are first.
        Then the varied sets; in order of decreases (if present) then increases (if present) both in order of variations
        Sets are included sequentially, each in order of nom_exptIdx


    :param inDB: [DataBase] the DataBase with the pts to run the sensitivity analysis on
    :param colsToVary: [list of strings] the labels for the columns to vary together
    :param retainAllInputs: [bool] (see above) Retain the inputDB's input values/columns?
    :param variations: [list of float] the variations to apply
    :param variationType: [one of 'fraction', 'f', 'value', 'v'] the type of variation
                                            fraction => varVals = nomVals * (1 +- var)   (ie. fractions are with respect to 1 (0.05 = 105%, -0.05 = 95%))
                                            value => varVals = nomVals +- var (value should be correct units, eg. should be supplied in um if colsToVary are the th cols)
    :param variationDirection: [one of 'b(oth)', 'i(ncrease)', 'd(ecrease)', 'u(p)', 'd(own)'] which direction to vary in
    :param save: [bool] save the grid db? (Or just return it, if False)
    :param saveLoc: [str or None] the folder to save the grid in, if None uses the folder of the inputDB
    :param saveName: [str or None] the filename to save the grid, if None uses <inputDB.dbfilename>_unifMultiSensAna.csv
    :return: griddb [DataBase] the generated grid db (no output values are retained), ordered as outlined above
    """
        
    # NOTE; the choice of output db ordering is consistent with the other sensitivity analysis functions allowing easier analysis of the surveyed grid, but is not necessarily the easiest to compute

    #(For aiirpower: colsToVary default =  arcThLabelsTtoB+absThLabelsTtoB_bottomHomo)


    #make input checks
    if not isinstance(variations, list):
        variations = [variations]

    if variationType.lower() not in ['fraction', 'f', 'value', 'v']:
        if variations[-1] < 1:
            print(f"ERROR :sensitivity analysis vary together uniformly: variationType '{variationType}' is not recognized. It must be one of 'f(raction)' or 'v(alue)'. Using fraction...")
            variationType='fraction'
        else:
            print(f"ERROR :sensitivity analysis vary together uniformly: variationType '{variationType}' is not recognized. It must be one of 'f(raction)' or 'v(alue)'. Using value...")
            variationType = 'value'

    if variationDirection.lower() not in ['increase', 'i', 'decrease', 'd', 'up', 'u', 'down', 'both', 'b']:
        print(f"ERROR :sensitivity analysis vary together uniformly: variationDirection '{variationDirection}' is not recognized. It must be one of ['increase', 'i', 'decrease', 'd', 'up', 'u', 'down', 'd', 'both', 'b']. Setting to both...")
        variationDirection = 'both'


    print(f"sensitivity analysis vary together uniformly: Creating sensitivity analysis grid by varying columns '{colsToVary}' together using '{variationType}' variations of '{variations}' in '{variationDirection}' direction.")


    #get df, reset index, and create the nom_exptIdx column
    inputDB = inDB.dbCopy()
    inputgrid = inDB.dbExtractGrid()
    griddb = inputgrid.dbCopy()

    try: x = griddb.dataframe['nom_exptIdx']
    except KeyError: nomExists = False
    else: nomExists = True

    if retainAllInputs:
        df = griddb.dataframe
        if not nomExists:
            df = cleanDFIdxs(df)
            df['nom_exptIdx'] = df.index
    else:
        df = pd.DataFrame(inputDB.dataframe[colsToVary])
        if not nomExists:
            df.reset_index(inplace=True, drop=True)
            df['nom_exptIdx'] = df.index


    #compute variation sets with decreases
    if variationDirection.lower().startswith('b') or variationDirection.lower().startswith('d'):
            if variationType.lower().startswith('f'):  # fractional
                for var in variations: # for each decrease variation run through nominal points and create the set
                    for ptidx in range(len(df.nom_exptIdx.unique())):
                        nomVals = df.loc[ptidx, colsToVary].values
                        varVals = nomVals * (1 - var)

                        if nomExists: nomExpt = df.loc[ptidx, 'nom_exptIdx'] #inherit nom expt idx if it exists
                        else: nomExpt = ptidx
                        
                        if retainAllInputs:
                            nomAllInputRowDF = pd.DataFrame(df.iloc[ptidx, :])
                            nomAllInputRowDF = nomAllInputRowDF.transpose().reset_index(drop=True)
                            nomAllInputRowDF.loc[0, colsToVary] = varVals
                            nomAllInputRowDF.loc[0, 'nom_exptIdx'] = nomExpt
                            df = pd.concat([df,nomAllInputRowDF], axis=0, ignore_index=True)
                        else:
                            df.loc[len(df.index)] = list(varVals) + [nomExpt]

            else:  # value
                for var in variations:
                    for ptidx in range(len(df.nom_exptIdx.unique())):
                        nomVals = df.loc[ptidx, colsToVary].values
                        varVals = nomVals - var

                        if nomExists: nomExpt = df.loc[ptidx, 'nom_exptIdx']
                        else: nomExpt = ptidx
                        
                        if retainAllInputs:
                            nomAllInputRowDF = pd.DataFrame(df.iloc[ptidx, :])
                            nomAllInputRowDF = nomAllInputRowDF.transpose().reset_index(drop=True)
                            nomAllInputRowDF.loc[0, colsToVary] = varVals
                            nomAllInputRowDF.loc[0, 'nom_exptIdx'] = nomExpt
                            df = pd.concat([df,nomAllInputRowDF], axis=0, ignore_index=True)
                        else:
                            df.loc[len(df.index)] = list(varVals) + [nomExpt]
                        

    #compute variation sets with increases
    if variationDirection.lower().startswith('b') or variationDirection.lower().startswith('i') or variationDirection.lower().startswith('u'):
            if variationType.lower().startswith('f'):  # fractional
                for var in variations: # for each decrease variation run through nominal points and create the set
                    for ptidx in range(len(df.nom_exptIdx.unique())):
                        nomVals = df.loc[ptidx, colsToVary].values
                        varVals = nomVals * (1 + var)

                        if nomExists: nomExpt = df.loc[ptidx, 'nom_exptIdx'] #inherit nom expt idx if it exists
                        else: nomExpt = ptidx
                        
                        if retainAllInputs:
                            nomAllInputRowDF = pd.DataFrame(df.iloc[ptidx, :])
                            nomAllInputRowDF = nomAllInputRowDF.transpose().reset_index(drop=True)
                            nomAllInputRowDF.loc[0, colsToVary] = varVals
                            nomAllInputRowDF.loc[0, 'nom_exptIdx'] = nomExpt
                            df = pd.concat([df,nomAllInputRowDF], axis=0, ignore_index=True)
                        else:
                            df.loc[len(df.index)] = list(varVals) + [nomExpt]

            else:  # value
                for var in variations:
                    for ptidx in range(len(df.nom_exptIdx.unique())):
                        nomVals = df.loc[ptidx, colsToVary].values
                        varVals = nomVals + var

                        if nomExists: nomExpt = df.loc[ptidx, 'nom_exptIdx']
                        else: nomExpt = ptidx
                        
                        if retainAllInputs:
                            nomAllInputRowDF = pd.DataFrame(df.iloc[ptidx, :])
                            nomAllInputRowDF = nomAllInputRowDF.transpose().reset_index(drop=True)
                            nomAllInputRowDF.loc[0, colsToVary] = varVals
                            nomAllInputRowDF.loc[0, 'nom_exptIdx'] = nomExpt
                            df = pd.concat([df,nomAllInputRowDF], axis=0, ignore_index=True)
                        else:
                            df.loc[len(df.index)] = list(varVals) + [nomExpt]                       
                        
                        
    #create the output grid db
    griddb = DataBase()
    griddb.dbFile = inDB.dbFile
    griddb.dbfilename = inDB.dbfilename
    griddb.grid = True
    griddb.lineage = [[time.strftime(timeStr), "Created from uniform variation together sensitivity analysis grid creator"]]
    griddb.dataframe = df
    if save:
        if saveName is not None:
            griddb.dbfilename = saveName
        else:
            griddb.dbfilename = inputDB.dbfilename[:-4]+"_unifMultiSensAna.csv"
        if saveLoc is not None:
            griddb.dbFile = os.path.join(os.path.abspath(saveLoc), griddb.dbfilename)
        else:
            griddb.dbFile = os.path.join(os.path.abspath(os.path.split(inputDB.dbFile)[0]), griddb.dbfilename)
        griddb.dbSaveFile(saveName=griddb.dbFile)



    print(f"sensitivity analysis vary together uniformly: Success :Created new grid db (input dataframe shape: '{inputDB.dataframe.shape}', input dataframe inputs shape: '{inputgrid.dataframe.shape}', output grid shape: '{griddb.dataframe.shape}')")

    return griddb



 # ------------------------------------------------------------------------------------------------------------------


def sensitivityAnalysis_varyOneSetMultipleColsLinearly(inDB, colsToVaryOrdered, retainAllInputs=True, variationBounds=[0,0.05], variationType='fraction', loLocation='last',  save=True, saveLoc=None, saveName=None):
    """
    Create a grid db where the values for the specified columns are varied according to a linear trend (creates one varied set)
        The linear trend is defined by the variationBounds [min/lo, max/hi variation] and the loLocation (where the lo bound is located in the list of cols to vary)
        loLocation == last or first; the variation is split evenly for the number of columns, starts at the given location and increases in the same order as the given col list
    Does this for each pt in the inputDB

    If retainAllInputs==False; Output includes only the varied columns and a column nom_exptidx which links each row to its nominal (unvaried) row
        Else; Output includes all input columns in the inputDB plus the nom_exptidx column. The non-colsToVary values are obtained from the nominal expt in the inputDB
    If the nom_exptIdx column exists then it's values are inherited by varied rows from the unvaried rows.

    Output db contains both the unvaried (inputs) and the (single) varied set; 
        Unvaried expt rows are first.

    Note: lo and hi are just names, there is no need for lo < hi

    :param inDB: [DataBase] the DataBase with the pts to run the sensitivity analysis on
    :param colsToVaryOrdered: [list of strings] the labels for the columns to vary with the linear trend (ordered, see above)
    :param retainAllInputs: [bool] (see above) Retain the inputDB's input values/columns?
    :param variationBounds: [list of 2 floats] The [lo,hi] bounds for the linear variation
    :param variationType: [str, one of 'f(raction)' or 'v(alue)'] the type of variation to apply (fractions are with respect to 1 ie. 0.05 = 105%, -0.05 = 95%)
    :param loLocation: [str, one of 'f(irst)' or 'l(ast)'] where the first supplied var bound is located in the cols list (the other is automatically placed at the other end)
    :param save: [bool] save the grid db? (Or just return it, if False)
    :param saveLoc: [str or None] the folder to save the grid in, if None uses the folder of the inputDB
    :param saveName: [str or None] the filename to save the grid, if None uses <inputDB.dbfilename>_linearSensAna.csv
    :return: griddb [DataBase] the generated grid db (no output values are retained), ordered as unvaried expts and then the varied expts both in nom_exptidx order
    """

    #(for aiirpower colsToVaryOrdered default = absThLabelsTtoB_bottomHomo)

    #make input checks
    if not isinstance(variationBounds, list):
        print(f"ERROR :sensitivity analysis vary linearly: The supplied variationBounds ('{variationBounds}') are not recognized. They must be a list of length two. Returning None...")
        return None
    else:
        if len(variationBounds) != 2:
            print(f"ERROR :sensitivity analysis vary linearly: The supplied variationBounds ('{variationBounds}') are not recognized. They must be a list of length two. Returning None...")
            return None

    if variationType.lower() not in ['fraction', 'f', 'value', 'v']:
        if variationBounds[-1] < 1:
            print(f"ERROR :sensitivity analysis vary linearly: variationType '{variationType}' is not recognized. It must be one of 'f(raction)' or 'v(alue)'. Using fraction...")
            variationType='fraction'
        else:
            print(f"ERROR :sensitivity analysis vary linearly: variationType '{variationType}' is not recognized. It must be one of 'f(raction)' or 'v(alue)'. Using value...")
            variationType = 'value'

    if loLocation not in ['f', 'first', 'l', 'last']:
        print(f"ERROR :sensitivity analysis vary linearly: loLocation '{loLocation}' is not recognized. It must be one of 'f(irst)' or 'l(ast)'. Using last...")
        loLocation = 'last'


    print(f"sensitivity analysis vary linearly: Creating sensitivity analysis grid by varying columns '{colsToVaryOrdered}' according to linear trend with bounds '{variationBounds}' ('{variationType}') with the first bound located '{loLocation}' in the col list.")


    #get df, reset indices, and create/check nom_exptIdx column
    inputDB = inDB.dbCopy() #get deep copy
    inputgrid = inDB.dbExtractGrid()
    griddb = inputgrid.dbCopy()

    try: x = griddb.dataframe['nom_exptIdx']
    except KeyError: nomExists = False
    else: nomExists = True

    if retainAllInputs:
        df = griddb.dataframe
        if not nomExists:
            df = cleanDFIdxs(df)
            df['nom_exptIdx'] = df.index
    else:
        df = pd.DataFrame(inputDB.dataframe[colsToVaryOrdered])
        if not nomExists:
            df.reset_index(inplace=True, drop=True)
            df['nom_exptIdx'] = df.index


    #determine linear trend steps
    steps = np.linspace(variationBounds[0], variationBounds[1], num=len(colsToVaryOrdered))

    # run thru the nominal points and create the variations, adding to the df as you go
    for ptidx in range(df.shape[0]):
        colVals = df.loc[ptidx, colsToVaryOrdered].values
        if loLocation == 'last': #the end of the list where the first bound is located is put first
            colVals = np.flip(colVals)

        if nomExists: nomExpt = df.loc[ptidx, 'nom_exptIdx'] #inherit nom expt idx if it exists
        else: nomExpt = ptidx

        if retainAllInputs:
            nomAllInputRowDF = pd.DataFrame(df.iloc[ptidx, :])
            nomAllInputRowDF = nomAllInputRowDF.transpose().reset_index(drop=True)

        #create varied col values
        for cidx in range(len(colsToVaryOrdered)):
            if variationType.lower().startswith('f'): #fractional
                colVals[cidx] = colVals[cidx] * (1 + steps[cidx])
            else: #value
                colVals[cidx] = colVals[cidx] + steps[cidx]

        #save to df
        if loLocation == 'last': #flip back
            colVals = np.flip(colVals)

        if retainAllInputs:
            nomAllInputRowDF.loc[0, colsToVaryOrdered] = colVals
            nomAllInputRowDF.loc[0, 'nom_exptIdx'] = nomExpt
            # df = df.append(nomAllInputRowDF, ignore_index=True)
            df = pd.concat([df, nomAllInputRowDF], axis=0, ignore_index=True)
        else:
            df.loc[len(df.index)] = list(colVals) + [nomExpt]


    # create the output grid db
    griddb = DataBase()
    griddb.dbFile = inDB.dbFile
    griddb.dbfilename = inDB.dbfilename
    griddb.grid = True
    griddb.lineage = [[time.strftime(timeStr), "Created from linear variation sensitivity analysis grid creator"]]
    griddb.dataframe = df
    if save:
        if saveName is not None:
            griddb.dbfilename = saveName
        else:
            griddb.dbfilename = inputDB.dbfilename[:-4] + "_linearSensAna.csv"
        if saveLoc is not None:
            griddb.dbFile = os.path.join(os.path.abspath(saveLoc), griddb.dbfilename)
        else:
            griddb.dbFile = os.path.join(os.path.abspath(os.path.split(inputDB.dbFile)[0]), griddb.dbfilename)
        griddb.dbSaveFile(saveName=griddb.dbFile)

    print(f"sensitivity analysis vary linearly: Success :Created new grid db (input dataframe shape: '{inputDB.dataframe.shape}', input dataframe inputs shape: '{inputgrid.dataframe.shape}', output grid shape: '{griddb.dataframe.shape}')")

    return griddb


# ------------------------------------------------------------------------------------------------------------------

def createResortedDB(DBs, sort_by, ascending=True, filenameAppend=None, saveDBs=False, saveDBFolder=None):
    """
    Creates resorted DBs which do not change the original DB sorting (ie. a deep copy is created and subsequently sorted)

    :param DBs: [DB or list thereof] DB(s) to resort
    :param sort_by: [str] column label to sort by
    :param ascending: [bool] Sort values in ascending (T) or descending (F) order
    :param filenameAppend: [str or None] String to append to DB filename (None: _sorted-by_<sort_by>)
    :param saveDBs: [bool] Save the resorted DB? Or just return it
    :param saveDBFolder: [str or None] Path to folder to save the DB in (None: same folder as input DB)
    :return: sortedDBs [DB or list thereof] the resorted DBs
    """
    if not isinstance(DBs, list):
        DBs = [DBs]

    if filenameAppend is None:
        filenameAppend = f"_sorted-by_{sort_by}"

    sortedDBs = []

    for db in DBs:
        if saveDBFolder is None:
            if db.dbFile is not None:
                saveDBFolder = os.path.split(db.dbFile)[0]
            else:
                saveDBFolder = os.path.join(dbdir)
        else:
            saveDBFolder = os.path.abspath(saveDBFolder)

        sortedDB = DataBase()
        if sortedDB.dbfilename is not None:
            sortedDB.dbfilename = db.dbfilename[:-4]
        else:
            sortedDB.dbfilename = f"DB_{datetime.now().strftime(timeStr)}"
        sortedDB.dbfilename += f"{filenameAppend}.csv"
        sortedDB.dbFile = os.path.join(saveDBFolder, sortedDB.dbfilename)
        sortedDB.lineage = db.lineage.copy()
        sortedDB.lineage.append([f"\n{datetime.now().strftime(timeStr)}", f"Sorted by {sort_by}"])
        sortedDB.dataframe = db.dataframe.copy()
        sortedDB.dataframe = sortedDB.dataframe.sort_values(by=sort_by, ascending=ascending)
        sortedDB.dbCleanIdxsAndGridness()
        sortedDBs.append(sortedDB)
        if saveDBs:
            sortedDB.dbSaveFile()

    if len(sortedDBs) == 1:
        return  sortedDBs[0]
    else:
        return sortedDBs


# ------------------------------------------------------------------------------------------------------------------

def createTruncateDB(DBs, truncate_label, truncate_condition, filenameAppend=None, saveDBs=False, saveDBFolder=None, verbose=amverbose):
    """
    Create new DB with deep copy of the input dataframe and filter/truncate dataframe according to desire
        Uses an eval string:
          if str.isnumeric; "trunDB.dataframe.nlargest({truncate_condition}, '{truncate_label}')"
          else: "trunDB.dataframe[db.dataframe['"+truncate_label+"'] "+truncate_condition+"]"

    :param DBs: [DB or list thereof] DB(s) to filter/truncate
    :param truncate_label: [str] the data column label to filter on
    :param truncate_condition: [str] the truncate condition for the eval string (eg. '>= 0.98' for value filter OR '100' for top int value-sorted results)
    :param filenameAppend: [str or None] String to append to DB filename (eg. _JphFOM_gt_98pc. None: _truncated-by_<truncate_label>)
    :param saveDBs: [bool] Save the resorted DB? Or just return it
    :param saveDBFolder: [str or None] Path to folder to save the DB in (None: same folder as input DB)
    :return: truncatedDBs [DB or list thereof] the truncated/filtered DBs
    """

    if not isinstance(DBs, list):
        DBs = [DBs]
    if filenameAppend is None:
        filenameAppend = f"_trunc-by_{truncate_label}_{truncate_condition}".replace(" ", "-").replace(".","p").replace("<","lt").replace(">","gt").replace("=","e")

    truncatedDBs=[]
    for db in DBs:
        if db.dbFile is not None:
            if saveDBFolder is None:
                saveDBFolder = os.path.split(db.dbFile)[0]
            else:
                saveDBFolder = os.path.abspath(saveDBFolder)
        else: saveDBFolder = dbdir
        if db.dbfilename is not None: dbsavefile = db.dbfilename[:-4]
        else: dbsavefile = f"{timeStr}_truncation"

        trunDB = DataBase()
        trunDB.dbfilename = dbsavefile
        trunDB.dbfilename += f"{filenameAppend}.csv"
        trunDB.dbFile = os.path.join(saveDBFolder, trunDB.dbfilename)
        trunDB.lineage = db.lineage.copy()
        trunDB.lineage.append([f"\n{datetime.now().strftime(timeStr)}", f"Truncated by {truncate_label} {truncate_condition}"])
        trunDB.dataframe = db.dataframe.copy()
        if truncate_condition.strip().isnumeric(): #pick top n/condition label-sorted results
            evalStr = f"trunDB.dataframe.nlargest({truncate_condition.strip()}, '{truncate_label}')"
        else: #pick according to label cutoff-value/equality condition
            evalStr = "trunDB.dataframe[db.dataframe['"+truncate_label+"'] "+truncate_condition+"]"
        trunDB.dataframe = eval(evalStr)
        trunDB.dbCleanIdxsAndGridness()
        truncatedDBs.append(trunDB)
        if saveDBs:
            trunDB.dbSaveFile()
        if verbose: print(f"truncate db: Success :{trunDB.dbfilename} truncated\n"
              f"truncate db: {truncate_condition} {truncate_label} | N={len(trunDB.dataframe)} | Top={trunDB.dataframe.nlargest(1, truncate_label)[truncate_label]} | Bot={trunDB.dataframe.nsmallest(1, truncate_label)[truncate_label]}" )

    if len(truncatedDBs) == 1:
        return truncatedDBs[0]
    else:
        return truncatedDBs






# ------------------------------------------------------------------------------------------------------------------
# ------------------------------------------------------------------------------------------------------------------
#!    HELPER FUNCTIONS
# ------------------------------------------------------------------------------------------------------------------
# ------------------------------------------------------------------------------------------------------------------



def csvFindHeader(csvFile):
    """
    Find the header row line idx within the database csv file using inputStartStrs, outputStartStrs, and 'db_idx'

    :param csvFile: path to the csv file (string)
    :return: headerLinesLen: number of lines of header in the csv file (int)
    """
    headerLinesLen = None
    with open(csvFile, 'r') as file:
        lines = file.readlines()
        for lidx in range(len(lines)):
            if any(header in lines[lidx] for header in inputStartStrs+outputStartStrs) or 'db_idx' in lines[lidx]:
                headerLinesLen = lidx
                break
            else:
                continue
        if headerLinesLen == None:
            print (f"WARNING :csv find header: Could not find header strings in the file '{csvFile}'. \nAssuming no header.")
            headerLinesLen = 0
    return headerLinesLen



def getNExptsAtEveryM(lengthDB, N=5, M=100, startM=0, endM=None, printResults=True):
    """
    Get a selection of indicies; N elements at every Mth index
    Specifically; M, M+1, M+2, ... M+N at every Mth index from startM up to a max of endM (inclusive)
    Will not include endM if endM is not startM + n*M with n an integer

    Note; For M=len(DB) will pull idxs len(DB)-N, len(DB)-N+1, ... len(DB) (also the same when M is within N of len(DB))


    :param lengthDB: int, number of experiments in the db
    :param N: int, number of experiments to select at each Mth index
    :param M: int, how many idxs to skip between each selection
    :param startM: int, index to start at selection at
    :param endM: int or None, index to end selection at (inclusive) , if None uses end of db
    :param printResults: bool, print the selected indicies
    :return idxSel: list, the selected indicies
    """

    if endM is None:
        endM = lengthDB - 1

    if endM > lengthDB - 1:
        print(
            f"ERROR :getNExptsAtEveryM: endM ({endM}) is greater than the last idx in the database ({lengthD - 1}). Setting endM to last idx in database...")
        endM = lengthDB - 1

    idxSel = []
    Ms = list(np.arange(startM, endM + 1, M))
    if Ms[-1] + N > lengthDB - 1:
        abutsEnd = True
    else:
        abutsEnd = False

    for Mi in Ms:
        for Ni in range(len(N)):
            if abutsEnd and Ms.index[Mi] == len(Ms) - 1:
                # handle end of db abutting case
                idxSel.append(lengthDB - N + Ni)
            else:
                # regular case
                idxSel.append(Mi + Ni)

    if printResults:
        print(f"getNExptsAtEveryM: Selected indicies are:\n{idxSel}")

    return idxSel
