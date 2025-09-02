
"""
Log-in, log-out, and auxiliary functions for the project runtime logger functionality.

The runtime logger is a csv file containing a few header lines and a pd.dataframe.
Entries into the runtime dataframe are made at the start of a run (log-in) and at the end (log-out).
The file is left closed in-between brief accesses, to minimize the risk of double access issues.

Log-in records start time, and run type information from the user input (inputColValues) and the run database
Log-out finds the run_descriptor entry, records the end_time, and fills the duration and expt rate cols
  If the run_descriptor entry is not found (or is not unique) a new row is added.

The runtime logger file is typically located in filingdir.
It can be created below using createRuntimeLoggerFile.
It's location can be set in config using runtimeLogFilePath (or passed directly to the log_in/out fncs using runtimeLogPath)

!NOTE!: The logger file has no built-in backup functionality. It is suggested to make your own BACKUPS.


-----------------------------------
RUNTIME LOGGER DATAFRAME COLUMNS
-----------------------------------
All logger file columns (ordered);
'run_idx', 'run_descriptor', 'start_time', 'end_time', 'duration_min', 'duration_hr', 'num_expts', 'expts_per_min', 'expts_per_hr', 'expts_per_hr_per_pfac', 'run_type','inSen_pfac', 'matl_pfac', 'num_opts', 'num_segments', 'eSim_type', 'OptiToSentaurus', 's4OptType', 'LC', 'comment'

Log_in editable columns;
-, 'run_descriptor', 'start_time', ---, 'num_expts', ---, 'run_type','inSen_pfac', 'matl_pfac', 'num_opts', 'num_segments', 'eSim_type', 'OptiToSentaurus', 's4OptType', 'LC', 'comment'

Log_out editable columns;
-, 'run_descriptor', 'start_time', end_time, ----------, 'num_segments', 'eSim_type', 'OptiToSentaurus', 's4OptType', 'LC', 'comment'



-----------------------------------
LOGGER LOCATIONS
-----------------------------------




2310
"""

# was this ever tested / fully implemented? ~Rob (2505)  ##TODO?


from config import *
import time
import pandas as pd
import numpy as np
from datetime import datetime



def main():
    #test the functions



    return





def runtimeLogger_log_in(rundb, runDescr=None, runType=None,  inputColValues={},
                           runtimeLogPath=runtimeLogFilePath):
    """
    Log in a project run to the runtime logger csv file

    Opens the runtime logger dataframe csv file,
    write a new line for runDescriptor, recording start_time and other run parameters
    save and close runtime logger

    See below for runtime column info.

    NOTE: use in project run functions with runtimeLogger_log_out

    @param rundb: DataBase, the DataBase with the experiments being run
    @param runDescr: string, descriptor for the run *will be used to identify the run in log_out, if None tries to use dbfilename/dbFile
    @param runType: string, what type of run / where this script was run from (eg. s-grid, matl-opt, ...)
    @param inputColValues: dict, input column values in form 'col_label': value, see labels below (overwrites rundb)
    @param runtimeLogPath: string, path to the runtime logger file
    @return: startTime: string, (yymmdd_hhmmss) the recorded start time



    RUNTIME DATAFRAME COLUMNS
    All logger file columns;
    'run_idx', 'run_descriptor', 'start_time', 'end_time', 'duration_min', 'duration_hr', 'num_expts', 'expts_per_min', 'expts_per_hr', 'expts_per_hr_per_pfac', 'run_type','inSen_pfac', 'matl_pfac', 'num_opts', 'num_segments', 'eSim_type', 'OptiToSentaurus', 's4OptType', 'LC', 'comment'

    Log_in editable columns;
    -, 'run_descriptor', 'start_time', ---, 'num_expts', ---, 'run_type','inSen_pfac', 'matl_pfac', 'num_opts', 'num_segments', 'eSim_type', 'OptiToSentaurus', 's4OptType', 'LC', 'comment'

    Notes:
    inputColValues supercedes rundb column value if column is given!
    pfacs and num_opts are only given through inputColValues
    runDescr and runType are used for their column values (the presence of these labels in inputColValues is ignored)


    """


    #prep; handle missing inputs
    if runDescr is None:
        if rundb.dbfilename is None:
            if rundb.dbFile is None:
                runDescr = f''
            else:
                runDescr = os.path.split(rundb.dbFile)[1][:-4]
        else:
            runDescr = rundb.dbfilename[:-4]

    #allow for user given start time to be set
    if 'start_time' not in inputColValues:
        startTime = time.strftime(timeStr)
    else: startTime=inputColValues['start_time']

    #run type specific inputs to be saved
    if 'inSen_pfac' not in inputColValues:
        inSen_pfac = None
    else: inSen_pfac=inputColValues['inSen_pfac']
    if 'matl_pfac' not in inputColValues:
        matl_pfac=None
    else: matl_pfac=inputColValues['matl_pfac']
    if 'num_opts' not in inputColValues:
        num_opts=None
    else: num_opts=inputColValues['num_opts']

    #pull from rundb if not supplied directly
    if 'num_segments' not in inputColValues:
        try: num_segs = rundb.dataframe['num_segments']
        except KeyError: num_segs = None
    else: num_segs=inputColValues['num_segments']

    if 'eSim_type' not in inputColValues:
        try: eSim_type = rundb.dataframe['eSim_type']
        except KeyError: eSim_type = None
    else: eSim_type=inputColValues['eSim_type']

    if 'OptiToSentaurus' not in inputColValues:
        try: OptiToSent = rundb.dataframe['OptiToSentaurus']
        except KeyError: OptiToSent = None
    else: OptiToSent=inputColValues['OptiToSentaurus']

    if 's4OptType' not in inputColValues:
        try: s4OptType = rundb.dataframe['s4OptType']
        except KeyError: s4OptType = None
    else: s4OptType=inputColValues['s4OptType']

    if 'LC' not in inputColValues:
        try: LC = rundb.dataframe['LC']
        except KeyError: LC = None
    else: LC=inputColValues['LC']

    #empty comment if not present
    if 'comment' not in inputColValues:
        comment=''
    else: comment=inputColValues['comment']

    #access the runtime logger csv file dataframe, find header bound, separate header and read dataframe
    with open(runtimeLogPath, 'r') as logFile:
        logFileLines = logFile.readlines()
    headlen = 0
    for lidx in range(len(logFileLines)):
        if 'start_time' in logFileLines[lidx]: headlen=lidx; break

    headerLines = logFileLines[:headlen]
    rtdf = pd.read_csv(runtimeLogPath, skiprows=headlen)

    #get data and write a new line
    # loginDataLabels = ['run_idx', 'run_descriptor', 'start_time', 'num_expts', 'run_type', 'inSen_pfac', 'matl_pfac', 'num_opts', 'num_segments', 'eSim_type', 'OptiToSentaurus', 's4OptType', 'LC', 'comment']
    # logoutDataLabels = ['end_time', 'duration_min', 'duration_hr', 'expts_per_min', 'expts_per_hr', 'expts_per_hr_per_pfac', 'comment']

    runtimeDataLabels_inOrder = ['run_idx', 'run_descriptor', 'start_time', 'end_time', 'duration_min', 'duration_hr', 'num_expts', 'expts_per_min', 'expts_per_hr', 'expts_per_hr_per_pfac', 'run_type','inSen_pfac', 'matl_pfac', 'num_opts', 'num_segments', 'eSim_type', 'OptiToSentaurus', 's4OptType', 'LC', 'comment']
    runData = [len(rtdf)+1, runDescr, startTime, None, None, None, len(rundb.dataframe), None, None, None, runType, inSen_pfac, matl_pfac, num_opts, num_segs, eSim_type, OptiToSent, s4OptType, LC, comment]

    rtdf = pd.concat([rtdf, pd.DataFrame(columns=runtimeDataLabels_inOrder, data=[runData])], axis=0, join='outer', ignore_index=True)
    rtdf.reset_index(inplace=True)

    #save
    with open(runtimeLogPath, 'w') as file:
        for line in headerLines: file.write(line)
        rtdf.to_csv(runtimeLogPath)

    #output to terminal
    print(f"RUNTIME LOGGER: Log-in for '{runDescr}'. start_time={startTime}, run_type={runType}, num_expts={len(rundb.dataframe)}")

    return startTime





def runtimeLogger_log_out(rundb, runDescr=None, outputColValues={}, overwriteEnd=False,
                           runtimeLogPath=runtimeLogFilePath):
    """
    Log-out functionality for the project runtime logger dataframe csv

    Uses the runDescr(iptor) to identify the correct entry in the runtime dataframe.
        As log as there is one matching entry;
            Pulls data for start_time and pfacs. Records end_time. Calculates duration and expt rate columns. Adds to row.
        Else (many, none);
            Create a new row using rundb for col values (unless they are given in outputColValues);
                if no start_time is supplied in outputColValues, output cols are None
                if start_time is supplied; use it to calculate the duration and rate values

    NOTE: use in project run functions with runtimeLogger_log_in

    @param rundb: DataBase, the DataBase with the experiments being run
    @param runDescr: string, descriptor for the run *will be used to identify the run in log_out, if None tries to use dbfilename/dbFile
    @param inputColValues: dict, input column values in form 'col_label': value, see labels below (overwrites rundb)
    @param runtimeLogPath: string, path to the runtime logger file
    @return: startTime: string, (yymmdd_hhmmss) the recorded start time
    @return: startTime: string, (yymmdd_hhmmss) the recorded start time
    @return: duration_min/hr: floats or None, duration between the start_time and end_time in min/hr
    @return: expts_per_min/hr: floats or None, number of experiments completed per min/hr on average (numExpts/duration)
    @return: expts_per_hr_pfac: float or None, number of experiments completed per min/hr on average
                                                                divided by the parallelization factor (pfac=inSen_pfac*matl_pfac) (numExpts/duration/pfac)



    RUNTIME DATAFRAME COLUMNS
    All logger file columns;
    'run_idx', 'run_descriptor', 'start_time', 'end_time', 'duration_min', 'duration_hr', 'num_expts', 'expts_per_min', 'expts_per_hr', 'expts_per_hr_per_pfac', 'run_type','inSen_pfac', 'matl_pfac', 'num_opts', 'num_segments', 'eSim_type', 'OptiToSentaurus', 's4OptType', 'LC', 'comment'

    Log_out editable columns;
    -, 'run_descriptor', 'start_time', end_time, ----------, 'num_segments', 'eSim_type', 'OptiToSentaurus', 's4OptType', 'LC', 'comment'

    Notes:
    outputColValues supercedes rundb column value if column is given!
    runDescr and runType are used for their column values (the presence of these labels in inputColValues is ignored)


    """

    #prep; handle missing inputs
    if runDescr is None:
        if rundb.dbfilename is None:
            if rundb.dbFile is None:
                runDescr = f''
            else:
                runDescr = os.path.split(rundb.dbFile)[1][:-4]
        else:
            runDescr = rundb.dbfilename[:-4]

    #allow for user given start/end times to be set
    if 'start_time' not in outputColValues:
        startTime = None
    else: startTime=outputColValues['start_time']
    if 'end_time' not in outputColValues:
        endTime = time.strftime(timeStr)
    else: endTime=outputColValues['end_time']


    #check rundb if not supplied directly
    try: num_segs = rundb.dataframe['num_segments']
    except KeyError: num_segs = None

    try: eSim_type = rundb.dataframe['eSim_type']
    except KeyError: eSim_type = None

    try: OptiToSent = rundb.dataframe['OptiToSentaurus']
    except KeyError: OptiToSent = None

    try: s4OptType = rundb.dataframe['s4OptType']
    except KeyError: s4OptType = None

    try: LC = rundb.dataframe['LC']
    except KeyError: LC = None


    #None comment if not present (will be ignored later)
    if 'comment' not in outputColValues:
        commentIn=None
    else: commentIN=outputColValues['comment']



    #access the runtime logger csv file dataframe, find header bound, separate header and read dataframe
    with open(runtimeLogPath, 'r') as logFile:
        logFileLines = logFile.readlines()
    headlen = 0
    for lidx in range(len(logFileLines)):
        if 'start_time' in logFileLines[lidx]: headlen=lidx

    headerLines = logFileLines[:headlen]
    rtdf = pd.read_csv(runtimeLogPath, skiprows=headlen)



    #look for runDescriptor line (exact)
    #count entries and run
    numEntr = (rtdf['run_descriptor'].eq(runDescr)).sum()

    # found one
    if numEntr == 1:

        #exit if already written and overwrite is False
        if rtdf.loc[rtdf['run_descriptor'] == runDescr]['end_time'] is not None and not overwriteEnd:
            print(f"ERROR: RUNTIME LOGGER: Found run. Run already logged out! Exiting without logging... (overwriteEnd=False, runDescr='{runDescr}', run_idx={ rtdf.loc[rtdf['run_descriptor'] == runDescr]['run_idx']})")
            return

        #get start time
        if startTime is None: #get from the runtime db, else if spplied use supplied
            if rtdf.loc[rtdf['run_descriptor'] == runDescr]['start_time'] is not None:
                startTime = rtdf.loc[rtdf['run_descriptor'] == runDescr]['start_time']

        #determine paralellization factor
        pfac_s = rtdf.loc[rtdf['run_descriptor'] == runDescr]['inSen_pfac']
        pfac_m = rtdf.loc[rtdf['run_descriptor'] == runDescr]['matl_pfac']
        if pfac_s is not None and pfac_m is not None: pfac = pfac_m*pfac_s
        elif pfac_s is not None: pfac = pfac_s
        elif pfac_m is not None: pfac = pfac_m
        else: pfac = None

        #calc output
        if startTime is not None:
            tdelta_min, tdelta_hr, expts_per_min, expts_per_hr, expts_per_hr_per_pfac = runtimeOutCalcs(startTime, endTime, len(rundb.dataframe), pfac=pfac)
        else:
            tdelta_min, tdelta_hr, expts_per_min, expts_per_hr, expts_per_hr_per_pfac =  None, None, None, None, None

        #handle comment
        if commentIn is None:
            comment = rtdf.loc[rtdf['run_descriptor'] == runDescr]['comment']
        else:
            comment = rtdf.loc[rtdf['run_descriptor'] == runDescr]['comment'] + " | " + commentIn

        #save to rtdf line
        rtdf.loc[rtdf['run_descriptor'] == runDescr]['duration_min', 'duration_hr', 'expts_per_min', 'expts_per_hr', 'expts_per_hr_per_pfac', 'comment'] \
            = [tdelta_min, tdelta_hr, expts_per_min, expts_per_hr, expts_per_hr_per_pfac, comment]

        print(f"RUNTIME LOGGER: Log-out found entry for run '{runDescr}' (run_idx={rtdf.loc[rtdf['run_descriptor'] == runDescr]['run_idx']})...")


    else:
        # found many or none
        if numEntr > 1:
            print(f"ERROR: RUNTIME LOGGER :Found multiple entries with run_descriptor={runDescr}. (run_idx={list(rtdf.loc[rtdf['run_descriptor'] == runDescr]['run_idx'].values)})")
            print(f"ERROR: RUNTIME LOGGER :Creating new end_time row for run_descriptor={runDescr}. (run_idx={len(rtdf)+1}, start_time={startTime})")
        else:
            print(f"ERROR: RUNTIME LOGGER :No entries with run_descriptor={runDescr}. Creating new entry (run_idx={len(rtdf)+1}) with start_time={startTime}...")

        # create new and set output to None unless startTime was supplied via input
        if startTime is not None:
            tdelta_min, tdelta_hr, expts_per_min, expts_per_hr, expts_per_hr_per_pfac = runtimeOutCalcs(startTime, endTime, len(rundb.dataframe), pfac=None)
        else:
            tdelta_min, tdelta_hr, expts_per_min, expts_per_hr, expts_per_hr_per_pfac = None, None, None, None, None

        if commentIn is None: comment = ''
        else: comment = commentIn

        runtimeDataLabels_inOrder = ['run_idx', 'run_descriptor', 'start_time', 'end_time', 'duration_min', 'duration_hr', 'num_expts', 'expts_per_min', 'expts_per_hr', 'expts_per_hr_per_pfac', 'run_type','inSen_pfac', 'matl_pfac', 'num_opts', 'num_segments', 'eSim_type', 'OptiToSentaurus', 's4OptType', 'LC', 'comment']
        runData = [len(rtdf)+1, runDescr, startTime, endTime, tdelta_min, tdelta_hr, len(rundb.dataframe), expts_per_min, expts_per_hr, expts_per_hr_per_pfac, 'log_out', None, None, None, rundb, eSim_type, OptiToSent, s4OptType, LC, comment]

        rtdf = pd.concat([rtdf, pd.DataFrame(columns=runtimeDataLabels_inOrder, data=[runData])], axis=0, join='outer', ignore_index=True)
        rtdf.reset_index(inplace=True)


    #save
    with open(runtimeLogPath, 'w') as file:
        for line in headerLines: file.write(line)
        rtdf.to_csv(runtimeLogPath)


    #output to terminal
    print(f"RUNTIME LOGGER: Log-out for '{runDescr}'  -------------------------------------------------\n "
          f"start_time={startTime},\t\tend_time={endTime},\t\tduration_hr={tdelta_hr}, num_expts={len(rundb.dataframe)}, expts/hr={expts_per_hr},")

    return startTime, endTime, tdelta_min, tdelta_hr, expts_per_min, expts_per_hr, expts_per_hr_per_pfac





def runtimeOutCalcs(start, end, numexpts, pfac=1):
    # take start and end times (in string format; yymmdd_hhmmss), the number of expts (int), and the parallelization factor (pfac; int)
    # calculate duration (min, hr), expts per min/hr, expts per min/hr per pfac

    tdelta_timedelta = datetime.strptime(end, timeStr) - datetime.strptime(start, timeStr)
    tdelta_min = tdelta_timedelta.total_seconds()/60
    tdelta_hr = tdelta_min/60
    expts_per_min = numexpts/tdelta_min
    expts_per_hr = numexpts/tdelta_hr
    if pfac is not None: expts_per_hr_per_pfac = expts_per_hr/pfac
    else: expts_per_hr_per_pfac = None

    return tdelta_min, tdelta_hr, expts_per_min, expts_per_hr, expts_per_hr_per_pfac



def createRuntimeLoggerFile(filepath = filingdir, location = ''):

    #create the runtime logger file header info

    # filepath = f'' \
    #            f'C:\\Users\\rhunt013\\Local-Work-Folder_sunlap\\flyingFolder_local_sunlap\\aiirmap_runtime_logfile.csv'
    # location = 'dyson'

    f= open(filepath, 'w')
    f.write('aiirmap project runtime dataframe csv\n'
            f'location,{location}\n'
            'This functionality is implemented using the aiirmapCommon.runtimeLogger_log_in/out functions.\n'
            'Login and logout is implemented in ...\n'
            'Include in further code for tracking, if desired.\n'
            'run_idx,run_descriptor,start_time,end_time,duration_min,duration_hr,num_expts,expts_per_min,expts_per_hr,expts_per_hr_per_pfac,run_type,inSen_pfac,matl_pfac,num_opts,num_segments,eSim_type,OptiToSentaurus,s4OptType,LC,comment\n')
    f.close()