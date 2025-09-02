import os
import sys

import matlab.engine
from aiirmapCommon import *

def main():
    """
    Terminal command wrapper for the run_gridInSingleSentaurusProject function from aiirmapCommon

    Runs a grid DB in the given sentaurus project.
        Splits the grid into multiple project runs according to the size of the grid DB and the defined maximum number of exp'ts per project

    Takes as input one to five linux terminal command arguments, but only in order (ie. supply input 2 to be able to supply inputs 3+, etc.);
        (str) The path to the grid DB to run (must be supplied)
        (str) The sentaurus working directory (swd) path (if not supplied; use current directory)
        (int) Locked tools switch: 0='find' (use current saved swb config), 1=lockedTools_default_s4Opt, 2=lockedTools_default_electrical (see config)
        (int) The maximum number of experiments per project run (if not supplied; uses default given below)
        (str) Run descriptor, used in the filing cabinet folders and filenames (if not supplied; uses the grid DB's dbfilename, will overwrite filed files if folders with that run descriptor exist, see below)


    Egs.
    Full suite of inputs
    python3 /path/to/this/script/ /path/to/gridDB.csv /path/to/swd/ lockedToolSwitch N_maxExptsPerProj "nice_run_description"

    Minimal inputs
    python3 /path/to/this/script/ /path/to/gridDB.csv
            With this setup, must run the command from the swd

    Aiirmap Assumptions:
    -Node number assignments are larger for tools later in the sim flow (within each expt)
    -Heirarchial sentaurus project organization

    :return: surveyedGridDB; (DataBase) DataBase with the grid's inputs and the results from the run
    """

    #TODO: Add runtime output and save it to the DB (In run_grid... function of amCommon)

    #defaults
    # overwritten by the terminal arguments
    wd = os.getcwd()
    maxExptsPerProj = 25
    lockedTools = 'find'  #'find' #'find' = try and extract locked tools from the templateGtree

    # for run_gridInSingleSentaurusProject() (inputs which are not overwritten by the terminal arguments)
    #for more info see that function
    templateGtree = None #None = use the gtree file in the wd
    gsubGEXPR = 'all' #gsub -e option
    gsubOptionsStr = '-verbose' #gsub options, see si.sentaurus_gsub for more info
    gcleanupOptionsStr = '-verbose -default -ren' #gcleanup options, see si.sentaurus_gcleanup for more info
    overwriteRunning = True #continue with the run even if the project is in the 'running' state
    logOutput = True #log gsub and gcleanup command output in some logfiles if True
    hushBashOutput = True #do not output gsub and gcleanup command output to the python/run terminal/output if True
    overwriteFiledFiles = True #overwrite the sentaurus_files and databases runDescriptor folders if they exist if True (or append datetime to runDesc if False)





    # interpret the bash inputs
    if len(sys.argv) < 2:
        print(f"ERROR :grid runner wrapper: Not enough command line arguments. You must supply the path to the gridDB. Cancelling...")
        return None
    elif len(sys.argv) == 2: #minimal inputs
        gridDBpath = os.path.abspath(sys.argv[1])
        runDescriptor = os.path.split(os.path.abspath(sys.argv[1]))[1][0:-4] #default
    elif len(sys.argv) == 3: #gridDBpath and wd supplied
        gridDBpath = os.path.abspath(sys.argv[1])
        wd = os.path.abspath(sys.argv[2])
        runDescriptor = os.path.split(os.path.abspath(sys.argv[1]))[1][0:-4]  # default
    elif len(sys.argv) == 4:  # gridDBpath, wd, and locked tool switch supplied
        gridDBpath = os.path.abspath(sys.argv[1])
        wd = os.path.abspath(sys.argv[2])
        lockedToolSwitch = int(sys.argv[3])
        runDescriptor = os.path.split(os.path.abspath(sys.argv[1]))[1][0:-4]  # default
    elif len(sys.argv) == 5:   # gridDBpath, wd, and locked tool switch supplied and num expts per project supplied
        gridDBpath = os.path.abspath(sys.argv[1])
        wd = os.path.abspath(sys.argv[2])
        lockedToolSwitch = int(sys.argv[3])
        maxExptsPerProj = int(sys.argv[4])
    elif len(sys.argv) == 6: # all inputs supplied
        gridDBpath = os.path.abspath(sys.argv[1])
        wd = os.path.abspath(sys.argv[2])
        lockedToolSwitch = int(sys.argv[3])
        maxExptsPerProj = int(sys.argv[4])
        runDescriptor = sys.argv[5]
    elif len(sys.argv) > 6:
        gridDBpath = os.path.abspath(sys.argv[1])
        wd = os.path.abspath(sys.argv[2])
        lockedToolSwitch = int(sys.argv[3])
        maxExptsPerProj = int(sys.argv[4])
        runDescriptor = sys.argv[5]
        print(f"WARNING :grid runner wrapper: Called with more than command line parameters than expected ('{len(sys.argv)} > 6'). "
              f"Using first parameter as path to the grid ('{gridDBpath}'), second as wd ('{wd}'), third is the locked tool switch ('{lockedToolSwitch}'), fourth as max number of experiments per project ('{maxExptsPerProj}'), and fifth as the run desciptor ('{runDescriptor}').")

    #load grid DB
    gridDB = dbh.loadDBs(gridDBpath)

    #handle locked tool switch
    if lockedToolSwitch == 1: lockedTools = lockedTools_default_s4Opt
    elif lockedToolSwitch == 2: lockedTools = lockedTools_default_electrical
    elif lockedToolSwitch > 2: print(f"WARNING :grid runner wrapper: Invalid lockedToolSwitch value ('{lockedToolSwitch}' > 2). Using lockedTools='find'...")

    #run the survey
    surveyedGridDB = run_gridInSingleSentaurusProject(gridDB, wd, maxExptsPerProj=maxExptsPerProj, runDescriptor=runDescriptor,
                                                        templateGtree=templateGtree, lockedTools=lockedTools, gsubGEXPR=gsubGEXPR, gsubOptionsStr=gsubOptionsStr, gcleanupOptionsStr=gcleanupOptionsStr,
                                                        overwriteRunning=overwriteRunning, logOutput=logOutput, hushBashOutput=hushBashOutput, overwriteFiledFiles=overwriteFiledFiles)


    return surveyedGridDB


if __name__ == '__main__':
    main()
