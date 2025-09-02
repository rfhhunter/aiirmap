"""
Sentaurus Project Read and Interact Utilities

Functions and variables for interacting with the Sentaurus project folder
Includes (#!):
- Collect Sentaurus project info and files
    =sPyToolPrepper, sCollectProject
- Sentaurus send to and run functions
- Sentaurus created file; read and interact functions
- AiirMap created file; read and interact functions
- Sentaurus project info querying
- Filing cabinet - Sentaurus project folder interaction

22.09.11
"""
#------------------------------------------------------------------------------------------------------------------
#------------------------------------------------------------------------------------------------------------------
#------------------------------------------------------------------------------------------------------------------
import time
import os
import sys
import csv
import glob
import shutil
import subprocess
import numpy as np
import pandas as pd

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
import databasing as dbh
sys.path.append("..")
from config import *

#------------------------------------------------------------------------------------------------------------------
#------------------------------------------------------------------------------------------------------------------
#!    COLLECT SENTAURUS PROJECT INFO AND FILES
#------------------------------------------------------------------------------------------------------------------
#------------------------------------------------------------------------------------------------------------------

# config's sToolName and sFileNameBody are used to set the py tool's name and the prefix for the files
# created by aiirmap in the sentaurus project folder
# in the below comments these are referred to as pyCollector and pyCollection for historical reasons



def sPyToolPrepper(wd):
    """
    Prepare the pyCollector_pyt.py command file.
    :param wd: (string) sentaurus project working directory path
    :return: None, writes the pyCollector_pyt.py command file in wd

    This is the python3 version of this function. It is not compatible with the gsub prologue python (which is py2)
    For the py2 version see py2prepper.py
    Functional changes to this (that) function should be propagated there (here)

    """
    #Previously pyPrepper.py

    #Assumes vertical tool flow orientation (should work with horizontal but needs to be tested
    #Sentaurus must be running in a linux environment (/ path divisor)
        #this is required since this function writes a file with sentaurus preprocessed path dependency and cannot use os.join.path
        #the "/" of paths is also used to help manipulate standard strings in the read_glog, read_goptlog, and read_epi

    print(f"pyCollector prepper: Preparing py-tool cmd file '{os.path.join(wd, f'{sToolName}_pyt.py')}'.")

    now = time.strftime(timeStr)

    # Get the parameter names and their variable types, then generate the pyCollector writer strings
    params, _, ptypes, _, toolLists, _ = read_gtree(os.path.join(wd, "gtree.dat"))

    headerStr = f"writer.writerow(['expt_num',"
    dataCallStr = f"writer.writerow([@experiment@,"
    for pidx in range(len(params)):
        headerStr = headerStr + f"'{params[pidx]}',"
        if ptypes[pidx] in ["string", "boolean"]:
            dataCallStr = dataCallStr + f"'@{params[pidx]}@',"  # pyCollection_swbVars.csv retains the specific swb boolean identifier used
        else:
            dataCallStr = dataCallStr + f"@{params[pidx]}@,"

    headerStr = headerStr + "'expt nodes'])\n"
    dataCallStr += "'@node|all@'])\n"

    # Write the sToolName_pyt.py command file
    with open(os.path.join(wd, f"{sToolName}_pyt.py"), 'w', newline='', encoding='UTF8') as file:
        file.write(
            f"#Sentaurus pyCollector cmd file\n#The tool collects input parameters and node information for each experiment\n"
            f"#It creates or overwrites results/nodes/#/n#_swbVars.csv files when run\n\n"
            f"#File autocreated by pyPrepper.py script\n#Command file creation time:{now}\n\n")
        file.write("import time\nimport csv\nimport os\n\nnow = time.strftime('%Y%m%d_%H%M%S')\n\n"
                   ## f"#setdep @node|-1@\n\n"
                   "#For all expts; write project info, headers and then extract data and print to line\n")
        file.write(f"expts = '@experiments@'\nexptStr = expts.replace(' ', '-')\n")
        file.write(
            "with open(f'@pwd@/results/nodes/@node@/n@node@_swbVars.csv', 'w', newline='', encoding='UTF8') as file:\n"
            "\tfile.write(f'@pwd@\\npreprocessing time:,{now}')\n"
            "\tfile.write(f'\\n\\nsimulation flow:\\n@tool_label|all@\\n\\n')\n"
            "\twriter = csv.writer(file)\n"
            f"\t{headerStr}\n\t{dataCallStr}\n")

    print(f"pyCollector prepper: Success :A new {sToolName}_pyt.py has been written.")
    return


#------------------------------------------------------------------------------------------------------------------


def sCollectProject(wd, fd, inheritedProjectData=None):
    """
    Save Sentaurus project data and files to the filing cabinet. For a single project.
    Collect and fill project information from glog.txt or gopt.log (for Info)
        If an optimization family which is running; supply inheritedProjectData = [hostname (str), user (str), sVersion (str), runTime (str), parent opt node (int)]
    Collect tool info from gtree.dat and use to anticipate the cmd file names using sentaurus defaults
    Collect and fill exp't information from n#_exp-#_swbVars and gvars (for Info)
    Collect and fill exp't information from swbVars and nodes sta files (for execTimes)
    Copy files from sentaurus project to filing cabinet
    :param wd: [path-string] working directory of sentaurus project
    :param fd: [path-string] directory for filing cabinet
    :param inheritedProjectData: see above, list of [hostname (str), user (str), sVersion (str), runTime (str), parent opt node (int)]
    :param family: [boolean] whether or not the project is part of an optimization family (parent or child)
    :param parent: [path-string] if project is part of a family this is the path to the parent project
    :return: None (creates files, copies files)
    :return inheritenceData: see above, list of [hostname (str), user (str), sVersion (str), runTime (str), parent opt node (int)] to pass on
    """

    #Previously called pyProjectCollector.py

    print(f"\nsentaurus project collection: Collecting sentaurus project '{wd}' and saving files to '{fd}'.")


    #Collect  Project Info

    #get project information
    _, _, _, _, toolsList, _ = read_gtree(os.path.join(wd, "gtree.dat"))
    inheritenceData = []
    if inheritedProjectData == None or len(inheritedProjectData) == 0:
        parentOptNode = -1
        #normal case, pull from file
        if os.path.exists( os.path.join(wd, "results", "logs", "glog.txt") ):
            #single or completed family project folders have this file
            hostname, user, sVersion, runTime, _ = read_glog(wd)
            child=False
        else:
            #probably a running optimization family parent, check for gopt.log
            if os.path.exists(os.path.join(wd, "results", "logs", "gopt.log")):
                hostname, user, sVersion, runTime, _ = read_goptlog(wd)
                child=False
            else:
                #a running child without an inheritance (, or, optimization parent with user gen expts, or,... (sigh-taurus))
                #try to pull from status file

                child=False #240503 gridrunner paradigm falls into this category, set False to collect all sFiles

                if os.path.exists(os.path.join(wd,".status")):
                    [runTime, hostname, user,_ ,_ ] = read_dotStatus(wd)
                    sVersion = None
                else:
                    print(f"WARNING :sentaurus project collection: There is no .status or (results/logs/)glog.txt or gopt.log file. Setting hostname,user,runTime,sVersion to None. (Are you trying to collect a child with the family still running? Supply an inheritence to sCollectProject).")
                    hostname, user, runTime, sVersion = None,None,None,None
    else:
        # use inheritence data (running optimization family child)
        [hostname, user, sVersion, runTime, parentOptNode] = inheritedProjectData
        child = True

    #Newer implementation of the command files (see config.py)
    cmdFiles = anticipate_cmdFiles(wd, toolsList)




    #Collect Experiment Information; from the node-specific swb vars files

    #pull all possible expt swbVar file paths
    swbFiles = glob.glob(os.path.join(wd, "results", "nodes", "*", "*_swbVars.csv"))
    #match the outputs to the experiments and screen for incomplete and old experiments
    toolLabelFlow, swbHeader, swbData, exptsNodes = readMatch_exptIO(swbFiles, gvarsFile=os.path.join(wd, "gvars.dat"))

    #(historical; project/swbVars implementation)
    # #pull inputs and node info
    # wd_fromSwb, toolLabelFlow, inputsLabels, exptInputs, exptNodes = read_swbVars()
    #pull outputs
    # outputsLabels, exptOutputs = readAssign_gvars(outputVarNodes, os.path.join(wd, "gvars.dat"))



    #Output and Transfer Files

    #prep file header info
    now = time.strftime(timeStr)
    projectHeader = f"Project Directory,sDir,{os.path.realpath(wd)}\n"
    if parentOptNode is not None: projectHeader += f"Project Parent Opt Node,par_opt_node,{parentOptNode}\n"
    projectHeader +=f"Hostname,hostname,{hostname}\n" \
                    f"User,user,{user}\nSentaurus " \
                    f"Version,sVer,{sVersion}\n" \
                    f"Simulation Start Time,time_sim,{runTime}\n" \
                    f"File Creation Time,,{now}\n\n"

    # write the pyCollection_execTimes.csv file and handle lockedTools if obtained
    print(f"sentaurus project collection: Writing {sFileNameBody}_execTimes.csv...")
    lockedTools = save_execTimeFile(wd, projectHeader, toolLabelFlow, exptsNodes, detectLockedTools=True, verbose=amverbose)
    # also pull lockedTools from gtree? (and compare)
    if lockedTools != None:
        for tidx in range(len(toolLabelFlow)):
            if lockedTools[tidx]:
                toolLabelFlow[tidx] = toolLabelFlow[tidx] + "-L"

    # write the pyCollection_Info.csv file
    print(f"sentaurus project collection: Writing {sFileNameBody}_Info.csv...")
    with open(os.path.join(wd, f"{sFileNameBody}_Info.csv"), 'w', newline='', encoding='UTF8') as file:
        file.write(f"#Sentaurus Project Info File\n{projectHeader}Tool Flow\n")
        writer = csv.writer(file)
        writer.writerow(toolLabelFlow)
        writer.writerow([])
        writer.writerow(swbHeader)
        for eidx in range(len(swbData)):
            writer.writerow(swbData[eidx])

    #copy files according to config.py
    print(f"sentaurus project collection: Copying files to the filing folder '{fd}'.")
    filingFolder = os.path.realpath(fd)
    file_sAndCmdFiles(filingFolder, wd, cmdFiles, child=child)

    print(f"sentaurus project collection: Success : Sentaurus project '{wd}' collected.")

    inheritenceData = [hostname, user, sVersion, runTime, parentOptNode]
    return inheritenceData


#------------------------------------------------------------------------------------------------------------------

def collectAndSave_SProj_DB(wd, sfd, dbSaveName,  inheritedProjectData=None):
    """
    Collect and copy a project's run info and files (sCollectProject) and save a database from the pyCollection_Info.csv file.
    :param wd: [str] path to the Sentaurus project working directory
    :param sfd: [str] path to the sentaurus file filing directory (sfdir)
    :param dbSaveName: [str] save path for the database
    :param inheritedProjectData: [list] a list of project info needed for logless children in running optimization families ([hostname (str), user (str), sVersion (str), runTime (str), parent opt node (int)])
    :return: db: [DataBase] the DataBase object
    :return inheritanceProjectData: [list] a list of project info to be passed to logless children in running opt families (see above)
    """
    inheritanceProjectData = sCollectProject(wd, sfd, inheritedProjectData)
    db = dbh.DataBase(dbFile=os.path.join(sfd,f"{sFileNameBody}_Info.csv"))
    db.dbSaveFile(saveName=dbSaveName)
    return db, inheritanceProjectData



#------------------------------------------------------------------------------------------------------------------


def collectAndSave_SParentChild_DBs(parent_wd, familyDescriptor=None, childPullTimeIdx=childPullTimeIdx_default):
    """
    Collect the files and run info from an optimization family, save to databases.
        Run sCollectProject on parent and children
        Copy the files to the sentaurus files directory (sfdir/familyDescriptor)
        Create databases from the pyCollection_Info.csv files in all the projects (dbdir/familyDescriptor)
        Create merged database from the childrem
        Create a opt-start-pts DB by merging all initial expts from the children
    :param parent_wd: [string] path to the parent sentaurus folder
    :param familyDescriptor: [string] descriptor string for the save folder and filenames
    :param childPullTimeIdx: [int] index of the child file path globs to use, usually want last (-1)
    :return: pdb, [cdbs,], mergecdb: [DataBases] DB objects for the parent, children (list), and merged children
    """

    # prep optimization family run name if not given and create folders if not already existing
    if familyDescriptor == None:
        familyDescriptor = f"{time.strftime(timeStr)}_{os.path.split(parent_wd)[1]}_genoptCollection"

    print(
        f"\ncollect s parent and children and save: Running collection '{familyDescriptor}' for parent project '{os.path.split(parent_wd)[1]}'")

    sfd = os.path.join(os.path.abspath(sfdir), familyDescriptor)  # sentaurus file directory for the family
    if not os.path.isdir(sfd):
        os.makedirs(sfd)

    dbd = os.path.join(os.path.abspath(dbdir), familyDescriptor)  # database directory for the family
    if not os.path.isdir(dbd):
        os.makedirs(dbd)

    print(f"collect s parent and children and save: Sentaurus files directory '{sfd}'")
    print(f"collect s parent and children and save: Database files directory '{dbd}'")

    # Get children info
    # get genopt nodes (pull all nodes associated with results/nodes/*/*_opt.out files)
    genoptNodes = get_genoptNodes(parent_wd)

    # find child project folder locations, check if present, save, pull runtime and opt node # if so
    print(
        f"collect s parent and children and save: Checking on children {genoptNodes} with timestamp idx {childPullTimeIdx}.")
    # uses the project name and genopt node numbers to find all possible child project folders for each node
    # choose the most recent (childPullTimeIdx=-1) or other (index) of possible child folders
    childFolders, childOptNodes, childRuntimes, childComplete = [], [], [], []
    with open(os.path.join(parent_wd, "gvars.dat"), 'r') as file:
        pargvars = file.read()
    for nidx in range(len(genoptNodes)):
        # check if node number is present in parent gvars, skip if not (child optimization is not complete)
        searchPath = os.path.join(parent_wd, f"..", f"{os.path.split(parent_wd)[1]}_{genoptNodes[nidx]}_*")
        # print("pull children search path:", os.path.join(parent_wd, "..", f"{os.path.split(parent_wd)[1]}_{genoptNodes[nidx]}_*"))
        # print("search glob:", glob.glob(os.path.join(parent_wd, "..", f"{os.path.split(parent_wd)[1]}_{genoptNodes[nidx]}_*")))
        if len(glob.glob(searchPath)) != 0:
            childFolders.append(glob.glob(searchPath)[childPullTimeIdx])
            childstr = os.path.split(childFolders[-1])[1][len(os.path.split(parent_wd)[1]) + 1:]
            childOptNodes.append(int(childstr[: childstr.find("_")]))
            rT = time.strptime(childstr[childstr.find("_") + 1:], '%Y-%m-%d_%H.%M.%S')
            childRuntimes.append(time.strftime(timeStr, rT))
            print(f"collect s parent and children and save: Child located '{glob.glob(searchPath)[childPullTimeIdx]}'")
        else:
            print(
                f"WARNING :collect s parent and children and save: Could not find child project at '{searchPath}'. Skipping node.")

        if not f" {genoptNodes[nidx]} " in pargvars:
            print(
                f"WARNING :collect s parent and children and save: Child project '{genoptNodes[nidx]}' has no results in the parent's gvars file. Assuming optimization is incomplete and addending child folder and DB filename...")
            childComplete.append(False)
        else:
            childComplete.append(True)

    # run pyProjectCollector and createAndSaveDb for the parent
    print(f"collect s parent and children and save: Collecting projects...")
    pdb, inheritance = collectAndSave_SProj_DB(parent_wd, os.path.join(sfd, "parent"),
                                               os.path.join(dbd, f"{familyDescriptor}_parent.csv"))
    # inheritence = list of [hostname (str), user (str), sVersion (str), runTime (str), parent opt node (int)] passed from parent to child
    # hostname, user and sVer are pulled from the parent and the runtime and opt node number are pulled from the child folder name

    # run pyProjectCollector and createAndSaveDB for each child
    cdbs = []
    for child in range(len(childFolders)):
        if childComplete[child]:
            complstr = ""
        else:
            complstr = "_incomplete"

        childInher = inheritance  # inheritence = [hostname, user, sVer, runtime, parent_opt_node]
        childInher[-2] = childRuntimes[child]
        childInher[-1] = childOptNodes[child]
        cdb, _ = collectAndSave_SProj_DB(childFolders[child],
                                         os.path.join(sfd, f"child_{genoptNodes[child]}{complstr}"), os.path.join(dbd,
                                                                                                                  f"{familyDescriptor}_child_{genoptNodes[child]}{complstr}.csv"),
                                         inheritedProjectData=childInher)
        cdbs.append(cdb)

    # merge all child dbs and save
    mergecdb = dbh.mergeDBs(cdbs, f"{familyDescriptor}_mergedChildren")
    mergecdb.dbSaveFile(saveName=os.path.join(dbd, f"{familyDescriptor}_mergedChildren.csv"))

    # Create start point db (db with all the local optimization starting points), create from first point in each DB
    print(
        f"\ncollect s parent and children and save: Creating optimizations start points database from first experiment in each child... ")
    stPtSeries = [cdbs[i].dataframe.iloc[0] for i in range(len(cdbs))]
    stPtSeries[0] = pd.DataFrame(stPtSeries[0])
    stPtDF = pd.concat(stPtSeries, axis=1, ignore_index=True, join='outer', sort=False)
    stPtDF = stPtDF.transpose(copy=True)
    dbh.cleanDFIdxs(stPtDF)
    starterParent = dbh.DataBase()
    starterParent.lineage.append(
        [time.strftime(timeStr), f"Created automatically during '{familyDescriptor}' optimization family collection"])
    starterParent.dataframe = stPtDF
    starterParent.dbfilename = f"{familyDescriptor}_opt-start-pts.csv"
    starterParent.dataframe.insert(0, 'opt_node', childOptNodes)
    starterParent.dbCleanIdxsAndGridness()
    starterParent.dbSaveFile(saveName=os.path.join(dbd, starterParent.dbfilename))

    # create end pt db
    print(f"\ncollect s parent and children and save: Creating optimizations end points database...")

    # Create end pt database
    # determine number of optimization parameters
    o, p, r = read_optOut(
        os.path.join(parent_wd, "results", "nodes", f"{childOptNodes[0]}", f"n{childOptNodes[0]}_opt.out"))
    numOptParam = p['num_opt_params']

    # determine method (for each cdb)
    if 'optimizer' in pdb.dataframe.columns:
        method = pdb.dataframe['optimizer'].tolist()
        if all(i == method[0] for i in method):
            mstr = method[0]
        else:
            mstr = 'various'
    else:
        method = [optMethod_default] * len(cdbs)
        mstr = optMethod_default

    # check to make sure that the extracted number of methods equals the number of children
    if len(method) < len(cdbs):
        print(
            f"WARNING :collect s parent and children and save: pyCollector has not yet been run for one or more experiments where genopt has started. Assuming default optimization algorithm ({optMethod_default}) for end point pull index.")
        method = method + [optMethod_default] * (len(cdbs) - len(method))
    elif len(method) > len(cdbs):
        print(
            f"WARNING :collect s parent and children and save: pyCollector has been run for one or more experiments where genopt has not started. Shortening end point pull index list accordingly (ASSUMES that these problematic experiments are the ones with highest experiment number!)")
        method = method[0:len(cdbs)]

    # set pull index from method and num opt params
    pullidx = get_pullIdxForOptChildren(method, numOptParam)

    # create end pt db
    print(
        f"\ncollect s parent and children and save: Creating optimizations end points database using expected indicies for opt method '{mstr}' with {numOptParam} opt params...")

    # print(f"method:{len(method)} and pullidx:{len(pullidx)} and cdbs:{len(cdbs)}")
    endPtSeries = [cdbs[i].dataframe.iloc[pullidx[i]] for i in range(len(cdbs))]

    if not all(
            childComplete):  # need to create a database with incomplete end pts (to match start pts db) and one with the incomplete children removed
        endPtSeries_inclIncomplete = endPtSeries.copy()
        endPtSeries_inclIncomplete[0] = pd.DataFrame(endPtSeries_inclIncomplete[0])
        endPtDF_inclIncompl = pd.concat(endPtSeries_inclIncomplete, axis=1, ignore_index=True, join='outer', sort=False)
        endPtDF_inclIncompl = endPtDF_inclIncompl.transpose(copy=True)
        enderParent_inclIncompl = dbh.DataBase()
        enderParent_inclIncompl.lineage.append([time.strftime(timeStr),
                                                f"Created automatically during '{familyDescriptor}' optimization family collection"])
        enderParent_inclIncompl.dataframe = endPtDF_inclIncompl
        enderParent_inclIncompl.dbfilename = f"{familyDescriptor}_opt-end-pts_incl-incomplete.csv"
        enderParent_inclIncompl.dataframe.insert(0, 'opt_node', childOptNodes)
        enderParent_inclIncompl.dbCleanIdxsAndGridness()
        enderParent_inclIncompl.dbSaveFile(saveName=os.path.join(dbd, enderParent_inclIncompl.dbfilename))

        # remove incomplete children from this end points database
        print(
            f'collect s parent and children and save: There are incomplete children, creating two end point databases (with and without incomplete children removed).')
        endPtSeries = [c for (c, remove) in zip(endPtSeries, childComplete) if remove]
        childOptNodes = [c for (c, remove) in zip(childOptNodes, childComplete) if remove]

    if len(endPtSeries) != 0:
        endPtSeries[0] = pd.DataFrame(endPtSeries[0])
        endPtDF = pd.concat(endPtSeries, axis=1, ignore_index=True, join='outer', sort=False)
        endPtDF = endPtDF.transpose(copy=True)
        enderParent = dbh.DataBase()
        enderParent.lineage.append([time.strftime(timeStr),
                                    f"Created automatically during '{familyDescriptor}' optimization family collection"])
        enderParent.dataframe = endPtDF
        enderParent.dbfilename = f"{familyDescriptor}_opt-end-pts.csv"
        enderParent.dataframe.insert(0, 'opt_node', childOptNodes)
        enderParent.dbCleanIdxsAndGridness()
        enderParent.dbSaveFile(saveName=os.path.join(dbd, enderParent.dbfilename))
    else:
        print(
            f"WARNING :collect s parent and children and save: All optimizations are incomplete. Skipping the creation of an end point database without incomplete children.")

    print(
        f"collect s parent and children and save: Success :Sentaurus files transferred to '{sfd}', dbs saved to '{dbd}'")
    return pdb, cdbs, mergecdb

#------------------------------------------------------------------------------------------------------------------
#------------------------------------------------------------------------------------------------------------------
#!    SEND TO AND RUN SENTAURUS
#------------------------------------------------------------------------------------------------------------------
#------------------------------------------------------------------------------------------------------------------


def run_gridInSingleSentaurusProject(gridDB, wd, optNum="", maxExptsPerProj=25, runDescriptor=None, templateGtree=None,
                                     lockedTools='find', gsubGEXPR='all', gsubOptionsStr='-verbose',
                                     gcleanupOptionsStr='-verbose -default -ren', overwriteRunning=False,
                                     logOutput=True, hushBashOutput=True, overwriteFiledFiles=True,
                                     saveFoldersSubFolder=None):
    """
    Run a gridDB in a single sentaurus project.
        The gridDB is split into subgrids according to the maxExptsPerProj.
        For each subgrid; write gtree in the wd, gcleanup, gsub, collect and save project and DB in folders given by runDescriptor
        At the end; create a DB for all expts in the gridDB by merging the results from each subgrid

    :param gridDB: Database, the grid DB object with the parameters of the experiments to run
    :param wd: string, path to the sentaurus working directory (wildcards accepted since terminal command, but must lead to a single project directory)
    :param optNum: string for optimization number differentiation
    :param maxExptsPerProj: int, the maximum number of experiments to include in a project run, if the grid has more expts than this value sequential project runs are used
    :param runDescriptor: string, the descriptor to use for the sentaurus_files and databases save location
    :param templateGtree: string or None, path to the gtree writer fnc's input gtree, used to fill missing parameters (and to extract locked tools list if lockedTools='find')
                                            if None uses the gtree in the wd
    :param lockedTools: one of [0, 1, list 0/1 len # tools in proj, 'find'], the lock state of the tools (0 unlocked, 1 locked)
    :param gsubGEXPR: string, the gexpr for the gsub command, for more info see sentaurus_gsub
    :param gsubOptionsStr: string or None, the options for the gsub command, for more info see sentaurus_gsub
    :param gcleanupOptionsStr: string or None, the options for the gcleanup command, for more info see sentaurus_gcleanup
    :param overwriteRunning: boolean, whether or not to run the command if the project is in the running state
    :param logOutput: boolean, whether or not to log the bash terminal output to a file
                                if True; logs for each project run in the project-folder/aiirmap-bash-output_gcleanup/gsub.log file
                                        And, logs for the full grid run, saved to sentaurus_files/runDescriptor/aiirmap-bash-output_grid-runner_runDescriptor.log
    :param hushBashOutput: boolean, whether or not to hush the output of the PIPE in the python terminal (if True then no bash output in python terminal)
    :param overwriteFiledFiles: boolean, whether to overwrite {runDescriptor} sentaurus_files and databases folders if they already exist (True) or append {runDescriptor} with datetime (False)
    :param saveFoldersSubFolder: str or none, sub-folder(s) to save results to
                                    if None results are saved to os.path.join(xdir,runDescriptor) w/ xdir=dbdir and sfdir,
                                    otherwise uses os.path.join(xdir, saveFolderSubFolder, runDescriptor)
    :return: mergedDB: Database, the database containing the results from all experiments in the grid
    """

    # TODO; add time to the terminal output at the start of each subgrid run

    # set inputs to defaults for inputs which are based on other inputs
    if runDescriptor == None:
        runDescriptor = gridDB.dbfilename[0:-4]
    if templateGtree == None:
        templateGtree = os.path.join(wd, "gtree.dat")

    # handle save folder
    if saveFoldersSubFolder != None:
        dbsavefolder = os.path.join(dbdir, saveFoldersSubFolder)
        sfsavefolder = os.path.join(sfdir, saveFoldersSubFolder)
    else:
        dbsavefolder = dbdir
        sfsavefolder = sfdir

    # check if filing cabinet folders with the name runDescriptor exist and append with datetime if overwriteFiledFiles is False
    if (os.path.isdir(os.path.join(dbsavefolder, runDescriptor)) or os.path.isdir(
            os.path.join(sfsavefolder, runDescriptor))) and optNum == "":
        if overwriteFiledFiles:
            print(
                f"\nWARNING :run grid in one s-project: A sentarus_files or databases folder with the name '{runDescriptor}' already exists, overwriting...")
            if os.path.isdir(os.path.join(dbsavefolder, runDescriptor)):
                shutil.rmtree(os.path.join(dbsavefolder, runDescriptor))
            if os.path.isdir(os.path.join(sfsavefolder, runDescriptor)):
                shutil.rmtree(os.path.join(sfsavefolder, runDescriptor))
        else:
            print(
                f"\nWARNING :run grid in one s-project: A sentarus_files or databases folder with the name '{runDescriptor}' already exists, saving to '{runDescriptor}_{time.strftime(timeStr)}'.")
            runDescriptor = f'{runDescriptor}_{time.strftime(timeStr)}'

    print(
        f"\nrun grid in one s-project: Running gridDB '{gridDB.dbfilename}' in sentaurus project '{wd}' (runDescriptor={runDescriptor}...  I fight for the Users!)\n"
        f"run grid in one s-project: Run options; overwriteRunning={overwriteRunning}, lockedTools={lockedTools}, gsubGEXPR={gsubGEXPR}, gsubOptionsStr={gsubOptionsStr}, gcleanupOptionsStr={gcleanupOptionsStr}, logOutput={logOutput}")

    # check project status
    [_, _, _, status, _] = read_dotStatus(wd)
    if status.lower() == 'running':
        if overwriteRunning:
            print(
                f"WARNING :run grid in one s-project: Project at '{wd}' is Running. overwriteRunning=True. Stopping the current project simulation run...")
        else:
            print(
                f"ERROR :run grid in one s-project: Project at '{wd}' is Running. Cannot start grid run (overwriteRunning=False). Cancelling...")
            return None

    # find out how many project runs are going to be needed to complete the whole grid
    totExpts = gridDB.dataframe.shape[0]
    numProjRuns = int(np.ceil(totExpts / maxExptsPerProj))
    print(
        f"run grid in one s-project: The input grid has {totExpts} experiments and the max experiments per project run is set to {maxExptsPerProj}. {numProjRuns} sequential project runs will be used.")

    # backup the template gtree if it is going to be overwritten during operation
    if os.path.abspath(os.path.split(templateGtree)[0]) == os.path.abspath(wd) and os.path.split(templateGtree)[
        1] == 'gtree.dat':
        templateGtree = shutil.copy2(templateGtree,
                                     os.path.join(wd, f"templateGtree_forGridRunner.dat"))  # _{runtime}.dat"))

    # set log filename if logging and name not given, use the same logFile for all runs
    if logOutput:
        if not os.path.exists(os.path.join(sfsavefolder, runDescriptor)):
            os.makedirs(os.path.join(sfsavefolder, runDescriptor))
        gridRunnerLogFilePath = os.path.join(sfsavefolder, runDescriptor,
                                             f"aiirmap-bash-output_grid-runner_{runDescriptor}.log")  # _{runtime}.log")
        print(f"run grid in one s-project: Log filename (for all runs); '{gridRunnerLogFilePath}'")
        gridRunnerLog = open(gridRunnerLogFilePath, 'a')

    # process each project run
    runDBs = []  # output DBs
    for ridx in range(numProjRuns):
        print(
            f"\nrun grid in one s-project: Starting project run {str(ridx + 1).zfill(len(str(numProjRuns)))} of {numProjRuns}.")
        # create temporary sub-grid gridDB
        tmpGridDB = dbh.DataBase(
            dbfilename=f'temp_grid_db_{str(ridx + 1).zfill(len(str(numProjRuns)))}_of_{numProjRuns}')
        if ridx is not numProjRuns - 1:
            tmpGridDB.dataframe = gridDB.dataframe.iloc[ridx * maxExptsPerProj:(ridx + 1) * maxExptsPerProj]
        else:
            tmpGridDB.dataframe = gridDB.dataframe.iloc[ridx * maxExptsPerProj:totExpts]

        print(
            f"run grid in one s-project: Project run {str(ridx + 1).zfill(len(str(numProjRuns)))} of {numProjRuns}. Number of experiments in run: {tmpGridDB.dataframe.shape[0]}")

        # write gtree
        write_gtreeFromGrid(tmpGridDB, templateGtree, lockedTools, os.path.join(wd, "gtree.dat"))

        # run gcleanup
        exitcode = sentaurus_gcleanup(wd, gcleanupOptionsStr, overwriteRunning=overwriteRunning, logOutput=logOutput,
                                         logFile=None, hushBashOutput=hushBashOutput)
        if logOutput:
            with open(os.path.join(wd, 'aiirmap-bash-output_gcleanup.log'), 'r') as runLog: runlogInfo = runLog.read(-1)
            gridRunnerLog.write(runlogInfo + '\n')
        if exitcode != 0:
            print(
                f"WARNING :run grid in one s-project: gcleanup failed (exitcode={exitcode}) for run {str(ridx + 1).zfill(len(str(numProjRuns)))} of {numProjRuns}, trying again.")
            exitcode = sentaurus_gcleanup(wd, gcleanupOptionsStr, overwriteRunning=overwriteRunning,
                                             logOutput=logOutput, logFile=None, hushBashOutput=hushBashOutput)
            if logOutput:
                with open(os.path.join(wd, 'aiirmap-bash-output_gcleanup.log'),
                          'r') as runLog: runlogInfo = runLog.read(-1)
                gridRunnerLog.write(runlogInfo + '\n')
            if exitcode != 0:
                print(
                    f"ERROR :run grid in one s-project: gcleanup failed again (exitcode={exitcode}) for run {str(ridx + 1).zfill(len(str(numProjRuns)))} of {numProjRuns}. Cannot continue, exiting...")
                return None

        # run gsub

        exitcode = sentaurus_gsub(wd, gexpr=gsubGEXPR, optionsStr=gsubOptionsStr, overwriteRunning=overwriteRunning,
                                     logOutput=logOutput, logFile=None, hushBashOutput=hushBashOutput)
        if logOutput:
            with open(os.path.join(wd, 'aiirmap-bash-output_gsub.log'), 'r') as runLog: runlogInfo = runLog.read(-1)
            gridRunnerLog.write(runlogInfo + '\n')
        if exitcode > 1:
            print(
                f"WARNING :run grid in one s-project: gsub failed (exitcode={exitcode}) for run {str(ridx + 1).zfill(len(str(numProjRuns)))} of {numProjRuns}, cleaning up and trying again.")
            exitcode = sentaurus_gcleanup(wd, gcleanupOptionsStr, overwriteRunning=overwriteRunning,
                                             logOutput=logOutput, logFile=None, hushBashOutput=hushBashOutput)
            if logOutput:
                with open(os.path.join(wd, 'aiirmap-bash-output_gcleanup.log'),
                          'r') as runLog: runlogInfo = runLog.read(-1)
                gridRunnerLog.write(runlogInfo + '\n')
            if exitcode != 0:
                print(
                    f"ERROR :run grid in one s-project: gcleanup failed after a gsub fail (exitcode={exitcode}), for run {str(ridx + 1).zfill(len(str(numProjRuns)))} of {numProjRuns}. Cannot continue, exiting...")
                return None
            exitcode = sentaurus_gsub(wd, gexpr=gsubGEXPR, optionsStr=gsubOptionsStr,
                                         overwriteRunning=overwriteRunning, logOutput=logOutput, logFile=None,
                                         hushBashOutput=hushBashOutput)
            if logOutput:
                with open(os.path.join(wd, 'aiirmap-bash-output_gsub.log'), 'r') as runLog: runlogInfo = runLog.read(-1)
                gridRunnerLog.write(runlogInfo + '\n')
            if exitcode > 1:
                print(
                    f"ERROR :run grid in one s-project: gsub failed again (exitcode={exitcode}) for run {str(ridx + 1).zfill(len(str(numProjRuns)))} of {numProjRuns}. Cannot continue, exiting...")
                return None

        # collect the project
        outDB, _ = collectAndSave_SProj_DB(wd, os.path.join(sfsavefolder, runDescriptor,
                                                            f"{runDescriptor}_ProjectRun_{str(ridx + 1).zfill(len(str(numProjRuns)))}_of_{numProjRuns}"),
                                           os.path.join(dbsavefolder, runDescriptor, optNum,
                                                        f"{runDescriptor}_ProjectRun_{str(ridx + 1).zfill(len(str(numProjRuns)))}_of_{numProjRuns}.csv"))

        runDBs.append(outDB)

    print(
        f"\nrun grid in one s-project: Completed all project runs! Now copying input grid and merging collected output DBs...")
    if False:
        try:
            shutil.copy(gridDB.dbFile, os.path.join(dbsavefolder, runDescriptor))
        except FileNotFoundError:
            print(
                f"WARNING :run grid in one s-project: Could not find the input grid to copy. Seems that its dbFile attribute is inaccurate ('{gridDB.dbFile}'). Cannot copy.")

    mergedDB = dbh.mergeDBs(runDBs, fileDescriptor=runDescriptor)
    mergedDB.dbSaveFile(saveName=os.path.join(dbsavefolder, runDescriptor, optNum, f"{runDescriptor}_SURVEYED.csv"))

    print(f"run grid in one s-project: Success :Completed gridDB '{gridDB.dbfilename}' sentaurus run... User!")

    return mergedDB


# ------------------------------------------------------------------------------------------------------------------


def write_gtreeFromGrid(gridDB, gtreeIn, lockedTools, gtreeOut):
    """
    Writes a new gtree.dat file based upon the parameters and tools of the input gtree and the experiments (parameter values) of the grid database.
        Locked tools can be specified (all 0, all 1, list of 0/1 of len of tools in project) or an attempt can be made to extract the tool's lock states from the gtree itself ('find')
        The input gtree's parameter list is taken as truth. If a parameter is missing from the grid, the input gtree's default is used for all experiments.
        The output gtree will have all possible nodes in the tree present except for the first param/input * which is shared for all experiments (* ie. the one defining the first tool).
    :param gridDB: DataBase, grid database object with the inputs to run in the experiment (parameter names must match gtreeIn's)
    :param gtreeIn: string, path to the input gtree.dat file. read_gtree will be used to extract parameter and tool information and locked tool states if selected.
    :param lockedTools: one of [0, 1, list 0/1 len # tools in proj, 'find'], the lock state of the tools (0 unlocked, 1 locked)
    :param gtreeOut: string, path for the output gtree.dat file
    :return: None: Writes gtreeOut.
    """

    print(
        f"write gtree: Writing gtree to '{gtreeOut}' from grid '{gridDB.dbfilename}' and input gtree '{gtreeIn}' using lockedTools '{lockedTools}'.   ")

    # pull parameter and project tool info from input gtree
    params, pdefs, ptypes, ptools, toolLists, inLockedTools = read_gtree(gtreeIn, True)

    # handle locked tools input
    if lockedTools == 0:  # all unlocked
        lockedTools = np.zeros(len(toolLists))
    elif lockedTools == 1:  # all locked
        lockedTools = np.ones(len(toolLists))
    elif isinstance(lockedTools, int):
        print(
            f"WARNING :write gtree: lockedTools value ('{lockedTools}') is not valid. Please use [0,1,'find',list of len({len(toolLists)})]. Using gtree locked states '{inLockedTools}'.")
        lockedTools = inLockedTools
    elif isinstance(lockedTools, list):
        if len(lockedTools) == len(toolLists):
            lockedTools = lockedTools
        else:
            print(
                f"WARNING :write gtree: lockedTools '{lockedTools}' list length should be the same as toolLists. Please use [0,1,'find',list of len({len(toolLists)})]. Using gtree locked states '{inLockedTools}'.")
            lockedTools = inLockedTools
    elif lockedTools.lower() == 'find' or lockedTools.lower() == 'f':
        lockedTools = inLockedTools
    else:
        print(
            f"WARNING :write gtree: lockedTools value '{lockedTools}' is not valid. Please use [0,1,'find',list of len({len(toolLists)})]. Using gtree locked states '{inLockedTools}'.")
        lockedTools = inLockedTools

    # extract parameters from the input grid dataframe
    # for each param, extract the values list and create the formatted string
    paramValuesLists = []  # lists of values for each parameter
    paramSimFlowStrings = []  # space-separated strings of parameter values for sim flow section
    for pidx in range(len(params)):
        try:
            plist = gridDB.dataframe[params[pidx]].tolist()
        except KeyError:  # gtree param not in the grid file
            listLen = len(gridDB.dataframe.iloc[:, 0].tolist())
            plist = np.full(listLen, pdefs[pidx])
            print(
                f"WARNING :write gtree: Input gtree parameter '{params[pidx]}' does not exist in the grid DB. Using default value ('{pdefs[pidx]}').")

        string = f"{ptools[pidx]} {params[pidx]} \"{plist[0]}\" " + "{"
        for lidx in range(len(plist)):
            string += str(plist[lidx]) + " "
        string = string[:-1] + "}\n"

        paramValuesLists.append(plist)
        paramSimFlowStrings.append(string)

    numExpts = len(paramValuesLists[0])

    # write new gtree file
    with open(gtreeOut, 'w', newline='', encoding='UTF8') as file:

        file.write("# --- simulation flow\n")

        # write sim flow using the prepared strings
        # prepare data for sim tree writing (interweave tool lines into param list)
        pidx = 0
        treeData = []  # list of param info; values, lock   ... includes the tools
        toolFillData = np.full(numExpts, "")  # empty values for tool nodes
        for tidx in range(len(toolLists)):
            # write tool line
            file.write(f'{toolLists[tidx][0]} {toolLists[tidx][1]} \"\" ' + "{}\n")
            treeData.append([toolFillData, lockedTools[tidx]])
            # write param lines using toolLists[2]=numAssocParams value
            for pidx2 in range(toolLists[tidx][2]):
                file.write(paramSimFlowStrings[pidx])
                treeData.append([paramValuesLists[pidx], lockedTools[tidx]])
                pidx += 1

        if pidx != len(params):
            print(
                f"ERROR :write gtree: Did not write all parameters (total={len(params)},written={pidx}). Please review the gtree file.")

        # write vars and scens portion
        file.write("# --- variables\n")
        file.write("# --- scenarios and parameter specs\n")
        for pidx in range(len(params)):
            file.write(f"scenario default {params[pidx]} \"\"\n")

        # write sim tree
        file.write("# --- simulation tree\n")
        file.write("0 1 0 {} {default} 0\n")

        numSwbRows = len(treeData)
        nidx = 2  # current node to write

        for eidx in range(len(paramValuesLists[0])):  # experiment index
            for ridx in range(
                    numSwbRows - 1):  # row index = tools&params / tree index (swb rows when expts are columns)
                if ridx == 0:
                    # write first row
                    file.write(
                        f"{ridx + 1} {nidx} {ridx + 1}" + " {" + f"{treeData[ridx + 1][0][eidx]}" + "} {default} " + f"{int(treeData[ridx + 1][1])}\n")
                    nidx += 1
                else:
                    # write rest of rows
                    file.write(
                        f"{ridx + 1} {nidx} {nidx - 1}" + " {" + f"{treeData[ridx + 1][0][eidx]}" + "} {default} " + f"{int(treeData[ridx + 1][1])}\n")
                    nidx += 1
            nidx += 1

    print(f"write gtree: Locked tools used; '{lockedTools}'.")

    print(f"write gtree: Success :Wrote new gtree file '{gtreeOut}'.")

    return None


# ------------------------------------------------------------------------------------------------------------------

def sentaurus_gcleanup(wd, optionsStr='-verbose -default -ren', overwriteRunning=False, logOutput=False, logFile=None, hushBashOutput=False):
    """
    Wrapper for the sentaurus terminal command gcleanup.

    :param: wd: string, path to the sentaurus project working directory (wildcards accepted since terminal command)
    :param: optionsStr: string or None, the gcleanup options to run, must be properly formatted for bash
    :param: overwriteRunning: boolean, whether or not to run the command if the project is in the running state
    :param logOutput: boolean, whether or not to log the bash terminal output to a file
    :param logFile: string or None, path to the file to save the bash terminal output if logOutput==True
                                    None defaults to wd/aiirmap-bash-output_gcleanup.log
    :param hushBashOutput: boolean, whether or not to hush the output of the PIPE in the python terminal (if True then no bash output in python terminal)
    :return: subprocess.CompletedProcess.returncode: int, the returncode of the command (typically 0 for no errors, <0 for errors)
    """

    print(f"sentaurus gcleanup: Running gcleanup for working directory '{wd}' (optionsStr={optionsStr}, logOutput={logOutput})")
    #check status and overwriteRunning
    [_,_,_,status,_] = read_dotStatus(wd)
    if status.lower() == 'running':
        if overwriteRunning:
            print(f"WARNING :sentaurus gcleanup: Project at '{wd}' is Running. overwriteRunning=True. Cleaning up, this will stop the project simulation run.")
        else:
            print(f"ERROR :sentaurus gcleanup: Project at '{wd}' is Running. Cannot cleanup (overwriteRunning=False). Command cancelled.")
            return -1

    #prep command string
    cmdStr = "gcleanup "
    if optionsStr is None:
        optionsStr = ""
    cmdStr += optionsStr.strip()
    cmdStr += " "
    cmdStr += wd
    #no &, process is Not run in the background (not currently supported (230118))

    #set log filename if logging and name not given
    if logOutput and logFile is None:
        logFile = os.path.join(wd, f"aiirmap-bash-output_gcleanup.log")#_{time.strftime(timeStr, time.localtime())}.log")

    #run command
    returncode = execute_bashCommand(cmdStr, logOutput=logOutput, logFile=logFile, hushBashOutput=hushBashOutput)

    if returncode == 0:
        print(f"sentaurus gcleanup: Success :gcleanup exited with returncode 0 (execute terminal command output duplicate).")
    else:
        print(f"ERROR :sentaurus gcleanup: gcleanup exited with returncode '{returncode}' (execute terminal command output duplicate).")

    return returncode



#------------------------------------------------------------------------------------------------------------------

def sentaurus_gsub(wd, gexpr='all', optionsStr='-verbose', overwriteRunning=False, logOutput=False, logFile=None, hushBashOutput=False):
    """
    Wrapper for the bash terminal command gsub
        Default run method is to run -e ie. run the nodes given by the gexpr input
        However; if optionsStr includes -n; then run the node list given therein and the -e run method is NOT used
                 if optionStr includes -e; then use the gexpr defined therein and not the one given by the gexpr input
                 if optionsStr includes both -n and -e then it will fail... make a choice and edit your input

    :param wd: string, path to the sentaurus working directory (wildcards accepted since it is a bash terminal command)
    :param gexpr: string, the sentaurus gexpr to define which nodes to run, for more information see the swb manual
    :param optionsStr: string or None, the gsub options to run (see note immd above), must be properly formatted for bash
    :param overwriteRunning: boolean, whether or not to run the command if the project is in the running state
    :param logOutput: boolean, whether or not to log the bash terminal output to a file
    :param logFile: string or None, path to the file to save the bash terminal output if logOutput==True
                                    None defaults to wd/aiirmap-bash-output_gsub.log
    :param hushBashOutput: boolean, whether or not to hush the output of the PIPE in the python terminal (if True then no bash output in python terminal)
    :return: subprocess.CompletedProcess.returncode: int, the returncode of the command (typically 0 for no errors, <0 for errors)
    """

    print(f"sentaurus gsub: Running gsub for working directory '{wd}' (gexpr={gexpr}, optionsStr={optionsStr}, logOutput={logOutput})")

    # check status and overwriteRunning
    [_, _, _, status, _] = read_dotStatus(wd)
    if status.lower() == 'running':
        if overwriteRunning:
            print(f"WARNING :sentaurus gsub: Project at '{wd}' is Running. overwriteRunning=True. Starting new gsub, this will stop the current project simulation run.")
        else:
            print(f"ERROR :sentaurus gsub: Project at '{wd}' is Running. Cannot submit new gsub (overwriteRunning=False). Command cancelled.")
            return -1

    # prep command string
    cmdStr = 'gsub '
    if optionsStr is None:
        optionsStr = ''
    cmdStr += optionsStr.strip()
    if '-n' in optionsStr or '-e' in optionsStr:
        pass
    else:
        cmdStr += f" -e {gexpr.strip()}"
    cmdStr += " "
    cmdStr += wd
    #running in background not currently supported (230118)

    # set log filename if logging and name not given
    if logOutput and logFile is None:
        logFile = os.path.join(wd, f"aiirmap-bash-output_gsub.log")#_{time.strftime(timeStr, time.localtime())}.log")

    #run command
    returncode = execute_bashCommand(cmdStr, logOutput=logOutput, logFile=logFile, hushBashOutput=hushBashOutput)

    if returncode == 0:
        print(f"sentaurus gsub: Success :gsub exited with returncode 0 (execute terminal command output duplicate).")
    elif returncode == 1:
        #probably a failed experiment
        print(f"WARNING :sentaurus gsub: gsub exited with returncode '{returncode}'. There is probably one or more failed experiments in the project. (execute terminal command output duplicate).")
    else:
        print(f"ERROR :sentaurus gsub: gsub exited with returncode '{returncode}' (execute terminal command output duplicate).")

    return returncode


#------------------------------------------------------------------------------------------------------------------


def execute_bashCommand(cmd, logOutput=False, logFile=None, hushBashOutput=False):
    """
    Execute a bash terminal command using the python subprocess module.
        Opens a PIPE between the bash terminal and the python terminal, outputs the bash STDOUT and STDERR into the python terminal
        And(/or) writes the output to the logFile if logOutput is true

        For more info see https://docs.python.org/3/library/subprocess.html

    :param cmd: string or list of strings, Popen command input. See subprocess documentation for more info.
                                        if a string; must be full command properly formatted,
                                        if a list; each string item is a single argument (formatting handled by system)
    :param logOutput: boolean, whether or not to save the output to a log file
    :param logFile: string or None, the path to the file in which the terminal output will be saved (will be overwritten if existent)
                                    None defaults to cwd/aiirmap-bash-output.log
    :param hushBashOutput: boolean, whether or not to hush the output of the PIPE in the python terminal (if True then no bash output in python terminal)
    :return: subprocess.CompletedProcess.returncode: int, the returncode of the command (typically 0 for no errors, <0 for errors)
    """


    print(f"execute terminal command: Running terminal command '{cmd}' with logOutput set to {logOutput}.")

    # open logfile to save output into
    if logOutput:
        if logFile is None:
            logFile = os.path.join(os.getcwd(),f"aiirmap-bash-output.log")#_{time.strftime(timeStr, time.localtime())}.log")
        fileForLog = open(logFile, "w")
        fileForLog.write(f"\n===================\n{time.strftime(timeStr, time.localtime())}\nCommand;{cmd}\n===================")
        print(f"execute terminal command: Log file is '{logFile}'")

    #get piped process running
    if isinstance(cmd, str):
        #run with shell=True
        popen = subprocess.Popen(cmd, shell=True, universal_newlines=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    else:
        popen = subprocess.Popen(cmd, universal_newlines=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    print(f"execute terminal command: '{cmd}' started. Process id is {popen.pid}.")

    #monitor output, send to python terminal, write to file.
    if not hushBashOutput:
        print(f"\nexecute terminal command: Terminal output begins ======================")
    if logOutput or not hushBashOutput:
        for stdout_line in iter(popen.stdout.readline, ""):
            if not hushBashOutput:
                print(stdout_line.strip())
            if logOutput:
                fileForLog.write(stdout_line)
        popen.stdout.close()
    if not hushBashOutput:
        print(f"execute terminal command: Terminal output ends ======================\n")

    returncode = int(popen.wait())

    #finish up
    if logOutput:
        fileForLog.close()

    if returncode == 0:
        print(f"execute terminal command: Success : Command '{cmd}' finished with return code of '{returncode}'.")
    else:
        print(f"ERROR :execute terminal command: Command '{cmd}' finished with return code of '{returncode}")

    return returncode


#------------------------------------------------------------------------------------------------------------------
#------------------------------------------------------------------------------------------------------------------
#!    SENTAURUS CREATED FILES, READ AND INTERACT FNCS
#------------------------------------------------------------------------------------------------------------------
#------------------------------------------------------------------------------------------------------------------

def read_gtree(filename, pullToolsLock=False):
    """
    Read the swb parameter and tool info from the gtree.dat file's simulation flow section.
    If pullToolsLock; Try to pull the tools' locked states from the last column of each tool's first row in simulation tree portion of the file.
        If the simulation tree is empty, and pullToolsLock=True, the lockedTools are set to all unlocked. (If pullToolsLock=False, lockedTools=[]
    :param: filename: string, path to the gtree.dat file
    :param: pullToolsLock: boolean, try to pull tool's locked state if True
    :return: params: list of strings, the swb parameter names
    :return: pdefs: list of values, the default parameter values read from the file
    :return: ptypes: list of strings, the variable type for each param ("string", "float", or "boolean")
    :return: ptools: list of strings, the labels of the tools that each parameter belongs to (correspondence to params)
    :return: toolLists: list of lists of form [toolLabel,toolType,numAssocParams], information for each tool in the project
    :return: toolsLocked: list of ints, the lock state for each tool (0=unlocked, 1=locked)

    This is the python3 version of this function. It is not compatible with the gsub prologue python (which is py2)
    For the py2 version see py2prepper.py
    Functional changes to this (that) function should be propagated there (here)
    """


    with open(filename, 'r') as inputFile:
        lines = inputFile.readlines()

    params = [] #list of parameter name strings
    pdefs = [] #list of default parameter values
    ptypes = [] #list of the variable type for each parameter ("float" or "string", element-wise correspondence to params)
    ptools = [] #list of the tool each parameter belongs to (same length and element-wise correspondence to params)
    toolLists = [] #[list of [toolLabel, toolType, # of associatiated params]], for each tool in the sim flow, where toolTypes are given by config.py's toolNameList
    toolsLocked = [] #list of tools lock state (0=unlocked, 1=locked)

    simflow = False #track the section of the file we care about
    simTree = False #ditto (for tools' locked state)
    lastToolLidx = 0 #track location of the last line defining a tool ('toolLabel toolType "" {}')
    streeToolLidxs = []#track location of the first line for each tool in the sim tree section (for locked state pull)

    #run lines
    for lidx in range(len(lines)):
        lines[lidx] = lines[lidx][:-1]  # remove \n

        #section start/end checks and actions
        if lines[lidx] == "# --- simulation flow":
            simflow = True
            lastToolLidx = lidx+1
        elif lines[lidx] == "# --- variables":
            simflow = False
            toolLists[-1][2] = lidx - lastToolLidx -1
            if not pullToolsLock: break
        elif lines[lidx] == "# --- simulation tree":
            simTree = True
            streeToolLidxs = [lidx+1]
            for tidx in range(len(toolLists)-1):
                streeToolLidxs.append(streeToolLidxs[-1]+toolLists[tidx][2]+1)
            if lastToolLidx+3 > len(lines):
                if pullToolsLock: print(f"WARNING :read gtree: No simulations in sim tree, could not read tools locked state. Setting all to unlocked.")
                toolsLocked = np.zeros(len(toolLists))
                break

        #sim flow data pull
        elif simflow == True:
            lineList = lines[lidx].split(" ")

            if lineList[1] not in toolNameList: #its a parameter!
                params.append(lineList[1])
                ptools.append(lineList[0])
                lineList[2] = lineList[2][1:-1] #get rid of the "s
                pdefs.append(lineList[2])
                try:
                    lineList[2] = float(lineList[2])
                except ValueError:
                    if lineList[2].lower() in (bool_true_lowerlist + bool_false_lowerlist):
                        ptypes.append("boolean")
                    else:
                        ptypes.append("string")
                else:
                    ptypes.append("float")
            else: #its a tool!
                numAssocParamsForLast = lidx - lastToolLidx - 1
                if len(toolLists) != 0: toolLists[-1][2] = numAssocParamsForLast
                lastToolLidx = lidx
                #save the toolLabel, toolType pair
                toolLists.append([lineList[0], lineList[1], np.nan])

        #sim tree data pull
        elif simTree == True:
            if lidx in streeToolLidxs:
                toolsLocked.append(lines[lidx].split(" ")[-1])
            if lidx > streeToolLidxs[-1]:
                break

        else:
            continue

    if len(toolsLocked) != len(toolLists):
        if pullToolsLock: print(f"WARNING :read gtree: The number of identified locked tool states ({len(toolsLocked)} does not match the number of tools ({len(toolLists)}). Setting all tools to unlocked.")
        toolsLocked = np.zeros(len(toolLists))

    return params, pdefs, ptypes, ptools, toolLists, toolsLocked


#------------------------------------------------------------------------------------------------------------------

def clear_gtree_simTree(gtreeIn, gtreeOut):
    """
    Clear the simulation tree portion of the sentaurus project file 'gtree.dat'
        This has the effect of clearing the experiments from the sentaurus workbench
    :param: gtreeIn: string, path to the input gtree file
    :param: gtreeOut: string, path for the output gtree file
    :return: None, writes the new gtree file to gtreeOut
    """

    print(f"clear gtree sim tree: Clearing the simulation tree section of '{gtreeIn}' and writing to '{gtreeOut}'.")
    with open(gtreeIn, 'r') as inputFile:
        lines = inputFile.readlines()

    for lidx in range(len(lines)):
        if lines[lidx] == "# --- simulation tree":
            lines = lines[0:lidx+1]
            break

    with open(gtreeOut, 'w', newline='', encoding='UTF8') as file:
        for lidx in range(len(lines)):
            file.write(lines[lidx])

    print(f"clear gtree sim tree: Success :The gtree.dat with cleared sim tree has been written to '{gtreeOut}'.")
    return None


#------------------------------------------------------------------------------------------------------------------

def read_glog(wd):
    """
    Obtain key project info from the glog.txt file (located in wd/results/logs/)
    :param wd: Sentaurus project working directory (path string)
    :return: hostname: name of the computer upon which the project is being run (string)
    :return: user: name of the linux user running the project (string)
    :return: sVersion: sentaurus version used (string)
    :return: runTime: Date and time at which the simulation was STARTED (string)
    :return: cmdFiles: the sentaurus tool command files used (list of path strings)
    """
    glogFile = os.path.join(wd, "results","logs","glog.txt")

    if not os.path.exists(glogFile):  # probably part of a family
        print(f"ERROR :sentaurus read glog: glog file does not exist at '{glogFile}'. Returning None.")
        return None, None, None, None, None

    with open(glogFile, 'r') as inputFile:
        lines = inputFile.readlines()
    cmdFiles= []

    for lidx in range(len(lines)):
        if lines[lidx].startswith("gsub is running on"):
            hostname = lines[lidx].split("'")[1]
            user =  lines[lidx].split("'")[3]
        elif lines[lidx].startswith("\tSTROOT="):
            sVersion = lines[lidx][lines[lidx].find("sentaurus"):lines[lidx].find("/bin")]
        elif "submitted to the batch system" in lines[lidx]:
            runTime = lines[lidx][0:lines[lidx].find("<")-1]
            rT = time.strptime(runTime, '%H:%M:%S %b %d %Y')
            runTime = time.strftime(timeStr, rT)
        elif lines[lidx].startswith("Reading file"):
            cmdFiles.append(lines[lidx][lines[lidx].find("/"):-1])
        elif lines[lidx].startswith("PREPROCESSING STEP 2:"):
            break
        else:
            continue

    return hostname, user, sVersion, runTime, cmdFiles


# ------------------------------------------------------------------------------------------------------------------

def read_goptlog(wd):
    """
    Obtain key project info from the gopt.log file (located in wd/results/logs/)

    :param wd: Sentaurus project family directory (path string)
    :return: hostname: name of the computer upon which the project is being run (string)
    :return: user: name of the linux user running the project (string)
    :return: sVersion: sentaurus version used (string)
    :return: runTime: Date and time at which the simulation was STARTED (string)
    :return: cmdFiles: the sentaurus tool command files used (list of path strings)
    """
    glogFile = os.path.join(wd, "results","logs", "gopt.log")
    if not os.path.exists(glogFile):  # probably a child
        print(f"ERROR :sentaurus read gopt: Log file does not exist at '{glogFile}'.")
        return None, None, None, None, None

    with open(glogFile, 'r') as inputFile:
        lines = inputFile.readlines()
    cmdFiles = []

    for lidx in range(len(lines)):
        if lines[lidx].startswith("genopt is running on"):
            hostname = lines[lidx].split("'")[1]
            user = lines[lidx].split("'")[3]
        elif lines[lidx].startswith("\tSTROOT="):
            sVersion = lines[lidx][lines[lidx].find("sentaurus"):lines[lidx].find("/bin")]
        elif lines[lidx].startswith("Compiled "):
            runTime = lines[lidx][lines[lidx].find(" ") + 1 : lines[lidx].find(" on ")]

            runTime = interpretPDTDateTime(runTime)

        elif lines[lidx].startswith("Reading file"):
            cmdFiles.append(lines[lidx][lines[lidx].find("/"):-1])
        elif lines[lidx].startswith("PREPROCESSING STEP 2:"):
            break
        else:
            continue

    return hostname, user, sVersion, runTime, cmdFiles


#------------------------------------------------------------------------------------------------------------------

def read_gvars(gvarsFile):
    """
    Read the gvars file and output all info for later assignment
    :param gvarsFile: path to the gvars.dat file (path string)
    :return: See below
    """
    varNodes = [] #list of all unique output variable nodes (list of ints)
    varLabels = [] #list of lists of variable name strings for each node, each inner list corresponds element wise to varNodes
    varValues = [] #list of lists of variable values (floats or strings) for each node, correspondence to varLabels
    uniqueVarLabels = [] #list of unique variable names, ie. variable space dimensions

    with open(gvarsFile, 'r') as inputFile:
        lines = inputFile.readlines()

    #build string lists from files
    for lidx in range(len(lines)):
        if lines[lidx].startswith('""') or lines[lidx].startswith('"hidden"'):
            continue
        lineList = lines[lidx].split(" ")
        if lineList[1] not in varNodes:
            varNodes.append(lineList[1])
            nidx = len(varNodes)-1
            varLabels.append([])
            varValues.append([])
        else:
            nidx = varNodes.index(lineList[1])
        varLabels[nidx].append(lineList[2])
        varValues[nidx].append(lineList[3])
        if lineList[2] not in uniqueVarLabels:
            uniqueVarLabels.append(lineList[2])

    #convert lists to correct variable type
    for vnidx in range(len(varNodes)):
        try: varNodes[vnidx] = int(varNodes[vnidx])
        except ValueError: raise ValueError(
                f"ERROR :read gvars info: Cannot convert {sFileNameBody}_execTimes.csv{varNodes[vnidx]} to int. Was expecting a node number. (Reading output variable nodes.)")
        for vvidx in range(len(varValues[vnidx])):
            try: varValues[vnidx][vvidx] = float(varValues[vnidx][vvidx])
            except ValueError: pass

    return varNodes, varLabels, varValues, uniqueVarLabels


#------------------------------------------------------------------------------------------------------------------

def readExtract_gvars(exptNodes, gvarsFile):
    """
    Match gvars.dat outputs to an experiment using the list of the experiment's nodes.
        If there are duplicates of an output label (unclean gvars) then keeps the latest result.
    :param exptNodes: (list of ints) list of node numbers for an experiment
    :param gvarsFile: (string) path to the sentaurus project gvars.dat file
    :return outputLabels: (list of strings) the labels for the output variables, ordered by appearance
    :return outputVars:  (list) the data values for the output variables (correspondent to outputLabels)
    :return outputTypes: (list) the data type ["boolean","string","float"] for each output
    """

    outputLabels = []
    outputVals = []
    outputTypes = []
    outputTimes = [] #track time of output write; for duplicates, take most recent result only


    with open(gvarsFile, 'r') as inputFile:
        lines = inputFile.readlines()

    if len(lines) <= 1: #empty gvars, no experiments have completed yet
        return [], [], []

    for lidx in range(len(lines)):
        if lines[lidx].startswith('""') or lines[lidx].startswith('"hidden"'):
            continue
        lines[lidx] = lines[lidx][:-1]
        lineList = lines[lidx].split(' ')
        if int(lineList[1]) not in exptNodes:
            continue
        else:
            if lineList[2] in outputLabels: #this output for this expt exists, keep newer
                try: t = int(lineList[0])
                except ValueError: t = 0 #(output is "define")
                if t <= outputTimes[outputLabels.index(lineList[2])]:
                    continue #older data, skip
                else:
                    #data is newer, delete old
                    outputTimes.pop(outputLabels.index(lineList[2]))
                    outputTypes.pop(outputLabels.index(lineList[2]))
                    outputVals.pop(outputLabels.index(lineList[2]))
                    outputLabels.pop(outputLabels.index(lineList[2]))

            outputLabels.append(lineList[2])
            outputVals.append(lineList[3])
            try: t = int(lineList[0])
            except ValueError: t = 0  # (output is "define")
            outputTimes.append(t)
            try:
                val = float(lineList[3])
            except ValueError:
                if lineList[3].lower()  in (bool_true_lowerlist + bool_false_lowerlist):
                    outputTypes.append('boolean')
                else:
                    outputTypes.append('string')
            else:
                outputTypes.append('float')


    #check for duplicates (should not be)
    if not len(outputLabels) == len(set(outputLabels)):
        print(f"ERROR :read extract gvars for expt: Something has gone wrong, there are duplicate labels.\n"
              f"... (expt nodes; '{exptNodes}', extracted output labels; '{outputLabels}').")

        outputLabels, outputVals, outputTypes = [],[],[] #expt nodes not found in gvars or duplicate variable labels (do not collect)


    return outputLabels, outputVals, outputTypes





#------------------------------------------------------------------------------------------------------------------

def readAssign_gvars(outputVarNodes, gvarsFile):
    """
    ! Historical; currently unused (project/swbVars.csv implementation) >> see readExtract_gvars !
    Reads the gvars.dat file to extract the swb output variables and their corresponding nodes.
    Uses the outputVarNodes info from the swbVars file to assign the variables to the correct experiment.
    :param outputVarNodes: list of lists with each expt's nodes with output variables
    :param gvarsFile: path to the gvars.dat file (string)
    :return: uniqueVarLabels: the labels for the output variables (list of strings) - uniqueness relates to the fact that each output is repeated number-of-expts times in gvars.dat
    :return: sortedVarValues: list of lists, see below.
    """

    #Prepare
    varNodes, varLabels, varValues, uniqueVarLabels = read_gvars(gvarsFile)
    sortedVarValues = []  # list of lists where variable/output values are recorded in order for each experiment
    # outer list is experiment number, inner list is the variable/output values in order of uniqueVarLabels
    missingNodesList = [] #nodes which are expected to have associated variables according to pyCol_swbVars but are missing from gvars (track to avoid multiple warnings)


    # extract sorted values from the gvars dat
    for uvidx in range(len(uniqueVarLabels)):

        for eidx in range(len(outputVarNodes)):
            if uvidx == 0:
                sortedVarValues.append([])

            for snidx in range(len(outputVarNodes[eidx])):  # search node idx (from swbVars)

                try:
                    oidx = varNodes.index(outputVarNodes[eidx][snidx]) #outer=node

                except ValueError: # the swb variable node does not appear in gvars nodelists
                    if outputVarNodes[eidx][snidx] not in missingNodesList:
                        missingNodesList.append(outputVarNodes[eidx][snidx])
                        print(f"WARNING :assign gvars: Output for expt #{eidx+1} variable producing node"
                            f" {outputVarNodes[eidx][snidx]} "
                            f"is missing from gvars.dat file. SETTING TO NaN.")
                    else: pass
                    sortedVarValues[eidx].append(np.nan)

                else:
                    try:
                        iidx = varLabels[oidx].index(uniqueVarLabels[uvidx]) #inner=variable
                    except ValueError:  # the variable is not from this node
                        if snidx == len(outputVarNodes[eidx]) - 1: #did not find variable for expt
                            if uniqueVarLabels[uvidx] != 'genopt_RMSDCost': #genopt_RMSDCost is a swb var but is not written to gvars by sentaurus
                               print(f"WARNING :assign gvars: Could not find variable '{uniqueVarLabels[uvidx]}' for expt #{eidx + 1}. SETTING TO NaN.")
                            sortedVarValues[eidx].append(np.nan)

                        continue
                    else:
                        sortedVarValues[eidx].append(varValues[oidx][iidx])
                        break

    return uniqueVarLabels, sortedVarValues


#------------------------------------------------------------------------------------------------------------------

def read_optOut(filePath, verbose=amverbose):
    """
    Read an n###_opt.out file and extract a bunch of information
    :param filePath: (str) path to the _opt.out file
    :param verbose: (bool) More print-to-screen output?
    :return: options: (dict) the optimization options used (eg. method, eps, ftol...)
    :return: params: (dict) the number of optimized parameters, the labels for the parameters optimized, and the evaluations data
    :return: results: (dict) the runtimes and the final results from the optimization (eg. time_start/end, num_iter,...)
    """

    print(f"read _opt.out: Reading data from '{filePath}'...")

    with open(filePath, 'r') as file:
        lines=file.readlines()

    #PREPARE

    options = {}
    results = {}
    labels = ["eval_num"]
    data = []
    #[eval #, opt params, opt targets, opt target errors, opt target nodes, total RMSD]

    #variables for tracking data extraction
    collectingEval=0
    collectingFin = False
    collectingFinVal = False
    end = False


    #RUN
    #run through the lines and process single line and multiline data, use switches to track multiline collection
    for lidx in range(len(lines)):

        ###switches and single line data

        if lines[lidx] == "\n": continue

        # if lines[lidx].startswith("Compiled"):
        #     results['time_start'] = interpretPDTDateTime( lines[lidx][9 : lines[lidx].find(" on")] )

        if lines[lidx].startswith("Current directory"):
            results['wd'] = lines[lidx][lines[lidx].find("'")+1 : -2]
            starttime = lines[lidx].strip()[-20 : -1]
            starttime = time.strptime(starttime, '%Y-%m-%d_%H.%M.%S')
            results['time_start'] = time.strftime(timeStr, starttime)

        elif lines[lidx].startswith("MINIMIZE: Initial solution"):
            #count number of params being optimized using initial solution list length
            numOptParams = lines[lidx][ lines[lidx].find("[")+1 : lines[lidx].find("]") ].split(" ")
            numOptParams = len(numOptParams)
            
        elif lines[lidx].startswith("MINIMIZE: Other parameters:"):
            pparams = lines[lidx][ 29 : lines[lidx].find(", 'jac'") ].split(",")
            for pidx in range(len(pparams)):
                if "{" in pparams[pidx]:
                    pparams[pidx]=pparams[pidx][pparams[pidx].find("{")+1:]
                elif "}" in pparams[pidx]:
                    pparams[pidx]=pparams[pidx][ : pparams[pidx].find("}")]
                ps = pparams[pidx].replace("'", " ").split(":")#.strip()
                try:
                    options[ps[0].strip()]=float(ps[1].strip())
                except ValueError:
                    options[ps[0].strip()] = ps[1].strip()

        elif lines[lidx].startswith("--- Eval"):
            if collectingEval > 0:
                data.append(datarow)
            collectingEval += 1
            datarow = [int(lines[lidx].split(" ")[2])]

        elif "solver converged" in lines[lidx] or "failed to converge" in lines[lidx]:
            if "solver converged" in lines[lidx]: results['converged'] = True
            else: results['converged'] = False
            collectingEval = -1  #stop collecting evaluations
            collectingFin = True

        elif "Enigma finished" in lines[lidx]:
            et = time.strptime(lines[lidx][19:].strip(), "%a %b %d %H:%M:%S %Y")
            results['time_end'] = time.strftime(timeStr, et)
            break

        ###multiline collection
        elif collectingEval > 0:
            if "Optimi" in lines[lidx] or "WARNING" in lines[lidx]: continue# or "Total weigh" in lines[lidx]: continue
            linelist = lines[lidx].strip().replace("\t", " ").split(" ")#.strip()

            if len(linelist) <= 3: #its a param line
                if collectingEval == 1:
                    labels.append(linelist[0])
                datarow.append(float(linelist[2]))
            else:
                if lines[lidx].startswith("Total weigh"):
                    if collectingEval == 1:
                        labels.append("RMSDCost")
                    datarow.append(float(lines[lidx][lines[lidx].strip().rfind(" "):]))
                else:
                    if collectingEval == 1:
                        labels += [f"{linelist[0]}", f"{linelist[0]}_error", f"{linelist[0]}_node"]
                    try: datarow += [float(linelist[6]), float(linelist[-1][:-1]), int(linelist[5][:-2])]
                    except IndexError: print(f"lidx:{lidx}\n{lines[lidx]}")

        elif collectingFin:
            if "function evaluations." in lines[lidx]:
                results['num_iter'] = int(lines[lidx][ lines[lidx].rfind(" ", 0, lines[lidx].find(" func")-1) : lines[lidx].find(" function") ] )

            elif "error" in lines[lidx]:
                results['error_final'] = float( lines[lidx].strip().split(" ")[-1] )

            elif "message" in lines[lidx]:
                results['stop_msg'] = lines[lidx][ lines[lidx].find("message:")+10 : ]

            elif "Fitting time:" in lines[lidx]:
                results['fitting_time_s'] = float( lines[lidx].strip().split(" ")[-2] )

            elif "final parameter values" in lines[lidx]:
                collectingFinVal = True
                results['final_labels'] = []
                results['final_values'] = []

            elif collectingFinVal:
                if lines[lidx].startswith("+") or "Parameter" in lines[lidx]:
                    continue
                elif not lines[lidx].startswith('|'):
                    collectingFinVal = False
                    collectingFin = False
                    continue
                linelist = lines[lidx].split("|")
                results[f'final_{linelist[1].strip()}'] = float(linelist[2].strip())
                results['final_labels'].append(linelist[1].strip())
                results['final_values'].append(float(linelist[2].strip()))


    #PROCESS

    params = {'num_opt_params': numOptParams, 'labels': labels, 'data': data}

    print(f"read _opt.out: Success :Completed read of '{filePath}'. '{len(options)}' options, '{len(params)}' params, and '{len(results)}' results objects obtained.")
    if verbose: print(f"read _opt.out: Output dictionaries...\n"
                      f"Options:\n{options}\n(Eval)Params:\n{params}\nResults:\n{results}\n\n")

    return options, params, results


#------------------------------------------------------------------------------------------------------------------

def read_epi(filename, verbose=amverbose):
    """
    Pull out variables from the epi command file.

    :param filename: epi command file name (eg. epi_epi.csv)
    :param verbose: desired verbosity of script print info
    :return: layers: list of layers, see below for list details
    :return: interfaces: list of interfaces, see below for list details
    :return: swb variables: list of swb variables found in the file
    """

    #THIS IS NOT IN USE, YET (220906)


    if verbose: print(f"read epi: Reading epi file '{filename}'.")

    with open(filename, 'r') as inputFile:
        lines = inputFile.readlines()
    layers=[] #list of layers, each layer a list of [id, material, tcl/par file, thickness in um, doping in cm^-3,
    # mole fraction, meshing], layers should be in order from top of device stack to bottom (as found in epi file)
    interfaces=[] #list of interfaces (not necessarily in device stack order)
    swbs=[] #list of swb variable strings in the layer or interface lines (stored in the order found in the file)

    for lidx in range(len(lines)):
        lines[lidx] = lines[lidx][:-1] #remove \n
        if len(lines[lidx]) == 0:
            continue
        elif (lines[lidx][0] == "#") or (lines[lidx][0] == "$"):
            continue
        elif (lines[lidx][0] == ","): #material and interface section
            if "/" in lines[lidx]: #interface
                lineList = lines[lidx].split(",")
                interfaces.append([lineList[1],lineList[2]])
                if verbose: print(f"read epi: interface {len(interfaces)-1}:\t{lineList[1]},\t{lineList[2]}")
            else:
                continue
        else: #layer
            lineList = lines[lidx].split(",")
            for pidx in range(len(lineList)):
                lineList[pidx]=lineList[pidx].strip()
                if 3 <= pidx <= 5: #thickness, doping, or mole fraction
                    if lineList[pidx] == "": lineList[pidx] = 0.0
                    else:
                        try: lineList[pidx] = float(lineList[pidx])
                        except ValueError:
                            swbs.append(lineList[pidx])
                            if verbose: print(f"read epi: Found swb variable '{lineList[pidx]}' in layer {len(layers)}.")
            layers.append(lineList)
            if verbose: print(f"layer {len(layers)-1}:\t{lineList}")


    return layers, interfaces, swbs


# ------------------------------------------------------------------------------------------------------------------

def read_dotStatus(wd):
    """
    Pull the sentaurus project's status info from the wd/.status file.
    Returns None if .status is not present. (The status None is a string to avoid later conversion.)
    Assumes .status has not been altered from sentaurus default format
    :param wd: string, path to the sentaurus working directory
    :return: list [start time, hostname, user, status, exec time], status info of the project
    """
    # eg wd/.status file:
    # 1637160917|dyson.rdc.uolocal|rhunt013|done|200382

    if not os.path.exists(os.path.join(wd, ".status")):
        return [None, None, None, 'None', None]

    with open(os.path.join(wd, ".status"), 'r') as inputFile:
        lines = inputFile.readlines()

    stList = lines[0].split("|")

    if len(stList) != 5:
        print(f"ERROR :read .status: Malformed .status file? Expected 5 items, got '{len(stList)}'; '{stList}'. Returning None...")
        return [None, None, None, 'None', None]

    return stList


def interpretPDTDateTime(inputRunTime):
    """"
    Try and interpret time strings of the format (eg.)'Fri May  7 15:52:27 PDT 2021'
    Convert to that of timeStr.
    :param inputRunTime: (str) the string to try and interpret (of the form given above)
    :result runTime: (str) the same time but now formatted to the config's timeStr format
    """

    runTime = inputRunTime
    try:
        rT = time.strptime(runTime, "%a %b %d %H:%M:%S %Z %Y")
    except ValueError:
        # handle time zone by hand and hope that it does not run on last day of the month
        rtstr = runTime[:runTime.rfind(":") + 2]
        rT = time.strptime(rtstr, '%a %b %d %H:%M:%S')
        rtstr = runTime[-4:]
        rT2 = time.strptime(rtstr, '%Y')

        rTc = time.struct_time((rT2[0],) + rT[1:9])
        """rT.tm_year = rT2.tm_year"""
        # convert to EST
        if runTime[runTime.rfind(":") + 4:-5] == "PDT":
            rTc = time.struct_time(rTc[0:3] + ((rTc[3] + 3),) + rTc[4:9])
    runTime = time.strftime(timeStr, rTc)
    return runTime

#------------------------------------------------------------------------------------------------------------------
#------------------------------------------------------------------------------------------------------------------
#!   AIIRMAP CREATED FILES, INTERACTION
#------------------------------------------------------------------------------------------------------------------
#------------------------------------------------------------------------------------------------------------------

def readMatch_exptIO(swbFiles, gvarsFile):
    """
    Matches experiment inputs and outputs using the node-specific swb vars files and the gvars.dat file
    Combines all experiment data into one collection
        Pulls all swbVar files,
        looks at the input parameters from the first one,
        looks at the gavrs variables associated with all expts to get a list of unique sorted outputs variables,
        then, creates a list of the sorted parameters (inputs) and variables (outputs) for all expts

        All experiments with swbVars files are considered. Even if they do not have any output in the gvars file. (allows for incomplete optimizations)
        Previously (<221213); included only expts with variables written to gvars in the output file.

    :param swbFiles: (list of strings) list of node-specific swbVars.csv filepaths
    :param gvarsFile: (string) path to the gvars.dat file
    :return toolLabelFlow: (list of strings) the labels for the tools in the sim flow, ordered according to the sim flow
    :return swbHeader: (list of strings) ordered list of the data column labels
    :return swbData: (list of lists) ordered list of input and output value (inner) for each expt (outer)
    :return exptsNodes: (list of lists of ints) all node numbers (inner) for each experiment (outer)
    """

    swbHeader = []
    swbData = []
    exptsNodes = []
    orderedLabels = [] #will hold the unique set of [[inputLabels], [outputLabels]] so that the data can be sorted
    uniqueOutputLabels = [] #will hold all unique output labels
    unorderedInputLabels, unorderedInputData = [], []
    unorderedOutputLabels, unorderedOutputData = [], []

    #get inputs list from first expt
    _, eToolLabelFlow, eInputLabels, eInputValues, _ = read_swbVars(swbFiles[0])
    orderedLabels.append(eInputLabels)
    swbHeader = ["experiment-inputs"] + eInputLabels
    toolLabelFlow = eToolLabelFlow

    #read swb vars for all expts and get all possible unique variable names, save outputs for later sort
    for eidx in range(len(swbFiles)):
        _, eToolLabelFlow, eInputLabels, eInputValues, eExptNodes = read_swbVars(swbFiles[eidx])
        unorderedInputLabels.append(eInputLabels)
        unorderedInputData.append(eInputValues)
        exptsNodes.append(eExptNodes)

        eOutputLabels, eOutputVals, _ = readExtract_gvars(eExptNodes, gvarsFile)
        unorderedOutputLabels.append(eOutputLabels)
        unorderedOutputData.append(eOutputVals)

        #get all unique variable labels
        if not set(eOutputLabels).issubset(uniqueOutputLabels):
            uniqueOutputLabels = list(set(eOutputLabels)|set(uniqueOutputLabels)) #union

    #save unique output labels for output
    orderedLabels.append(uniqueOutputLabels)
    swbHeader += ["experiment-outputs"] + uniqueOutputLabels

    #sort and save sorted to output list of lists
    for eidx in range(len(swbFiles)):
        #print(f"eidx:{eidx}\nunique:{uniqueOutputLabels}\nexpt:{unorderedOutputLabels[eidx]}")
        #add missing outputs as nans
        if set(unorderedOutputLabels[eidx]) != set(uniqueOutputLabels):
            missingLabels = set(uniqueOutputLabels) - set(unorderedOutputLabels[eidx])
            unorderedOutputLabels[eidx] += list(missingLabels)
            unorderedOutputData[eidx] += [np.nan] * len(missingLabels)

        #sort and save
        labelDataTups = list(tuple(zip(unorderedInputLabels[eidx], unorderedInputData[eidx])))
        labelDataTups.sort(key=lambda i: orderedLabels[0].index(i[0]))
        exptData = [eidx + 1] + [labelDataTups[i][1] for i in range(len(labelDataTups))]

        labelDataTups = list(tuple(zip(unorderedOutputLabels[eidx], unorderedOutputData[eidx])))
        labelDataTups.sort(key=lambda i: orderedLabels[1].index(i[0]))
        exptData += [eidx + 1] + [labelDataTups[i][1] for i in range(len(labelDataTups))]

        swbData.append(exptData)

    return toolLabelFlow, swbHeader, swbData, exptsNodes


#------------------------------------------------------------------------------------------------------------------

def read_swbVars(swbFile):
    """
    Reads the results/nodes/#/n#_exp-#(s)_swbVars.csv file and extracts information in preparation for saving in projectCollection_dbVars and post-processing.
    :param: swbFile: path to the swbFile (string)
    :return: wd: Sentaurus project working directory path (string)
    :return: toolLabelFlow: ordered simulation flow tool labels (list of strings)
    :return: inputsLabels: the swb parameter/input names (list of strings)
    :return: exptInputs: the param/input values for each expt, correspondence to inputsLabels (list of lists of floats and strings)
    :return: exptInputs: the output variable producing nodes for each expt (list of lists of ints)
    :return: exptInputs: all nodes for each expt, numerically sorted (list of lists of ints)
    In these last three returns each inner list is one expt, stored in order of experiments.
    """

    with open(swbFile, 'r') as inputFile:
        lines = inputFile.readlines()

    wd = lines[0][:-1] #remove \n
    toolLabelFlow = lines[4][:-1].strip(",").split(" ")
    inputsLabels = lines[6].split(",")[:-1]  # remove the expt nodes entries
    exptInputs = []  # list of lists for each expt's param values
    # outputVarNodes = []  # list of lists for each expt's nodes with output variables
    exptNodes = []  # list of lists containing all nodes for each expt (nodes are sorted numerically)

    for lidx in range(7, len(lines), 1):

        lineList = lines[lidx].split(",")
        for pidx in range(len(lineList) - 1):  # convert param floats
            try:
                lineList[pidx] = float(lineList[pidx])
            except ValueError:
                lineList[pidx] = lineList[pidx]
        exptInputs = lineList[:-1]

        nodeList = lineList[-1].split(" ")
        for nidx in range(len(nodeList)):
            try:
                nodeList[nidx] = int(nodeList[nidx])
            except ValueError:
                raise ValueError(
                    f"ERROR :read_swbVars: Cannot convert {nodeList[nidx]} to int. Was expecting a node number. (Reading expt nodes.)")
        nodeList = sorted(nodeList)
        exptNodes = nodeList

    return wd, toolLabelFlow, inputsLabels, exptInputs, exptNodes

#------------------------------------------------------------------------------------------------------------------

def save_execTimeFile(wd, projectHeader, toolLabelFlow, exptNodes, detectLockedTools=False, verbose=amverbose):
    """
    Extracts the node/tool execution times for each expt from the results/nodes/#/<filename>.sta files. Save these and the expt total to the file pyCollection_execTimes.csv
    :param wd: Sentaurus working directory path (string)
    :param projectHeader: header info for the file (string)
    :param toolLabelFlow: the labels for the swb tools in the simulation (list of strings)
    :param exptNodes: all nodes for each expt (list of lists of ints (and np.nan), outer list is by expt, inner is nodes for each expt)
    :param verbose: desired verbosity for the script
    :return: lockedTools: boolean list of the locked tools in the sim flow
    :return: None (file is saved)
    """

    # build and save a file recording the execution time for each expt and each node in each expt detect if locked from np.nan if desired
    lockedTools = None
    if detectLockedTools: lockedTools = [True] * len(toolLabelFlow) #detect unlocked tools by presence of a value (if all np.nans then it is locked)
    with open(os.path.join(wd, f"{sFileNameBody}_execTimes.csv"), 'w', newline='', encoding='UTF8') as file:
        file.write(f"#Sentaurus Project Experiment and Node Execution Times in Seconds\n{projectHeader}")

        writer = csv.writer(file)
        headerRow = ["expt #", "expt Total"]
        for tidx in range(len(toolLabelFlow)):
            headerRow.append(toolLabelFlow[tidx]+"_node#")
            headerRow.append(toolLabelFlow[tidx]+"(s)")
        writer.writerow(headerRow)

        for eidx in range(len(exptNodes)):
            exptRow = []
            exptTot = 0
            for tidx in range(len(toolLabelFlow)):
                nTime = extract_nodeExecTime(wd, exptNodes[eidx][tidx], verbose)
                exptRow.append(exptNodes[eidx][tidx])
                exptRow.append(nTime)
                if nTime is not np.nan: 
                    exptTot = exptTot + nTime
                    if detectLockedTools and lockedTools[tidx]:
                        lockedTools[tidx] = False

            exptRow.insert(0, exptTot)
            exptRow.insert(0, eidx + 1)
            writer.writerow(exptRow)

        if verbose: print(f"save exec times csv: swb execution time file '{wd}{sFileNameBody}_execTimes.csv' has been written.")
    return lockedTools 

#------------------------------------------------------------------------------------------------------------------

def extract_nodeExecTime(wd, nodeNo, verbose=amverbose):
    """
    Use the nodes status file to extract the execution time for a node. In seconds
    :param wd: Sentaurus project working directory (path string)
    :param nodeNo: node number (int)
    :return: execTime: how long it took for the node to execute, in seconds (int)
    """
    staFilePath = glob.glob(os.path.join(wd, f"results","nodes",f"{nodeNo}","*.sta"))

    if len(staFilePath) > 1:
        print(f"WARNING :extract node exec time: There is more than one .sta file for node {nodeNo}. This is unexpected. Returning nan.")
        return np.nan

    if len(staFilePath) == 0:
        if verbose: print(f"WARNING :extract node exec time: Missing node {nodeNo} status file (.sta). Setting exec time to nan.")
        return np.nan

    with open(staFilePath[0], 'r') as inputFile:
        lines = inputFile.readlines()
    if lines[0].split("|")[2] != "done":
        if verbose: print(f"WARNING :extract node exec time: Node {nodeNo} has not completed running. This is unexpected. Has expt 1 completed yet? Setting to nan.")
        execTime = np.nan
    else:
        execTime = int(lines[0].split("|")[4])
    return execTime



#------------------------------------------------------------------------------------------------------------------
#------------------------------------------------------------------------------------------------------------------
#!    SENTAURUS PROJECT INFO QUERYING
#------------------------------------------------------------------------------------------------------------------
#------------------------------------------------------------------------------------------------------------------

def get_genoptNodes(wd):
    """
    Gets the node numbers for the genopt tools for all experiments
    Uses the presence of results/nodes/*/*_opt.out to extract node numbers and then checks to see if the node is in gvars to make sure it is current
    >> UPDATE; Sentaurus sometimes will fail to write opt to parent gvars, take all nodes with _opt.out files
    :param wd: [string-path] path to the swbVars file to be analyzed
    :param toolLabelFlow: [list] the labels of the tools in the sim flow, ordered by sim flow
    :return: toolLists: list of lists of form [toolLabel,toolType,numAssocParams], information for each tool in the project
    :return: toolsLocked: list of ints, the lock state for each tool (0=unlocked, 1=locked)

    :return: genoptNodeNumbers: [list] genopt tool node for each exp't
    """

    files = glob.glob(os.path.join(wd, "results", "nodes", "*", "*_opt.out"))
    nodes = []

    for fidx in range(len(files)):
        poss_node = int(os.path.split(files[fidx])[1][1 : os.path.split(files[fidx])[1].find('_')])
        nodes.append(poss_node) #take all possible genopt nodes
        # chk, _, _ = readExtract_gvars([poss_node], os.path.join(wd,"gvars.dat"))
        # if len(chk) != 0:
        #     nodes.append(poss_node)
    if len(nodes) == 0:
        print(f"ERROR :get genopt nodes: Could not find any genopt nodes for '{wd}'!")

    return nodes


#------------------------------------------------------------------------------------------------------------------


def get_optoutParFolderAndNodeStr(optoutStr):
    """
    Reads the opt out filepath and extracts the 2-level-up folder name and the child opt node number
    (used for sorting of opt out files for start and end pt db generation)
    2-level-up: Since typically source the optout files from sentaurus_files/<projects>/parent/ folders
    :param optoutStr: (str) path to the n###_opt.out file
    :return: out (list; [grandparent folder name, child opt node number] )
    """
    out = []
    p = str(os.path.split(optoutStr)[0])
    p2 = str(os.path.split(p)[0])
    out.append(str( os.path.split(p2)[1] ))
    out.append(int(optoutStr[optoutStr.rfind("n")+1:optoutStr.rfind("_")]))
    return out


#------------------------------------------------------------------------------------------------------------------

def get_optoutNodeStr(optoutStr):
    """
    Reads the opt out filepath and extracts the child's opt node number
    (used for sorting of opt out files for start and end pt db generation)
    :param optoutStr: (str) path to the n###_opt.out file
    :return: (int) the child node number for the given opt out file
    """
    return int(optoutStr[optoutStr.rfind("n")+1:optoutStr.rfind("_")])


#------------------------------------------------------------------------------------------------------------------

def get_childDateAndNodeStr(childStr):
    """
    Reads the child project folder name and extracts the run start date and child node number
    (used for sorting of child project dbs for start and end pt db generation)
    :param childStr: [str] the path to the child db
    :return: [start date (yymmdd, int), child node number (int)]
    """
    out = []
    out.append(int(os.path.split(childStr)[1][0:6]))
    out.append(int(childStr[childStr.rfind("_")+1:childStr.rfind(".")]))
    return out


#------------------------------------------------------------------------------------------------------------------


def get_childNodeStr(childStr):
    """
    Reads the child project folder name and extracts the child node number
    (used for sorting of child project dbs for start and end pt db generation)
    :param childStr: [str] the path to the child db
    :return: child node number (int)
    """
    return int(childStr[childStr.rfind("_")+1:childStr.rfind(".")])


#------------------------------------------------------------------------------------------------------------------

def get_gitRevInfo(wd):
    """
    Gets the current git revision hash as hex string and the current branch name. If the git executable is missing or git is unable to get the revision, None is returned
    :param: wd: sentaurus working directory path (string)
    :returns: A hex string for the revision tag and a string for the branchName or None
    """
    wd = os.path.realpath(wd)
    # if not check_executable('git'):
    #     print(f"WARNING :get git rev: 'git' command not found, git revision not detectable. Returning None.")
    #     return None
    revHex = subprocess.check_output(['cd', wd, ';', 'git', 'rev-parse', 'HEAD']).strip()
    branchName = subprocess.check_output(['cd', wd, ';', 'git', 'rev-parse', '--abbrev-ref', 'HEAD']).strip()
    if not revHex:
        print(f"WARNING :get git rev: Couldn't detect git revision (not a git repository?) Returning None.")
        return None
    return revHex, branchName



#------------------------------------------------------------------------------------------------------------------
#------------------------------------------------------------------------------------------------------------------
#!   FILING CABINET - SENTAURUS PROJECT INTERACTION
#------------------------------------------------------------------------------------------------------------------
#------------------------------------------------------------------------------------------------------------------

def file_sAndCmdFiles(filingFolder, wd, cmdFiles, child=False):
    """
    Copy files from the sentaurus working folder to the filing cabinet's sentaurus files directory
    Copy config's sfilesToSave and the command files anticipated by the provided patterns.
    :param filingFolder: [string-path] the datetime specificfiling cabinet folder to copy to
    :param wd:  [string-path] the sentaurus project working directory
    :param cmdFiles: [list of path-strings] the project command files
    :return: None (copies files)
    """

    #create filing folder
    #filingFolder = os.path.join(os.path.realpath(filingFolder), os.path.split(os.path.realpath(wd))[1] + "__" + now)
    if not os.path.exists(filingFolder):
        os.makedirs(filingFolder)

    #get list of files to copy
    files = []
    if child:
        for fidx in range(len(sfilesToSave_optChildren)):
            files += glob.glob(os.path.join(wd, sfilesToSave_optChildren[fidx]))
    else:
        for fidx in range(len(sfilesToSave)):
            files += glob.glob(os.path.join(wd, sfilesToSave[fidx]))
    files += cmdFiles

    #copy
    for fidx in range(len(files)):

        try:
            shutil.copy(files[fidx], filingFolder)
        except IsADirectoryError:
            try:
                shutil.copytree( files[fidx], os.path.join(filingFolder, os.path.split(files[fidx])[1]) )
            except FileExistsError:
                print(f"WARNING :file sentaurus files: Overwriting existing folder '{os.path.join(filingFolder, os.path.split(files[fidx])[1])}'")
                shutil.rmtree(os.path.join(filingFolder, os.path.split(files[fidx])[1]))
                shutil.copytree(files[fidx], os.path.join(filingFolder, os.path.split(files[fidx])[1]))
        except FileNotFoundError:
            if amverbose: print(f"WARNING :file sentaurus files: File not found, cannot copy ; '{files[fidx]}'")
    return None




#------------------------------------------------------------------------------------------------------------------

def anticipate_cmdFiles(wd, toolsList):
    """
    Takes toolsList and returns a list of expected tool cmd files using sentaurus naming defaults.
    :param toolsList: [list of [toolLabel, toolType]] for all tools in the project (type matches config toolNameList)
    :return: cmdFiles: [list] the paths of the tool command files for filing
    """

    #Uses config.py's toolNameList and cmdFileEndings (matched) lists

    cmdFiles=[]
    for tidx in range(len(toolsList)):
        toolTypeIdx = toolNameList.index(toolsList[tidx][1])
        if isinstance(cmdFileEndings[toolTypeIdx], list):
            for fidx in range(len(cmdFileEndings[toolTypeIdx])):
                cmdFiles.append(os.path.join(wd, f"{toolsList[tidx][0]}{cmdFileEndings[toolTypeIdx][fidx]}"))
        else:
            cmdFiles.append(os.path.join(wd, f"{toolsList[tidx][0]}{cmdFileEndings[toolTypeIdx]}"))

    return cmdFiles


#------------------------------------------------------------------------------------------------------------------

def get_pullIdxForOptChildren(methods, numOptParam):
    """
    Get the expected idx for the optimal experiment in child dbs
    :param methods: list, optimizer algorithm used for each cdb
    :param numOptParam: int, number of optimization parameters in all cdbs
    :return: pullidx: list, the expected index for the optimal result for each algorithm given in methods
    """


    pullidx = []
    for i in range(len(methods)):
        if methods[i] == 'L-BFGS-B' or methods[i] == 'SLSQP': #optimal expt is located one gradiant eval from bottom (ie. numOptParams+1 from bottom)
            pullidx.append(int(numOptParam*(-1)-1))
        elif methods[i] == 'Nelder-Mead': #optimal expt is the last one
            pullidx.append(-1)
        else:
            if not isinstance(optFinalPullIdx_default, int):
                pullidx.append(int(numOptParam*(-1)-1))
            else:
                pullidx.append(optFinalPullIdx_default)

    return pullidx


#------------------------------------------------------------------------------------------------------------------
#------------------------------------------------------------------------------------------------------------------
#------------------------------------------------------------------------------------------------------------------
