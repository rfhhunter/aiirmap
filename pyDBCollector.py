
import matlab.engine
from aiirmapCommon import *

def main():
    """
    Terminal command wrapper for the collectAndSave functions from aiirmapcommon
    Collects the sentaurus info and exec times and writes them to file, copies all important sentaurus files to the filing cabinet
    And, creates and saves DB csv file
    Tries to detect if the project is an optimization family, a single db, or a child in a still-running family and runs the correct function config
    (Note; runs full family collection for optimization family)


    Takes as input up to two linux terminal command arguments, but only in order;
        (str) The sentaurus working directory path (default, if none; use current directory)
        (str) Run descriptor, used in the folders and filenames (default, if none; %y%m%d_%H%M%S__projectName<_optimization>)
    by default saves DBs
        for optimization families; to dbdir/runDescriptor/
        for single dbs; to dbdir/runDescriptor.csv


    Eg.
    python3 /path/to/script/ /path/to/swd/ nice_run_description
        /swd/ sentaurus working directory (string)
        nice_run_description used in folder and filenames of the saved files (no spaces)


    Aiirmap Assumptions:
    -Node number assignments are larger for tools later in the sim flow (within each expt)
    -Heirarchial sentaurus project organization

    :return: DBs; list of the created DBs (for optimization families; ordered parent,children[i],mergedChildren )
    """
    wd = os.getcwd()
    runDescriptor = f"{time.strftime(timeStr)}__"

    if len(sys.argv) == 1: #just path to script supplied
        runDescriptor += os.path.split(wd)[1]
    elif len(sys.argv) == 2: #wd supplied
        wd = os.path.abspath(sys.argv[1])
        runDescriptor += os.path.split(wd)[1]
    elif len(sys.argv) == 3: #wd and runDescriptor supplied
        wd = os.path.abspath(sys.argv[1])
        runDescriptor = sys.argv[2]
    elif len(sys.argv) > 3: #too many args supplied
        wd = os.path.abspath(sys.argv[1])
        runDescriptor = sys.argv[2]
        print(f"WARNING :pyDBCollector wrapper: Called with more than two command line parameter. Using first parameter as wd ('{sys.argv[1]}') and second as runDescriptor ('{sys.argv[2]}').")

    # dbd = os.path.join(dbdir, runDescriptor) #dbdir for project
    sfd = os.path.join(sfdir, runDescriptor) #sfdir for project

    DBs = []

    #try and interpret what type of project is present
    optFiles = glob.glob(os.path.join(wd, "results", "nodes", "*", "*_opt.out"))
    if len(optFiles) > 0:
        #optimization parent
        pdb,cdbs,mdb = collectAndSave_SParentChild_DBs(wd, runDescriptor)
        DBs.append(pdb); DBs += cdbs; DBs.append(mdb)

    elif os.path.exists(os.path.join(wd, "results", "logs", "glog.txt")):
        #regular project
        db = collectAndSave_SProj_DB(wd, sfd, dbSaveName=os.path.join(dbdir, runDescriptor+".csv"))
        DBs.append(db)

    else:
        #child of a still-running optimization family
        #needs inheritence ([hostname (str), user (str), sVersion (str), runTime (str), parent opt node (int)])
        #if not supplied will try and pull all but cmdFiles from the wd/.status file (sRaI.sCollectProject)
        db = collectAndSave_SProj_DB(wd, sfd, dbSaveName=os.path.join(dbdir, runDescriptor+".csv"))
        DBs.append(db)


    return DBs


if __name__ == '__main__':
    main()
