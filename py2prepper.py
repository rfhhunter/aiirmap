"""
py2prepper.py

python2 compatible functions for creating the pyCollector_pyt.py command file.
(vS (2021) of Sentaurus requires py2)

Operation;
1. Create
A python tool in your sentaurus project with the name <sToolName>
2. Add
set WB_tool(gsub,prologue) { exec  python $wdir/path/to/py2prepper.py $wdir }
to the project tooldb (swb automatically creates the _pyt.py command file at the start of every run of gsub to get the accurate parameter list)




pyPrepper Assumptions:
    -linux/unix filesystem (/ path divisor)
    -Node number assignments are larger for tools later in the sim flow (within each expt)
    -Heirarchial sentaurus project organization


-----

These functions have been copied and edited from the py3 versions;
sReadAndInteract.sPyToolPrepper
sReadAndInteract.read_gtree
py3prepper.py

The config variables are copied from
config.py
(Config can be recoupled if it is made py2-friendly.... currently is not)
If functional changes are made here (or there) they should be copied there (or here).

Last edited: 221108

"""

import os
import time
import sys
import numpy as np


#config variables
#from config import *
timeStr = '%Y%m%d_%H%M%S'
sToolName='pyCollector'
toolNameList = ["epi", "sde", "mpr", "tdx", "sdevice", "svisual", "python", "genopt"]
bool_true_lowerlist = ["true", "on", "t", "yes"]
bool_false_lowerlist = ["false", "off", "f", "no"]


def main():
    """
    Wrapper for the sPyToolPrepper_py2 function below
    Writes a new command file for the pyCollector tool. (pyCollector_pyt.py)
    Use python2
    Will take up to one linux command line input when run. (see below for example terminal command)
        Takes as input sentaurus working directory
        Defaults; current directory

    Eg.
    python2 /path/to/pyPrepper.py /path/to/sentaurus/project

    Assumptions;
    -linux/unix filesystem (/ path divisor)
    -Node number assignments are larger for tools later in the sim flow (within each expt)
    -Heirarchial sentaurus project organization

    :return: None, writes the new pyCollector_pyt.py command file
    """

    wd = os.getcwd()
    # outputGeneratingTools = outputGeneratingTools_default

    if len(sys.argv)>1:
            wd = sys.argv[1]
    elif len(sys.argv)>2:
        raise Warning("WARNING :pyPrepper wrapper: pyPrepper called with more than one command line parameters. Using first parameter after script name as sentaurus working directory.")

    sPyToolPrepper_py2(wd)





def sPyToolPrepper_py2(wd):
    """
    Prepare the pyCollector_pyt.py command file.
    :param wd: (string) sentaurus project working directory path
    :return: None, writes the pyCollector_pyt.py command file in wd

    python2 compatible
    Copied from sRaI.sPyToolPrepper on 221107
    if functional edits are made here they should be copied there

    """

    #Assumes vertical tool flow orientation (should work with horizontal but needs to be tested
    #Sentaurus must be running in a linux environment (/ path divisor)
        #this is required since this function writes a file with sentaurus preprocessed path dependency and cannot use os.join.path
        #the "/" of paths is also used to help manipulate standard strings in the read_glog, read_goptlog, and read_epi

    print("pyCollector prepper: Preparing py-tool cmd file '"+os.path.join(wd, sToolName+'_pyt.py')+"'.")

    now = time.strftime(timeStr)

    # Get the parameter names and their variable types, then generate the pyCollector writer strings
    params, _, ptypes, _, toolLists, _ = read_gtree_py2(os.path.join(wd, "gtree.dat"))

    headerStr = "writer.writerow(['expt_num',"
    dataCallStr = "writer.writerow([@experiment@,"
    for pidx in range(len(params)):
        headerStr = headerStr + "'" + params[pidx] + "',"
        if ptypes[pidx] in ["string", "boolean"]:
            dataCallStr = dataCallStr + "'@" + params[pidx] + "@',"  # pyCollection_swbVars.csv retains the specific swb boolean identifier used
        else:
            dataCallStr = dataCallStr + "@" + params[pidx] + "@,"

    headerStr = headerStr + "'expt nodes'])\n"
    dataCallStr += "'@node|all@'])\n"

    # Write the sToolName_pyt.py command file
    filename = sToolName + "_pyt.py"
    with open(os.path.join(wd, filename), 'w') as file:
        file.write(
            "#Sentaurus pyCollector cmd file\n#The tool collects input parameters and node information for each experiment\n"
            "#It creates or overwrites results/nodes/#/n#_swbVars.csv files when run\n\n"
            "#File autocreated by pyPrepper.py script\n#Command file creation time:" + now + "\n\n")
        file.write("import time\nimport csv\nimport os\n\nnow = time.strftime('%Y%m%d_%H%M%S')\n\n"
                   ## "#setdep @node|-1@\n\n"
                   "#For all expts; write project info, headers and then extract data and print to line\n")
        file.write("expts = '@experiments@'\nexptStr = expts.replace(' ', '-')\n")
        file.write(
            "with open(f'@pwd@/results/nodes/@node@/n@node@_swbVars.csv', 'w', newline='', encoding='UTF8') as file:\n"
            "\tfile.write(f'@pwd@\\npreprocessing time:,{now}')\n"
            "\tfile.write(f'\\n\\nsimulation flow:\\n@tool_label|all@\\n\\n')\n"
            "\twriter = csv.writer(file)\n\t"+headerStr+"\n\t"+dataCallStr+"\n")

    print("pyCollector prepper: Success :A new" + sToolName + "_pyt.py has been written.")
    return





def read_gtree_py2(filename, pullToolsLock=False):
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

    python2 compatible
    Copied from sRaI.read_gtree on 221107
    if functional edits are made here they should be copied there

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
                streeToolLidxs.append(streeToolLidxs[-1]+toolLists[tidx+1][2]+1)
            if lastToolLidx+3 > len(lines):
                if pullToolsLock: print("WARNING :read gtree: No simulations in sim tree, could not read tools locked state. Setting all to unlocked.")
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
        if pullToolsLock: print("WARNING :read gtree: The number of identified locked tool states ("+len(toolsLocked)+") does not match the number of tools ("+len(toolLists)+"). Setting all tools to unlocked.")
        toolsLocked = np.zeros(len(toolLists))

    return params, pdefs, ptypes, ptools, toolLists, toolsLocked



if __name__ == '__main__':
    main()
