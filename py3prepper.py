
from aiirmapCommon import *

def main():
    """
    See py2prepper.py

    Wrapper for the sPyToolPrepper function from sReadAndInteract
    Writes a new command file for the pyCollector tool. (pyCollector_pyt.py)
    Will take up to one linux command line input when run. (see below for example terminal command)
        Takes as input sentaurus working directory
        Defaults; current directory

    Eg.
    python3 /path/to/pyPrepper.py /path/to/sentaurus/project

    Assumptions;
    -Node number assignments are larger for tools later in the sim flow (within each expt)
    -Heirarchial sentaurus project organization

    :return: None, writes the new pyCollector_pyt.py command file


    This calls the python3 version.
    For the py2 version see py2prepper.py
    Functional changes to this (that) script should be propagated there (here)

    """

    wd = os.getcwd()
    # outputGeneratingTools = outputGeneratingTools_default

    if len(sys.argv)>1:
            wd = sys.argv[1]
    elif len(sys.argv)>2:
        raise Warning(f"WARNING :pyPrepper wrapper: pyPrepper called with more than one command line parameters. Using first parameter after script name as sentaurus working directory.")

    si.sPyToolPrepper(wd)



if __name__ == '__main__':
    main()





"""
Below is the original pyPrepper.py
!!!
OBSOLETE
!!!
"""
"""
#File to create the command file for the pyCollector tool
#pyCollector collects the sentaurus workbench inputs for each expt, all the nodes for each expt, and the nodes which generate the output variables
#This file allows the list of swb inputs/parameters to be automatically generated from the gtree file.

#Run this file within the sentaurus project working directory or pass that working directory as unix terminal input:
#eg for latter: "python3 pyPrepper.py ../path/to/project"


#!!!! Assumptions !!!!
#pyCollector tool label is "pyCollector" (such that its command file is pyCollector_pyt.py)
#linux/unix filesystem (/ path divisor)
#Vertical tool flow orientation in swb (should work with horizontal, but is untested)
#Output variables are generated in the tools given by outputGeneratingTools



from utilities.sReadAndInteract import read_gtree_params, bool_true_lowerlist, bool_false_lowerlist

import time
import sys
import os



#Space separated list of tools which generate the output variables (and which are used to setdep of the pyCollector nodes)
#Use the format @node|x@ where x is the relative location of the generating tool wrt pyCollector
#Ex; output variables generated in tools directly before and five tools before pyCollector in the sim flow
# outputGeneratingTools = "@node|-5@ @node|-1@"
outputGeneratingTools = "@node|-1@ @node|1@"


def main(wd="./"):

    #Pull wd from command line arguments, format it, get time now
    if len(sys.argv)>1:
        wd = sys.argv[1]
    elif len(sys.argv)>2:
        raise Warning(f"pyPrepper called with more than one command line parameter. Using first parameter after script name as sentaurus working directory('{sys.argv[1]}').")

    wd = os.path.realpath(wd)
    if wd[-1] != "/":
        wd = wd + "/"

    now = time.strftime('%Y%m%d_%H%M%S')


    #Get the parameter names and their variable types, then generate the pyCollector writer strings
    params, ptypes, ptools, _ = read_gtree_params(wd+"gtree.dat")

    headerStr = f"writer.writerow(['experiment',"
    dataCallStr = f"writer.writerow([@experiment@,"
    for pidx in range(len(params)):
        headerStr = headerStr + f"'{params[pidx]}',"
        if ptypes[pidx] in ["string", "boolean"]:
            dataCallStr = dataCallStr + f"'@{params[pidx]}@'," #pyCollection_swbVars.csv retains the specific swb boolean identifier used
        else:
            dataCallStr = dataCallStr + f"@{params[pidx]}@,"
    headerStr = headerStr + "'variables out nodes','expt nodes'])\n"
    dataCallStr = dataCallStr + "'" + outputGeneratingTools + "','@node|all@'])\n"


    #Write the pyCollector_pyt.py command file
    with open(wd + f"pyCollector_pyt.py", 'w', newline='', encoding='UTF8') as file:
        file.write(f"#Sentaurus pyCollector cmd file\n#Tool collects input parameters and node information for all experiments\n"
                   f"#Tool creates or overwrites pyCollection_swbVars.csv file on first expt and appends to it in following expts\n\n"
                   f"#File autocreated by pyPrepper.py script\n#Command file creation time:{now}\n\n")
        file.write("import time\nimport csv\nimport os\n\nnow = time.strftime('%Y%m%d_%H%M%S')\n\n"
                   f"#setdep {outputGeneratingTools}\n\n"
                   "#First expt; overwrite or create write project info, headers and then extract data\n")
        file.write("if @experiment@ == 1 or not os.path.exists(os.path.join('@pwd@','pyCollection_swbVars.csv')):\n"
                   "\twith open('@pwd@/pyCollection_swbVars.csv', 'w', newline='', encoding='UTF8') as file:\n"
                   "\t\tfile.write(f'@pwd@\\npreprocessing time:,{now}')\n"
                   "\t\tfile.write(f'\\n\\nsimulation flow:\\n@tool_label|all@\\n\\n')\n"
                   "\t\twriter = csv.writer(file)\n"
                   f"\t\t{headerStr}\n\t\t{dataCallStr}\n\n")
        file.write("else:\n"
                   "\twith open('@pwd@/pyCollection_swbVars.csv', 'a', newline='', encoding='UTF8') as file:\n"
                   "\t\twriter = csv.writer(file)\n"
                   f"\t\t{dataCallStr}")


if __name__ == '__main__':
    main()
"""
