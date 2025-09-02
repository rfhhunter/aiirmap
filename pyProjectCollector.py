import matlab.engine
from aiirmapCommon import *

def main():
    """
    Linux terminal wrapper for the sCollectProject function from sReadAndInteract
    Collects the sentaurus info and exec times and writes them to file, copies all important sentaurus files to the filing cabinet
    Takes as input up to one linux terminal command argument;
        The sentaurus working directory path (if None; use current wd)

    Eg.
    python3 /path/to/script/ /path/to/swd/
        /swd/ sentaurus working directory (string)

    Aiirmap Assumptions:
    -Node number assignments are larger for tools later in the sim flow (within each expt)
    -Heirarchial sentaurus project organization

    :return: None, writes the pyCollection_Info/execTime.csv files and copies the config defined files to the sentaurus_files filing cabinet
    """

    if len(sys.argv) == 1:
        wd = os.getcwd()
    elif len(sys.argv)>1:
        wd = os.path.abspath(sys.argv[1])
    elif len(sys.argv)>2:
        print(f"WARNING :pyProjectCollector wrapper: Called with more than one command line parameter. Using first parameter as wd ('{sys.argv[1]}').")
        wd = os.path.abspath(sys.argv[1])

    si.sCollectProject(wd,filingdir)



if __name__ == '__main__':
    main()



