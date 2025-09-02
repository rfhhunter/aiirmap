"""
config
The aiirmap configuration file.

Includes
- Filing
- DR related
- Plot related
- DB related
- Sentaurus related


Important config values to set for a new setup :
    - filingdir
    - nk_path (aiirpower
for implementation within Sentaurus; refer to py2prepper



"""


"""
Also:

Nice git logging in your bash terminal?

Put the following commands in your .bashrc (and source it):
alias gitlog='git log --graph --abbrev-commit --decorate --format=format:"%C(bold blue)%h%C(reset) - %C(bold cyan)%aD%C(reset) %C(bold green)(%ar)%C(reset)%C(bold yellow)%d%C(reset)%n          %C(white)%s%C(reset) %C(dim white)- %an%C(reset)" --all'
alias githead='gitlog | head -n 25'
alias gittail='gitlog | tail -n 25'

And run command 
git config color.diff always

(you may have to remove the --all from the first command)
"""


"""
Also:

Running on windows? 
Are you running into issues where the scripts cannot find or save files with long filepaths?
Windows has a 260 character limit for filepaths (facepalm), but this limit can be overwritten!
Google 'enabling win32 long paths for windows <ver>' ... it kinda works, sometimes *shrugs

"""


import os





amverbose = False #if True; gives more python terminal output (partially implemented ##TODO? very low priority)


#------------------------------------------
#Filing related

filingdir= "/raidC/rhunt013/stdb/flying-cabinet_aiirmap" ##dyson 2506
filingdir = "D:\\OneDrive - University of Ottawa\\Documents\\solar-cells_sunlab_onedrive\\aiirpower_ppc_freespace\\aiirmap_flying_folder"  ##sunwin 2506
filingdir = "C:\\Users\\Rob_F\\OneDrive - University of Ottawa\\Documents\\solar-cells_sunlab_onedrive\\aiirpower_ppc_freespace\\aiirmap_flying_folder" ##the rog 250507
filingdir = "C:\\Users\\rhunt013\\OneDrive - University of Ottawa\\Documents\\solar-cells_sunlab_onedrive\\aiirpower_ppc_freespace\\aiirmap_flying_folder" #sunlap2 250610

#simulation_files, databases, plots folder locations and path to the project runtime logger csv file
sfdir = os.path.join(filingdir,"sentaurus_files") #simulation_files (location to copy files output by the simulation)
dbdir = os.path.join(filingdir,"databases") #location to save databases
pldir = os.path.join(filingdir,"plots") #location to save plots and figures (partially implemented, most scripts with output just print to screen for user manip or screenshot ##TODO? low priority)
runtimeLogFilePath = os.path.join(filingdir, f"aiirmap_project_runtime_logger.csv") #for tracking databases/runs and their completion, path to dataframe csv file with header (partially implemented, see runtimeLogger utility ##TODO?)


timeStr = '%Y%m%d_%H%M%S'
msTimeStr = f"{timeStr}_%f"


#------------------------------------------
#Dimensionality reduction related


#default settings for dimensionality reduction mapping
mapSettings_default = {
    #algorithm
    'dr': {
        'type': 'pca',  # One of ['pca']; The type of DR algorithm to run
        'n_components': 2,  # (int) Number of reduced dimensions (for PCA)
        'scale': True,  # (bool) scale training data before DR
    },
    #colummns to include/cut
    'cols': {
        'include': None,  # (list of col labels, None) Supercedes cols.drop if not None! The input cols to include in the DR. All others are ignored. Pass only inputs.
        'drop': [], #(list of col labels) columns to drop before the DR; reference, outputs, non-number, and columns with only one unique value are already excluded, combined with dr_colsToDrop_baseline,
    },
    #figure of merit filter
    'filter': {
        'fom': 'Jph_norm0',  # (str, None) figure of merit column label
        'lo': 0.99,  # (float, None) the lower bound cutoff for the figure of merit (expts w fom < val are cut)
        'hi': None,  # (float,None) the upper bound cutoff for the fom (expts with fom > val are cut)
        'stage': None,  # (str,None) include only expts with 'sim_type' == val
    },
    #reduced space sweep grid
    'sweep': {
        'distance': None, #(float, array, None) dr.pca.subspace_mesh distance; Manhattan distance, in original space, between mesh pts (array is original component-wise)
        'n_points': None, #(int,array, None) dr.pca.subspace_mesh; Number of sampling points. (array is reduced space component-wise)
        'boundaries': None #(array (2,n_redSpaceComps)) dr.pca.subspace_mesh; The lower and upper bounnds of the region to be sampled in the reduced space. [reduced space units] (default; +-5 times the std dev of the projected training data)
    }
}

#Excess columns to drop from the DR training set,
#Reference, non-number, and columns with only one unique value are already excluded
#cols.drop from mapSettings is combined with this list
#Note; cols.drop+colsToDrop_baseline is superceded by cols.include if it is present
dr_colsToDrop_baseline = []





#------------------------------------------
#Plot Related

dpi = 300



#------------------------------------------
#DataBase Related

#Inherit lineage lines from ancestors in DB Merge?
#How many lines?
inheritLineage = True
inheritLineageLength = 3

# dataframe index columns / csv save file related
inputStartStrs = ["experiment-inputs"] #strings used in first header column of csv files (idx column)
outputStartStrs = ["experiment-outputs"] #strings used to indicate the start of output columns (idx column)




#------------------------------------------
#Sentaurus related

#Which folder to use in the case of multiple optimization children folders for a node of interest
#Assumes that the child names are Sentaurus default such that the glob is ordered by runtime
#Pull the latest ; -1
childPullTimeIdx_default = -1


#tool name
sToolName='pyCollector'
#file names
#will be addended with _Info.csv,_execTime.csv
sFileNameBody='pyCollection'


#.lower() strings used to define booleans in swb params
bool_true_lowerlist = ["true", "on", "t", "yes"]
bool_false_lowerlist = ["false", "off", "f", "no"]



#Sentaurus file copying defaults and tool info

#tool names are the types of tools one could have
#!these are not the tool labels (which are, rather, instances of these tools in a sim flow)
#in gtree read: used to identify nodes/params which are the tool itself
toolNameList = ["epi", "sde", "mpr", "tdx", "sdevice", "svisual", "python", "genopt"]

#this info is used to anticipate the command file names for each tool in the project
#matched list of default file name endings for the tool types above
cmdFileEndings = ["_epi.csv", ["_dvs.cmd","_epi.csv"], "_mpr.cmd", "_tdx.tcl", "_des.cmd", "_vis.tcl", "_pyt.py", ["_opt.py", "_stg_opt.py"]]

#list of other sentaurus files to try and save (path globs relative to sentaurus working directory)
sfilesToSave = ["gvars.dat", "gtree.dat", "gscens.dat", "genopt.py", "results/logs", "results/nodes/optimizer/genopt*", "results/nodes/*/*_opt.out",
                "results/nodes/*/*swbVars.csv", "results/nodes/*/*OptProg.csv", f"{sFileNameBody}_Info.csv", f"{sFileNameBody}_execTimes.csv", 
                "aiirmap-bash-output*.log", "results/nodes/*/*LC.csv", "results/nodes/*/*_pyt.out", "results/nodes/*/J*.csv",
                "results/nodes/*/*GenRate.csv", "results/nodes/*/*GenRatePlot*","results/nodes/*/*LayerThs.csv" ]

# Locked Tool Lists for Stage 1 and Stage 3
# as of 230816 (added LCplotter after LCMatrix)
lockedTools_default_s4Opt = [0,0,1,0,0,1,1,1,1,1,1,1,1,1,1,1,1,0]
#lockedTools_default_electrical = [0,0,1,0,0,0,0,1,1,1,1,0,0,1,1,0,1,0]
# 240501; include the JV svisual to generate the J and its contributions data
lockedTools_default_electrical = [0,0,1,0,0,0,0,1,1,1,1,0,0,1,0,0,1,0]
# lockedTools_default_electrical = [0,0,0,0,0,0,1,1,1,1,0,0,1,0,0,1,0] #does not include the structure svisual tool

# maximum number of experiments to run in sentaurus project for grid runner functionality
maxExptsPerProj = 20


##Parameters for Sentaurus built-in optimization functionality (gopt) which creates child projects for each expt optimization
##(functionality when optimization expts are added into the parent folder is not implemented, use children option)

#list of other sentaurus files to try and save for optimization child projects (path globs relative to sentaurus working directory)
sfilesToSave_optChildren = sfilesToSave.copy()
sfilesToSave_optChildren.remove("results/nodes/*/*swbVars.csv")

# Default settings to pull the optimal expts from the child dbs
# (see si.get_pullIdxForOptChildren and aiirmapCommon.collectAndSave_sParentChild_DBs)
optMethod_default = 'Nelder-Mead' #'L-BFGS-B' #if 'optimizer' parameter is not included in the parent project then the code will assume all children use this opt algorithm
optFinalPullIdx_default = -1 #'numParams+1' #what child db expt index to use as the optimal result if the optimizer method is not recognized (if it is not L-BFGS-B, SLSQP, or Nelder-Mead)
#set optFinalPullIdx_default to an int to hard code the idx (eg. -1 => last), set to anything else to use the number of opt parameters plus one




