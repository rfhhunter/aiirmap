"""
Functions for applying clustering using the algorithms imported immediately below


~2304+
"""


from aiirmapCommon import  *
from sklearn.cluster import AgglomerativeClustering as agcl
from sklearn.cluster import DBSCAN
from sklearn.cluster import KMeans
import hdbscan




def dbClusterTagging(indb, clusterCols, algorithm='hdbscan', algInputs={}, metric='euclidean', overwriteExistingLabels=True ):
    """
    Apply clustering to the data in the clusterCols of the input database.
        Add the column (cl-idx_<alg>...) with cluster indices to the database

    !!! Works inPlace... ie. the input db is altered !!!

    See below or output to screen on exact construction of the column label for each algorithm

    @param indb: [DataBase] input database to apply clustering to
    @param clusterCols: [list of string] the columns of the indb.database to include in the clustering analysis
    @param algorithm: [string, one of aggl(omerative),dbscan,hdbscan,kmeans] the clustering algorithm to use
    @param algInputs:[dict] inputs to send to the clustering algorithms. Some critical ones are filled from the defaults below if not supplied.
    @param metric: [str] the distance metric to use (eg. 'euclidean')
    @param overwriteExistingLabels: [bool] overwrite the column if it exists already?
    @return: indb [DataBase] the db with the column added
    @return: clusterObj [cluster object] the return from whichever of the four cluster algs was run
    @return: colLabel [str] the label used for the cluster idx column
    """
    #automatically is inPlace (ie. runs upon the input db)
    #generate the cl-idx_{alg}-{nCl} column(s) desired

    #(aiirpower defaults clusterCols = arcThLabelsTtoB_S4+absThLabelsTtoB_bottomHomo_S4)

    #defaults to be used if algInputs is empty or missing one of these entries
    agglDefaults = {'linkage': 'single', 'n_clusters': 3,}
    dbscanDefaults = {'eps': 0.12, 'min_samples': 5,}
    hdbscanDefaults = {'min_cluster_size': 5, 'min_samples': 5,}
    kmeansDefaults = {'n_clusters': 3}

    #check for algorithm
    algorithm = algorithm.lower()
    alg = algorithm[0:4]
    if alg not in ['aggl', 'dbsc', 'hdbs', 'kmea']:
        print(f"WARNING :db cluster tagging: Algorithm '{algorithm}' not recognized. Should be one of ['aggl', 'dbscan', 'hdbscan', 'kmeans']. Using 'hdbscan' with parameters {hdbscanDefaults}...")
        algorithm = 'hdbscan'
        algInputs = hdbscanDefaults

    #fill required alg inputs missing from algInputs from the defaults
    colLabel = "cl-idx_" + alg + "_"
    labelKey = "Cluster Index: algorithm "
    if algorithm == 'aggl':
        if algInputs is None:
            algInputs = agglDefaults
        else:
            if not 'linkage' in algInputs:
                algInputs['linkage'] = agglDefaults['linkage']
            if not 'n_clusters' in algInputs and not 'distance_threshold' in algInputs:
                algInputs['n_clusters'] = agglDefaults['n_clusters']
        colLabel += f"{algInputs['linkage'][0:1]}-{algInputs['n_clusters']}"
        labelKey += "linkage n_clusters"
    elif algorithm == 'dbscan':
        if algInputs  is None:
            algInputs = dbscanDefaults
        else:
            if not 'eps' in algInputs:
                algInputs['eps'] = dbscanDefaults['eps']
            if not 'min_samples' in algInputs:
                algInputs['min_samples'] = dbscanDefaults['min_samples']
        colLabel += f"{algInputs['eps']}-{algInputs['min_samples']}"
        labelKey += "eps min_samples"
    elif algorithm == 'hdbscan':
        if algInputs is None:
            algInputs = hdbscanDefaults
        else:
            if not 'min_cluster_size' in algInputs:
                algInputs['min_cluster_size'] = hdbscanDefaults['min_cluster_size']
            if not 'min_samples' in algInputs:
                algInputs['min_samples'] = hdbscanDefaults['min_samples']
        colLabel += f"{algInputs['min_cluster_size']}-{algInputs['min_samples']}"
        labelKey += "min_cluster_size min_samples"
    elif algorithm == 'kmeans':
        if algInputs  is None:
            algInputs = kmeansDefaults
        elif not 'n_clusters' in algInputs:
            algInputs['n_clusters'] = kmeansDefaults['n_clusters']
        colLabel += f"{algInputs['n_clusters']}"
        labelKey += "n_clusters"


    print(f"db cluster tagging: Applying '{algorithm}' clustering algorithm to {indb.dbfilename}.")
    print(f"db cluster tagging: Column indexing will be located in '{colLabel}' indicating '{labelKey}'.")

    #check if the col exists
    if colLabel in indb.dataframe:
        if overwriteExistingLabels:
            print(f"WARNING :db cluster tagging: The column '{colLabel}' already exists in the input DataBase. Overwriting...")
        else:
            print(f"ERROR :db cluster tagging: The column '{colLabel}' already exists in the input DataBase. overwriteExistingLabels is False. Cannot continue. Returning input db as is...")
            return indb


    #do the tagging
    if alg == 'aggl':
        clusterObj = agcl(**algInputs, metric=metric)
        clusterObj.fit(indb.dataframe[clusterCols])
        indb.dataframe[colLabel] = clusterObj.labels_

    elif alg == 'dbsc':
        clusterObj = DBSCAN(**algInputs, metric=metric)
        clusterObj.fit(indb.dataframe[clusterCols])
        indb.dataframe[colLabel] = clusterObj.labels_

    elif alg == 'hdbs':
        clusterObj = hdbscan.HDBSCAN(**algInputs, metric=metric, gen_min_span_tree=True)
        clusterObj.fit(indb.dataframe[clusterCols])
        indb.dataframe[colLabel] = clusterObj.labels_

    elif alg == 'kmea':
        clusterObj = KMeans(**algInputs)
        clusterObj.fit(indb.dataframe[clusterCols].to_numpy())
        indb.dataframe[colLabel] = clusterObj.labels_


    print(f"db cluster tagging: Success : Column {colLabel} added to the input database. (number of clusters in labels = {len(clusterObj.labels_)}) Database is altered and is not saved.")

    return indb, clusterObj, colLabel




def getClusterStats(db, clIdxCol, statCols, meanOnly=False, printStats=True, save=True, savePath=None, plot=True):
    """
    Get stats for each cluster in the db
        Clustering indexing according to clIdxCol (must be present, use dbClusterTagging to tag)

    Returns dataframe with
    clusters as rows
    and columns;
        cluster label, number of pts, means for each statCol, std-devs for each statCol, mins, medians, maxs for each statCol


    @param db: [DataBase] dbh.DataBase to analyze
    @param clIdxCol: [str] the label of the column in the db with the cluster indices
    @param statCols: [str or list of str or None] the columns to compute the stats for, if none tries to compute for all cols but the clIdxCol
    @param meanOnly: [bool] the output datafram will contain the mean columns only
    @param printStats: [bool] print to screen the stats?
    @param save: [bool] save the stats (as csv)
    @param savePath: [str or None] path to save the csv file, if None uses dbdir with a preconfigured name
    @param plot: [bool] run the plotClusterStatsDF function at the end of cluster stat gathering
    @return: statsDF: [pd.DataFrame] the stats dataframe described above
    """
    #(aiirpower defaults for clIdxCol='cl-idx_hdbs_5-5', statCols=arcThLabelsTtoB_S4+absThLabelsTtoB_bottomHomo_S4+['Jph_norm1'])

    #prep
    if save and savePath is None:
        if db.dbfilename is None:
            savePath = os.path.join(dbdir, f'{time.strftime(timeStr)}_cluster-stats.csv')
        else:
            savePath = os.path.join(dbdir, f'{db.dbfilename[-4:]}_cluster-stats.csv')

    if save and savePath[-4:] != '.csv': savePath = savePath + ".csv"

    if clIdxCol is None:
        print('ERROR :get cluster stats: Must supply clIdxCol. Cannot continue. Exiting...')
        return None

    if not clIdxCol in db.dataframe:
        print(f'ERROR :get cluster stats: Column "{clIdxCol}" does not exist in the database. Run dbClusterTagging with the necessary parameters. Exiting...')
        return None

    if statCols is None: #use all!
        statCols = list(db.dataframe.columns.values)
        # statCols.remove(clIdxCol)

    if not isinstance(statCols, list): statCols = [statCols]


    clusterLabels = db.dataframe[clIdxCol].unique()
    clusterLabels.sort()
    numClusters = len(db.dataframe[clIdxCol].unique())

    print(f"get cluster stats: Getting '{clIdxCol}' cluster population, means, std-devs, mins, medians, and maxs for DB {db.dbfilename}... \n"
          f"get cluster stats: Stat columns; {statCols}")


    #prep column headings, create empty dataframe
    statsLabels = [clIdxCol, 'num_pts']
    for didx in range(len(statCols)): statsLabels += [f'mean_{statCols[didx]}']
    if not meanOnly:
        for didx in range(len(statCols)): statsLabels += [f'std-dev_{statCols[didx]}']
        for didx in range(len(statCols)): statsLabels += [f'min_{statCols[didx]}']
        for didx in range(len(statCols)): statsLabels += [f'median_{statCols[didx]}']
        for didx in range(len(statCols)): statsLabels += [f'max_{statCols[didx]}']
    statsDF = pd.DataFrame(columns = statsLabels)

    # get data for each cluster and append to dataframe
    for clidx in range(numClusters):
        clDF = db.dataframe[db.dataframe[clIdxCol] == clusterLabels[clidx]]

        clData  = [clusterLabels[clidx], len(clDF)]
        for didx in range(len(statCols)):
            try: clData += [clDF[statCols[didx]].mean()]
            except TypeError: clData += [np.nan]
        if not meanOnly:
            for didx in range(len(statCols)):
                try: clData += [clDF[statCols[didx]].std()]
                except TypeError: clData += [np.nan]
            for didx in range(len(statCols)):
                try:clData += [clDF[statCols[didx]].min()]
                except TypeError: clData += [np.nan]
            for didx in range(len(statCols)):
                try: clData += [clDF[statCols[didx]].median()]
                except TypeError: clData += [np.nan]
            for didx in range(len(statCols)):
                try: clData += [clDF[statCols[didx]].max()]
                except TypeError: clData += [np.nan]

        statsDF = pd.concat([statsDF, pd.DataFrame([clData], columns=statsLabels)], axis=0, join='outer')
        statsDF.index = range(len(statsDF))

    header = f"Cluster Stats Data\nCreated,{time.strftime(timeStr)},\nDB,{db.dbFile},\ntotal_expts,{len(db.dataframe)},\n"

    #print and/or save
    if printStats:
        print(f"-----------------------------------------------------------------------------------------------------")
        print(header)
        with pd.option_context('display.max_rows', None, 'display.max_columns', None):  # more options can be specified also
            print(statsDF)
        print(f"-----------------------------------------------------------------------------------------------------")

    if save:
        if not os.path.exists(os.path.split(savePath)[0]):
            os.makedirs(os.path.split(savePath)[0])
        with open(savePath, 'w') as f:
            f.write(header)
            statsDF.to_csv(path_or_buf=savePath, sep=',')
        print(f"get cluster stats: Saved file; '{savePath}'")

    print(f"get cluster stats: Success :Cluster stats for column '{clIdxCol}' and database '{db.dbfilename}' computed.")

    if plot:
        plotClusterStatsDF(statsDF, statCols=statCols)

    return statsDF




def plotClusterStatsDF(statsDF):
    #Very basic
    #No save functionality
    #make a num_pts histogram
    #make a plot of every other dimension showing all data

    # ^ = max
    # . = mean
    # v = min
    # _ = median
    # * = std dev
    allCols = statsDF.columns.values
    dimCols = allCols[2:]
    clIdxLabel = allCols[0]

    f, ax = plt.subplots()
    f.canvas.manager.set_window_title(f"Num Pts Histo")
    cl_idxs = statsDF.iloc[:, 0] #cl-idx_...
    num_pts = statsDF.loc[:,'num_pts']

    statsDF.plot.bar(x=clIdxLabel,y='num_pts')


    if len(statsDF) == 1:
        ptLocs= [0.5]
    elif len(statsDF) == 2:
        ptLocs = [0.33, 0.67]
    elif len(statsDF) == 3:
        ptLocs = [0.25, 0.5, 0.75]
    elif len(statsDF) == 4:
        ptLocs = [0.2, 0.4, 0.6, 0.8]
    elif len(statsDF) == 5:
        ptLocs = [0.17, 0.33, 0.5, 0.67, 0.83]
    elif len(statsDF) == 6:
        ptLocs = [0.14, 0.28, 0.42, 0.56, 0.7, 0.84]
    elif len(statsDF) == 7:
        ptLocs = [0.125, 0.25,0.375,0.5,0.625,0.75,0.875]
    else:
        print(f"ERROR :plot cluster stats df: Stats DF is too large... Dataframe length must be 7 or less... Exiting...")
        return

    for cidx in range(int(len(dimCols)/5)):
        f, ax = plt.subplots()
        f.canvas.manager.set_window_title(f"{dimCols[cidx]} centroid stats")
        for pidx in range(len(ptLocs)):
            plt.plot(ptLocs[pidx], statsDF.loc[pidx, f"{dimCols[cidx]}"], marker="."  )
            plt.plot(ptLocs[pidx], statsDF.loc[pidx, f"{dimCols[cidx+len(ptLocs)]}"], marker="*"  )
            plt.plot(ptLocs[pidx], statsDF.loc[pidx, f"{dimCols[cidx+len(ptLocs)*2]}"], marker="v"  )
            plt.plot(ptLocs[pidx], statsDF.loc[pidx, f"{dimCols[cidx+len(ptLocs)*3]}"], marker="_"  )
            plt.plot(ptLocs[pidx], statsDF.loc[pidx, f"{dimCols[cidx+len(ptLocs)*4]}"], marker="^"  )




def dbSplitOnClusters(indb, outputOnlyCentroid=False, outdbCols=None, clIdxCol=None, algorithm='hdbscan', algInputs={},
                      clusterCols=None, metric='euclidean', fileDescriptor=None, save=False, saveLoc=dbdir):
    """
    #output several dbs based upon the clusters in the input db;
        #the cluster centroids db
        #the cluster dbs (the original db datapoints belonging to each cluster, split to its own db)

    #include only outdbCols in the output dbs, if this is None include all columns

    #run clustering using dbClusterTagging if the clIdxCol is None or is not present (clusterCols must not be None in this case)
        # if clIdxCol interpretable using set algorithm dependent structure of cl-idx col labels; use clIdxCol
        # otherwise; use the algorithm and algInputs params

    :param indb: [DataBase] input database to analyze
    :param outputOnlyCentroid: [bool] the clusterDBs are not output (second return variable is None)
    :param outdbCols: [list of str] the labels for the cols to include in the output dbs, all if None
    :param clIdxCol: [str] label for the cluster indexing column
    :param algorithm: [str] algorithm to use for clustering if clIdxCol is not interpretable or is None
    :param algInputs: [dict] inputs for algorithm
    :param clusterCols: [list of str] list of col labels to use for clustering again only if clIdxCol is None or is not interpretable
    :param metric: [str] the metric  for the clustering (only used if new clustering is needed)
    :param fileDescriptor: [str] file descriptor for output db names
    :param save: [bool] save (True) or just create (False)
    :param saveLoc: [str] path to save folder, only used if save is True
    :return: centroidDB, [DataBase] a database with the cluster centroid locations
    :return: clusterDBs, [list of DataBases] a database for each cluster, only including the points in that cluster
    """


    #prep
    #handle clIdxCol, ensure column is valid and in db, running clustering as needed
    if clIdxCol is None:
        print(f"db split on clusters: clIdxCol is None. Clustering according to algorithm and algInputs parameters...")
        if clusterCols is None:
            print("ERROR :db split on clusters: clIdxCol is None but so is clusterCols. Cannot cluser and thus cannot split! Returning None...")
            return None, None
        indb, _, clIdxCol = dbClusterTagging(indb, clusterCols=clusterCols, algorithm=algorithm, algInputs=algInputs, metric=metric)
    else:
        if not clIdxCol in indb.dataframe:
            print(f"WARNING :db split on clusters: Column '{clIdxCol}' is not in the input db.")

            alg, algIns = translateClIdxColLabel(clIdxCol)
            if not alg == None:
                print(f"db split on clusters: clIdxCol interpretable. Clustering according to missing clIdxCol '{clIdxCol}' ...")
                indb, _, clIdxCol = dbClusterTagging(indb, clusterCols=clusterCols, algorithm=alg, algInputs=algIns, metric=metric)
            else:
                print(f"db split on clusters: clIdxCol un-interpretable. Clustering according to algorithm and algInputs parameters...")
                indb, _, clIdxCol = dbClusterTagging(indb, clusterCols=clusterCols, algorithm=algorithm, algInputs=algInputs, metric=metric)


    #cont prep
    if fileDescriptor is None:
        if indb.dbFile is not None:
            fileDescriptor = os.path.split(indb.dbFile)[1][0:-4]+f"_{clIdxCol}"
        else:
            fileDescriptor = f"{time.strftime(timeStr)}_{clIdxCol}"

    if saveLoc is None: saveLoc=dbdir
    if not outputOnlyCentroid: saveLoc = os.path.join(saveLoc,fileDescriptor) #use folder if multiple plots
    if save and not os.path.exists(saveLoc): os.makedirs(saveLoc)




    #column found or added can now split,
    # pull some info and then create db for each cluster
    clusterLabels = indb.dataframe[clIdxCol].unique()
    clusterLabels.sort()
    numClusters = len(indb.dataframe[clIdxCol].unique())

    clusterDBs = []
    if not outputOnlyCentroid:
        for clidx in range(numClusters):
            cldb = indb.dbCopy(new_dbFile=os.path.join(saveLoc, f"{fileDescriptor}_{clusterLabels[clidx]}.csv"))
            cldb.dataframe = indb.dataframe[indb.dataframe[clIdxCol] == clusterLabels[clidx]]
            if outdbCols is not None:
                cldb.dataframe = cldb.dataframe.loc[:, outdbCols]
            cldb.dbCleanIdxsAndGridness()
            clusterDBs.append(cldb)
            if save:
                cldb.dbSaveFile(saveName=os.path.join(saveLoc, f"{fileDescriptor}_{clusterLabels[clidx]}.csv"))

    centroidDB = dbh.DataBase()
    centroidDB.dbFile = os.path.join(saveLoc, f"{fileDescriptor}_centroids.csv")
    centroidDB.dbfilename = os.path.split(centroidDB.dbFile)[1]
    clusterDataDF = getClusterStats(indb, clIdxCol=clIdxCol, statCols=outdbCols, printStats=False, save=False, meanOnly=True)
    oldLabels = list(clusterDataDF.columns.values)
    labelSwitchDict = {}
    for colidx in range(len(oldLabels)): #remove "mean_" from start of colu headers
        if oldLabels[colidx][0:5] == "mean_":
            labelSwitchDict[oldLabels[colidx]] = oldLabels[colidx][5:]
        else:
            labelSwitchDict[oldLabels[colidx]] = oldLabels[colidx]
    clusterDataDF.rename(columns=labelSwitchDict,inplace=True)
    centroidDB.dataframe = clusterDataDF
    centroidDB.dbCleanIdxsAndGridness()
    if save:
        centroidDB.dbSaveFile(saveName=os.path.join(saveLoc, f"{fileDescriptor}_centroids.csv"))

    return centroidDB, clusterDBs




def translateClIdxColLabel(clIdxColIn):
    """
    Obtain the algorithm and algInputs from a cl-idx column label using established formats

    Format;
    cl-idx_algo_par-ams
        where algo = aggl, dbsc, hdbs, or kmea
        and par-ams =
                l(inkage)-n_clusters for aggl
                eps-min_samples for dbsc
                min_cluster_size-min_samples for hdbs
                n_clusters for kmea

    eg. 'cl-idx_hdbs_5-5'

    Any malformed results return None


    @param clIdxColIn: [str] the cluster index column label to interpret
    @return: alg: [str] 'aggl', 'dbsc', 'hdbs', or 'kmea'
    @return: algInputs: [dict] the key algorithm inputs given by the column label (as above)
    """


    clIdxCol = clIdxColIn
    clIdxCol = clIdxCol[7:] #remove cl-idx_
    alg = clIdxCol[0:4]
    if alg not in ['aggl', 'dbsc', 'hdbs', 'kmea']:
        print(f"ERROR :translate clIdxCol: clIdxCol '{clIdxColIn}' is malformed. Returning None...")
        return None, None
    clIdxCol = clIdxCol[5:]
    algInputValStrs = clIdxCol.split('-')
    if alg == 'aggl':
        if len(algInputValStrs) != 2:
            print(f"ERROR :translate clIdxCol: clIdxCol '{clIdxColIn}' is malformed. Returning None...")
            return None, None
        else:
            if algInputValStrs[0] == 's': link = 'single'
            elif algInputValStrs[0] == 'c': link = 'complete'
            elif algInputValStrs[0] == 'a': link = 'average'
            else: link = 'ward'
            algInputs={'linkage':link, 'n_clusters':int(algInputValStrs[1]),}
    elif alg == 'dbsc':
        if len(algInputValStrs) != 2:
            print(f"ERROR :translate clIdxCol: clIdxCol '{clIdxColIn}' is malformed. Returning None...")
            return None, None
        else:
            algInputs = {'eps':float(algInputValStrs[0]), 'min_samples':int(algInputValStrs[1]),}
    elif alg == 'hdbs':
        if len(algInputValStrs) != 2:
            print(f"ERROR :translate clIdxCol: clIdxCol '{clIdxColIn}' is malformed. Returning None...")
            return None, None
        else:
            algInputs = {'min_cluster_size':int(algInputValStrs[0]), 'min_samples':int(algInputValStrs[1]),}
    elif alg == 'kmea':
        if len(algInputValStrs) != 1:
            print(f"ERROR :translate clIdxCol: clIdxCol '{clIdxColIn}' is malformed. Returning None...")
            return None, None
        else:
            algInputs = {'n_clusters':int(algInputValStrs[0]),}
    return alg, algInputs




def dbClusterAndSplit(db, numClusters=3, algorithm='Kmeans', fileDescriptor=None, save=True, saveLoc=dbdir):
    """
    #!THIS FNC MAY BE ONLY VALID WITH algorithm='Kmeans' I AM NOT SURE!

    #take a single db, run clustering, output a list of dbs containing the cluster points
    # which clustering alg? how many clusters to choose? --> This must be supplied via user
    # db analyzed as is (no truncation or sorting)

    # Results will be stored as {saveLoc}/{fileDescriptor}_cluster_{algorithm}-{numClusters}_cluster-{n}.csv
    #  if None, fileDescriptor default is dbFile's dbfilename, or time string if dbFile is unavailable

    :param db: [DataBase] input database to run clustering on
    :param numClusters: [int] how many clusters to identify
    :param algorithm: [str] clustering algorithm to use
    :param fileDescriptor: [str] for the save filename (see above)
    :param save: [bool] to save? or just create (if False)
    :param saveLoc: [str] path to folder to save file (see above)
    :return: outDBs [list of DataBases] databases with the points in each cluster
    """


    #handle fileDesc
    if fileDescriptor is None:
        if db.dbFile is not None:
            fileDescriptor = os.path.split(db.dbFile)[1]
            fileDescriptor = fileDescriptor[:-4]
        else:
            fileDescriptor = f'{time.strftime(timeStr)}'

    if saveLoc is None:
        saveLoc = dbdir

    if save:
        print(f'db cluster and split: Splitting db using {algorithm} with {numClusters} clusters. Results will be stored as {saveLoc}/{fileDescriptor}_cluster_{algorithm}-{numClusters}_cl-n.csv...')
        if not os.path.exists(saveLoc):
            os.makedirs(saveLoc)
        else:
            print(f"WARNING :db cluster and split: Overwriting databases in existing folder '{saveLoc}'...")
    else:
        print(f'db cluster and split: Splitting db using {algorithm} with {numClusters} clusters. save is False. Results are not saved...')


    #check if cl-idx_<algorithm>-{numClusters} column exists in the db
    idxColLabel = f'cl-idx_{algorithm}-{numClusters}'
    try:
        x = db.dataframe[idxColLabel]
    except KeyError:
        #run clustering tagging algorithm
        print(f'db cluster and split: Column "{idxColLabel}" does not exist. Running cluster tagging...')
        db = dbClusterTagging(algorithm, numClusters)


    #check/get stats
    statDF = getClusterStats(db, algorithm, numClusters, print=True, save=True, savePath=os.path.join(f'{saveLoc}', f'{fileDescriptor}_cluster_{algorithm}-{numClusters}_stats.csv'))
    clStatLens = list(statDF['num_pts'].values)


    #split db based on cluster idx
    outDBs = []
    for clidx in range(len(numClusters)):

        #create db
        outdb = db.dbCopy()
        outdb.dbFile = os.path.join(saveLoc, f'{fileDescriptor}_cluster_{algorithm}-{numClusters}_cl-{clidx}.csv')
        outdb.dbfilename = os.path.split(outdb.dbFile)[1]
        outdb.grid = False
        outdb.lineage.append([time.strftime(timeStr), f'cluster split using {algorithm} with {numClusters} clusters, this db is cluster {clidx} data'])

        #filter db for cluster idx
        outdb.dataframe = outdb.dataframe[outdb.dataframe[f'cl-idx_{algorithm}-{numClusters}'] == {clidx}]

        #clean and store
        outdb.dbCleanIdxsAndGridness()
        outDBs.append(outdb)

        #run length check
        if len(outdb.dataframe) != clStatLens[clidx]:
            print(f"WARNING :db cluster and split: The number of points in cluster {clidx} does not match for the input ({clStatLens[clidx]}) and output ({len(outdb.dataframe)}) DBs.")

    if save:
        for dbidx in range(len(outDBs)):
            outDBs[dbidx].dbSaveFile(saveName={outDBs[dbidx].dbfilename})
        print(f'db cluster and split: Success : Cluster split DBs {saveLoc}/{fileDescriptor}_cluster_{algorithm}-{numClusters}... created.')
    else:
        print(f'db cluster and split: Success : Cluster split DBs using {algorithm} with {numClusters} clusters... created but NOT SAVED.')

    return outDBs




