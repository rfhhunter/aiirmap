"""
Plotting Functions

Includes;
- AutoViz (Dataframes, DataBases, csvs)
- DB plotting
- DR plotting
- Plot Helper functions

22.06.29
"""
import itertools
import os.path
import sys
import time

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import sklearn.metrics
from scipy.cluster.hierarchy import dendrogram, linkage

sys.path.append("..")
from aiirmapCommon import *
from config import *
import utilities.BeerLambertTools as bl




linestyle_tuple = [
     ('loosely dotted',		   (0, (1, 10))),
     ('dotted',				   (0, (1, 1))),
     ('densely dotted',		   (0, (1, 1))),
     ('loosely dashed',		   (0, (5, 10))),
     ('dashed',				   (0, (5, 5))),
     ('densely dashed',		   (0, (5, 1))),
     ('loosely dashdotted',    (0, (3, 10, 1, 10))),
     ('dashdotted',			   (0, (3, 5, 1, 5))),
     ('densely dashdotted',    (0, (3, 1, 1, 1))),
     ('dashdotdotted',		   (0, (3, 5, 1, 5, 1, 5))),
     ('loosely dashdotdotted', (0, (3, 10, 1, 10, 1, 10))),
     ('densely dashdotdotted', (0, (3, 1, 1, 1, 1, 1)))]
#https://matplotlib.org/stable/gallery/lines_bars_and_markers/linestyles.html


plt.ioff()
# plt.ion()


#------------------------------------------------------------------------------------------------------------------
#------------------------------------------------------------------------------------------------------------------
#	 AutoViz
#------------------------------------------------------------------------------------------------------------------
#------------------------------------------------------------------------------------------------------------------


def AutoViz(data, saveFolder=pldir, chart_format='png', verbose=2, **kwargs):
    """
    Use the AutoViz module to quickly plot data. Plots each data object separately
    Accepted data type is single or list of csv path strings, dataframes, or databases
    :param data: [see above] The data to plot.
    :param saveFolder: [string] path to folder in which to create an AutoViz save folder
    :param chart_format: [string] output type, see link below
    :param verbose: [int=0,1,or 2] how much output is given to console, ALSO if the plots are inline (0,1) or silently saved (2), see link
    :param kwargs: The (other) keyword arguments of the AutoViz_Class
    https://github.com/AutoViML/AutoViz/blob/master/README.md
            depVar; the target variable in your dataset
    :return: None: outputs plots according to AutoViz chart_format option
    """

    # prep save folder if not given
    if saveFolder == pldir or saveFolder == None:
        saveFolder = os.path.join(saveFolder, f"{time.strftime(timeStr)}_AutoViz")

    #Detect data type
    #single or list of csvs, dataframes, databases
    if not isinstance(data, list):
        data = [data]

    dataType = None
    if isinstance(data[0], pd.DataFrame):
        dataType = "df"
    elif isinstance(data[0], str):
        dataType = "csv"
    else:
        dataType = "db"


    print(f"autoviz: AutoViz '{dataType}' with chart_format={chart_format}...")

    if len(data) >= 3:
        print(f"WARNING :autoviz: Several data objects to visualize, this may take some time depending upon the data size. ")


    #Run AutoViz for each data object based upon file type
    from autoviz.AutoViz_Class import AutoViz_Class
    from .databasing import csvFindHeader
    AV = AutoViz_Class()

    if dataType == "df":
        for didx in range(len(data)):
            dSaveFolder = os.path.join(saveFolder, f"dataframe_{didx}")
            if not os.path.exists(dSaveFolder):
                os.makedirs(dSaveFolder)
            datadisplay = AV.AutoViz("", dfte = data[didx], save_plot_dir=dSaveFolder, chart_format=chart_format, verbose=verbose, **kwargs)

    elif dataType == "csv":
        for didx in range(len(data)):
            dSaveFolder = os.path.join(saveFolder, f"db_{didx}_{os.path.split(data[didx])[1][:-4]}")
            if not os.path.exists(dSaveFolder):
                os.makedirs(dSaveFolder)
            headerlen = csvFindHeader(data[didx])
            datadisplay = AV.AutoViz(data[didx], save_plot_dir=dSaveFolder, chart_format=chart_format, header=headerlen, verbose=verbose, **kwargs)

    else: #db
        for didx in range(len(data)):
            dSaveFolder = os.path.join(saveFolder, f"db_{didx}_{data[didx].dbfilename[:-4]}")
            if not os.path.exists(dSaveFolder):
                os.makedirs(dSaveFolder)
            datadisplay = AV.AutoViz("", dfte = data[didx].dataframe, save_plot_dir=dSaveFolder, chart_format=chart_format, verbose=verbose, **kwargs)

    print(f"autoviz: Success :AutoViz complete, '{chart_format}' plots if created are in '{saveFolder}'")


#------------------------------------------------------------------------------------------------------------------
#------------------------------------------------------------------------------------------------------------------
#	 Plotting DBs
#------------------------------------------------------------------------------------------------------------------
#------------------------------------------------------------------------------------------------------------------

def plotDBs(DBs, x=None, y='Eff', multiY=True,  multiDB=True, saveName=None, xlabel=None, ylabel=None, title=None, labels=None, colours=None, legend=True, yscale='linear', func_fit=None, pcolormesh='',pcolormesh_label=None, secondary=None, **kwargs):

    """
    Plot DB data using pandas.dataframe.plot().
    Using the multiDB/Y options it can be configured to combine data from different DBs, from different columns, or both.
    If multiDB=True and DBs is a list; plot all DBs in the same plot
    If multiY=True and y is a list; plot all columns in the same plot.
    In the case that multiDB/or/Y=False and the input is a list; plot each DB or y's data in a separate plot

    :param DBs: [DataBase or list of DataBases] the DBs to plot
    :param x: [string] the column name for the x axis data
    :param y: [string or list of strings] the column name(s) for the y-axis data
    :param multiY: [boolean] plot all y data (ie. y columns) in the same plot (True) or give each its own plot (False)
    :param multiDB: [boolean] plot all DBs in the same plot (True) or give each DB its own plot (False)
    :param saveName: [string] save filename or folder path (if the wrong type is supplied then the input is coerced), default is a folder or filename similar to ~'<timeStr>_<y>_<DB>'
    :param xlabel: [string] user supplied x label for plots, default/None uses the x data label
    :param ylabel: [string or list of strings] user supplied y label for plots' axis, default/None uses the y data label(s)
    :param title: [string], user supplied title string base (will be addended with y labels and db idxs as needed)
    :param labels: [list of 2 lists of strings], [[y labels],[db labels]], user supplied data labels for plot legends, default/None uses the format "DB<dbidx> <yLabel>" omitting a clarifier if possible
    :param colours: [list of 2 lists of strings], [[y colours],[db colours]], user supplied colours for the trends, if multiY And multiDB then uses DB colours and varies y line style, otherwise uses the given colours for whatever trend type is multi
    :param yscale: [string] plt.yscale type applied in plot
    :param func_fit: [function ], provide function you want to fit data to. Can only have 1 DBs
    :param pcolormesh [string ], Z component of a pcolor plot. empty string means do not use this.
    :param pcolormesh_label [string ], label for the colorbar axis
    :param secondary: [boolean list], follow order of ys, True --> on secondary
    :param kwargs: the keyword arguments of pandas.dataframe.plot (incorporates matplotlib kwargs)
    https://pandas.pydata.org/pandas-docs/stabsetttle/reference/api/pandas.DataFrame.plot.html
    :return: None: plots the figures
    """

    ##This function is a bit messy;
    #The multiY and multiDB functions are considered along with the number of dbs,ys
    #these are used to extract whether it is desired to plotOnlyOnePerFig of each type of data  ('dbs',and/or 'y') in the created plots
    #The four cases are handled separately in the preparation and plotting stage

    plt.rcParams['font.size'] = 16


    #VARIABLE HANDLING AND DATA PREP
    #create prep data (listify if singular) and default saveName (timeStr_y_DB)
    inputSaveName = saveName #save for later
    saveName = os.path.join(pldir, f"{time.strftime(timeStr)}_")
    plotOnlyOnePerFig=[] #Which dimensions to isolate in the plots (ie. whether to a create a plot for each value, or only one value)

    if not isinstance(y, list):
        y = [y]
        saveName += f"{y[0]}"
    else:
        if not multiY:
            plotOnlyOnePerFig.append('Y')
        saveName += f"_multiY"

    if not isinstance(DBs,list):
        DBs=[DBs]
        saveName += f"_{DBs[0].dbfilename[:-4]}"
    else:
        if not multiDB:
            plotOnlyOnePerFig.append('DB')
        saveName += f"_multiDB"

    if len(plotOnlyOnePerFig) == 0:
        saveName += f".png"

    # check type and use input saveName
    if inputSaveName != None:
        if inputSaveName[-4:-3] == "." and len(plotOnlyOnePerFig) != 0:
            saveName = inputSaveName[:-4]
        elif inputSaveName[-4:-3] != "." and len(plotOnlyOnePerFig) == 0:
            saveName = f"{inputSaveName}.png"
        else:
            saveName = inputSaveName

    if len(plotOnlyOnePerFig) != 0:
        if not os.path.isdir(saveName):
            os.makedirs(saveName)

    print(f"\nplot DBs: Plotting DB data to '{saveName}'. {y} vs. {x} will be plotted for {len(DBs)} DBs (Plot settings; multiY={multiY}, multiDB={multiDB}, plotOnlyOnePerFig={plotOnlyOnePerFig}).")

    #handle labels and colours inputs for multi plotting
    if ylabel is not None:
        if not isinstance(ylabel, list):
            ylabel = [ylabel]
        if len(ylabel) != 1 and len(ylabel) != len(y):
            print(f"WARNING :plot DBs: Length of supplied ylabel ({len(ylabel)}) is not 1 and does not match the number of y columns ({len(y)}). Setting to default...")
            ylabel = None

    if title is None:
        title = os.path.split(saveName)[1][:-4]

    if labels is not None:
        #Check to see if the labels input is the correct dimensions,
        #4 cases; single or multi, y or DB
        #allow certain subset inputs (string, list of strings) in certain scenarios
        if len(y) == 1 and len(DBs) == 1:
            if isinstance(labels,str):
                labels = [[""], [labels]]
            elif isinstance(labels,list):
                if len(labels) == 1:
                    labels=[[""], labels[0]]
                elif len(labels) > 2:
                    print(f"WARNING :plot DBs: Input labels dimensions ({len(labels[0])},{len(labels[1])}) do not match input DB ({len(DBs)}) and y-data ({len(y)}) dimensions. Setting to default... ")
                    labels=None
            else:
                print(f"WARNING :plot DBs: Input labels dimensions ({len(labels[0])},{len(labels[1])}) do not match input DB ({len(DBs)}) and y-data ({len(y)}) dimensions. Setting to default... ")
                labels = None

        elif len(DBs) == 1:
            if isinstance(labels,list):
                if len(labels) == len(y):
                    labels = [labels, [""]]
                elif len(labels) == 2:
                    if len(labels[0]) != len(y):
                        print(f"WARNING :plot DBs: Input labels dimensions ({len(labels[0])},{len(labels[1])}) do not match input DB ({len(DBs)}) and y-data ({len(y)}) dimensions. Setting to default... ")
                        labels = None
                else:
                    print(f"WARNING :plot DBs: Input labels dimensions ({len(labels[0])},{len(labels[1])}) do not match input DB ({len(DBs)}) and y-data ({len(y)}) dimensions. Setting to default... ")
                    labels = None
            else:
                print(f"WARNING :plot DBs: Input labels dimensions ({len(labels[0])},{len(labels[1])}) do not match input DB ({len(DBs)}) and y-data ({len(y)}) dimensions. Setting to default... ")
                labels = None

        elif len(y) == 1:
            if isinstance(labels, list):
                if len(labels) == len(DBs):
                    labels = [[""],labels]
                elif len(labels) == 2:
                    if len(labels[1]) != len(DBs):
                        print(f"WARNING :plot DBs: Input labels dimensions ({len(labels[0])},{len(labels[1])}) do not match input DB ({len(DBs)}) and y-data ({len(y)}) dimensions. Setting to default... ")
                        labels = None
                else:
                    print(f"WARNING :plot DBs: Input labels dimensions ({len(labels[0])},{len(labels[1])}) do not match input DB ({len(DBs)}) and y-data ({len(y)}) dimensions. Setting to default... ")
                    labels = None
            else:
                print(f"WARNING :plot DBs: Input labels dimensions ({len(labels[0])},{len(labels[1])}) do not match input DB ({len(DBs)}) and y-data ({len(y)}) dimensions. Setting to default... ")
                labels = None

        else: #multiple ys and multiple DBs
            if isinstance(labels, list):
                if len(labels) == 2:
                    if len(labels[1]) != len(DBs) or len(labels[0]) != len(y):
                        print(f"WARNING :plot DBs: Input labels dimensions ({len(labels[0])},{len(labels[1])}) do not match input DB ({len(DBs)}) and y-data ({len(y)}) dimensions. Setting to default... ")
                        labels = None
                else:
                    print(f"WARNING :plot DBs: Input labels dimensions ({len(labels[0])},{len(labels[1])}) do not match input DB ({len(DBs)}) and y-data ({len(y)}) dimensions. Setting to default... ")
                    labels = None
            else:
                print(f"WARNING :plot DBs: Input labels dimensions ({len(labels[0])},{len(labels[1])}) do not match input DB ({len(DBs)}) and y-data ({len(y)}) dimensions. Setting to default... ")
                labels = None

    if colours is not None:
        #copied from labels handler
        if len(y) == 1 and len(DBs) == 1:
            if isinstance(colours, str):
                colours = [[colours], [""]]
            elif isinstance(colours, list):
                if len(colours) == 1:
                    colours = [[colours[0]], ["k"]]
                elif len(colours) > 2:
                    print(f"WARNING :plot DBs: Input colours dimensions ({len(colours[0])},{len(colours[1])}) do not match input DB ({len(DBs)}) and y-data ({len(y)}) dimensions. Setting to default... ")
                    colours = None
            else:
                print(f"WARNING :plot DBs: Input colours dimensions ({len(colours[0])},{len(colours[1])}) do not match input DB ({len(DBs)}) and y-data ({len(y)}) dimensions. Setting to default... ")
                colours = None

        elif len(DBs) == 1:
            if isinstance(colours, list):
                if len(colours) == len(y):
                    colours = [colours, ["k"]]
                elif len(colours) == 2:
                    if len(colours[0]) != len(y):
                        print(f"WARNING :plot DBs: Input colours dimensions ({len(colours[0])},{len(colours[1])}) do not match input DB ({len(DBs)}) and y-data ({len(y)}) dimensions. Setting to default... ")
                        colours = None
                else:
                    print(f"WARNING :plot DBs: Input colours dimensions ({len(colours[0])},{len(colours[1])}) do not match input DB ({len(DBs)}) and y-data ({len(y)}) dimensions. Setting to default... ")
                    colours = None
            else:
                print(f"WARNING :plot DBs: Input colours dimensions ({len(colours[0])},{len(colours[1])}) do not match input DB ({len(DBs)}) and y-data ({len(y)}) dimensions. Setting to default... ")
                colours = None

        elif len(y) == 1:
            if isinstance(colours, list):
                if len(colours) == len(DBs):
                    colours = [["k"], colours]
                elif len(colours) == 2:
                    if len(colours[1]) != len(DBs):
                        print(f"WARNING :plot DBs: Input colours dimensions ({len(colours[0])},{len(colours[1])}) do not match input DB ({len(DBs)}) and y-data ({len(y)}) dimensions. Setting to default... ")
                        colours = None
                else:
                    print(f"WARNING :plot DBs: Input colours dimensions ({len(colours[0])},{len(colours[1])}) do not match input DB ({len(DBs)}) and y-data ({len(y)}) dimensions. Setting to default... ")
                    colours = None
            else:
                print(f"WARNING :plot DBs: Input colours dimensions ({len(colours[0])},{len(colours[1])}) do not match input DB ({len(DBs)}) and y-data ({len(y)}) dimensions. Setting to default... ")
                colours = None

        else:  # multiple ys and multiple DBs
            if isinstance(colours, list):
                if len(colours) == 2:
                    if len(colours[1]) != len(DBs) or len(colours[0]) != len(y):
                        print(f"WARNING :plot DBs: Input colours dimensions ({len(colours[0])},{len(colours[1])}) do not match input DB ({len(DBs)}) and y-data ({len(y)}) dimensions. Setting to default... ")
                        colours = None
                else:
                    print(f"WARNING :plot DBs: Input colours dimensions ({len(colours[0])},{len(colours[1])}) do not match input DB ({len(DBs)}) and y-data ({len(y)}) dimensions. Setting to default... ")
                    colours = None
            else:
                print(f"WARNING :plot DBs: Input colours dimensions ({len(colours[0])},{len(colours[1])}) do not match input DB ({len(DBs)}) and y-data ({len(y)}) dimensions. Setting to default... ")
                colours = None

    if secondary is None: secondary = [False]*len(y)



    #OUTPUT

    if not os.path.exists(os.path.split(saveName)[0]):
        os.makedirs(saveName)

    #print the list of database names and their indicies to screen and file (default labels use db-idx)

    if labels is None and len(DBs) > 1:
        if not plotOnlyOnePerFig:
            txtFile = os.path.join(os.path.split(saveName)[0], f"{time.strftime(timeStr)}_DB_ids.txt")
        else:
            txtFile = os.path.join(saveName, f"{time.strftime(timeStr)}_NumExpts.txt")
        with open(txtFile, 'w') as file:
            print(f"plot DBs: Databases and their indicies...")
            file.write(f"Plot databases and their indicies:\n{time.strftime(timeStr)}")
            for dbidx in range(len(DBs)):
                print(f"\t\t\tdb-{dbidx}:\t\t{DBs[dbidx].dbfilename}")
                file.write(f"db-{dbidx},{DBs[dbidx].dbfilename}")


    ###PLOTTING###
    #Handle each case separately; all trends in one plot, plots for each y, each DB, plots for each y and DB

    if len(plotOnlyOnePerFig) == 0: #plot all in the same plot or one y and one DB
        time.sleep(0.5)
        #plt.figure(title)
        ax = plt.gca()#axes()
        #plot first (originally separate to get axis)
        if labels is not None:
            label = f"{labels[0][0]} {labels[1][0]}"
        else:
            label = f"{y[0]} DB-{0} "
        plotStr = "DBs[0].dataframe.plot(x, y=y[0], ax=ax, label=label"
        if colours is not None:
            if len(y) > len(DBs):
                plotStr += ", color=colours[0][0]"
                if len(DBs) > 1:
                    plotStr += ", linestyle=linestyle_tuple[0][1]"
            else:
                plotStr += ", color=colours[1][0]"
        plotStr += ", **kwargs)"
        # ax = eval(plotStr)
        if pcolormesh:
            piv = pd.pivot_table(DBs[0].dataframe,columns=x,index=y[0],values=pcolormesh)
            pcm = ax.pcolormesh(piv.columns.values.tolist(),piv.index.values.tolist(),piv,shading='gouraud',**kwargs)
            cbar = plt.colorbar(pcm,label=pcolormesh_label,extend='neither')
            cbar.set_ticks(cbar.ax.get_yticks())
            plt.setp(cbar.ax.get_yticklabels()[1:-1:2],visible=False) #hide every second tick label (less crowded)
            cs = ax.contour(piv.columns.values.tolist(), piv.index.values.tolist(), piv, levels=cbar.ax.get_yticks(), colors='k')
            ax.clabel(cs,inline=True,fontsize=12)
        else:
            eval(plotStr)


        #plot the rest
        for dbidx in range(len(DBs)):
            for yidx in range(len(y)):
                if dbidx == 0 and yidx == 0:
                    if len(y) == 1: break
                    else: continue

                plotStr = 'DBs[dbidx].dataframe.plot(x, y=y[yidx], ax=ax'
                if secondary[yidx]:
                    plotStr+= ', secondary_y=True'
                if labels is not None:
                    plotStr += ', label=f"{labels[0][yidx]} {labels[1][dbidx]}"'
                else:
                    plotStr += ', label=f"{y[yidx]} DB-{dbidx}"'
                if colours is not None:
                    if len(y) > len(DBs):
                        plotStr += ', color=colours[0][yidx]'
                        if len(DBs) > 1:
                            plotStr += ', linestyle=linestyle_tuple[yidx%len(linestyle_tuple)][1]'
                    else:
                        plotStr += ', color=colours[1][dbidx]'
                plotStr += ",legend=False, **kwargs)"
                try: eval(plotStr)
                except: print(f"ERROR :plot dbs: Issue with eval string... \n'{plotStr}'")

        #handle the frame
        plt.title(title)

        if yscale !='linear':
                plt.yscale(yscale)

        if func_fit: # fit function to dataset
            from scipy.optimize import curve_fit
            res, cov = curve_fit(func_fit, DBs[0].dataframe[x], DBs[0].dataframe[y[0]])
            print('Fit parameters and uncertainties = ', res, np.diag(cov))
            plt.plot(DBs[0].dataframe[x], func_fit(DBs[0].dataframe[x], *res), 'k', label='Fit',linewidth=2)

        if not pcolormesh:
            if legend:
                h1,l1 = ax.get_legend_handles_labels()
                h2 =[]; l2 = []
                if secondary and True in secondary:
                    h2,l2 = ax.right_ax.get_legend_handles_labels()
                plt.legend(h1+h2,l1+l2)
            else:
                ax.get_legend().remove()

        if xlabel is not None:
            ax.set_xlabel(xlabel)
        else:
            ax.set_xlabel(x)

        if multiY:
            myl = ""
            myl2 = ""
            if ylabel is not None:
                for ylidx in range(len(ylabel)):
                    if secondary and secondary[ylidx]:
                        myl2 += f"{ylabel[ylidx]}, "
                    else:
                        myl += f"{ylabel[ylidx]}, "

            else:
                for yidx in range(len(y)):
                    if secondary and secondary[yidx]:
                        myl2 += f"{y[yidx]}, "
                    else:
                        myl += f"{y[yidx]}, "
            myl = myl[:-2]
            myl2 = myl2[:-2]
            ax.set_ylabel(myl)
            if secondary and True in secondary:
                ax.right_ax.set_ylabel(myl2)
        else:
            if ylabel is not None:
                plt.ylabel(ylabel[0])
            else:
                plt.ylabel(y[0])

        #save/end
        # plt.show(block=False)
        plt.tight_layout()
        plt.savefig(saveName, dpi = dpi)
        print(f"plot DBs: Success :Plotting complete. Plot saved to '{saveName}'")


    elif plotOnlyOnePerFig == ['Y']: #one plot for each y or only one y
        for yidx in range(len(y)):
            plt.figure(f"{y[yidx]}__{title}")
            ax = plt.axes()
            # get axis, plot first DB
            plotStr = "DBs[0].dataframe.plot(x, y=y[yidx], ax=ax, label="
            if labels is not None:
                plotStr += "f'{labels[1][0]}'"
            else:
                plotStr += "'DB-0'"
            if colours is not None:
                if len(DBs) > 1:
                    plotStr += ", color=colours[1][0]"
                else:
                    plotStr += ", color=colours[0][yidx]"
            plotStr += ", **kwargs)"
            # ax = eval(plotStr)
            eval(plotStr)
            # plot the rest
            for dbidx in range(len(DBs)):
                if dbidx == 0:
                    continue
                plotStr = 'DBs[dbidx].dataframe.plot(x, y=y[yidx], ax=ax'
                if labels is not None:
                    plotStr += ', label=f"{labels[1][dbidx]}"'
                else:
                    plotStr += ', label=f"DB-{dbidx}"'
                if colours is not None:
                    if len(DBs)>1:
                        plotStr += ', color=colours[1][dbidx]'
                    else:
                        plotStr += ', color=colours[0][yidx]'
                plotStr += ", **kwargs)"
                eval(plotStr)

            # handle the frame
            plt.title(f"{y[yidx]}__{title}")

            if legend: plt.legend(loc='best')
            else: ax.get_legend().remove()

            if xlabel is not None: plt.xlabel(xlabel)
            else: plt.xlabel(x)

            if ylabel is not None: plt.ylabel(ylabel[yidx])
            else: plt.ylabel(y[yidx])

            #save plot
            if multiDB: dbsavestr = '__multiDB'
            else: dbsavestr = ''
            plt.savefig(os.path.join(saveName, f"{time.strftime(timeStr)}__{y[yidx]}{dbsavestr}.png"), dpi=dpi)

        #end
        # plt.show(block=False)
        print(f"plot DBs: Success :Plotting complete. Plots saved to '{saveName}'")

    elif plotOnlyOnePerFig == ['DB']: # one plot for each DB or only one DB
        for dbidx in range(len(DBs)):
            plt.figure(f"db-{dbidx}__{title}")
            ax = plt.axes()
            # get axis, plot first y
            plotStr = "DBs[dbidx].dataframe.plot(x, y=y[0], ax=ax, label="
            if labels is not None:
                plotStr += "f'{labels[0][0]}'"
            else:
                plotStr += "y[0]"
            if colours is not None:
                plotStr += ", color=colours[0][0]"
            plotStr += ", **kwargs)"
            # ax = eval(plotStr)
            eval(plotStr)
            # plot the rest
            for yidx in range(len(y)):
                if yidx == 0:
                    continue
                plotStr = 'DBs[dbidx].dataframe.plot(x, y=y[yidx], ax=ax'
                if labels is not None:
                    plotStr += ', label=f"{labels[0][yidx]}"'
                else:
                    plotStr += ', label=y[yidx]'
                if colours is not None:
                    plotStr += ', color=colours[0][yidx]'
                plotStr += ", **kwargs)"
                eval(plotStr)

            # handle the frame
            plt.title(f"db-{dbidx}__{title}")

            if legend: plt.legend(loc='best')
            else: ax.get_legend().remove()

            if xlabel is not None: plt.xlabel(xlabel)
            else: plt.xlabel(x)

            if ylabel is not None and len(ylabel) == 1:
                plt.ylabel(ylabel[0])
                ysavestr=y[0]
            elif multiY:
                myl = ""
                for yidx in range(len(y)):
                    if ylabel is not None and len(ylabel) > 1:
                        myl += f"{ylabel[yidx]}, "
                    else:
                        myl += f"{y[yidx]}, "
                myl = myl[:-2]
                plt.ylabel(myl)
                ysavestr = 'multiY'
            else:
                if ylabel is not None:
                    plt.ylabel(ylabel[0])
                else:
                    plt.ylabel(y[0])
                ysavestr=y[0]

            # save plot
            plt.savefig(os.path.join(saveName, f"{time.strftime(timeStr)}__{ysavestr}__db-{dbidx}.png"), dpi=dpi)

        # end
        # plt.show(block=False)
        print(f"plot DBs: Success :Plotting complete. Plots saved to '{saveName}'")


    else: #a plot for each y and each DB!
        for yidx in range(len(y)):
            for dbidx in range(len(DBs)):
                plotStr = "f = plt.figure(f'{y[yidx]}__db-{dbidx}__{title}'); a=f.add_subplot(111); "

                plotStr += 'DBs[dbidx].dataframe.plot(x, y=y[yidx], ax=a'
                if labels is not None:
                    plotStr += ', label=f"{y[yidx]} DB-{dbidx}"'
                if colours is not None:
                    plotStr += ', color=colours[1][dbidx]'
                plotStr += ", **kwargs)"
                exec(plotStr)

                # handle the frame
                plt.title(f'{y[yidx]}__db-{dbidx}__{title}')

                if legend: plt.legend(loc='best')
                else: plt.legend().remove()

                if xlabel is not None: plt.xlabel(xlabel)
                else: plt.xlabel(x)

                if ylabel is not None: plt.ylabel(ylabel[yidx])
                else: plt.ylabel(y[yidx])

                # save plot
                plt.savefig(os.path.join(saveName, f"{time.strftime(timeStr)}__{y[yidx]}__db-{dbidx}.png"), dpi=dpi)

        # end
        # if len(y)*len(DBs) <= 20:
        #     plt.show(block=False)
        print(f"plot DBs: Success :Plotting complete. Plots saved to '{saveName}'")




#------------------------------------------------------------------------------------------------------------------


def plotDBStack(DBs, x, yStackBtoT=['seg1_em_t','seg2_em_t'], saveName=None, DBlabels=None, xlabel=None, ylabel=None, title=None, yStackLabelsBtoT=None, ycoloursBtoT=None, legend=True, grid=False, **kwargs):
    """
    Create a plot where the y-variables are stacked on-top of one another. Uses pyplot.stackplot (https://matplotlib.org/stable/api/_as_gen/matplotlib.pyplot.stackplot.html).
    Do so sequentially for each DB.
        !Assumes that all data is in the same units!
        eg usage; thickness vs optimization run
    :param DBs: DataBase or list of DataBases, the Databases from which to source the data
    :param x: string, the label of the column to use for the x-data (should be in all DBs)
    :param yStackBtoT: list of strings, the labels for the y-data to be stacked, in order of bottom -> up
    :param saveName: string, path to the save folder or filename (coerced if input type does not match num of DBs)
    :param DBlabels: list of strings, length DBs, the labels for the DBs for saving and title purposes (default is db-<dbidx> and dbfilename)
    :param xlabel: string, the label for the plot's x axis
    :param ylabel: string, the label for the plot's y axis
    :param title: string, the title for the plot (db
    :param yStackLabelsBtoT: list of strings, length yStackBtoT, the labels for the yStackBtoT for the legend
    :param ycoloursBtoT: list of strings, length yStackBtoT, the colours for the yStackBtoT dara
    :param legend: boolean, whether or not to include a legend
    :param grid: boolean, whether or not to include the xy grid on the plot.
    :param kwargs: dict, keyword arguments to plt.stackplot (https://matplotlib.org/stable/api/_as_gen/matplotlib.pyplot.stackplot.html)
    :return: None: saves the plot
    """

    plt.rcParams['font.size']= 11


    #HANDLE INPUTS AND PREP DATA
    if not isinstance(DBs,list):
        DBs = [DBs]

    yStackStr=''
    for yidx in range(len(yStackBtoT)): yStackStr += f"{yStackBtoT[yidx]}_"
    yStackStr=yStackStr[:-1]

    if saveName == None:
        saveName = os.path.join(pldir, f"{time.strftime(timeStr)}_yStacker_{yStackBtoT[0]}-etc")
        if len(DBs) == 1: saveName += ".png"
    if len(DBs) == 1 and saveName[-4:] != ".png": saveName = os.path.join(saveName, f"{time.strftime(timeStr)}_yStacker_{yStackBtoT[0]}-etc.png") #if one db then saveName is a file
    elif len(DBs) != 1 and saveName[-4:] == ".png": saveName = saveName[:-4] #if more than one db then saveName is a folder
    if saveName[-4:] != ".png" and not os.path.exists(saveName):
        os.makedirs(saveName)

    print(f"plot db stack-plot: Plotting y-stack '{yStackBtoT}' for {len(DBs)} DBs and saving to '{saveName}'.")

    if not isinstance(DBs,list):
        DBs = [DBs]
    if DBlabels is not None and len(DBlabels) != len(DBs):
        print(f"WARNING :plot db stack-plot: Length of DBlabels ({len(DBlabels)}) does not match of DBs ({len(DBs)}). Using defaults.")
        DBlabels = None
    if len(yStackLabelsBtoT) != len(yStackBtoT):
        print(f"WARNING :plot db stack-plot: Length of yStackLabelsBtoT ({len(yStackLabelsBtoT)}) does not match that of yStackBtoT ({len(yStackBtoT)}). Using y-labels. ")
        yStackLabelsBtoT = yStackBtoT
    if len(ycoloursBtoT) != len(yStackBtoT):
        print(f"WARNING :plot db stack-plot: Length of ycoloursBtoT ({len(ycoloursBtoT)}) does not match that of yStackBtoT ({len(yStackBtoT)}). Using defaults.")
        ycoloursBtoT = None

    if title is None:
        title = os.path.split(saveName)[1][:-4]

    if DBlabels == None:
        DBlabels = [f"db-{dbidx}" for dbidx in range(len(DBs))]

    try: #handle common kwarg which may be passed to this fnc but is not supported below
        grid
    except NameError:
        grid = False


    ###STACK PLOT###
    for dbidx in range(len(DBs)):
        if title is not None: plt.figure(title)
        else: plt.figure()

        try: x_series = DBs[dbidx].dataframe[x]
        except KeyError: x_series = DBs[dbidx].dataframe.index.to_series()
        data_df = DBs[dbidx].dataframe[yStackBtoT]
        pltStr = "plt.stackplot(x_series.to_numpy(), data_df.to_numpy().T, baseline='zero', labels=yStackLabelsBtoT"
        if ycoloursBtoT is not None:
            pltStr += ", colors=ycoloursBtoT"
        pltStr += ", **kwargs)"
        eval(pltStr)

        # handle the frame
        try: plt.title(f"{DBlabels[dbidx]}__{title}\n{DBs[dbidx].dbfilename[:-4]}")
        except: plt.title(f"dbStack_{DBlabels[dbidx]}__{time.strftime(timeStr)}")

        if legend:
            plt.legend(loc='best')
        else:
            plt.legend().remove()

        if grid:
            plt.grid()

        if xlabel is not None:
            plt.xlabel(xlabel)
        else:
            plt.xlabel(x)

        if ylabel is not None:
            plt.ylabel(ylabel)
        else:
            plt.ylabel(yStackStr)

        # save plot
        if len(DBs) == 1:
            plt.savefig(saveName, dpi=dpi)
        else:
            plt.savefig(os.path.join(saveName, f"{time.strftime(timeStr)}_yStack_{dbidx}_{DBlabels[dbidx]}.png"), dpi=dpi)


    print(f"plot db stack-plot: Success :Finished plotting, results saved to '{saveName}'")

    return None




#------------------------------------------------------------------------------------------------------------------
#------------------------------------------------------------------------------------------------------------------
#	 Dimensionality Reduction Plotting
#------------------------------------------------------------------------------------------------------------------
#------------------------------------------------------------------------------------------------------------------

def plot2dSlices_InFS(dbs, colsToInclude, descr=None,
                           dbMarkerSize=None, dbMarkerStyles=None, dbCmaps=None,
                           res='Eff', resLim=None, resTicks=None, resAscending=False,
                           zoom=[False,"(-1,1)","(-1,1)"], vallims=None, refPoints=None,
                           oneFigure=True, cowabunga=True, cowabungaOld=False,
                           save=False, saveFolder=pldir, **kwargs):

    """
    Plot the colsToInclude data from the DB(s) in 2d slices. (In the Full Space)
    All dbs in same plot(s)
    Use oneFigure to get one figure with many subfigures (False for many figures)

    (Originally based off of plot2dSlices_InDRSpace, expanded)

    :param dbs: [DataBase or list thereof] database(s) to plot in the aiirmap's reduced dimension space, all are included in the same plot
    :param descr: [str or None] a descriptor to use withing the plot window names (None => '')
    :param dbMarkerSize: [int or None] size of markers for all databases' points, None uses default set below
    :param dbMarkerStyles: [str or list of str or None] marker shape for each database's points (correspondent to dbs), None uses default set below
    :param dbCmaps: [cmap str or list thereof or None] cmap for each database's points (correspondent to dbs), None uses default set below
    :param res: [string] column label name of the result which is to be plotted
    :param resLim: len 2 list of [vmin,vmax] optional plot limits
    :param resTicks: list or None, the tick values to use for results trends (None; use matplotlib default)
    :param resAscending: bool, if True flips db so top results are plotted on top, if False, no flip to maintain top-focused ordering
    :param zoom: len 3 list of [bool, (xmin,xmax), (ymin,ymax)] produce a zoomed version of the plot alongside the full version if bool=True (if oneFigure, zoom is disabled)
    :param vallims: list of [ (valmin,valmax), or None, ...] or None, the plot limits for each dimension, in the order of colsToInclude
                        if length is not the same as the length of colsToInclude fills the end in with Nones to match
                        None implies using the matplotlib defaults for that dimension or for all dimensions
    :param refPoints: None or list of lists of floats, reference points to plot, points are plotted with the marker size and style defined below
                        outer list are the different reference points to be included
                        inner list is ref point coordinate in each colsToInclude dimension
    :param oneFigure: bool, put all slices in one figure
    :param cowabunga: bool, make the single figure a 2d 2d slice slice (oneFigure must be True)
    :param cowabungaOld: bool, make the 2d 2d slice slice plot but with original styling (oneFigure==True, cowabunga==False)
    :param save: [bool] Save plots or just create?
    :param saveFolder: [path-string] filepath to folder for plot saves
    :param kwargs: [dict] keyword args for plt.scatter
    :return:
    """

    #(aiirpower default  = arcThLabelsTtoB_S4+absThLabelsTtoB_bottomHomo_S4)

    markersize_default = 2
    markerstyle_default = '.'
    cmap_default = 'viridis'

    #can set per ref point
    refPt_marker = 'x'
    refPt_markersize = 2
    refPt_markercolor_palette = ['red', 'k', 'purple', 'orange','gray', 'magenta']*5
    if refPoints is not None:
        if not isinstance(refPoints[0], list): refPoints = [refPoints] #one ref point has been passed straight in
        refPt_marker = [refPt_marker] * len(refPoints)
        refPt_markersize = [refPt_markersize] * len(refPoints)
        refPt_markercolor = refPt_markercolor_palette[0:len(refPoints)]
        # refPt_markercolor = [refPt_markercolor] * len(refPoints)
    # refPt_markercolor = ['r', 'k']

    fontsize_labels=10
    fontsize_ticklabels=6

    if not isinstance(dbs,list):
        prstr = f'1 DB'
        dbs = [dbs]
    else:
        prstr = f"{len(dbs)} DBs"
    print(f"plot in FS 2d slices: Plotting 2d slices for {prstr} {colsToInclude} data with {res} colouring.")

    if dbMarkerSize is None: dbMarkerSize = markersize_default

    if dbMarkerStyles is None: dbMarkerStyles = [markerstyle_default] * len(dbs)
    if not isinstance(dbMarkerStyles, list): dbMarkerStyles =[dbMarkerStyles]
    if len(dbMarkerStyles) != len(dbs): dbMarkerStyles = [markerstyle_default] * len(dbs)

    if dbCmaps is None: dbCmaps = [cmap_default] * len(dbs)
    if not isinstance(dbCmaps, list): dbCmaps =[dbCmaps]
    if len(dbCmaps) != len(dbs): dbCmaps = [cmap_default] * len(dbs)

    if descr is None: descr = ''

    if vallims is not None:
        if len(vallims) > len(colsToInclude):
            print(f"WARNING :plot in FS 2d slices: Length of vallims ({len(vallims)}) is greater than length of colsToInclude ({len(colsToInclude)}). Truncating vallims list...")
            vallims = vallims[0:len(colsToInclude)]
        elif len(vallims) < len(colsToInclude):
            print(f"WARNING :plot in FS 2d slices: Length of vallims ({len(vallims)}) is less than length of colsToInclude ({len(colsToInclude)}). Adding Nones to end of vallims...")
            vallims += [None]*(len(colsToInclude)-len(vallims))

    if 'sc1_em2_d' in colsToInclude or 'seg1_em2_t' in colsToInclude:
        bottomHomo=True
    else: bottomHomo=False

    if res is None:
        res='db_idx'
        for dbidx in range(len(dbs)):
            try: dbs[dbidx].dataframe.loc[:,res]
            except KeyError: dbs[dbidx].dataframe['db_idx'] = np.linspace(0,len(dbs[dbidx].dataframe)-1,len(dbs[dbidx].dataframe))

    dbResData = []
    dbData = []

    for DB in dbs:
        if res is not None:
            resData = DB.dataframe[res]
            dbResData.append(resData)
            ##This flipping is not needed since we later generate a 'sortorder' based on the resData; it's there that the flip is used
            # if not resAscending:
            #     #flip dbs so that if they are sorted then the best reuslts are plotted on top
            #     flippeddata = DB.dataframe.copy(deep=True)
            #     flippeddata = flippeddata[::-1]
            #     flippeddata.reset_index(inplace=True, drop=True)
            #     # flippeddata = flippeddata.reindex(index=flippeddata.index[::-1])
            #     # DB.dataframe = flippeddata
            #     dbData.append(flippeddata[colsToInclude].values)
            # else:
            #     dbData.append(np.array(DB.dataframe[colsToInclude].values))

        else:
            dbResData.append(['b']*len(DB.dataframe))

        dbData.append(np.array(DB.dataframe[colsToInclude].values))

    # Begin plotting
    plt.rcParams['font.size'] = 6
    mpl.rcParams['lines.markersize'] = dbMarkerSize

    if save and saveFolder == pldir:
        saveFolder = os.path.join(pldir, f"{descr}_OSin2dSlices_{time.strftime(timeStr)}")
    if save and not os.path.exists(saveFolder):
        os.makedirs(saveFolder)

    cmap = mpl.cm.plasma
    dbNorms = []
    for dbidx in range(len(dbs)):
        if resLim is not None:
            dbNorms.append(mpl.colors.Normalize(vmin=resLim[0], vmax=resLim[1]))
        elif res is not None:
            dbNorms.append(mpl.colors.Normalize(vmin=np.nanmin(dbResData[dbidx].values), vmax=np.nanmax(dbResData[dbidx].values)))
        else:
            dbNorms.append(None)



    #Plot according to dimensionality
    dim = dbData[0].shape[1]
    num2dSlices = (dim*(dim-1))/2 #n choose 2
    print(f"plot in FS 2d slices: Now plotting {dim}-dimensional data in {num2dSlices} 2d slices...")

    sliceLabels = list(itertools.combinations(list(np.arange(0,dim)), 2))
    dimLabels = list(itertools.combinations(colsToInclude, 2))
    sliceLabels = list(set(sliceLabels))
    sliceLabels = sorted(sliceLabels)
    if vallims is not None:
        vallimsCopy = list(itertools.combinations(vallims, 2))
    else:
        vallims = [None]*len(colsToInclude)
        vallimsCopy = list(itertools.combinations(vallims, 2))

    if refPoints is not None:
        refPointsCopy = copy.deepcopy(refPoints)
        for rptidx in range(len(refPoints)): #Assume ref points are well formed (ie. right number of dimensions)
            refPointsCopy[rptidx] = list(itertools.combinations(refPoints[rptidx],2))
    else: refPointsCopy = None

    if len(sliceLabels) != num2dSlices:
        print(f"WARNING: plot in FS 2d slices: Length of slide labels ({len(sliceLabels)}) does not match num2dSlices ({num2dSlices})")

    if oneFigure:
        if zoom != None and zoom[0]:
            print(f"WARNING :plot in FS 2d slices: zoom is not possible for oneFigure configuration. Consider vallims instead or oneFigure==False.")

        if cowabunga: #2d 2dslice slice plot
            print(f"plot in FS 2d slices: One {dim}d 2d 2d-slice slice plot coming up... ")
            fig,axes = plt.subplots(dim-1, dim-1, constrained_layout=True, figsize=(15.5,12.6)) #figsize set for 13d space, not sure if it works generally
            fig.canvas.manager.set_window_title(f'{descr}; {dim}d data 2d 2d slice slice plot ({time.strftime(timeStr)})')
            # ax=axes.ravel()
            pidx = 0 #plotidx
            for rowi in range(dim-1):
                for coli in range(dim-1):
                    if dim==2: plt.sca(axes) #if 2d space
                    else: plt.sca(axes[rowi,coli])
                    if coli < rowi: #blank square
                        axes[rowi,coli].axis('off')
                        if rowi == coli+1: #off-diagonal by one, include the labels
                            axes[rowi,coli].text(0.5, 0.5, getResultStr(dimLabels[pidx][0], multiline=True, shortVer=True, bottomHomo=bottomHomo) , ha='center', va='center', fontsize=fontsize_labels)
                    else:
                        for dbidx in range(len(dbs)):
                            if res is not None:
                                sortOrder = np.argsort(dbResData[dbidx].values) #will plot lowest first (highest on top)
                                if resAscending: sortOrder = sortOrder[::-1] #will plot highest first
                                plt.scatter(dbData[dbidx][:, sliceLabels[pidx][1]][sortOrder],
                                            dbData[dbidx][:, sliceLabels[pidx][0]][sortOrder],
                                            c=dbResData[dbidx][sortOrder].values, norm=dbNorms[dbidx],
                                            marker=dbMarkerStyles[dbidx], cmap=dbCmaps[dbidx])
                            else:
                                sortOrder = np.linspace(0,len(dbs[dbidx].dataframe)-1,len(dbs[dbidx].dataframe)).astype(int)
                                plt.scatter(dbData[dbidx][:, sliceLabels[pidx][1]][sortOrder],
                                            dbData[dbidx][:, sliceLabels[pidx][0]][sortOrder],
                                            c='b', norm=None,
                                            marker=dbMarkerStyles[dbidx], cmap=dbCmaps[dbidx])

                        if refPointsCopy is not None:
                            for rpt in range(len(refPointsCopy)):
                                plt.plot(refPointsCopy[rpt][pidx][1], refPointsCopy[rpt][pidx][0], color=refPt_markercolor[rpt], marker=refPt_marker[rpt], markersize=refPt_markersize[rpt])
                        plt.xlim(vallimsCopy[pidx][1])
                        plt.ylim(vallimsCopy[pidx][0])
                        if coli == rowi: #diagonal, so have tick labels and have axis labels for first row y and last row x
                            # resStr = getResultStr(res)1111
                            if coli==(dim-2):
                                plt.xlabel(f'{getResultStr(dimLabels[pidx][1], multiline=True, shortVer=True, bottomHomo=bottomHomo)}', fontsize=fontsize_labels)
                            if rowi==0:
                                plt.ylabel(f'{getResultStr(dimLabels[pidx][0], multiline=True, shortVer=True, bottomHomo=bottomHomo)}', fontsize=fontsize_labels)
                            # axes[rowi,coli].tick_params(labelbottom=False, labelleft=False)
                            axes[rowi,coli].tick_params(labelsize=fontsize_ticklabels)
                        else: #non-diagonal , remove tick labels? controlled by commenting in/out line directly below
                            # axes[rowi,coli].tick_params(labelbottom=False, labelleft=False)
                            axes[rowi,coli].tick_params(labelsize=fontsize_ticklabels)

                        if dbidx == 0: pidx += 1


        elif cowabungaOld: #original 2d 2d slice slice plot
            print(f"plot in FS 2d slices: One {dim}d 2d 2d-slice slice plot coming up... ")
            fig,axes = plt.subplots(dim-1, dim-1, constrained_layout=True)
            fig.canvas.manager.set_window_title(f'{descr}; {dim}d data 2d 2d slice slice plot ({time.strftime(timeStr)})')
            # ax=axes.ravel()
            pidx = 0 #plotidx
            for rowi in range(dim-1):
                for coli in range(dim-1):
                    if dim==2: plt.sca(axes) #if 2d space
                    else: plt.sca(axes[rowi,coli])
                    if coli < rowi: #blank square
                        axes[rowi,coli].axis('off')
                    else:
                        for dbidx in range(len(dbs)):
                            sortOrder = np.argsort(dbResData[dbidx].values) #will plot lowest first (highest on top)
                            if resAscending: sortOrder = sortOrder[::-1] #will plot highest first
                            plt.scatter(dbData[dbidx][:, sliceLabels[pidx][1]][sortOrder], dbData[dbidx][:, sliceLabels[pidx][0]][sortOrder], c=dbResData[dbidx][sortOrder].values, norm=dbNorms[dbidx], marker=dbMarkerStyles[dbidx], cmap=dbCmaps[dbidx])
                        if refPointsCopy is not None:
                            for rpt in range(len(refPointsCopy)):
                                plt.plot(refPointsCopy[rpt][pidx][1], refPointsCopy[rpt][pidx][0], color=refPt_markercolor[rpt], marker=refPt_marker[rpt], markersize=refPt_markersize[rpt])
                        # resStr = getResultStr(res)
                        plt.xlim(vallimsCopy[pidx][1])
                        plt.ylim(vallimsCopy[pidx][0])
                        plt.xlabel(f'{dimLabels[pidx][1]}')
                        plt.ylabel(f'{dimLabels[pidx][0]}')
                        if dbidx == 0: pidx += 1
            print(f"plot in FS 2d slices: Remember;  a wise man once said, 'Forgiveness is divine, but never pay full price for late pizza.' Cowabunga!")




        else: #square grid oneFigure
            nrow,ncol = getOnePlotDims(num2dSlices)
            fig, axes =  plt.subplots(nrow,ncol, constrained_layout=True)#, figsize=(ncol*4, nrow*4))
            fig.canvas.manager.set_window_title(f'{descr}; {dim}d data in 2d Slices ({time.strftime(timeStr)}) ')
            ax = axes.ravel()
            for pidx in range(len(sliceLabels)):
                plt.sca(ax[pidx])
                for dbidx in range(len(dbs)):
                    sortOrder = np.argsort(dbResData[dbidx].values) #will plot lowest first (highest on top)
                    if resAscending: sortOrder = sortOrder[::-1] #will plot highest first
                    plt.scatter(dbData[dbidx][:, sliceLabels[pidx][1]][sortOrder], dbData[dbidx][:, sliceLabels[pidx][0]][sortOrder], c=dbResData[dbidx][sortOrder].values, norm=dbNorms[dbidx], marker=dbMarkerStyles[dbidx], cmap=dbCmaps[dbidx])
                if refPointsCopy is not None:
                    for rpt in range(len(refPointsCopy)):
                        plt.plot(refPointsCopy[rpt][pidx][1], refPointsCopy[rpt][pidx][0], color=refPt_markercolor[rpt], marker=refPt_marker[rpt], markersize=refPt_markersize[rpt])
                plt.xlim(vallimsCopy[pidx][1])
                plt.ylim(vallimsCopy[pidx][0])
                resStr = getResultStr(res, bottomHomo=bottomHomo)
                # for dbidx in range(len(dbs)):
                #     plt.colorbar(mpl.cm.ScalarMappable(norm=dbNorms[dbidx], cmap=dbCmaps[dbidx]),  label=resStr)
                plt.xlabel(f'{dimLabels[pidx][1]}')
                plt.ylabel(f'{dimLabels[pidx][0]}')
                # plt.xlabel(f"OS Dim. {sliceLabels[pidx][0]}, $x_{sliceLabels[pidx][0]}$")
                # plt.ylabel(f"OS Dim. {sliceLabels[pidx][1]}, $x_{sliceLabels[pidx][1]}$")
                # plt.tight_layout(h_pad=0.1, w_pad=0.1)



        if res is not None:
            fig = plt.figure(f'{descr}; {dim}d data in 2d slices ({time.strftime(timeStr)}) colorbar')
            for dbidx in range(len(dbs)):
                sortOrder = np.argsort(dbResData[dbidx].values) #will plot lowest first (highest on top)
                if resAscending: sortOrder = sortOrder[::-1] #will plot highest first
                plt.scatter(dbData[dbidx][:, sliceLabels[0][1]][sortOrder], dbData[dbidx][:, sliceLabels[0][0]][sortOrder], c=dbResData[dbidx][sortOrder].values, norm=dbNorms[dbidx], marker=dbMarkerStyles[dbidx], cmap=dbCmaps[dbidx]) #todo? seems to be missing the zoom/limit parameter
                cb = plt.colorbar(cmap=dbCmaps[dbidx])
                cb.ax.tick_params(labelsize=16)
                if len(dbs) ==1:
                    cb.set_label(f'{descr}', size=16)
                    if resTicks is not None: cb.set_ticks(resTicks)
                else:
                    cb.set_label(f'{descr} DB{dbidx}', size=16)
                    if resTicks is not None: cb.set_ticks(resTicks)
            plt.tight_layout()


    else:
        if cowabunga: print(f"WARNING :plot in FS 2d slices: Cannot create 2d 2d slice slice plot when oneFigure is False...\n"
                            f"Remember, to be a true ninja you must become one with the shadows. Darkness gives the ninja power, while light reveals the ninja's presence.\n"
                            f"Embrace the darkness. Embrace the oneFigure.")
        # sliceLabels = dimLabels
        plt.rcParams.update({'font.size': 16})

        for pidx in range(len(sliceLabels)):
            print(f"Creating plot: {descr}; S: {dimLabels[pidx]}  ({time.strftime(timeStr)})")
            fig = plt.figure(f"{descr}; S: {dimLabels[pidx]}  ({time.strftime(timeStr)})")
            ax = plt.axes()

            if zoom == None or not zoom[0]:
                for dbidx in range(len(dbs)):
                    sortOrder = np.argsort(dbResData[dbidx].values) #will plot lowest first (highest on top)
                    if resAscending: sortOrder = sortOrder[::-1] #will plot highest first
                    plt.scatter(dbData[dbidx][:, sliceLabels[pidx][1]][sortOrder], dbData[dbidx][:, sliceLabels[pidx][0]][sortOrder], c=dbResData[dbidx][sortOrder].values, norm=dbNorms[dbidx], marker=dbMarkerStyles[dbidx], cmap=dbCmaps[dbidx], **kwargs)
                if refPointsCopy is not None:
                    for rpt in range(len(refPointsCopy)):
                        plt.plot(refPointsCopy[rpt][pidx][1], refPointsCopy[rpt][pidx][0], color=refPt_markercolor[rpt], marker=refPt_marker[rpt], markersize=refPt_markersize[rpt])
                plt.xlabel(f"{getResultStr(dimLabels[pidx][1], bottomHomo=bottomHomo)}")
                plt.ylabel(f"{getResultStr(dimLabels[pidx][0], bottomHomo=bottomHomo)}")
                plt.xlim(vallimsCopy[pidx][1])
                plt.ylim(vallimsCopy[pidx][0])
                # plt.ylim([0.1485,0.1510])
                # plt.ylim([0.13,0.20])
                # plt.ylim([0.10,0.17])
                # plt.ylim([0.0,0.45])
                # plt.yticks([0.149,0.150,0.151])
                # plt.xlim([0.0, 0.4])
                # plt.xlim([0.22,0.32])
                # plt.xlim([0.17,0.23])
                # plt.xlim([0.0,0.3])
                # plt.xticks([0.206,0.207,0.208,0.209])
                plt.tight_layout()

            else:
                plt.subplot(1,2,1)
                for dbidx in range(len(dbs)):
                    sortOrder = np.argsort(dbResData[dbidx].values) #will plot lowest first (highest on top)
                    if resAscending: sortOrder = sortOrder[::-1] #will plot highest first
                    plt.scatter(dbData[dbidx][:, sliceLabels[pidx][1]][sortOrder], dbData[dbidx][:, sliceLabels[pidx][0]][sortOrder], c=dbResData[dbidx][sortOrder].values, norm=dbNorms[dbidx], marker=dbMarkerStyles[dbidx], cmap=dbCmaps[dbidx], **kwargs)
                if refPointsCopy is not None:
                    for rpt in range(len(refPointsCopy)):
                        plt.plot(refPointsCopy[rpt][pidx][1], refPointsCopy[rpt][pidx][0], color=refPt_markercolor[rpt], marker=refPt_marker[rpt], markersize=refPt_markersize[rpt])
                plt.xlabel(f"{getResultStr(dimLabels[pidx][1], bottomHomo=bottomHomo)}")
                plt.ylabel(f"{getResultStr(dimLabels[pidx][0], bottomHomo=bottomHomo)}")
                plt.xlim(vallimsCopy[pidx][1])
                plt.ylim(vallimsCopy[pidx][0])
                plt.tight_layout()

                plt.subplot(1,2,2)
                for dbidx in range(len(dbs)):
                    sortOrder = np.argsort(dbResData[dbidx].values) #will plot lowest first (highest on top)
                    if resAscending: sortOrder = sortOrder[::-1] #will plot highest first
                    plt.scatter(dbData[dbidx][:, sliceLabels[pidx][1]][sortOrder], dbData[dbidx][:, sliceLabels[pidx][0]][sortOrder], c=dbResData[dbidx][sortOrder].values, norm=dbNorms[dbidx], marker=dbMarkerStyles[dbidx], cmap=dbCmaps[dbidx], **kwargs)
                if refPointsCopy is not None:
                    for rpt in range(len(refPointsCopy)):
                        plt.plot(refPointsCopy[rpt][pidx][1], refPointsCopy[rpt][pidx][0], color=refPt_markercolor[rpt], marker=refPt_marker[rpt], markersize=refPt_markersize[rpt])
                execStr = f"plt.xlim{zoom[1]}"
                eval(execStr)
                execStr = f"plt.ylim{zoom[2]}"
                eval(execStr)
                plt.xlabel(f"{getResultStr(dimLabels[pidx][1], bottomHomo=bottomHomo)}")
                plt.ylabel(f"{getResultStr(dimLabels[pidx][0], bottomHomo=bottomHomo)}")
                plt.tight_layout()


            resStr = getResultStr(res)

            # for dbidx in range(len(dbs)):
            #     cb= plt.colorbar(mpl.cm.ScalarMappable(norm=dbNorms[dbidx], cmap=dbCmaps[dbidx]),  label=resStr)
            #     cb.ax.tick_params(labelsize=16)
            #     if resTicks is not None: cb.set_ticks(resTicks)

            #ax = plt.contourf(dbData[:,0], dbData[:,1], resData, **kwargs)
            if save:
                plt.savefig(os.path.join(saveFolder, f"dim-{sliceLabels[pidx][0]}_dim-{sliceLabels[pidx][1]}.png"), dpi=dpi)

    print(f"plot in FS 2d slices: Success : {len(sliceLabels)} 2d scatter plots created.")

    return



def plot2dSlices_InDRSpace(aiirmap, dbs, descr=None,
                      dbMarkerSizes=None, dbMarkerStyles=None, dbCmaps=None,
                      res='Eff', resLim=None, resAscending=False,
                      zoom=[False,"(-1,1)","(-1,1)"] ,
                      oneFigure=True, cowabunga=True,
                      save=False, saveFolder=pldir, **kwargs):
    """
    Plot the data (DB) in the reduced space (RS) of the dimensional reduction mapping object.
    Plot the RS as 2d slices

    :param aiirmap: [aiirMapping] the dimensional reduction mapping object
    :param dbs: [DataBase or list thereof] database(s) to plot in the aiirmap's reduced dimension space
    :param descr: [str or None] a descriptor to use withing the plot window names (None => '')
    :param dbMarkerSizes: [int or list of ints or None] size of markers for each database's points (correspondent to dbs), None uses default set below
    :param dbMarkerStyles: [str or list of str or None] marker shape for each database's points (correspondent to dbs), None uses default set below
    :param dbCmaps: [cmap str or list thereof or None] cmap for each database's points (correspondent to dbs), None uses default set below
    :param res: [string] column label name of the result which is to be plotted
    :param resLim: len 2 list of [vmin,vmax] optional plot limits
    :param resAscending: bool, orders which data is plotted first, False; highest res plotted on top/last, True; lowest plotted last/ on top
    :param zoom: len 3 list of [bool, (xmin,xmax), (ymin,ymax)] produce a zoomed version of the plot alongside the full version if bool=True (if both oneFigure and zoom then plots only zoomed versions)
    :param oneFigure: bool, put all slices in one figure
    :param cowabunga: bool, make the single figure a 2d 2d slice slice (oneFigure must be True)
    :param save: [bool] Save plots or just create?
    :param saveFolder: [path-string] filepath to folder for plot saves
    :param kwargs: [dict] keyword args for
    :return:
    """

    markersize_default = 5
    markerstyle_default = 'o'
    cmap_default = 'viridis'


    if not isinstance(dbs,list):
        prstr = f'DB {dbs.dbfilename}'
        dbs = [dbs]
    else:
        prstr = f"{len(dbs)} DBs"
    print(f"plot in RS 2d slices: Plotting {prstr} {res} data in {aiirmap.mapName} {aiirmap.ml.n_components}d reduced space in 2d slices.")
    print(f"plot in RS 2d slices: Extracting lower dimensional information and transforming data...")

    if dbMarkerSizes is None: dbMarkerSizes = [markersize_default] * len(dbs)
    if not isinstance(dbMarkerSizes, list): dbMarkerSizes =[dbMarkerSizes]
    if len(dbMarkerSizes) != len(dbs): dbMarkerSizes = [markersize_default] * len(dbs)

    if dbMarkerStyles is None: dbMarkerStyles = [markerstyle_default] * len(dbs)
    if not isinstance(dbMarkerStyles, list): dbMarkerStyles =[dbMarkerStyles]
    if len(dbMarkerStyles) != len(dbs): dbMarkerStyles = [markerstyle_default] * len(dbs)

    if dbCmaps is None: dbCmaps = [cmap_default] * len(dbs)
    if not isinstance(dbCmaps, list): dbCmaps =[dbCmaps]
    if len(dbCmaps) != len(dbs): dbCmaps = [cmap_default] * len(dbs)

    if descr is None: descr = ''


    #Pull reduced design space components and training parameters
    components = aiirmap.ml.model.components_ #shape is [# reduced dims, # training parameters / # original dims]
    #scaling = aiirmap.ml._scaler.scale_ #shape is [# original dims]

    trainingParameters = aiirmap.filteredDRInputGrid.dataframe.columns

    if components.shape[1] != len(trainingParameters):
        print(f"ERROR :plot in RS 2d slices: Size of reduced space vector does not match training parameters. Map is malformed. Exiting.")
        return

    dbResData = []
    dbProjData = []
    for DB in dbs:
        # Check DB, ready data
        #Check DB for presence of Inputs and Result
        for iidx in range(len(trainingParameters)):
            if not trainingParameters[iidx] in DB.dataframe:
                print(f"WARNING :plot in RS 2d slices: Reduced design space column '{trainingParameters[iidx]}' not found in {DB.dbfilename}.")
        if not res in DB.dataframe:
            print(f"ERROR :plot in RS 2d slices: Reduced design space column '{res}' not found in {DB.dbfilename}. Cannot plot. Exiting...")
            return

        #split database into DR inputs and result-to-plot, and the rest of the columns
        drInputs = DB.dataframe[trainingParameters]
        resData = DB.dataframe[res]
        unusedData = DB.dataframe.drop(columns=trainingParameters.values)
        unusedData.drop(columns=[res])

        #check if unused columns are of no consequence
        #print(unusedData.nunique(axis=0))
        headerCol=True
        for cidx in range(len(unusedData.columns)):
            #print(f"cidx;{cidx}\n" + unusedData.columns[cidx])
            if unusedData.columns[cidx] in inputStartStrs:
                headerCol=False
                continue
            elif unusedData.columns[cidx] in outputStartStrs:
                break
            elif headerCol:
                continue
            elif unusedData.nunique(axis=0)[cidx] != 1:
                print(f"WARNING :plot in RS 2d slices: Column '{unusedData.columns[cidx]}' is not included in the DR and is not monotonic.")

        #Create data points in reduced space.
        #shape is (# pts, # dims in reduced space)
        #reducedDimInputs = (drInputs.values / scaling) @ components.T
        projectedData = aiirmap.ml._project_pca(drInputs)

        dbResData.append(resData)
        dbProjData.append(projectedData)


    #Begin plotting
    plt.rcParams['font.size'] = 14

    if save and saveFolder == pldir:
        saveFolder = os.path.join(pldir, f"DataInRS__{time.strftime(timeStr)}")
    if save and not os.path.exists(saveFolder):
        os.makedirs(saveFolder)

    cmap = mpl.cm.plasma
    dbNorms = []
    for dbidx in range(len(dbs)):
        if resLim is not None:
            dbNorms.append(mpl.colors.Normalize(vmin=resLim[0], vmax=resLim[1]))
        else:
            dbNorms.append(mpl.colors.Normalize(vmin=np.nanmin(dbResData[dbidx].values), vmax=np.nanmax(dbResData[dbidx].values)))

    #Plot according to dimensionality
    rsDim = dbProjData[0].shape[1]
    num2dSlices = (rsDim*(rsDim-1))/2 #n choose 2
    print(f"plot in RS 2d slices: Data transformed.\nplot in RS 2d slices: Now plotting {rsDim}-dimensional data in {num2dSlices} 2d slices...")

    sliceLabels = list(itertools.combinations(list(np.arange(0,rsDim)), 2))
    sliceLabels = list(set(sliceLabels))
    sliceLabels = sorted(sliceLabels)

    if len(sliceLabels) != num2dSlices:
        print(f"WARNING: plot in RS 2d slices: Length of slide labels ({len(sliceLabels)}) does not match num2dSlices ({num2dSlices})")

    if oneFigure:
        if cowabunga: #2d 2dslice slice plot
            print(f"plot in RS 2d slices: One {rsDim}d 2d 2d-slice slice plot coming up... ")
            fig,axes = plt.subplots(rsDim-1, rsDim-1, constrained_layout=True)
            fig.canvas.manager.set_window_title(f'{descr}; {rsDim}d data 2d 2d slice slice plot ({time.strftime(timeStr)})')
            # ax=axes.ravel()
            pidx = 0 #plotidx
            for rowi in range(rsDim-1):
                for coli in range(rsDim-1):
                    if rsDim==2: plt.sca(axes) #if 2d space
                    else: plt.sca(axes[rowi,coli])
                    if coli < rowi: #blank square
                        axes[rowi,coli].axis('off')
                    else:
                        for dbidx in range(len(dbs)):
                            sortOrder = np.argsort(dbResData[dbidx].values) #will plot lowest first (highest on top)
                            if resAscending: sortOrder = sortOrder[::-1] #will plot highest first
                            if zoom == None or not zoom[0]:
                                plt.scatter(dbProjData[dbidx][:, sliceLabels[pidx][1]][sortOrder], dbProjData[dbidx][:, sliceLabels[pidx][0]][sortOrder], c=dbResData[dbidx][sortOrder].values, norm=dbNorms[dbidx], marker=dbMarkerStyles[dbidx], cmap=dbCmaps[dbidx])
                            else:
                                plt.scatter(dbProjData[dbidx][:, sliceLabels[pidx][1]][sortOrder], dbProjData[dbidx][:, sliceLabels[pidx][0]][sortOrder], c=dbResData[dbidx][sortOrder].values, norm=dbNorms[dbidx], marker=dbMarkerStyles[dbidx], cmap=dbCmaps[dbidx]) #todo? seems to be missing the zoom/limit parameter
                        # resStr = getResultStr(res)
                        plt.xlabel(f"PCA Component {sliceLabels[pidx][1]}, $x_{sliceLabels[pidx][1]}$")
                        plt.ylabel(f"PCA Component {sliceLabels[pidx][0]}, $x_{sliceLabels[pidx][0]}$")
                        if dbidx == 0: pidx += 1
            print(f"plot in RS 2d slices: Enjoy the za... Remember;  a wise man once said, 'Forgiveness is divine, but never pay full price for late pizza.' Cowabunga!")

        else: #grid
            nrow,ncol = getOnePlotDims(num2dSlices)

            fig, axes =  plt.subplots(nrow,ncol)
            fig.canvas.manager.set_window_title(f'{descr}; Data in 2d Slices ({time.strftime(timeStr)}) ')
            ax = axes.ravel()
            for pidx in range(len(sliceLabels)):
                plt.sca(ax[pidx])
                for dbidx in range(len(dbs)):
                    if zoom == None or not zoom[0]:
                        plt.scatter(dbProjData[dbidx][:, sliceLabels[pidx][1]], dbProjData[dbidx][:, sliceLabels[pidx][0]], c=dbResData[dbidx].values, norm=dbNorms[dbidx], marker=dbMarkerStyles[dbidx], cmap=dbCmaps[dbidx])
                    else:
                        plt.scatter(dbProjData[dbidx][:, sliceLabels[pidx][1]], dbProjData[dbidx][:, sliceLabels[pidx][0]], c=dbResData[dbidx].values, norm=dbNorms[dbidx], marker=dbMarkerStyles[dbidx], cmap=dbCmaps[dbidx]) #todo? seems to be missing the zoom/limit parameter
                resStr = getResultStr(res)
                # for dbidx in range(len(dbs)):
                #     plt.colorbar(mpl.cm.ScalarMappable(norm=dbNorms[dbidx], cmap=dbCmaps[dbidx]),  label=resStr)
                plt.xlabel(f"PCA Component {sliceLabels[pidx][1]}, $x_{sliceLabels[pidx][1]}$")
                plt.ylabel(f"PCA Component {sliceLabels[pidx][0]}, $x_{sliceLabels[pidx][0]}$")

        fig = plt.figure(f'{descr}; {rsDim}d data in 2d slices ({time.strftime(timeStr)}) colorbar')
        for dbidx in range(len(dbs)):
            sortOrder = np.argsort(dbResData[dbidx].values) #will plot lowest first (highest on top)
            if resAscending: sortOrder = sortOrder[::-1] #will plot highest first
            plt.scatter(dbProjData[dbidx][:, sliceLabels[0][1]][sortOrder], dbProjData[dbidx][:, sliceLabels[0][0]][sortOrder], c=dbResData[dbidx][sortOrder].values, norm=dbNorms[dbidx], marker=dbMarkerStyles[dbidx], cmap=dbCmaps[dbidx]) #todo? seems to be missing the zoom/limit parameter
            cb = plt.colorbar(cmap=dbCmaps[dbidx])
            cb.ax.tick_params(labelsize=16)
            if len(dbs) ==1:
                cb.set_label(f'{descr}')
                # if resTicks is not None: cb.set_ticks(resTicks)
            else:
                cb.set_label(f'{descr} DB{dbidx}')
                # if resTicks is not None: cb.set_ticks(resTicks)
        plt.tight_layout()


    else:
        if cowabunga: print(f"WARNING :plot in FS 2d slices: Cannot create 2d 2d slice slice plot when oneFigure is False...\n"
                            f"Remember, to be a true ninja you must become one with the shadows. Darkness gives the ninja power, while light reveals the ninja's presence.\n"
                            f"Embrace the darkness. Embrace the oneFigure.")

        for pidx in range(len(sliceLabels)):

            fig = plt.figure(f"{descr}; RS: {sliceLabels[pidx]}  ({time.strftime(timeStr)})")
            ax = plt.axes()

            if zoom == None or not zoom[0]:
                for dbidx in range(len(dbs)):
                    plt.scatter(dbProjData[dbidx][:, sliceLabels[pidx][1]], dbProjData[dbidx][:, sliceLabels[pidx][0]], c=dbResData[dbidx].values, norm=dbNorms[dbidx], marker=dbMarkerStyles[dbidx], cmap=dbCmaps[dbidx], **kwargs)
                plt.xlabel(f"PCA Component {sliceLabels[pidx][1]}, $x_{sliceLabels[pidx][1]}$")
                plt.ylabel(f"PCA Component {sliceLabels[pidx][0]}, $x_{sliceLabels[pidx][0]}$")

            else:
                plt.subplot(1,2,1)
                for dbidx in range(len(dbs)):
                    plt.scatter(dbProjData[dbidx][:, sliceLabels[pidx][1]], dbProjData[dbidx][:, sliceLabels[pidx][0]], c=dbResData[dbidx].values, norm=dbNorms[dbidx], marker=dbMarkerStyles[dbidx], cmap=dbCmaps[dbidx],**kwargs)
                plt.xlabel(f"PCA Component {sliceLabels[pidx][1]}, $x_{sliceLabels[pidx][1]}$")
                plt.ylabel(f"PCA Component {sliceLabels[pidx][0]}, $x_{sliceLabels[pidx][0]}$")

                plt.subplot(1,2,2)
                for dbidx in range(len(dbs)):
                    plt.scatter(dbProjData[dbidx][:, sliceLabels[pidx][1]], dbProjData[dbidx][:, sliceLabels[pidx][0]], c=dbResData[dbidx].values, norm=dbNorms[dbidx], marker=dbMarkerStyles[dbidx], cmap=dbCmaps[dbidx], **kwargs)
                execStr = f"plt.xlim{zoom[1]}"
                eval(execStr)
                execStr = f"plt.ylim{zoom[2]}"
                eval(execStr)
                plt.xlabel(f"PCA Component {sliceLabels[pidx][1]}, $x_{sliceLabels[pidx][1]}$")
                plt.ylabel(f"PCA Component {sliceLabels[pidx][0]}, $x_{sliceLabels[pidx][0]}$")



            resStr = getResultStr(res)

            for dbidx in range(len(dbs)):
                plt.colorbar(mpl.cm.ScalarMappable(norm=dbNorms[dbidx], cmap=dbCmaps[dbidx]),  label=resStr)

            #ax = plt.contourf(dbProjData[:,0], dbProjData[:,1], resData, **kwargs)
            if save:
                plt.savefig(os.path.join(saveFolder, f"dim-{sliceLabels[pidx][0]}_dim-{sliceLabels[pidx][1]}.png"), dpi=dpi)

    print(f"plot in RS 2d slices: Success : {len(sliceLabels)} 2d scatter plots created.")

    return


def getOnePlotDims(numSlices):
    if numSlices==1: nrow=1; ncol=1
    elif numSlices==3: nrow=1; ncol=1
    elif numSlices==6: nrow=2; ncol=3
    elif numSlices==10: nrow=2; ncol=5
    elif numSlices==15: nrow=3; ncol=5
    elif numSlices==21: nrow=3; ncol=7
    elif numSlices==28: nrow=4; ncol=7
    elif numSlices== 78: nrow=6; ncol=13
    return nrow,ncol



# ------------------------------------------------------------------------------------------------------------------


def plotMap_DRComponentsHistogram(aiirmap, numComponentsToIncludeInPlot = None, variation = 'standard'):#, thLayerLabelsTtoB=arcThLabelsTtoB + absThLabelsTtoB_bottomHomo):
    """
    Plot the coefficients of the ML DR model used in the aiirmapping.

    This code plots;
        standard = a_ij ; the coefficients for each RS<->FS term [unitless]
        unnormalized = a_ij * s_j ; the coefficients * their standard deviations in the full space [full space units]

    PCA eqn;
    t_k = sum ( a_j * x_j ) = sum ( a_j * (x'_j - u_j)/s_j )
    where
    x'_j = actual original dimensions
    u_j = mean of training data in dim j
    s_j = std dev of training data in dim j
    x_j = "standardized" dimensions
    a_j = PCA coefficients
    t_k = PCA RS dimension k


    @param aiirmap: dbh.aiirmap, The mapping object to plot
    @param numComponentsToIncludeInPlot: int or None, number of red. space components to include in the plot, if None or greater than avail number of components plot them all
    @param variation: string, one of 'standard' or 'unnormalized', what to plot, see above for more info
    @return:
    """
    #TODO show, save functionality
    print(f"plot dr components: Plotting DR component histogram for aiirmap '{aiirmap.mapName}'")
    # scaling = aiirmap.ml._scaler.scale_ #shape is [# original dims]

    origSpaceParams = aiirmap.filteredDRInputGrid.dataframe.columns.values
    components = aiirmap.ml.model.components_  # shape is [# reduced dims, # training parameters / # original dims]

    numRedSpaceParamsToPlot = numComponentsToIncludeInPlot
    if numComponentsToIncludeInPlot is None:
        numRedSpaceParamsToPlot=components.shape[0] #plot all
    elif numComponentsToIncludeInPlot > components.shape[0]:
        numRedSpaceParamsToPlot=components.shape[0] #plot all

    origSpaceParams = list(origSpaceParams)
    # origSpaceParams.reverse()
    colourPaletteBtoT = getThStackColoursTtoB(origSpaceParams)
    # colourPaletteBtoT.reverse()
    colors=colourPaletteBtoT
    # colors=colourPaletteBtoT[0:len(origSpaceParams)]

    if 'em2' in origSpaceParams[-1]: bottomHomojnc = True
    else: bottomHomojnc = False
    fancyNames = [getResultStr(layer, True, False, bottomHomojnc) for layer in origSpaceParams]
    fancyNames.reverse()

    componentNames = [f"$y_{i+1}$" for i in range(components.shape[0])]
    reducSpaceParams = [f"PCA_dim{i}" for i in range(components.shape[0])]

    osdf = pd.DataFrame(components, columns=origSpaceParams)
    rindex = pd.Index(componentNames)
    osdf = osdf.set_index(rindex)
    rsdf = pd.DataFrame(components.T, columns=reducSpaceParams)
    oindex = pd.Index(fancyNames)
    rsdf = rsdf.set_index(oindex)



    # plot a few variants
    if variation.lower()[0] not in ['s', 'u']:
        print(f"ERROR :plot dr components: Variation '{variation}' is not recognized. Using 'standard'.")

    if variation.lower()[0] == 'u': #unnormed
        scaling = aiirmap.ml._scaler.scale_
        for col in range(osdf.shape[1]):
            for row in range(osdf.shape[0]):
                osdf.iloc[row, col] = osdf.iloc[row, col] * scaling[col] #/ baselines[col]
                # osdf.iloc[row, col] = osdf.iloc[row, col]  / baselines[col]
                # osdf.iloc[row, col] = (osdf.iloc[row, col] * scaling[col] ) + means[col]
        ylabel = "PCA Coefficient * Input Dimension Standard Deviation, $a_j \sigma_i$ [$\mu m$]"
        ylabel = "PCA Coeff. * Input Dim. Std. Dev., $a_j \sigma_i$ [$\mu m$]"
    else: #standard
        ylabel="PCA Coefficient, $a_j$"




    plt.rcParams.update({'font.size':16})
    osdf.iloc[0:numRedSpaceParamsToPlot,:].plot(kind='bar', color=colors, ylabel=ylabel, width=0.7)#, index=componentNames)
    plt.xlabel('Reduced Space Component')
    # plt.set_xticks(np.linspace())
    plt.legend().remove()
    plt.tight_layout()

    plt.rcParams.update({'font.size':10})
    rsdf.iloc[:,0:numRedSpaceParamsToPlot].plot(kind='bar', ylabel=ylabel, width=0.7)#, index=fancyNames)
    plt.xlabel('Input Dimension')
    plt.legend().remove()
    plt.tight_layout()

# ------------------------------------------------------------------------------------------------------------------


def plotMap_Info(aiirmap, saveName=pldir, decplaces=3, titleDecplaces=1, latex=False, colors=None):
    """
    Make a figure showing some of the key dimensionality reduction information.
    Includes

    :param aiirmap: [AiirMap] the aiirmap object for the DR run of interest
    :param saveName: [str] the path to the saveFolder
    :param decplaces: [int] the number of decimal places to include in the numbers in the projected dimension equations
    :param latex: [boolean] whether to use latex (T) or not (F)
    :param colors: [list of string or string or None] matplotlib colors to use for the y_i, a_j, x_j, mu_j, s_j, if a single color use it for all 5 (default, None, use black)
    :return: None: creates plot
    """


    if colors is not None and not latex:
        print(f"WARNING: dr info plot :Cannot use colors without latex. Turning latex on...")
        latex=True

    #plots dr info from aiirmap object
    if latex:
        plt.rcParams.update({
            "text.usetex": True,
            "font.family": "sans-serif",
            "font.sans-serif": ["Helvetica"]})
        plt.rc("text.latex", preamble='\\usepackage{xcolor}')


    if saveName == pldir:
        saveName = os.path.join(pldir, f"{aiirmap.mapName}_{time.strftime(timeStr)}_DR-Info-Plot.png")


    print(f"dr info plot: Plotting figure with DR info ('{saveName}')... ")


    fig = plt.figure(f"DR_Info_Plot_{time.strftime(timeStr)}")
    ax = fig.add_subplot(111)
    ax.set_axis_off()

    #prepare info strings to plot
    percentVarInfo = "Variance by component: "
    trainingParams = f"Training parameters, $x_j$:\n{aiirmap.filteredDRInputGrid.dataframe.columns.values}"
    compStrs = getMap_DRComponentStrings(aiirmap, decplaces=decplaces, latex=latex, colors=colors)
    multiCompStr = ""
    for dim in range(aiirmap.ml.n_components):
        percentVarInfo += f"  $y_{dim+1}$: {aiirmap.ml.model.explained_variance_ratio_[dim]*100:.{titleDecplaces}f}%\t"
        multiCompStr += f"{compStrs[dim]}\n\n"
    multiCompStr= multiCompStr[:-2]
    percentVarInfo=percentVarInfo[:-1]

    totPercentVar = np.sum(aiirmap.ml.model.explained_variance_ratio_)

    #plot the prepared strings
    #Num reduced dimensions
    txt = ax.text(0.5, 1, f"Reduced Space Information\nDimensionality: {aiirmap.ml.n_components}D\nTotal training variance captured: {totPercentVar:.{titleDecplaces}%}", ha='center', va='top', fontsize=16)
    #percent variation accounted ofr by each dim
    txt2 = ax.text(0.5, 0.9, percentVarInfo, ha='center', va='top', fontsize=13, wrap=True)
    #the training parameter names
    txt2pt5 = ax.text(0.5,0.85, trainingParams,ha='center', va='top', fontsize=11, wrap=True)
    #the reduced dimension component strings
    txt3 = ax.text(0.5, 0.67, multiCompStr, ha='center', va='top', fontsize=10, wrap = True)
    txt3.set_clip_on(False)


    plt.savefig(saveName)
    print(f"dr info plot: Success :DR info plot created; '{saveName}'")
    return



# ------------------------------------------------------------------------------------------------------------------


def getMap_DRComponentStrings(aiirmap=None, drobj=None, decplaces=4, latex=False, colors=None):
    """
    Takes lists of DR components and creates a string formula for display.

    !takes one of aiirmap or drobj, aiirmap supercedes drobj!

    :param aiirmap: dbh.aiirmapping, to pull the components and values from (from it's machine learning dr object)
    :param drobj: utilities.dimensionalityreduction object to display
    :param components: ndarray for DR inputs transformation, shape is (# reduced dims, # original dims)
    :param colors: [list of string or string or None] matplotlib colors to use for the y_i, a_j, x_j, mu_j, s_j, if a single color use it for all 5 (default, None, use black)
    :return: outStrs: list of strings giving the eqn defining each reduced dim (x) in terms of the original dims (X)
    """

    #! PCA ONLY
    # makes assumptions about scaler settings (with_mean/std = True)
    #TODO construct the string formula based upon use of scaler's with_mean/std attributes
    #https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.StandardScaler.html?highlight=scaler#sklearn.preprocessing.StandardScaler


    # Prescaling of the input dimensions
    # if with_mean/std = True;
    #	scaled X = X' = ( X - U ) / S
    # with_mean; subtracts by mean value of column, U
    # with_std; normalizes using std deviation, S

    # formula then for a pca dimension is;
    #	x_j = sum_i ( c_ij * X'_i )
    # c_ij are components for given reduced dimension x_j


    if colors is not None:
        if isinstance(colors, str): colors = [colors]*5
    else: colors = ['black']*5

    if aiirmap==None and drobj==None:
        print(f"ERROR :get dr component strs: One of aiirmap or drobj must be given.")
    elif aiirmap!=None and drobj!=None:
        print(f"WARNING :get dr component strs: Both aiirmap and drobj have been supplied. aiirmap supercedes. drobj will not be used.")
        drobj=aiirmap.ml
    elif aiirmap !=None :
        drobj = aiirmap.ml

    components = drobj.model.components_
    means = drobj._scaler.mean_
    if means is None: means = [0]*components.shape[1]
    scaling = drobj._scaler.scale_
    if scaling is None: scaling = [1]*components.shape[1]

    outStrs = []
    for dim in range(drobj.model.components_.shape[0]):
        dimStr = f"$y_{dim+1}$ = "
        for indim in range(components.shape[1]):
            if latex:
                if colors:
                    str1 = f'{components[dim][indim]:.{decplaces}f}'
                    coloredCommand = "$\\color{"+f"{colors[2]}"+"}{"+f"{components[dim][indim]:.{decplaces}f}"+"}"+  \
                                     " (\\frac{"+ "\\textcolor{"+f"{colors[1]}"+"}{x_"+"{"+f"{indim+1}"+"}} - " +\
                                     "\\textcolor{"+f"{colors[3]}"+"}{"+ f"{means[indim]:.{decplaces}f}"+"} }{\\textcolor{"+f"{colors[4]}"+"}{"+f"{scaling[indim]:.{decplaces}f}"+"}} )$ + "
                    dimStr += coloredCommand
                    # dimStr += "$\\frac{"+r"\textcolor{}{(str1)}-({means[indim]}:.{decplaces}f)"+"}{("+f"{scaling[indim]:.{decplaces}f}"+")} x_" + "{" + f"{indim+1}" + "}$ + " #this looks wrong
                else:
                    dimStr += "$\\frac{"+f"({components[dim][indim]:.{decplaces}f})-({means[indim]}:.{decplaces}f)"+"}{("+f"{scaling[indim]:.{decplaces}f}"+")} x_" + "{" + f"{indim+1}" + "}$ + " #this looks wrong
            else:
                dimStr += f"({components[dim][indim]:.{decplaces}f})[[$x_"+"{"+f"{indim+1}"+"}$ - " + f"({means[indim]:.{decplaces}f})]/({scaling[indim]:.{decplaces}f})] + " #this looks right
        dimStr = dimStr[:-3]
        outStrs.append(dimStr)

    return outStrs


#------------------------------------------------------------------------------------------------------------------
#------------------------------------------------------------------------------------------------------------------
#	 Plot Helper Fncs
#------------------------------------------------------------------------------------------------------------------
#------------------------------------------------------------------------------------------------------------------

def getResultStr(resultParamName, shortVer=False, multiline=False, bottomHomo=False):#, units=["%", "", "V", "mA/cm$^2$"]):
    """
    Return the nicely formatted (plotting axis titles) name string for the configured results or parameters.
    :param resultParamName: [string] key name of result to fetch
    :param shortVer: [bool] return the shortened version of the nicely formatted string
    :param multiline: [bool] use a line break for some of the longer nicely formatted strings
    :return: resultPlotName: [string] nicely formatted name string
    """
    if "em2" in resultParamName: bottomHomo=True
    resultPlotName=None
    if not shortVer and not multiline:
        if resultParamName=='Eff': resultPlotName = 'Efficiency, $\eta$ [%]'
        elif resultParamName=='QE': resultPlotName = 'External Quantum Efficiency, $EQE$'
        elif resultParamName=='Voc': resultPlotName = 'Open Circuit Voltage, $V_{oc}$ [V]'
        elif resultParamName=='FF': resultPlotName = 'Fill Factor, $FF$ [%]'
        elif resultParamName=='Jsc': resultPlotName = 'Short Circuit Current Density, $J_{sc}$ [mA/cm$^2$]'
        elif resultParamName=='Jph_norm1': resultPlotName = 'Photocurrent FOM'
        elif resultParamName=='Jph_norm0': resultPlotName = 'Photocurrent FOM'
        elif resultParamName in ['ARC2_d','ARC2_t']: resultPlotName = 'Upper ARC Thickness [$\mu m$]'
        elif resultParamName in ['ARC1_d','ARC1_t']: resultPlotName = 'Lower ARC Thickness [$\mu m$]'
        elif resultParamName in ['sc10_em_d','seg10_em_t']: resultPlotName = 'Segment 10 Thickness [$\mu m$]'
        elif resultParamName in ['sc9_em_d','seg9_em_t']: resultPlotName = 'Segment 9 Thickness [$\mu m$]'
        elif resultParamName in ['sc8_em_d','seg8_em_t']: resultPlotName = 'Segment 8 Thickness [$\mu m$]'
        elif resultParamName in ['sc7_em_d','seg7_em_t']: resultPlotName = 'Segment 7 Thickness [$\mu m$]'
        elif resultParamName in ['sc6_em_d','seg6_em_t']: resultPlotName = 'Segment 6 Thickness [$\mu m$]'
        elif resultParamName in ['sc5_em_d','seg5_em_t']: resultPlotName = 'Segment 5 Thickness [$\mu m$]'
        elif resultParamName in ['sc4_em_d','seg4_em_t']: resultPlotName = 'Segment 4 Thickness [$\mu m$]'
        elif resultParamName in ['sc3_em_d','seg3_em_t']: resultPlotName = 'Segment 3 Thickness [$\mu m$]'
        elif resultParamName in ['sc2_em_d','seg2_em_t']: resultPlotName = 'Segment 2 Thickness [$\mu m$]'
        if bottomHomo:
            if resultParamName in ['sc1_em_d','seg1_em_t']: resultPlotName = 'Segment 1 Emitter Thickness [$\mu m$]'
            elif resultParamName in ['sc1_em2_d','seg1_em2_t']: resultPlotName = 'Segment 1 Base Thickness [$\mu m$]'
        else:
            if resultParamName in ['sc1_em_d','seg1_em_t']: resultPlotName = 'Segment 1 Thickness [$\mu m$]'
    elif multiline and not shortVer: #multiline full
        if resultParamName=='Eff': resultPlotName = 'Efficiency, $\eta$ [%]'
        elif resultParamName=='QE': resultPlotName = 'External Quantum Efficiency, $EQE$'
        elif resultParamName=='Voc': resultPlotName = 'Open Circuit Voltage, $V_{oc}$ [V]'
        elif resultParamName=='FF': resultPlotName = 'Fill Factor, $FF$ [%]'
        elif resultParamName=='Jsc': resultPlotName = 'Short Circuit Current Density, $J_{sc}$ [mA/cm$^2$]'
        elif resultParamName=='Jph_norm1': resultPlotName = 'Photocurrent FOM'
        elif resultParamName=='Jph_norm0': resultPlotName = 'Photocurrent FOM'
        elif resultParamName in ['ARC2_d','ARC2_t']: resultPlotName = 'Upper ARC\nThickness [$\mu m$]'
        elif resultParamName in ['ARC1_d','ARC1_t']: resultPlotName = 'Lower ARC\nThickness [$\mu m$]'
        elif resultParamName in ['sc10_em_d','seg10_em_t']: resultPlotName = 'Segment 10\nThickness [$\mu m$]'
        elif resultParamName in ['sc9_em_d','seg9_em_t']: resultPlotName = 'Segment 9\nThickness [$\mu m$]'
        elif resultParamName in ['sc8_em_d','seg8_em_t']: resultPlotName = 'Segment 8\nThickness [$\mu m$]'
        elif resultParamName in ['sc7_em_d','seg7_em_t']: resultPlotName = 'Segment 7\nThickness [$\mu m$]'
        elif resultParamName in ['sc6_em_d','seg6_em_t']: resultPlotName = 'Segment 6\nThickness [$\mu m$]'
        elif resultParamName in ['sc5_em_d','seg5_em_t']: resultPlotName = 'Segment 5\nThickness [$\mu m$]'
        elif resultParamName in ['sc4_em_d','seg4_em_t']: resultPlotName = 'Segment 4\nThickness [$\mu m$]'
        elif resultParamName in ['sc3_em_d','seg3_em_t']: resultPlotName = 'Segment 3\nThickness [$\mu m$]'
        elif resultParamName in ['sc2_em_d','seg2_em_t']: resultPlotName = 'Segment 2\nThickness [$\mu m$]'
        if bottomHomo:
            if resultParamName in ['sc1_em_d','seg1_em_t']: resultPlotName = 'Segment 1\nEmitter\nThickness [$\mu m$]'
            elif resultParamName in ['sc1_em2_d','seg1_em2_t']: resultPlotName = 'Segment 1\nBase Thickness [$\mu m$]'
        else:
            if resultParamName in ['sc1_em_d','seg1_em_t']: resultPlotName = 'Segment 1\nThickness [$\mu m$]'
    elif multiline and shortVer:
        if resultParamName=='Eff': resultPlotName = 'PCE [%]'
        elif resultParamName=='QE': resultPlotName = 'EQE'
        elif resultParamName=='Voc': resultPlotName = '$V_{oc}$ [V]'
        elif resultParamName=='FF': resultPlotName = '$FF$ [%]'
        elif resultParamName=='Jsc': resultPlotName = '$J_{sc}$ [mA/cm$^2$]'
        elif resultParamName=='Jph_norm1': resultPlotName = '$J_{ph} FOM$'
        elif resultParamName=='Jph_norm0': resultPlotName = '$J_{ph} FOM$'
        elif resultParamName in ['ARC2_d','ARC2_t']: resultPlotName = 'Upper ARC\nTh. [$\mu m$]'
        elif resultParamName in ['ARC1_d','ARC1_t']: resultPlotName = 'Lower ARC\nTh. [$\mu m$]'
        elif resultParamName in ['sc10_em_d','seg10_em_t']: resultPlotName = 'Seg. 10\nTh. [$\mu m$]'
        elif resultParamName in ['sc9_em_d','seg9_em_t']: resultPlotName = 'Seg. 9\nTh. [$\mu m$]'
        elif resultParamName in ['sc8_em_d','seg8_em_t']: resultPlotName = 'Seg. 8\nTh. [$\mu m$]'
        elif resultParamName in ['sc7_em_d','seg7_em_t']: resultPlotName = 'Seg. 7\nTh. [$\mu m$]'
        elif resultParamName in ['sc6_em_d','seg6_em_t']: resultPlotName = 'Seg. 6\nTh. [$\mu m$]'
        elif resultParamName in ['sc5_em_d','seg5_em_t']: resultPlotName = 'Seg. 5\nTh. [$\mu m$]'
        elif resultParamName in ['sc4_em_d','seg4_em_t']: resultPlotName = 'Seg. 4\nTh. [$\mu m$]'
        elif resultParamName in ['sc3_em_d','seg3_em_t']: resultPlotName = 'Seg. 3\nTh. [$\mu m$]'
        elif resultParamName in ['sc2_em_d','seg2_em_t']: resultPlotName = 'Seg. 2\nTh. [$\mu m$]'
        if bottomHomo:
            if resultParamName in ['sc1_em_d','seg1_em_t']: resultPlotName = 'Seg. 1\nEmitter\nTh. [$\mu m$]'
            elif resultParamName in ['sc1_em2_d','seg1_em2_t']: resultPlotName = 'Seg. 1 Base\nTh. [$\mu m$]'
        else:
            if resultParamName in ['sc1_em_d','seg1_em_t']: resultPlotName = 'Seg. 1\nTh. [$\mu m$]'
    else:
        if resultParamName=='Eff': resultPlotName = 'PCE [%]'
        elif resultParamName=='QE': resultPlotName = 'EQE'
        elif resultParamName=='Voc': resultPlotName = '$V_{oc}$ [V]'
        elif resultParamName=='FF': resultPlotName = '$FF$ [%]'
        elif resultParamName=='Jsc': resultPlotName = '$J_{sc}$ [mA/cm$^2$]'
        elif resultParamName=='Jph_norm1': resultPlotName = '$J_{ph} FOM$'
        elif resultParamName=='Jph_norm0': resultPlotName = '$J_{ph} FOM$'
        elif resultParamName in ['ARC2_d','ARC2_t']: resultPlotName = 'Upper ARC Th. [$\mu m$]'
        elif resultParamName in ['ARC1_d','ARC1_t']: resultPlotName = 'Lower ARC Th. [$\mu m$]'
        elif resultParamName in ['sc10_em_d','seg10_em_t']: resultPlotName = 'Seg. 10 Th. [$\mu m$]'
        elif resultParamName in ['sc9_em_d','seg9_em_t']: resultPlotName = 'Seg. 9 Th. [$\mu m$]'
        elif resultParamName in ['sc8_em_d','seg8_em_t']: resultPlotName = 'Seg. 8 Th. [$\mu m$]'
        elif resultParamName in ['sc7_em_d','seg7_em_t']: resultPlotName = 'Seg. 7 Th. [$\mu m$]'
        elif resultParamName in ['sc6_em_d','seg6_em_t']: resultPlotName = 'Seg. 6 Th. [$\mu m$]'
        elif resultParamName in ['sc5_em_d','seg5_em_t']: resultPlotName = 'Seg. 5 Th. [$\mu m$]'
        elif resultParamName in ['sc4_em_d','seg4_em_t']: resultPlotName = 'Seg. 4 Th. [$\mu m$]'
        elif resultParamName in ['sc3_em_d','seg3_em_t']: resultPlotName = 'Seg. 3 Th. [$\mu m$]'
        elif resultParamName in ['sc2_em_d','seg2_em_t']: resultPlotName = 'Seg. 2 Th. [$\mu m$]'
        if bottomHomo:
            if resultParamName in ['sc1_em_d','seg1_em_t']: resultPlotName = 'Seg. 1 Emitter Th. [$\mu m$]'
            elif resultParamName in ['sc1_em2_d','seg1_em2_t']: resultPlotName = 'Seg. 1 Base Th. [$\mu m$]'
        else:
            if resultParamName in ['sc1_em_d','seg1_em_t']: resultPlotName = 'Seg. 1 Th. [$\mu m$]'



    if resultPlotName is None:
        print(f"WARNING :plot get result str: Result param name '{resultParamName}' is not recognized. Cannot get nice string for plots.")
        resultPlotName=resultParamName

    return resultPlotName




# ------------------------------------------------------------------------------------------------------------------





def plotPairwiseDistances(DBs, cols, baseline=None, metrics=['cityblock'], resultCols=None, exptAxisQualifier="", title=None, metricsPlotStyle='together', plot=True, savePlots=False, saveFolder=pldir, **kwargs):
    """
    Plots pairwise distances for experiments in each DB.
    Distances are calculated using only the columns supplied in cols.
    They are calculated using sklearn.metrics.pairwise_distances which allows different metrics to be used (see link below)
        These can be passed in parallel using metrics, with the appearance of the plots conntrolled using metricsPlotStyle
    A baseline-normalized version is plotted if baseline is not None (supply a list; the normalization values for each of the cols)
        The values for each experiment are normalized before the pairwise distances are computed
    If resultsCols is not None includes a plot for each resultCol where that performance metric is included along the distance matrix diagonal
    Assumes columns are numeric

    :param DBs: (DataBase or list thereof) DataBase(s) containing the data
    :param cols: (list of strings) the labels of the columns to include in the distance calculation
    :param baseline: (list of floats/ints) the col values to normalize the data with (correspondant to cols)
    :param metrics: (list of strs) the metrics to use to calculate the pwds (see sklearn link for details)
    :param resultCols: (list of strings) the labels of performance metrics to include in plots
    :param exptAxisQualifier: (str) a descriptor for the xy axis labels, will be added after 'Expt Idx By '
    :param title: (str or None) string to include in the titles for the plots
    :param plot: (bool) whether to plot the results (or just generate the pairwise distance matricies if False)
    :param metricsPlotStyle: (str; one of 'together','t','separate','s') whether to plot all metrics in the same figure or in separate figures
    :param savePlots: (bool) whether to save the plots, plot must be True or else this is ignored    
    :param saveFolder: (str) the path to the folder where results will be saved (filenames will be yymmdd_hhmmss__dbfilename__pairwisedistances(_res).png)
    :param kwargs: keyword arguments for sklearn.metrics.pairwise_distances (see link below)
    :return: distanceMatrix: (square numpy array) the absolute distance matrix
    :return: distMatrix_rel: (square numpy array) distance matrix where the inputs were normalized

    sklearn.metrics.pairwise_distances
    https://scikit-learn.org/stable/modules/generated/sklearn.metrics.pairwise_distances.html
    """



    if not isinstance(DBs, list):
        DBs = [DBs]

    if not isinstance(metrics,list):
        metrics = [metrics]
        if metrics[0] == None: metrics[0] = 'cityblock'
    

    print(f'pairwise distances: Calculating pairwise distances for {len(DBs)} DBs. Using distance metrics; {metrics}...')

    if not os.path.exists(saveFolder):
        os.makedirs(saveFolder)

    if baseline is not None and len(baseline) != len(cols):
        print(f"WARNING :pairwise distances: Length of baseline ('{len(baseline)}') does not match the len of cols ('{len(cols)}'). Setting baseline to None...")
        baseline = None
    
    if len(metrics) > 1:
        if not metricsPlotStyle.lower() in ['together', 't', 'separate', 's']:
            print(f"WARNING :pairwise distances: Metrics plot style is not recognized. Setting to 'together'...")
            metricsPlotStyle = 't'
    else:
        metricsPlotStyle = 't'


    distMats_abs_perDB = []
    distMats_rel_perDB = []

    for dbidx in range(len(DBs)):

        distMats_abs_permetric = []
        distMats_rel_permetric = []
        
        DB = DBs[dbidx]

        print(f'pairwise distances: Calculating pairwise distances for {DB.dbfilename}...')

        #input checks and preperation
        if isinstance(cols, str):
            cols = [cols]
        try:
            colData = DB.dataframe[cols]
        except KeyError:
            print(f"ERROR :pairwise distances: Not all distance columns ('{cols}') are in DB '{DB.dbfilename}'. Cannot compute distances. Skipping DB...")
            continue

        if resultCols is not None:
            if isinstance(resultCols,str):
                resultCols = [resultCols]
            try:
                resData = DB.dataframe[resultCols]
            except KeyError:
                print(f"WARNING :pairwise distances: Not all results columns ('{resultCols}') are in the DB. Cannot extract. Skipping plots which include results...")
                resultCols = None


        #compute distance matrix
        if 'n_jobs' not in kwargs:
            kwargs['n_jobs'] = 10


        for midx in range(len(metrics)):
            
            distanceMatrix = sklearn.metrics.pairwise_distances(pd.DataFrame.to_numpy(colData), metric=metrics[midx], **kwargs)

            if baseline is not None:
                colData_rel = colData / baseline
                distMatrix_rel = sklearn.metrics.pairwise_distances(pd.DataFrame.to_numpy(colData_rel), metric=metrics[midx], **kwargs)
            else:
                distMatrix_rel = None
            
            
            distMats_abs_permetric.append(distanceMatrix)
            distMats_rel_permetric.append(distMatrix_rel)
    
        distMats_abs_perDB.append(distMats_abs_permetric)
        distMats_rel_perDB.append(distMats_rel_permetric)
    
        
        
        #plot distance matricies for this db
        if plot:

            #prepare to plot
            if baseline is None: numRows=1
            else: numRows=2

            exptAxisLabel = f"Expt Idx"
            if exptAxisQualifier != None:
                if exptAxisQualifier.strip() != "":
                    exptAxisLabel += f" by {exptAxisQualifier}"

            if baseline is not None:
                bltext = f"\nBaseline values:{baseline}"
            else:
                bltext = ""
            text = f"Columns included:{cols}{bltext}"

            if title is not None:
                titlestr = f"{title} ; PWD for DB {DB.dbfilename[0:-4]}"
            else:
                titlestr = f"PWD for DB {DB.dbfilename[0:-4]}"


            if metricsPlotStyle.lower()[0:1]=='t': #metricsPlotStyle = together
                # one fig with all metrics
                if len(metrics) == 1:
                    mtext = metrics[0]
                else:
                    mtext = "metrics"
                
                #plot the pairwiwse distance matricies
                createPairwiseDistancePlot(distMats_abs_permetric, metrics=metrics, relPwdData=distMats_rel_permetric, resData=None, title=titlestr, exptAxisLabel=exptAxisLabel, \
                                           text=text, save=savePlots, saveName=os.path.join(saveFolder,f"{time.strftime(timeStr)}__{DB.dbfilename[:-4]}__{mtext}_pairwise-distances.png"))

                # plot distance matricies with the diagonal filled by result data
                # plot for each result column input separately
                if resultCols is not None:
                    for ridx in range(resData.shape[1]):
                        createPairwiseDistancePlot(distMats_abs_permetric, metrics=metrics, relPwdData=distMats_rel_permetric, resData=resData.iloc[:,ridx], resDataLabel=resData.columns[ridx], title=titlestr, exptAxisLabel=exptAxisLabel, \
                                           text=text, save=savePlots, saveName=os.path.join(saveFolder,f"{time.strftime(timeStr)}__{DB.dbfilename[:-4]}__{mtext}_pairwise-distances_{resData.columns[ridx]}.png"))

            else:
                 #each metric gets its own fig
                for midx in range(len(metrics)):
                    metr = metrics[midx]
                    metr = [metr]

                    createPairwiseDistancePlot(distMats_abs_permetric, metrics=metr, relPwdData=distMats_rel_permetric, resData=None, title=titlestr, exptAxisLabel=exptAxisLabel, \
                                           text=text, save=savePlots, saveName=os.path.join(saveFolder,f"{time.strftime(timeStr)}__{DB.dbfilename[:-4]}__{metr[0]}_pairwise-distances.png"))

                    if resultCols is not None:
                        for ridx in range(resData.shape[1]):
                            createPairwiseDistancePlot(distMats_abs_permetric, metrics=metr, relPwdData=distMats_rel_permetric, resData=resData.iloc[:, ridx], resDataLabel=resData.columns[ridx], title=titlestr, exptAxisLabel=exptAxisLabel, \
                                                       text=text, save=savePlots, saveName=os.path.join(saveFolder,f"{time.strftime(timeStr)}__{DB.dbfilename[:-4]}__{metr[0]}_pairwise-distances_{resData.columns[ridx]}.png"))

            print(f"pairwise distances: Completed plots for DB {DB.dbfilename}. Results saved to '{saveFolder}'.")

    print(f"pairwise distances: Success :Completed pairwise distance calculation.")

    return distMats_abs_perDB, distMats_rel_perDB



def createPairwiseDistancePlot(pwdData, metrics=['cityblock'], relPwdData=None, resData=None, resDataLabel=None, title=None, exptAxisLabel="", text=None, save=True, saveName=os.path.join(pldir, f"{time.strftime(timeStr)}__pairwise-distances.png")):
    """
    Create (and save) a pairwise distance matrix plot.
        Plot will have metrics along the horiz direction
        It  will have the baseline normalized plots in a second row (if the data is supplied)
            Unless there is only one metric, then the 2 plots are produced on the same row
        It will create the result data filled plots is resData is not None, otherwise, the pure pwd matricies are generated

    :param pwdData: list of pairwise distance matrix data, for each metric
    :param metrics: list of strings, the metrics labels for the pwdData
    :param relPwdData: list of normalized pairwise distance matrix data, for each metric (likewise ordering)
    :param resData: a 1d numpy.array, data to put along the diagonal of the pwd matrix (ordered as the pwd dims)
    :param resDataLabel: string, label for the colorbar of the resData diagonal
    :param title: string, title for the plot window
    :param exptAxisLabel: string, used as a sorting qualifier in the xy-axis labels (Expt Idx by exptAxisLabel)
    :param text: string, some info string to be placed at the top left corner of the plot in small text (feed the cols and baseline values here for display)
    :param save: boolean, save the plot?
    :param saveName: sting, path to save the plot
    :return: fig, axes: the matplotlib objects
    """

    #check if a relative plot should be included
    if (not isinstance(relPwdData, list) and relPwdData is not None) or (isinstance(relPwdData, list) and relPwdData[0] is not None):
        inclRelPlot=True
    else:
        inclRelPlot = False

    #determine the plot grid
    if len(metrics) == 1:
        numRows = 1
        if inclRelPlot:
            numCols = 2
        else:
            numCols = 1
    else:
        numCols = len(metrics)
        if inclRelPlot:
            numRows = 2
        else:
            numRows = 1


    #create figure
    fig, axes = plt.subplots(numRows, numCols, figsize=(6.5 * numCols, 5.2 * numRows + 2))
    try:
        ax = axes.ravel()
    except:
        ax = [axes]
    if title is not None:
        fig.canvas.manager.set_window_title(title)

    #prep diagonal matrix if present
    if resData is not None:
        diagData = np.zeros((pwdData[0].shape))
        np.fill_diagonal(diagData, resData)
        if np.max(resData) / np.min(resData) >= 100:
            useLog = True
        else:
            useLog = False


    #PLOT
    #plot absolute pwd for each metric and then normalized for each

    for midx in range(len(metrics)):


        if resData is None:
            p1 = ax[midx].imshow(pwdData[midx], cmap='viridis')
        else:
            p1 = ax[midx].imshow(np.ma.masked_array(pwdData[midx], diagData > 0), cmap='viridis')
            if useLog:
                p11 = ax[midx].imshow(np.ma.masked_array(diagData, diagData == 0), cmap='RdPu', norm=mpl.colors.LogNorm())
            else:
                p11 = ax[midx].imshow(np.ma.masked_array(diagData, diagData == 0), cmap='RdPu')
        b1 = plt.colorbar(p1, ax=ax[midx])  # , fraction=0.046, pad=0.04)
        b1.set_label(f'Absolute Pairwise Distances; {metrics[midx]}')
        if resData is not None:
            b11 = plt.colorbar(p11, ax=ax[midx])  # , fraction=0.046, pad=0.04)
            b11.set_label(resDataLabel)
        ax[midx].set_title(f"Absolute Pairwise Distances; {metrics[midx]}")
        ax[midx].set_xlabel(exptAxisLabel)
        ax[midx].set_ylabel(exptAxisLabel)
        ax[midx].xaxis.set_label_position('top')
        ax[midx].xaxis.set_ticks_position('top')

    if inclRelPlot:
        for midx in range(len(metrics)):

            if resData is None:
                p2 = ax[len(metrics) + midx].imshow(relPwdData[midx], cmap='viridis')
            else:
                p2 = ax[len(metrics) + midx].imshow(np.ma.masked_array(relPwdData[midx], diagData > 0), cmap='viridis')
                if useLog:
                    p21 = ax[len(metrics) + midx].imshow(np.ma.masked_array(diagData, diagData == 0), cmap='RdPu', norm=mpl.colors.LogNorm())
                else:
                    p21 = ax[len(metrics) + midx].imshow(np.ma.masked_array(diagData, diagData == 0), cmap='RdPu')
            b2 = plt.colorbar(p2, ax=ax[len(metrics) + midx])
            b2.set_label(f'Normalized Pairwise Distances; {metrics[midx]}')
            if resData is not None:
                b21 = plt.colorbar(p21, ax=ax[len(metrics) + midx])  # , fraction=0.046, pad=0.04)
                b21.set_label(resDataLabel)

            ax[len(metrics) + midx].set_title(f"Normalized Pairwise Distances; {metrics[midx]}")
            ax[len(metrics) + midx].set_xlabel(exptAxisLabel)
            ax[len(metrics) + midx].set_ylabel(exptAxisLabel)
            ax[len(metrics) + midx].xaxis.set_label_position('top')
            ax[len(metrics) + midx].xaxis.set_ticks_position('top')

    if text is not None:
        ax[0].text(0.02, 0.05, text, fontsize='small', transform=plt.gcf().transFigure)


    if save:
        plt.savefig(saveName)

    return fig, axes


# ------------------------------------------------------------------------------------------------------------------



def plotPairwiseDistanceScatter(db1_sorted, db2_sorted, cols, baseline=None, metric='cityblock', dblabels=['Start', 'End'], pwdPlots=True, saveFolder=pldir, nbins=50,  yaxisDB1=True, resultCols=None, exptAxisQualifier="", **pwdkwargs):
    """
    Plot scatter and density plots of the pairwise distances in one db vs another
        Assumes the two dbs are comparable; eg. same experiments at different points in an optimization

    Will also plot the pairwise distance matricies if pwdPlots is True.

    :param db1_sorted: (DataBase) the first database (comparable to db2)
    :param db2_sorted: (DataBase) the second database (comparable to db1)
    :param cols: (list of strings) the labels of the columns to include in the p.w. distance calculation
    :param baseline: (list of floats/ints) the col values to normalize the data with (correspondant to cols)
    :param metric: (string) the pairwise_distance metric to use for the plots/calculations
    :param dblabels: (list of str of len(2)) the labels for the two databases (eg. 'Start' and 'End') for plot and axes titles
    :param pwdPlots: (bool) Produce pwd plots too?
                                False; plot the absolute and baseline-normalized (if present) scatter and hist2d plots.
                                True; plot the scatter plots, plot hist2ds, plot the pairwise distance plots
    :param saveFolder: (str) Path to the folder to save the results
    :param yaxisDB1: {for the scatter and density plots} (bool) Which p.w. distances are put on the y axis. (True; use DB1 (first input), False; use DB2)
    :param nbins: {for the density plot} (int or other) the number of bins to use; follows the format of plt.hist2d's bins parameter (see link below)
    :param resultCols: {for pairwise distance plots} (list of strings) the labels of performance metrics to include in p.w. plots (not needed for scatter)
    :param exptAxisQualifier: {for pairwise distance plots} (str) a descriptor for the xy axis labels in the pairwise dist plots, will be added after 'Expt Idx By '
    :param pwdkwargs: keyword arguments for sklearn.metrics.pairwise_distances (see link below)
    :return:


    sklearn.metrics.pairwise_distances
    https://scikit-learn.org/stable/modules/generated/sklearn.metrics.pairwise_distances.html
    plt.hist2d (density plot)
    https://matplotlib.org/stable/api/_as_gen/matplotlib.pyplot.hist2d.html
    """


    #Check and prepare variables
    if not isinstance(dblabels, list):
        print(f"WARNING :pairwise distance scatter: Invalid dblabels '{dblabels}'. dblabels should be a list of strings of length 2. Setting to ['Start','End']...")
        dblabels = ['Start', 'End']
    else:
        if len(dblabels) != 2:
            print(f"WARNING :pairwise distance scatter: Invalid dblabels '{dblabels}'. dblabels should be a list of strings of length 2. Setting to ['Start','End']...")
            dblabels = ['Start', 'End']

    if yaxisDB1:
        y = db1_sorted
        ylabel = dblabels[0]
        x = db2_sorted
        xlabel = dblabels[1]
    else:
        x = db1_sorted
        xlabel = dblabels[0]
        y = db2_sorted
        ylabel = dblabels[1]


    print(f"pairwise distance scatter: Plotting {ylabel} vs {xlabel} pairwise distance scatter plots. (pwdPlots = {pwdPlots})\n"
          f"pairwise distance scatter: Note that input dbs should be pre-sorted so that experiments match along the db-idx.")

    if not os.path.exists(saveFolder):
        os.makedirs(saveFolder)

    if baseline is not None:
        if len(baseline) != len(cols):
            print(f"WARNING :pairwise distances: Length of baseline ('{len(baseline)}') does not match the len of cols ('{len(cols)}'). Setting baseline to None...")
            baseline = None


    print("pairwise distance scatter: Getting pairwise distance matricies...")
    #Get pairwise distance matricies
    pwd_y, pwd_y_rel = plotPairwiseDistances(y, cols, baseline=baseline, metrics=metric, resultCols=resultCols, exptAxisQualifier=exptAxisQualifier, plot=pwdPlots, savePlots=pwdPlots, saveFolder=saveFolder, **pwdkwargs)
    pwd_x, pwd_x_rel = plotPairwiseDistances(x, cols, baseline=baseline, metrics=metric, resultCols=resultCols, exptAxisQualifier=exptAxisQualifier,  plot=pwdPlots, savePlots=pwdPlots, saveFolder=saveFolder, **pwdkwargs)

    ## some code for colouring the points based upon the absoluter difference in their db-idx.
    # differenceMatrix = np.zeros(pwd_y.shape)
    # for row in range(differenceMatrix.shape[0]):
    #     for col in range(differenceMatrix.shape[1]):
    #         differenceMatrix[row, col] = np.abs(row - col)
    # colours = differenceMatrix[np.triu_indices_from(differenceMatrix, k=1)]

    # # plt.figure()
    # # plt.imshow(differenceMatrix)
    # # plt.show(block=True)


    # extract the unique pairwise distance values
    ypts = pwd_y[0][0][np.triu_indices_from(pwd_y[0][0], k=1)]
    xpts = pwd_x[0][0][np.triu_indices_from(pwd_x[0][0], k=1)]
    yx = zip(ypts, xpts)
    if baseline is not None:
        ypts_rel = pwd_y_rel[0][0][np.triu_indices_from(pwd_y_rel[0][0], k=1)]
        xpts_rel = pwd_x_rel[0][0][np.triu_indices_from(pwd_x_rel[0][0], k=1)]
        yx_rel = zip(ypts_rel, xpts_rel)
    else:
        ypts_rel = None ; xpts_rel=None


    Npts = len(ypts)
    Nexpts = 0.5 + 0.5 * np.sqrt(1+8*Npts)
    # print(f"shape:{pwd_x.shape}")

    print("pairwise scatter plots: Data extracted, plotting beginning...")

    if baseline is None: numPlots = 1
    else: numPlots = 2

    fig, axes = plt.subplots(1, numPlots, figsize=(6*numPlots, 4))
    if baseline is not None: ax = axes.ravel()
    else: ax = [axes]
    fig.canvas.manager.set_window_title(f"{ylabel} vs {xlabel} pairwise distance scatter plots")

    ax[0].scatter(xpts, ypts, marker='.', s=1.1,  alpha=0.5)
    ax[0].set_xlabel(f"{xlabel} Absolute Pairwise Distances")
    ax[0].set_ylabel(f"{ylabel} Absolute Pairwise Distances")
    ax[0].set_title("Absolute")
    if baseline is not None:
        ax[1].scatter(xpts_rel, ypts_rel, marker='.', s=1.1, alpha=0.5)
        ax[1].set_xlabel(f"{xlabel} Baseline-Normalized Pairwise Distances")
        ax[1].set_ylabel(f"{ylabel} Baseline-Normalized Pairwise Distances")
        ax[1].set_title("Baseline-Normalized")

    if baseline is not None:
        bltext = f"\nBaseline values:{baseline}"
    else:
        bltext = ""

    ax[0].text(0.02, 0.92, f"Columns included:{cols}{bltext}\nNpts:{Npts}    Nexpts:{Nexpts}", fontsize='small', transform=plt.gcf().transFigure)

    plt.savefig(os.path.join(saveFolder,f"{time.strftime(timeStr)}_{ylabel}-vs-{xlabel}_pairwise-distance-scatter.png"))


    # plot a density plot as well
    fig, axes = plt.subplots(1, numPlots, figsize=(6 * numPlots, 4))
    if baseline is not None: ax = axes.ravel()
    else: ax = [axes]
    fig.canvas.manager.set_window_title(f"{ylabel} vs {xlabel} pairwise distance density plots")

    h = ax[0].hist2d(xpts, ypts, bins=nbins, range=[[0,max(xpts)],[0,max(ypts)]])
    ax[0].set_xlabel(f"{xlabel} Absolute Pairwise Distances")
    ax[0].set_ylabel(f"{ylabel} Absolute Pairwise Distances")
    ax[0].set_title("Absolute")
    b = plt.colorbar(h[3],ax=ax[0])
    b.set_label("Experiment Density")


    if baseline is not None:
        h = ax[1].hist2d(xpts_rel, ypts_rel, bins=nbins, range=[[0, max(xpts_rel)], [0, max(ypts_rel)]])
        ax[1].set_xlabel(f"{xlabel} Baseline-Normalized Pairwise Distances")
        ax[1].set_ylabel(f"{ylabel} Baseline-Normalized Pairwise Distances")
        ax[1].set_title("Baseline-Normalized")
        b = plt.colorbar(h[3],ax=ax[1])
        b.set_label("Experiment Density")


    ax[0].text(0.02, 0.92, f"Columns included:{cols}{bltext}\nNpts:{Npts}    Nexpts:{Nexpts}    nbins:{nbins}", fontsize='small', transform=plt.gcf().transFigure)

    plt.savefig(os.path.join(saveFolder, f"{time.strftime(timeStr)}_{ylabel}-vs-{xlabel}_pairwise-distance-density.png"))

    return [pwd_y,pwd_x], [pwd_y_rel,pwd_x_rel], [ypts, xpts], [ypts_rel, xpts_rel]


# ------------------------------------------------------------------------------------------------------------------


def getThStackColoursTtoB(thColsTtoB):
    """
    Get the colours for a stack plot from the provided thCols list

    :param thColsTtoB: [list of str] th column labels from top to bottom
    :return: coloursTtoB: [list of str] matplotlib colours for each variable, top to bottom
    """

    coloursTtoB = []
    thColsTtoBCopy = thColsTtoB.copy()

    #arc?, dlarc?
    if thColsTtoB[0][0:3]=='ARC':
        coloursTtoB.append("gray")
        thColsTtoBCopy.pop(0)
    if thColsTtoB[0][0:3]=='ARC':
        coloursTtoB.append("lightgray")
        thColsTtoBCopy.pop(0)

    #abs layers
    colourPaletteBtoT = ['black','darkred','r', 'orange','y', 'greenyellow', 'g', 'c', 'royalblue', 'darkblue', 'darkviolet']

    if len(thColsTtoBCopy) < 11: #don't use black unless it is a 10J with bottom homojnc
        colourPaletteBtoT.pop(0)

    colourPaletteBtoT.reverse()
    absColoursBtoT = colourPaletteBtoT[0:len(thColsTtoBCopy)]
    # absColoursBtoT.reverse()
    coloursTtoB = coloursTtoB + absColoursBtoT

    return coloursTtoB




# ------------------------------------------------------------------------------------------------------------------


def plotDB_clusteringDendrogram_Agglomerative(db, colsToIncl, metric='euclidean', linkMethod='single', xlabel="Expt Idx", title=None, save=False, saveName=os.path.join(pldir, f"{time.strftime(timeStr)}__agglomerative-cluster-dendrogram.png"), **kwargs):
    """
    Use scipy.cluster.hierarchy.linkage to create a hierarchical/agglomerative clustering linkage matrix for the colsToIncl data from the dataframe
    Plot the cluster heirarchy using scipy.cluster.hierarchy.dendrogram

    Assumes db is already truncated and resorted if that is desired (note; x-axis labels is idx)

    https://docs.scipy.org/doc/scipy/reference/generated/scipy.cluster.hierarchy.linkage.html#scipy.cluster.hierarchy.linkage
    https://docs.scipy.org/doc/scipy/reference/generated/scipy.cluster.hierarchy.dendrogram.html#scipy.cluster.hierarchy.dendrogram

    :param db: [DataBase] The database to apply the agglomerative/herirarchial clustering to
    :param colsToIncl: [list of str] The db.dataframe column labels to include in the clusteringadded
    :param metric: [str, one of 'euclidean',...] The distance metric. See linkage link above for more information and valid inputs
    :param linkMethod: [str, one of 'single', 'complete', 'average', 'weighted', 'centroid', 'median','ward']
                            The method to link clusters. "method" parameter for linkage fnc. See linkage link above for more info
    :param xlabel: [str] The label to use for the x axis.
    :param title: [str or None] The title to use for the plot.
    :param save: [bool] Save the figure?
    :param saveName: [str] The path to save the figure. Includes filename.
    :param kwargs: [dict] Keyword arguments for scipy.cluster.heirarchy.dendrogram. See link above for more info.
    :return: linked [np.ndarray] The hierarchical clustering encoded as a linkage matrix. (Output of linkage function.)
    """

    #(aiirpower default = arcThLabelsTtoB_S4+absThLabelsTtoB_bottomHomo_S4)

    print(f"plot db agglomerative clusters: Plotting agglomerative clusters for DB '{db.dbfilename}' using '{linkMethod}' linkage and '{metric}' metric. Columns included; '{colsToIncl}'.")

    if save and not os.path.exists(os.path.split(saveName)[0]):
        os.makedirs(os.path.split(saveName)[0])

    #drop any bad data and then get the data values in np.array form
    db.dataframe.dropna(subset=colsToIncl, inplace=True)
    X = db.dataframe[colsToIncl].values

    #run linkage
    linked = linkage(X, method=linkMethod, metric=metric, optimal_ordering=True)

    #create dendrogram plot
    plt.figure(f"{title}: Agglomerative dendrogram for {db.dbfilename[:-4]} with {metric} {linkage} linking ({time.strftime(timeStr)}).")
    dendrogram(linked, **kwargs)
    plt.xlabel(xlabel)
    plt.ylabel(f"{metric} {linkMethod} linkage distance")
    if title is not None:
        plt.title(title)

    print(f"plot db agglomerative clusters: Success :Plot created.")

    if save:
        plt.savefig(saveName=saveName)

    return linked


# ------------------------------------------------------------------------------------------------------------------

def createDB_histogram_comparison(dbs, dbCols, labels=None, colors=None, xlabel=None, ylabel="Frequency", legend=True, inclSeparate=False, numDecPl=4, save=False, saveName=os.path.join(pldir, f"{time.strftime(timeStr)}__comparison_histogram.png"), **kwargs):
    """
    Use pandas dataframe.plot.hist to create a histogram comparing different databases or data within one database
        Histogram stats are print to screen and a figure.

    Note: To plot multiple columns from the same database; pass the db multiple times in dbs, set dbCols accordingly
        eg. Jph_norm0 and Jph_norm1 from one db: dbs=[egdb, egdb], dbCols=['Jph_norm0','Jph_norm1']

    https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.plot.hist.html


    :param dbs: [Database or list thereof] The databases to plot. Correspondent to dbCols. Must match.
    :param dbCols: [str or list of str] The column from each DB to plot. Correspondent to dbs. Must match.
    :param labels: [str, list of str, or None] The labels for each data series. Correspondent to dbs/dbCols. Must match. If None, uses dbfilename_dbCol.
    :param colors: [str, list of str, or None] The color to use for each data series. Correspondent to dbs/dbCols. Must match. If None, uses colorPalette set below.
    :param xlabel: [str or None] The label for the x-axis. If None, uses 'Figure of Merit'.
    :param ylabel: [str] The label for the y-axis.
    :param legend: [bool] Include a legend in the comparison histogram figure?
    :param inclSeparate: [bool] Include a figure where all data series are plotted separately? Ignored if only one data series is input.
    :param numDecPl: [int] Number of decimal places to format the histogram stats output.
    :param save: [bool] Save the figures? (or just create them)
    :param saveName: [str] Path to save the comparison histogram figure. Filename included. (Stats and separate figure have automatic filename appends.)
    :param kwargs: [dict] Keyword arguments for pandas dataframe.plot.hist
    :return: tableData [list of lists] For each dataset (outer list idx); the [label, numData, mean, stddev, stddevofthemean] (inner lists)
    """


    #check and preapare inputs
    if not isinstance(dbs, list):
        dbs = [dbs]

    if not isinstance(dbCols, list):
        dbCols = [dbCols] * len(dbs) #same col for all dbs (or one db)

    if labels is None:
        labels = []
        for dbidx in range(len(dbs)):
            labels.append(dbs[dbidx].dbfilename[:-4] + "_" + dbCols[dbidx])

    if not isinstance(labels, list):
        labels = [labels]

    if colors is None:
        colorPalette = ['blue', 'orange', 'green', 'red', 'm', 'c', 'y', 'k', 'navy', 'crimson']
        colors = colorPalette[0:len(dbs)]

    if not isinstance(colors, list):
        colors = [colors]

    if len(dbs) != len(dbCols) or len(dbs) != len(labels) or len(dbs) != len(colors):
        print(f"ERROR :create comparison histogram: Input list lengths do not match. len(dbs)={len(dbs)}, len(dbCols)={len(dbCols)}, len(labels)={len(labels)}, len(colors)={len(colors)}. Cannot continue. Exiting...")
        return

    if xlabel is None:
        xlabel = "Figure of Merit"

    if save and not os.path.exists(os.path.split(saveName)[0]):
        os.makedirs(os.path.split(saveName)[0])



    print(f"create comparison histogram: Creating comparison histogram for {len(dbs)} datasets; '{labels}'")

    #Plot
    plt.rcParams.update({'font.size': 16})

    #combo plot
    fig = plt.figure(f"Comparison Histogram Combo {time.strftime(timeStr)}")
    for pidx in range(len(dbs)):
        dbs[pidx].dataframe[dbCols[pidx]].plot.hist(label=labels[pidx], color=colors[pidx], **kwargs)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    if len(dbs) > 1: plt.title("Combo")
    else: plt.title(labels[0])
    if legend: plt.legend()
    plt.tight_layout()
    if save:
        plt.savefig(saveName)

    fig = plt.figure(f"Comparison Histogram Combo w Same Bins {time.strftime(timeStr)}")
    ax = plt.gca()
    absMin = dbs[0].dataframe[dbCols[0]].min()
    for pidx in range(len(dbs)):
        dataMin = dbs[pidx].dataframe[dbCols[pidx]].min()
        if dataMin < absMin: absMin = dataMin
    if 'bins' not in kwargs: bins=25

    roundMin = absMin
    step = (1.0-0.3)/(bins)
    bins = list(np.arange(start=0.3, stop=1.0+step, step=step))#int(bins+1))

    for pidx in range(len(dbs)):
        plt.hist(x=np.array(dbs[pidx].dataframe[dbCols[pidx]].values), label=labels[pidx], color=colors[pidx],  bins=bins, **kwargs)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    if len(dbs) > 1:
        plt.title("Combo")

    else:
        plt.title(labels[0])
    if legend: plt.legend()
    plt.tight_layout()
    if save:
        plt.savefig(saveName)

    #separate plots if there is more than one to plot and it is desired
    if len(dbs) > 1 and inclSeparate:
        fig, axes = plt.subplots(1, len(dbs))
        fig.canvas.manager.set_window_title(f"Comparison Histogram Separate {time.strftime(timeStr)}")
        ax = axes.ravel()
        for pidx in range(len(dbs)):
            plt.axes(ax[pidx])
            ax[pidx].set_title(labels[pidx])
            ax[pidx].set_xlabel(xlabel)
            ax[pidx].set_ylabel(ylabel)
            dbs[pidx].dataframe[dbCols[pidx]].plot.hist(color=colors[pidx], **kwargs)
        plt.tight_layout()
        if save:
            plt.savefig(saveName[:-4]+"_Separate.png")

    print(f"create comparison histogram: Histograms created.")


    #histo stats
    print(f"create comparison histogram: HISTOGRAM STATS --------------------------------------------------")
    print(f"DATASET\t\t|NUM DATA\t\t|MEAN\t\t|STD DEV\t\t|STD DEV OF THE MEAN")
    numdata, means, stddevs, stddevmeans = [], [], [], []
    tableData = []
    for pidx in range(len(dbs)):
        numdata.append(dbs[pidx].dataframe[dbCols[pidx]].count())
        means.append(dbs[pidx].dataframe[dbCols[pidx]].mean())
        stddevs.append(dbs[pidx].dataframe[dbCols[pidx]].std())
        stddevmeans.append(stddevs[-1] / np.sqrt(numdata[-1]))
        print(f"{labels[pidx]}\t|{numdata[-1]}\t\t|{format(means[-1], f'.{numDecPl}f')}\t|{format(stddevs[-1], f'.{numDecPl}f')}\t|{format(stddevmeans[-1], f'.{numDecPl}f')}")
        tableData.append([labels[pidx], numdata[-1], format(means[-1], f".{numDecPl}f"), format(stddevs[-1], f'.{numDecPl}f'),  format(stddevmeans[-1], f'.{numDecPl}f')])
    print(f"------------------------------------------------------------------------------------------------")

    #histo stats table figure
    tablefig = plt.figure(f"Comparison Histogram STATS {time.strftime(timeStr)}")
    plt.box(on=None)
    ax = plt.gca()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
    colLabels = ["Dataset", "N", "Mean", "S.D.", "S.D. Mean"]
    table = plt.table(cellText=tableData, colLabels=colLabels, loc='center')
    if save:
        plt.savefig(saveName[:-4] + f"_STATS.png")

    print(f"create comparison histogram: Success :Created comparison histogram for data; '{labels}'")

    return tableData






# ------------------------------------------------------------------------------------------------------------------


def createDRInv_matrixPlot(invDF, xlabel='n_components', ylabel='lo', zlabel='var_tot_pc',
                           xPlotLabel='Number of Reduced Dimensions', yPlotLabel='Jph FOM Cut Off', zPlotLabel='Total Explained Variance [frac]',
                           plotDescriptor = None,
                           inclTablePlot=True, show=False, save=False, saveName=None, cmap='viridis'):
    """
    Creates a DR investigation (stage 2) performance matrix;
        ie. The investigation dataframe (invDF) column 'zlabel' as a function of 'xlabel' and 'ylabel'
        eg. fraction of total variance in training data as a function of number of reduced dimensions and FOM cut-off

    Also creates a data table figure with # experiments in the training data as a function of the 'ylabel' (if inclTablePLot)

    :param invDF: [Dataframe] The DR investigation dataframe with the data to plot
    :param xlabel: [str] Data column label for the x dimension (eg. 'n_components')
    :param ylabel: [str] Data column label for the y dimension (eg. 'hi')
    :param zlabel: [str] Data column label for the z dimension (eg. 'var_tot_pc')
    :param xPlotLabel: [str] Axis label for x dimension (eg. 'Number of Reduced Dimension')
    :param yPlotLabel: [str] Axis label for y dimension (eg. 'Jph FOM Cut Off')
    :param zPlotLabel: [str] Axis label for z dimension (eg. 'Total Explained Variance [frac]')
    :param plotDescriptor: [str] descriptor to include in the plot window title
    :param inclTablePlot: [bool] Output a figure with the number of training expts table (still printed to screen if False)
    :param show: [bool] Show the plots ... !!Pauses computation!!
    :param save: [bool] Save the plots?
    :param saveName: [str] Path and filename for the matrix, should be png or other matplotlib outputable format (table gets _trainingExptTable appended to filename)
    :return: X[0,:]: [1d np array] The X values from the matrix
    :return: Y[:,0]: [1d np array] The y values from the matrix
    :return: Z: [2d np array] The performance values from the matrix.
    """

    #prep inputs
    if xPlotLabel is None: xPlotLabel = xlabel
    if yPlotLabel is None: yPlotLabel = ylabel
    if zPlotLabel is None: zPlotLabel = zlabel

    if save and saveName is None:
        saveName = os.path.join(pldir, f"{datetime.now().strftime(msTimeStr)}_stage-2_matrix.png")

    print(f"create DR investigation matrix: Creating STAGE 2 result matrix using DR investigation columns x='{xlabel}', y='{ylabel}', z='{zlabel}'...")

    if not show and not save:
        print(f"WARNING :create DR investigation matrix: Neither show nor save are selected within this function.")

    if save and not os.path.exists(os.path.split(saveName)[0]):
        os.makedirs(os.path.split(saveName)[0])

    if plotDescriptor is None:
        if saveName is not None:
            plotDescriptor = os.path.split(saveName)[1]
        else:
            plotDescriptor = time.strftime(timeStr)

    #define mesh and result np.objects from investigation dataframe
    # eg. xyz = invDFout[['n_components','hi','var_tot_pc']].to_numpy()
    xyz = invDF[[xlabel,ylabel,zlabel]].to_numpy()
    X,Y = np.meshgrid(np.unique(xyz[:,0]), np.unique(xyz[:,1]))
    Z = np.zeros((X.shape[0],X.shape[1]))

    #get data from dataframe and fill into result array
    for xi in range(X.shape[1]):
        for yi in range(Y.shape[0]):
            Z[yi,xi] = invDF.query(f"{xlabel} == {X[0][xi]} & {ylabel} == {Y[yi][0]}")[zlabel]


    #create figure with matrix and colourbar
    plotSizeMultiplier=1.5
    fig = plt.figure(f"DR investigation matrix: {plotDescriptor}", figsize=[6.4*plotSizeMultiplier, 4.8*plotSizeMultiplier])
    p = plt.pcolormesh(X,Y,Z, cmap=cmap)
    b = plt.colorbar(p)

    ax = fig.gca()
    matrixHeight = ax.get_position().height

    alt = False
    if alt:
        plt.xticks([])
        plt.yticks([])
        b.set_ticks([])
    else:
        plt.xlabel(xPlotLabel)
        plt.ylabel(yPlotLabel)
        b.set_label(zPlotLabel)
        plt.xticks(X[0])
        plt.yticks(Y[:,0])

        #add value labels to matrix
        for xi in range(X.shape[1]):
            for yi in range(Y.shape[0]):
                ax.text( X[0][xi], Y[yi][0], '{:0.2f}'.format(Z[yi,xi]), ha='center', va='center')

    if save:
        plt.savefig(saveName)
        print(f"create DR investigation matrix: Saving plot... '{saveName}'")


    #pull training data
    colLabels = [yPlotLabel, "Training Data Variance", "# Training Expts"]
    colData = []
    for di in range(len(Y[:,0])):
        colData.append([Y[di,0], format(invDF['data_var'].iloc[di], ".4f"), invDF['n_expts_train'].iloc[di]])
    if colData[0][0] < colData[-1][0]: colData.reverse()
    # print(f"\n\n\n# training expts\n{invDF['n_expts_train']}\n\nY\n{Y}\n\n")
    # print(invDF['data_var'])

    #print training data
    print(f"create DR investigation matrix: Number of training experiments per dataset -------------------------------\n"
          f"|\t{colLabels[0]}\t|\t{colLabels[1]}\t|")
    for ridx in range(len(colData)):
        print(f"|\t{colData[ridx][0]}\t|\t{colData[ridx][1]}\t|")
    print(f"----------------------------------------------------------------------------------------------------------")

    #add figure with table of # training experiments
    if inclTablePlot:
        tablefig = plt.figure(f"DR Investigation Training Data Table: {plotDescriptor}")
        plt.box(on=None)
        ax = plt.gca()
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)

        table = plt.table(cellText=colData, colLabels=colLabels, loc='center')

        cellHeight = matrixHeight/(len(colData))
        cellDict = table.get_celld()
        for i in range(0, len(colLabels)):
            # cellDict[(0, i)].set_height(cellHeight)  #header row (set j loop start to 1)
            for j in range(0, len(colData) + 1):
                cellDict[(j, i)].set_height(cellHeight) #data rows

        if save:
            plt.savefig(saveName[:-4]+"_trainingExptTable.png")
            print(f"create DR investigation matrix: Saving plot... '{saveName[:-4]}_trainingExptTable.png'")


    print(f"create DR investigation matrix: Success :Stage 2 matrix plot completed.")

    if show:
        plt.show(block=True)

    return X[0,:], Y[:,0], Z


# ------------------------------------------------------------------------------------------------------------------

def plotDRInv_DRComponents(invDFpath):
    #todo
    #somehow find a nice plot of the 12 DR coeffients for each pca dim for each set of training data in the DR investigation
    return


# ------------------------------------------------------------------------------------------------------------------

