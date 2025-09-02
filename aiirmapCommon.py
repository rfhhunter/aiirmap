"""
The AiirMap Common Functions

Import this file for aiirmap functionality.

250825
"""

##import packages
import shutil
import sys
import time
import os
import glob
import copy
import pickle
from datetime import datetime
import itertools
import numpy as np
import pandas as pd
from sklearn.metrics import pairwise_distances
import scipy.optimize as opt
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator,AutoMinorLocator
import matplotlib as mpl


##import utilities
import utilities.databasing as dbh
import utilities.sReadAndInteract as si
import utilities.plotting as pl
import utilities.BeerLambertTools as bl
import utilities.Matlab as matl
import utilities.runtimeLogger as rtl
import utilities.ml.machine_learning as ml


##import configuration
from config import *


