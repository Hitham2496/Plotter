#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Plotting commands

@author: hhassan
"""
import Plotter as plotter
from Plotter import unpack as unpack
from Plotter import plot_stacked as plot_stacked
from Plotter import plot_single as plot_single
from Plotter import plot_scale_bands as plot_scale_bands
from Plotter import plot_stacked_ratio as plot_stacked_ratio
from matplotlib import pyplot as plt # plotting
import matplotlib.gridspec as gridspec # more plotting
from matplotlib.ticker import (MultipleLocator) # even more plotting
from matplotlib.lines import Line2D # yet more plotting

#dy_bins = [".","v","^","<",">","2","1","x"]
#dy_bins_marks = [Line2D([], [], color='black', marker=mark) for mark in dy_bins]
#dy_corr = [r"$0\leq \Delta y \leq 1$", r"$1\leq \Delta y \leq 2$",r"$2\leq \Delta y \leq 3$",r"$3\leq \Delta y \leq 4$",r"$4\leq \Delta y \leq 5$",
#        r"$5\leq \Delta y \leq 6$",r"$6\leq \Delta y \leq 7$",r"$7\leq \Delta y \leq 8$"]

plot_single([unpack("h_dy_jet12_2j.dat")], r"$\Delta y_{12}$", r"$d\sigma_{W+\geq 2j}/d \Delta y_{12}$", "h_dy_jet12_2j", (1.,0.2), (7e-4,80))
