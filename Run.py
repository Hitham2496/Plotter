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
from Plotter import plot_stacked_ratio as plot_stacked_ratio
from matplotlib import pyplot as plt # plotting
import matplotlib.gridspec as gridspec # more plotting
from matplotlib.ticker import (MultipleLocator) # even more plotting
from matplotlib.lines import Line2D # yet more plotting

dy_bins = [".","v","^","<",">","2","1","x"]
dy_bins_marks = [Line2D([], [], color='black', marker=mark) for mark in dy_bins]
dy_corr = [r"$0\leq \Delta y \leq 1$", r"$1\leq \Delta y \leq 2$",r"$2\leq \Delta y \leq 3$",r"$3\leq \Delta y \leq 4$",r"$4\leq \Delta y \leq 5$",
        r"$5\leq \Delta y \leq 6$",r"$6\leq \Delta y \leq 7$",r"$7\leq \Delta y \leq 8$"]

f1 = ["d13-x01-y01.dat","d14-x01-y01.dat","d15-x01-y01.dat","d16-x01-y01.dat","d17-x01-y01.dat","d18-x01-y01.dat","d19-x01-y01.dat","d20-x01-y01.dat"]
f2 = ["d21-x01-y01.dat","d22-x01-y01.dat","d23-x01-y01.dat","d24-x01-y01.dat","d25-x01-y01.dat","d26-x01-y01.dat","d27-x01-y01.dat","d28-x01-y01.dat"]

# plot_stacked([unpack(A) for A in f1], 1., r"$\Delta \phi / \pi$", r"$d^{2} \sigma / d \Delta \phi d \Delta y$"r"    [pb/rad]    (inclusive)", "XS-inclusive", dy_bins,
#         dy_bins_marks, dy_corr, (0.1, 0.02))
# plot_stacked_ratio([unpack(A) for A in f1], r"$\Delta \phi / \pi$", r"Theory / Data", "XS-inclusive_ratio", dy_bins,
#         dy_bins_marks, dy_corr, (0.1, 0.02), r"Inclusive Events")
# 
# plot_stacked([unpack(A) for A in f2], 1., r"$\Delta \phi / \pi$", r"$d^{2} \sigma / d \Delta \phi d \Delta y$"r"    [pb/rad]    (gap)", "XS-gap", dy_bins,
#         dy_bins_marks, dy_corr, (0.1, 0.02))
# plot_stacked_ratio([unpack(A) for A in f2], r"$\Delta \phi / \pi$", r"Theory / Data", "XS-gap_ratio", dy_bins,
#         dy_bins_marks, dy_corr, (0.1, 0.02), r"Gap Events")
# 
plot_single([unpack("d03-x01-y01.dat")], r"$|\Delta y_{12}|$", r"$\langle N_{\mathrm{jets}} \textrm{in rapidity interval} \rangle$" , "NJDY", (1.,0.2))
