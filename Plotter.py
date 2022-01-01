#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Plotting routine for .dat files created by rivet-mkhtml

@author: Hitham2496
"""

import numpy as np  # numerical python
from matplotlib import pyplot as plt  # plotting
from matplotlib.ticker import MultipleLocator
from itertools import groupby

outputDirectory = 'plots/'
dataStr = r'Data'
legendStr = r'Theoretical Predictions'
printOuts = True
eps = 1e-20

class plot_env:

    def __init__(self, t, xL, yL, **kwargs):
        allowed_keys = ['logY', 'ratio']
        self.Title = t
        self.xLabel = xL
        self.yLabel = yL
        self.logY = 0
        self.ratio = 1
        self.__dict__.update((k, v)
                             for k, v in kwargs.items() if k in allowed_keys)

    plots = []

    def addPlot(self, p):
        self.plots.append(p)

    @property
    def plotList(self):
        return [j.title for j in self.plots]

    def clearPlots(self):
        self.plots = []

    def get_axes(self, Title, bins, xTup, yTup, xLab="", yLab="") :
        """Returns graph and axes for plotting"""

        graph, ax = plt.subplots(nrows=2, ncols=1, sharex=True, sharey=False,
                                 gridspec_kw={'height_ratios': [2.5, 1]},
                                 figsize=(6, 6))
        if (self.ratio != 1):
            graph, ax = plt.subplots(nrows=1, ncols=1, sharex=True,
                                     sharey=False, figsize=(6, 6))

            if (xLab != ""):
                ax.set_xlabel(xLab)
            else:
                ax.set_xlabel(self.xLabel)
            if (yLab != ""):
                ax.set_ylabel(yLab)
            else:
                ax.set_ylabel(self.yLabel)
            ax.set_xlim(bins[0], bins[-1])
            ax.set_ylim(yTup[0], yTup[1])

            ax.yaxis.set_ticks_position('both')
            ax.xaxis.set_ticks_position('both')
            ax.xaxis.set_major_locator(MultipleLocator(xTup[0]))
            ax.xaxis.set_minor_locator(MultipleLocator(xTup[1]))

            if(self.logY == 1):
                ax.set_yscale('log')

        else:

            if (xLab != ""):
                ax[1].set_xlabel(xLab)
            else:
                ax[1].set_xlabel(self.xLabel)
            if (yLab != ""):
                ax[0].set_ylabel(yLab)
            else:
                ax[0].set_ylabel(self.yLabel)

            ax[1].set_ylabel("Ratio")
            ax[1].set_xlim(bins[0], bins[-1])
            ax[0].set_ylim(yTup[0], yTup[1])
            ax[1].set_ylim(0., 2.)

            ax[0].yaxis.set_ticks_position('both')
            ax[0].xaxis.set_ticks_position('both')
            ax[1].xaxis.set_major_locator(MultipleLocator(xTup[0]))
            ax[1].xaxis.set_minor_locator(MultipleLocator(xTup[1]))

            ax[1].yaxis.set_ticks_position('both')
            ax[1].xaxis.set_ticks_position('both')
            ax[1].yaxis.set_minor_locator(MultipleLocator(0.1))
            ax[1].yaxis.set_major_locator(MultipleLocator(0.2))

            plt.subplots_adjust(hspace=0)
            if(self.logY == 1):
                ax[0].set_yscale('log')

        return graph, ax

    @property
    def n_plots(self):
        return len(self.plots)


class dataset:

    def __init__(self, t, xl, xh, y, **kwargs):
        allowed_keys = ['col', 'errsM', 'errs', 'errsP', 'norm']
        self.title = t
        self.Xl = xl
        self.Xh = xh
        self.Y = y
        self.col = 'red'
        self.norm = 1.
        self.errs = 0
        self.__dict__.update((k, v)
                             for k, v in kwargs.items() if k in allowed_keys)

    def integral(self):
        return "{:.3f}".format(np.sum(((np.array(self.Xh)-np.array(self.Xl))*np.array(self.Y))))


def unpack(filename):
    """Unpacks the plot data file"""
    plot_list = []
    ll = []
    ld = []

    with open(filename) as f:
        # create a new array every new line
        for k, g in groupby(f, lambda x: x.startswith('# BEGIN')):
            if not k:
                # combine each array of triples to a list of arrays
                plot_list.append(np.array([[str(x) for x in d.split('\n')]
                                 for d in g if len(d.strip())]))

    # for j in range(0, len(plot_list)):
    #     print(plot_list[j],'\n\n\n')
    for k in range(0, len(plot_list[0])-1):
        arr = np.array([[str(x) for x in plot_list[0][k][0].split('=')]])
        ll.append(arr[0, 0])
        ld.append(arr[0, 1])

    ti = ld[ll.index('Title')]
    xLa = ld[ll.index('XLabel')]
    yLa = ld[ll.index('YLabel')]
    logy = int(ld[int(ll.index('LogY'))])
    r = int(ld[int(ll.index('RatioPlot'))])

    p = plot_env(ti, xLa, yLa, logY=logy, ratio=r)
    p.clearPlots()
    for l in range(1, len(plot_list)):
        TEMP = []
        arr_t = []
        arr_p = []
        arr_d = []
        xl = []
        xh = []
        y = []
        errsm = []
        errsp = []
        kstar = 0
        for k in range(0, len(plot_list[l])-1):
            TEMP.append(np.array([[str(x)
                                   for x in plot_list[l][k][0].split()]]))
            if(TEMP[k][0][0] == '#'):
                # k* is the index at which the plot data begins
                kstar = k
        # (length of the plot info list - 1) - (k* - 1)
        z = len(plot_list[l]) - kstar - 2
        res = TEMP[-z:]
        for j in range(0, kstar):
            arr_t.append(np.array([str(x) for x in TEMP[j][0][0].split('=')]))
            arr_p.append(arr_t[j][0])
            arr_d.append(arr_t[j][1])

        t = arr_d[arr_p.index('Title')]
        if (len(TEMP[arr_p.index('Title')][0]) > 1):
            for j in range(1, len(TEMP[arr_p.index('Title')][0])):
                t += ' '
                t += TEMP[arr_p.index('Title')][0][j]
        no = 1.
        if ('Scale' in arr_p):
            no = arr_d[arr_p.index('Scale')]
        c = 'Black'
        if (t != 'Data'):
            c = arr_d[arr_p.index('LineColor')]
        for n in range(0, len(res)):
            xl.append(float(res[n][0][0]))
            xh.append(float(res[n][0][1]))
            y.append(float(res[n][0][2]))
            errsm.append(float(res[n][0][3]))
            errsp.append(float(res[n][0][4]))

        p.addPlot(dataset(t, xl, xh, y, col=c, errs=0,
                          errsM=errsm, errsP=errsp, norm=no))
    return p


def sort_env(p_env, sv_list):
    """Sorts the data in a plot_env if scale bands are present
       currently only implemented for central and two variations"""
    scale_vars = []
    rest = []
    j = 0
    if(sv_list == 0):
        return 0, p_env.plots
    while j < len(p_env.plots):
        if (j in sv_list):
            scale_vars.append([p_env.plots[j], p_env.plots[j+1],
                               p_env.plots[j+2]])
            j += 3
        else:
            rest.append(p_env.plots[j])
            j += 1
    return scale_vars, rest


def plot_single(p_env, Title, xTup, yTup, sv_list=0, xLab="", yLab=""):
    """Plots all of the distributions corresponding to a single histogram"""
    plt.rc('text', usetex=True)
    plt.rc('font', family='serif')

    bins = p_env.plots[0].Xl[:]
    bins.append(p_env.plots[0].Xh[-1])

    graph, ax = p_env.get_axes(Title, bins, xTup, yTup, xLab, yLab)

    pts = []
    new_bins = p_env.plots[0].Xh[:]
    new_bins.insert(0, p_env.plots[0].Xl[0])
    x_errors = []
    for j in range(0, len(new_bins)-1):
        pts.append((new_bins[j+1] + new_bins[j]) / 2.)
        x_errors.append((new_bins[j+1] - new_bins[j]) / 2.)

    sv, rest = sort_env(p_env, sv_list)
    ax2 = 0
    
    if printOuts: print("Plotting %s to %s" % (Title, outputDirectory+Title+".pdf"))

    if (sv != 0):
        if (p_env.ratio == 1):
            ax2 = ax[1]
            for pl in sv:
                if printOuts: print("\tPlotting curve with scale uncertainty %s for %s" % (pl.title, Title))
                ax[0], ax2 = plot_scale_bands(pl, ax[0], ax2, pts, x_errors, data=p_env.plots[0]) #data=sv[0][0])
        else:
            for pl in sv:
                if printOuts: print("\tPlotting curve with scale uncertainty %s for %s" % (pl.title, Title))
                ax, ax2 = plot_scale_bands(pl, ax, ax2, pts, x_errors, data=p_env.plots[0]) #data=sv[0][0])

    for p in rest:
        if printOuts: print("\tPlotting curve %s for %s" % (pl.title, Title))
        if (p_env.ratio != 1):
            if (p.title == "Data"):
                ax.errorbar(pts, [z for z in p.Y], xerr = x_errors,
                             yerr = (p.errsM, p.errsP), fmt = '.',
                             color = "black", markersize='5', linewidth = .75,
                             label = dataStr+" : "+str(p.integral())+" fb")
            else:
                ax.step(p.Xl, [z for z in p.Y], where='post', color=p.col, linewidth=.75, label=p.title+" : "+str(p.integral())+" fb", linestyle='-')
                ax.errorbar(pts, [z for z in p.Y], xerr=x_errors, yerr=(p.errsM, p.errsP), color=p.col, fmt='',
                            linewidth=.75, ls='none')
        else:
            if (p.title == "Data"):
                ax[0].errorbar(pts, [z for z in p.Y], xerr = x_errors,
                             yerr = (p.errsM, p.errsP), fmt = '.',
                             color = "black", markersize='5', linewidth = .75,
                             label = dataStr+" : "+str(p.integral())+" fb")
                ax[1].errorbar(pts, [(x+eps)/(y+eps) for x, y in zip(p.Y, p.Y)],
                                   xerr = x_errors, yerr = ([(x+eps)/(y+eps)
                                   for x, y in zip(p.errsM, p.Y)],[(x+eps)/(y+eps)
                                   for x, y in zip(p.errsP, p.Y)]),
                             fmt = '.', markersize='5', color = "black",
                             linewidth = .75,
                             label = dataStr+" : "+str(p.integral())+" fb")
            else:
                ax[0].step(p.Xl, [z for z in p.Y], where='post', color=p.col, linewidth=.75, label=p.title+" : "+str(p.integral())+" fb", linestyle='-')
                ax[0].errorbar(pts, [z for z in p.Y], xerr=x_errors, yerr=(p.errsM, p.errsP), color=p.col, fmt='',
                               linewidth=.75, ls='none')
                ax[1].step(p.Xl,
                           [x/(y+eps) for x, y in zip(p.Y, p_env.plots[0].Y[:len(p.Y)])], where='post',
                           color=p.col, linewidth=.75, label=p.title)
                ax[1].errorbar(pts, [x/(y+eps) for x, y in zip(p.Y, p_env.plots[0].Y[:len(p.Y)])], xerr=x_errors,
                               yerr=([x/(y+eps) for x, y in zip(p.errsM, p.Y)], [x/(y+eps) for x, y in zip(p.errsP, p.Y)]),
                               color=p.col, fmt='', linewidth=1., ls='none')

    if (p_env.ratio != 1):
        ax.legend(loc='upper right', title=legendStr, frameon=False)
        ax.get_legend()._legend_box.align = "left"
    else:
        ax[0].legend(loc='upper right', title=legendStr, frameon=False)
        ax[0].get_legend()._legend_box.align = "left"
    plt.savefig(outputDirectory+Title+".pdf", bbox_inches="tight")

def plot_scale_bands(plots, ax1, ax2, pts, x_errors, **kwargs):
    """Plots a single histogram with scale variation bands"""
    pts = []
    new_bins = plots[0].Xh[:]
    new_bins.insert(0, plots[0].Xl[0])
    x_errors = []
    y_minmax = []
    data = plots[0]
    for key, value in kwargs.items():
        if key == 'data':
            data = value

    for j in range(0, len(new_bins)-1):
        pts.append((new_bins[j+1] + new_bins[j]) / 2.)
        x_errors.append((new_bins[j+1] - new_bins[j]) / 2.)

    # plot data
    for p in plots:
        xa = 1
        if plots.index(p) == 0:
            for j in range(0, len(new_bins)-1):
                y_minmax.append([p.Y[j]*xa, p.Y[j]*xa])
            ax1.errorbar(pts, [z*xa for z in p.Y],
                         xerr=x_errors, yerr=(p.errsM, p.errsP),
                         fmt='.', color=p.col, markersize='0.01',
                         linewidth=.75, label=p.title+" : "+str(p.integral())+" fb")
            ax1.step(p.Xl, [z*xa for z in p.Y], where='post', color=p.col,
                     linewidth=.25, alpha=0.5, linestyle='-')
            if (ax2 != 0):
                ax2.errorbar(pts,
                             [(x*xa+eps)/(y+eps) for x, y in zip(p.Y, data.Y)],
                             xerr=x_errors, yerr=([(x*xa+eps)/(y+eps)
                                                  for x, y in
                                                  zip(p.errsM, data.Y)],
                             [(x*xa+eps)/(y+eps)
                              for x, y in zip(p.errsP, data.Y)]), fmt='.',
                             markersize='0.01', color=p.col, linewidth=.75,
                             label=p.title+" : "+str(p.integral())+" fb")

                ax2.step(p.Xl, [(x*xa+eps)/(y+eps) for x, y in zip(p.Y, data.Y)],
                                where='post', color=p.col, linewidth=.25,
                                alpha=0.5, linestyle='-')

        else:
            for j in range(0, len(new_bins)-1):
                if (p.Y[j]*xa < y_minmax[j][0]):
                    y_minmax[j][0] = p.Y[j]*xa
                if (p.Y[j]*xa > y_minmax[j][1]):
                    y_minmax[j][1] = p.Y[j]*xa

    ax1.fill_between(plots[0].Xl, [y[0] for y in y_minmax],
                     [y[1] for y in y_minmax], color=plots[0].col,
                     step="post", alpha=0.5, linewidth=0)
    ax1.fill_between([plots[0].Xl[-1], plots[0].Xh[-1]], y_minmax[-1][0],
                     y_minmax[-1][1], color=plots[0].col, step="pre",
                     alpha=0.5, linewidth=0)
    if (ax2 != 0):
        ax2.fill_between(plots[0].Xl, [x[0]/(y+eps) for x, y
                                       in zip(y_minmax, data.Y[:len(p.Y)])],
                         [x[1]/(y+eps) for x, y
                          in zip(y_minmax, data.Y[:len(p.Y)])],
                         color=plots[0].col, step="post", alpha=0.5,
                         linewidth=0)

        ax2.fill_between([plots[0].Xl[-1], plots[0].Xh[-1]],
                         y_minmax[-1][0]/(data.Y[-1]+eps),
                         y_minmax[-1][1]/(data.Y[-1]+eps),
                         color=plots[0].col, step="post", alpha=0.5,
                         linewidth=0)

    return ax1, ax2
