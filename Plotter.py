#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Plotting routine for .dat files created by rivet-mkhtml

@author: hhassan
"""

from random import * # random numbers
import os, subprocess # to check and create directories
import math # python math
import numpy as np # numerical python
import scipy # scientific python
from scipy import optimize # for numerical solution of equations
from matplotlib import pyplot as plt # plotting
import matplotlib.gridspec as gridspec # more plotting
from matplotlib.ticker import (MultipleLocator, LogLocator) # even more plotting
from matplotlib.lines import Line2D # yet more plotting
import argparse
import copy
from itertools import groupby

outputDirectory = 'plots'

parser = argparse.ArgumentParser(description="Usage: ./Plotter.py -f [INPUT .dat FILES] [options]")
parser.add_argument('--files', '-f', nargs="+", type=str, default="Rivet.yoda")
parser.add_argument('--runcard', '-r', type=str, default="Runcard.py")
parser.add_argument('--output', '-o', default=outputDirectory)
parser.add_argument('--debug', '-d')

args = parser.parse_args()

eps = 1e-20

if not args.files:   # if filename is not given
    parser.error('file(s) not given')

# set command line arguments
debug = args.debug
outputDirectory = args.output + '/'

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

    @property 
    def n_plots(self):
        return len(self.plots)

class dataset:

    def __init__(self, t, xl, xh, y, **kwargs):
        allowed_keys = ['col', 'errs', 'errsM', 'errsP', 'norm']
        self.title = t
        self.Xl = xl
        self.Xh = xh
        self.Y = y
        self.col = 'red'
        self.errs = 0
        self.norm = 1.
        self.__dict__.update((k, v)
        for k, v in kwargs.items() if k in allowed_keys)

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
                plot_list.append(np.array([[str(x) for x in d.split('\n')] for d in g if len(d.strip())]))

    #for j in range(0, len(plot_list)):
    #    print(plot_list[j],'\n\n\n')
    for k in range(0,len(plot_list[0])-1):
        arr = np.array([[str(x) for x in plot_list[0][k][0].split('=')]])
        ll.append(arr[0,0])
        ld.append(arr[0,1])

    ti = ld[ll.index('Title')]
    xLa = ld[ll.index('XLabel')]
    yLa = ld[ll.index('YLabel')]
    logy = int(ld[int(ll.index('LogY'))])
    r = ld[int(ll.index('RatioPlot'))]

    p = plot_env(ti, xLa, yLa, logY=logy, ratio=r)
    p.clearPlots()
    for l in range(1, len(plot_list)):
        TEMP=[]
        arr_t=[]
        arr_p=[]
        arr_d=[]
        xl=[]
        xh=[]
        y=[]
        errsm=[]
        errsp=[]
        kstar = 0
        for k in range(0,len(plot_list[l])-1):
            TEMP.append(np.array([[str(x) for x in plot_list[l][k][0].split()]]))
            if(TEMP[k][0][0]=='#'):
                # k* is the index at which the plot data begins
                kstar = k
        z = len(plot_list[l])-kstar-2 # (length of the plot info list - 1) - (k* - 1)
        res = TEMP[-z:]
        for j in range(0,kstar):
            arr_t.append(np.array([str(x) for x in TEMP[j][0][0].split('=')]))
            arr_p.append(arr_t[j][0])
            arr_d.append(arr_t[j][1])
        #print(arr_d,"\n\n")#,res,"\n\n")

        t = arr_d[arr_p.index('Title')]
        c = 'Black'
        if (t != 'Data'):
            c = arr_d[arr_p.index('LineColor')]
        for n in range(0, len(res)):
            xl.append(float(res[n][0][0]))
            xh.append(float(res[n][0][1]))
            y.append(float(res[n][0][2]))
            errsm.append(float(res[n][0][3]))
            errsp.append(float(res[n][0][4]))
        p.addPlot(dataset(t, xl, xh, y, col=c, errs=0, errsM=errsm, errsP=errsp))
    return p

def sort_env(p_env, sv_list):
    """Sorts the data in a plot_env if scale bands are present
       currently only implemented for central and two variations"""
    scale_vars = []
    rest = []
    j=0
    while j < len(p_env.plots):
        if (j in sv_list):
            scale_vars.append([p_env.plots[j], p_env.plots[j+1], p_env.plots[j+2]])
            j+=3
        else:
            rest.append(p_env.plots[j])
            j+=1
    return scale_vars, rest

def plot_single(p_env_l, xLab, yLab, Title, xTup, yTup):
    """Plots a single histogram"""
    plt.rc('text', usetex=True)
    plt.rc('font', family='serif')
    #graph, ax1 = plt.subplots(nrows=1, ncols=1, sharex=True, sharey=False, figsize=(4,4))
    graph, (ax1, ax2) = plt.subplots(nrows=2, ncols=1, sharex=True, sharey=False, gridspec_kw = {'height_ratios':[2.5, 1]}, figsize=(6,6))
    plt.subplots_adjust(hspace=0)

    bins = p_env_l[0].plots[0].Xl[:]
    bins.append(p_env_l[0].plots[0].Xh[-1])
    ax1.set_ylabel(yLab)
    #ax1.set_xlabel(xLab)
    ax2.set_xlabel(xLab)
    ax2.set_ylim(0.4, 1.7)
    #ax2.set_ylabel("Ratio to Data")
    ax1.set_xlim(bins[0], bins[-1])
    ax1.set_ylim(yTup[0], yTup[1])
    #if(p_env_l[0].logY == 1):
    #ax1.set_yscale('log')

    ax1.yaxis.set_ticks_position('both')
    ax1.xaxis.set_ticks_position('both')
    ax1.xaxis.set_major_locator(MultipleLocator(xTup[0]))
    ax1.xaxis.set_minor_locator(MultipleLocator(xTup[1]))
    #ax1.yaxis.set_minor_locator(MultipleLocator(0.5))
    #ax1.yaxis.set_minor_locator(MultipleLocator(0.1))
    #ax1.yaxis.set_minor_locator(MultipleLocator(0.2))

    ax2.yaxis.set_minor_locator(MultipleLocator(0.1))
    ax2.yaxis.set_major_locator(MultipleLocator(0.2))
    ax2.yaxis.set_ticks_position('both')
    ax2.xaxis.set_ticks_position('both')

    p_env=p_env_l[0]
    pts=[]
    p_env = p_env_l[0]
    new_bins = p_env.plots[0].Xh[:]
    new_bins.insert(0, p_env.plots[0].Xl[0])
    x_errors=[]
    for j in range(0, len(new_bins)-1):
        pts.append((new_bins[j+1]+new_bins[j])/2.)
        x_errors.append((new_bins[j+1]-new_bins[j])/2.)
  #  # plot data
  #  for p in p_env.plots:
  #      if p.title == "Data":
  #          ax1.errorbar(pts, [z for z in p.Y], xerr = x_errors, yerr = (p.errsM, p.errsP), fmt = '.', color = "black", markersize='5', linewidth = .75, label = r'Data')
  #          #ax2.errorbar(pts, [(x+eps)/(y+eps) for x, y in zip(p.Y, p.Y)], xerr = x_errors, yerr = ([(x+eps)/(y+eps) for x, y in zip(p.errsM, p.Y)],[(x+eps)/(y+eps) for x, y in zip(p.errsP, p.Y)]),
  #          #        fmt = '.', markersize='5', color = "black", linewidth = .75, label = r'ATLAS: arXiv/1407.5756 $\sqrt{s}=7$ TeV')
  #      else:

    sv,rest = sort_env(p_env_l[0],[0])
    for pl in sv:
        ax1, ax2 = plot_scale_bands(pl, ax1, ax2, pts, x_errors, data=sv[0][0])
    for p in rest:
         ax1.step(p.Xl, [z for z in p.Y], where = 'post', color=p.col, linewidth = .75, label=p.title, linestyle='-')
         ax2.step(p.Xl, [x/(y+eps) for x, y in zip(p.Y, p_env.plots[0].Y[:len(p.Y)])], where = 'post', color=p.col, linewidth = .75, label=p.title)
         #if(p.errs == 1):
         ax1.errorbar(pts, [z for z in p.Y], xerr = x_errors, yerr = (p.errsM, p.errsP), color=p.col, fmt = '', linewidth = .75, ls = 'none')
         ax2.errorbar(pts, [x/(y+eps) for x, y in zip(p.Y, p_env.plots[0].Y[:len(p.Y)])], xerr = x_errors,
              yerr = ([x/(y+eps) for x, y in zip(p.errsM, p.Y)],[x/(y+eps) for x, y in zip(p.errsP, p.Y)]), color=p.col, fmt = '', linewidth = 1., ls = 'none')


    #custom_lines = [Line2D([0], [0], color=p.col, linestyle='-', lw=.75) for p in p_env.plots] # with data: p_env.plots[1:]]
    #legend1 = plt.legend(custom_lines, [p.title for p in p_env.plots], loc='upper left', frameon = False)  # with data: [p.title for p in p_env.plots[1:]], loc='upper right', frameon = False)
    #ax1.add_artist(legend1)
    ax1.legend(loc='upper left', title=r'anti-$k_T$ jets, $R=0.4$, $p_{\perp} > 60$ GeV, $\sqrt{s}= 13$ TeV, $y_j<4.5$', frameon=False)
    ax1.get_legend()._legend_box.align = "left"
    plt.savefig(outputDirectory+Title+".pdf", bbox_inches="tight")

   # plt.show()

def plot_scale_bands(plots, ax1, ax2, pts, x_errors, **kwargs):
    """Plots a single histogram with scale variation bands"""
    pts=[]
    new_bins = plots[0].Xh[:]
    new_bins.insert(0, plots[0].Xl[0])
    x_errors=[]
    y_minmax=[]
    data=plots[0]
    for key, value in kwargs.items():
        if key == 'data':
            data=value
    for j in range(0, len(new_bins)-1):
        pts.append((new_bins[j+1]+new_bins[j])/2.)
        x_errors.append((new_bins[j+1]-new_bins[j])/2.)
    # plot data
    for p in plots:
        if plots.index(p) == 0:
            for j in range(0, len(new_bins)-1):
                y_minmax.append([p.Y[j], p.Y[j]])
            ax1.errorbar(pts, [z for z in p.Y], xerr = x_errors, yerr = (p.errsM, p.errsP), fmt = '.', color = p.col, markersize='0.01', linewidth = .75, label = p.title)
            ax2.errorbar(pts, [(x+eps)/(y+eps) for x, y in zip(p.Y, data.Y)], xerr = x_errors, yerr = ([(x+eps)/(y+eps) for x, y in zip(p.errsM, data.Y)], [(x+eps)/(y+eps) for x, y in zip(p.errsP, data.Y)]),
                    fmt = '.', markersize='0.01', color = p.col, linewidth = .75, label = p.title)
            ax1.step(p.Xl, [z for z in p.Y], where = 'post', color=p.col, linewidth = .25, alpha=0.5, linestyle='-')
            ax2.step(p.Xl, [(x+eps)/(y+eps) for x, y in zip(p.Y, data.Y)], where = 'post', color=p.col, linewidth = .25, alpha=0.5, linestyle='-')
        else:
            for j in range(0, len(new_bins)-1):
                if (p.Y[j] < y_minmax[j][0]):
                    y_minmax[j][0] = p.Y[j]
                if (p.Y[j] > y_minmax[j][1]):
                    y_minmax[j][1] = p.Y[j]

    ax1.fill_between(plots[0].Xl, [y[0] for y in y_minmax], [y[1] for y in y_minmax], color=plots[0].col, step="post", alpha=0.5, linewidth=0)
    ax1.fill_between([plots[0].Xl[-1],plots[0].Xh[-1]], y_minmax[-1][0],  y_minmax[-1][1], color=plots[0].col, step="pre", alpha=0.5, linewidth=0)
    ax2.fill_between(plots[0].Xl, [x[0]/(y+eps) for x, y in zip(y_minmax, data.Y[:len(p.Y)])], [x[1]/(y+eps) for x, y in zip(y_minmax, data.Y[:len(p.Y)])],
            color=plots[0].col, step="post", alpha=0.5, linewidth=0)
    ax2.fill_between([plots[0].Xl[-1],plots[0].Xh[-1]], y_minmax[-1][0]/(data.Y[-1]+eps), y_minmax[-1][1]/(data.Y[-1]+eps), color=plots[0].col,
            step="post", alpha=0.5, linewidth=0)

    #custom_lines = [Line2D([0], [0], color=p.col, linestyle='-', lw=.75) for p in p_env.plots] # with data: p_env.plots[1:]]
    #legend1 = plt.legend(custom_lines, [p.title for p in p_env.plots], loc='upper left', frameon = False)  # with data: [p.title for p in p_env.plots[1:]], loc='upper right', frameon = False)
    #ax1.add_artist(legend1)
    #ax1.legend(loc='upper left', title=r'anti-$k_T$ jets, $R=0.4$, $p_{\perp} > 30$ GeV', frameon=False)
    #ax1.get_legend()._legend_box.align = "left"
    #plt.savefig(outputDirectory+Title+".pdf", bbox_inches="tight")
    return ax1, ax2


def plot_stacked(p_env_l, step, xLab, yLab, Title, name_marks, bin_marks, bin_corr, xTup):
    """Plots histograms on top of each other, separated by a user-input step"""
    plt.rc('text', usetex=True)
    plt.rc('font', family='serif')
    graph, ax1 = plt.subplots(nrows=1, ncols=1, sharex=True, sharey=False, figsize=(6,9))

    bins = p_env_l[0].plots[0].Xl
    bins.append(p_env_l[0].plots[0].Xh[-1])
    ax1.set_ylabel(yLab)
    ax1.set_xlabel(xLab)
    ax1.set_xlim(bins[0], bins[-1])
    ax1.set_ylim(1e-8,1e9)
    #if(p_env_l[0].logY == 1):
    ax1.set_yscale('log')
    
    ax1.yaxis.set_ticks_position('both')
    ax1.xaxis.set_ticks_position('both')
    ax1.xaxis.set_major_locator(MultipleLocator(xTup[0]))
    ax1.xaxis.set_minor_locator(MultipleLocator(xTup[1]))
    #ax1.yaxis.set_minor_locator(MultipleLocator(0.2))
    ax1.yaxis.set_minor_locator(LogLocator(base=10, subs=(10,)))

    for q in range(0, len(p_env_l)):
        pts=[]
        p_env = p_env_l[q]
        new_bins = p_env.plots[0].Xh
        new_bins.insert(0, p_env.plots[0].Xl[0])
        x_errors=[]
        for j in range(0, len(new_bins)-1):
            pts.append((new_bins[j+1]+new_bins[j])/2.)
            x_errors.append((new_bins[j+1]-new_bins[j])/2.)
        # plot data
        for p in p_env.plots:
            if p.title == "Data":
                ax1.errorbar(pts, [z*10**(-q*step) for z in p.Y], xerr = x_errors, yerr = ([E*10**(-q*step) for E in p.errsM], [E*10**(-q*step) for E in p.errsP]),
                        fmt = name_marks[q], color = "black", markersize='5', linewidth = .75, label = r'Data')
            else:
                ax1.step(p.Xl, [z*10**(-q*step) for z in p.Y], where='post', color=p.col, linewidth = .75, label=p.title)
                #if(p.errs == 1):
                ax1.errorbar(pts, [z*10**(-q*step) for z in p.Y], xerr = x_errors, yerr = ([E*10**(-q*step) for E in p.errsM], [E*10**(-q*step) for E in p.errsP]),
                        color=p.col, fmt = '', linewidth = .75, ls = 'none')
    
    custom_lines = [Line2D([0], [0], color=p.col, lw=.75) for p in p_env.plots[1:]]
    legend1 = plt.legend(custom_lines, [p.title for p in p_env.plots[1:]], loc='upper right', frameon = False)
    legend2 = plt.legend((bin_marks), (bin_corr), loc='upper left', frameon = False)
    ax1.add_artist(legend1)
    ax1.add_artist(legend2)
    plt.savefig(outputDirectory+Title+".pdf", bbox_inches="tight")

    #plt.show()


def plot_stacked_ratio(p_env_l, xLab, yLab, Title, name_marks, bin_marks, bin_corr, xTup, phrase):
    """Plots histograms on top of each other, separated by a user-input step"""
    plt.rc('text', usetex=True)
    plt.rc('font', family='serif')
    graph, ax = plt.subplots(nrows=len(p_env_l)+1, ncols=1, sharex=True, sharey=False, figsize=(6,9))

    bins = p_env_l[0].plots[0].Xl
    bins.append(p_env_l[0].plots[0].Xh[-1])
    ax[0].xaxis.set_visible(False)
    ax[0].yaxis.set_visible(False)
    ax[1].set_ylabel(yLab)
    ax[-1].set_xlabel(xLab)
    ax[1].set_xlim(bins[0], bins[-1])
    
    for q in range(0, len(p_env_l)):
        pts=[]
        ax[-q-1].yaxis.set_ticks_position('both')
        ax[-q-1].set_ylim(0.,2.)
        ax[-q-1].xaxis.set_ticks_position('both')
        ax[-q-1].xaxis.set_major_locator(MultipleLocator(xTup[0]))
        ax[-q-1].xaxis.set_minor_locator(MultipleLocator(xTup[1]))
        ax[-q-1].yaxis.set_minor_locator(MultipleLocator(0.2))
        p_env = p_env_l[q]
        new_bins = p_env.plots[0].Xh
        new_bins.insert(0, p_env.plots[0].Xl[0])
        x_errors=[]
        for j in range(0, len(new_bins)-1):
            pts.append((new_bins[j+1]+new_bins[j])/2.)
            x_errors.append((new_bins[j+1]-new_bins[j])/2.)
        # plot data
        for p in p_env.plots:
            if p.title == "Data":
                ax[q+1].errorbar(pts, [(x+eps)/(y+eps) for x, y in zip(p.Y, p.Y)], xerr = x_errors, yerr = ([(x+eps)/(y+eps) for x, y in zip(p.errsM, p.Y)],[(x+eps)/(y+eps) for x, y in zip(p.errsP, p.Y)]),
                        fmt = name_marks[q], markersize='5', color = "black", linewidth = .75, label = r'Data')
            else:
                 ax[q+1].step(p.Xl, [x/(y+eps) for x, y in zip(p.Y, p_env_l[q].plots[0].Y[:len(p.Y)])], where = 'post', color=p.col, linewidth = .75, label=p.title)
            #    if(p.errs == 1):
                 ax[q+1].errorbar(pts, [x/(y+eps) for x, y in zip(p.Y, p_env_l[q].plots[0].Y[:len(p.Y)])], xerr = x_errors,
                         yerr = ([x/(y+eps) for x, y in zip(p.errsM, p.Y)],[x/(y+eps) for x, y in zip(p.errsP, p.Y)]), color=p.col, fmt = '', linewidth = 1., ls = 'none')

    custom_lines = [Line2D([0], [0], color=p.col, lw=.75) for p in p_env.plots[1:]]
    ax[0].legend(custom_lines, [p.title for p in p_env.plots[1:]], loc = "upper right", frameon = False)
    ax[0].text(1.5*bins[0], .8, phrase, horizontalalignment='left', verticalalignment='top')
    plt.savefig(outputDirectory+Title+".pdf", bbox_inches="tight")

    #plt.show()
