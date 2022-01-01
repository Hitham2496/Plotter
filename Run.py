#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Plotting commands
@author: hhassan
"""
import Plotter

def main():
    Plotter.outputDirectory = 'Plots/'
    Plotter.dataStr = r'Data'
    Plotter.legendStr = r'$pp\rightarrow jj$ at $\sqrt{s}= 7$ TeV'+'\n'+r'$\overline{k_T}$ jets, $R=0.4$, $p_{T} > 60$ GeV, $y_j<2.8$'

    Plotter.plot_single([Plotter.unpack("h_dy_jet12_2j.dat")], r"$\Delta y_{12}$", r"$d\sigma_{W+\geq 2j}/d \Delta y_{12}$", "h_dy_jet12_2j", (1.,0.2), (7e-4,80), [0])

if __name__ == '__main__':
  main()
