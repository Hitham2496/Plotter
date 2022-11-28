#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Plotting commands
@author: hhassan
"""
from Plotter import plotter

def main():
    LEGSTR = r'$pp\rightarrow Wjj at $\sqrt{s}=13$ TeV'+'\n'+r'anti-$k_\perp$ jets, $R=0.4$, $<k_{\perp} > 60$ GeV, $y_j<2.8$'
    LEGPOS = 'lower left'
    plot_routine = plotter(LEGPOS, LEGSTR)
    plot_routine.plot_single(Plotter.unpack("h_dy_jet12_2j.dat"), r"$\Delta y_{12}$", r"$d\sigma_{W+\geq 2j}/d \Delta y_{12}$",
                                            "h_dy_jet12_2j", (1.,0.2), (7e-4,80), [0]))

if __name__ == '__main__':
    main()
