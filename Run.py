#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Plotting commands
@author: hhassan
"""
import plotter

def main():

    Plotter.plot_single(Plotter.unpack("h_dy_jet12_2j.dat"), "h_dy_jet12_2j", (1.,0.2), (7e-4,80), [0],
                                       r"$\Delta y_{12}$", r"$d\sigma_{W+\geq 2j}/d \Delta y_{12}$",
                                       r"$pp \rightarrow Wjj$ at $\sqrt{s}=$13~TeV\par anti-$k_\perp$ jets, $R=$~0.4, $k_\perp>$~60 GeV, $y_j<$~2.8")

if __name__ == '__main__':
  main()
