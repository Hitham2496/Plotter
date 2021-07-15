#            Plotter

A python module for producing publication-presentable plots from the data files
produced with ```rivet-mkhtml``` with compatibility to plot scale variation
uncertainty bands, histograms, stacked plots, relative impact distributions (ratio
plots) with ample room for customisation.

##  Usage

  * Download the source code from this repository and copy the ```dat``` files to be
  plotted into the same directory.
  * Create a runcard based on ```Run.py``` replacing the names of the ```dat``` files
  with those to be plotted, customising the output as desired.
  * Run the runcard, the plots will be (by default) produced as pdf files in a subdirectory
  named ```plots```.

You may add the directory containing```Plotter``` to your ```$PYTHONPATH``` for more flexibility

This program requires python>3 and has been tested with python3.7. No local
installation of ```rivet``` is required. The python packages required are:
  
  * numPy
  * matplotlib
  * itertools
