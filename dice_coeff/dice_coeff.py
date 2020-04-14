#!/usr/bin/env python                                            
#
# dice_coeff ds ChRIS plugin app
#
# (c) 2016-2019 Fetal-Neonatal Neuroimaging & Developmental Science Center
#                   Boston Children's Hospital
#
#              http://childrenshospital.org/FNNDSC/
#                        dev@babyMRI.org
#


import os
import sys
import numpy as np
from skimage.io import imread
import SimpleITK as sitk
import math
import matplotlib.pyplot as plt

sys.path.append(os.path.dirname(__file__))

# import the Chris app superclass
from chrisapp.base import ChrisApp


Gstr_title = """

Generate a title from 
http://patorjk.com/software/taag/#p=display&f=Doom&t=dice_coeff

"""

Gstr_synopsis = """

(Edit this in-line help for app specifics. At a minimum, the 
flags below are supported -- in the case of DS apps, both
positional arguments <inputDir> and <outputDir>; for FS apps
only <outputDir> -- and similarly for <in> <out> directories
where necessary.)

    NAME

       dice_coeff.py 

    SYNOPSIS

        python dice_coeff.py                                         \\
            [-h] [--help]                                               \\
            [--json]                                                    \\
            [--man]                                                     \\
            [--meta]                                                    \\
            [--savejson <DIR>]                                          \\
            [-v <level>] [--verbosity <level>]                          \\
            [--version]                                                 \\
            <inputDir>                                                  \\
            <outputDir> 

    BRIEF EXAMPLE

        * Bare bones execution

            mkdir in out && chmod 777 out
            python dice_coeff.py   \\
                                in    out

    DESCRIPTION

        `dice_coeff.py` ...

    ARGS

        [-h] [--help]
        If specified, show help message and exit.
        
        [--json]
        If specified, show json representation of app and exit.
        
        [--man]
        If specified, print (this) man page and exit.

        [--meta]
        If specified, print plugin meta data and exit.
        
        [--savejson <DIR>] 
        If specified, save json representation file to DIR and exit. 
        
        [-v <level>] [--verbosity <level>]
        Verbosity level for app. Not used currently.
        
        [--version]
        If specified, print version number and exit. 

"""


class Dice_coeff(ChrisApp):
    """
    An app to calculate model testing accuracy by dice coefficient.
    """
    AUTHORS                 = 'Sandip Samal (sandip.samal@childrens.harvard.edu)'
    SELFPATH                = os.path.dirname(os.path.abspath(__file__))
    SELFEXEC                = os.path.basename(__file__)
    EXECSHELL               = 'python3'
    TITLE                   = 'Model accuracy app'
    CATEGORY                = ''
    TYPE                    = 'ds'
    DESCRIPTION             = 'An app to calculate model testing accuracy by dice coefficient'
    DOCUMENTATION           = 'http://wiki'
    VERSION                 = '0.1'
    ICON                    = '' # url of an icon image
    LICENSE                 = 'Opensource (MIT)'
    MAX_NUMBER_OF_WORKERS   = 1  # Override with integer value
    MIN_NUMBER_OF_WORKERS   = 1  # Override with integer value
    MAX_CPU_LIMIT           = '' # Override with millicore value as string, e.g. '2000m'
    MIN_CPU_LIMIT           = '' # Override with millicore value as string, e.g. '2000m'
    MAX_MEMORY_LIMIT        = '' # Override with string, e.g. '1Gi', '2000Mi'
    MIN_MEMORY_LIMIT        = '' # Override with string, e.g. '1Gi', '2000Mi'
    MIN_GPU_LIMIT           = 0  # Override with the minimum number of GPUs, as an integer, for your plugin
    MAX_GPU_LIMIT           = 0  # Override with the maximum number of GPUs, as an integer, for your plugin

    # Use this dictionary structure to provide key-value output descriptive information
    # that may be useful for the next downstream plugin. For example:
    #
    # {
    #   "finalOutputFile":  "final/file.out",
    #   "viewer":           "genericTextViewer",
    # }
    #
    # The above dictionary is saved when plugin is called with a ``--saveoutputmeta``
    # flag. Note also that all file paths are relative to the system specified
    # output directory.
    OUTPUT_META_DICT = {}

    def define_parameters(self):
        """
        Define the CLI arguments accepted by this plugin app.
        Use self.add_argument to specify a new app argument.
        """

    def run(self, options):
        """
        Define the code to be run by this plugin app.
        """
        print(Gstr_title)
        print('Version: %s' % self.get_version())

        print("-------------PLOTTING GRAPH----------------")
        self.plot_accuracy(options)
        print("-------------GRAPH SAVED ------------------")


    def show_man_page(self):
        """
        Print the app's man page.
        """
        print("-------------PLOTTING GRAPH----------------")
        plot_accuracy(options)
        print("-------------GRAPH SAVED ------------------")
   
    def dice_coeff(self,ground_truth,pred):
        pred=pred.astype('float32')
        ground_truth=ground_truth.astype('float32')
        labelPred=sitk.GetImageFromArray(pred, isVector=False)
        labelTrue=sitk.GetImageFromArray(ground_truth, isVector=False)
        overlap_measures_filter=sitk.LabelOverlapMeasuresImageFilter()
        overlap_measures_filter.Execute(labelTrue>0.5, labelPred>0.5)
        result=overlap_measures_filter.GetDiceCoefficient()

        if result==math.inf:
            return 0.0
        else:
            return result

    def plot_accuracy(self,options):
        pred_dir=options.outputdir+'/'
        ground_truth_dir=options.inputdir+'/'
        img_len=len(os.listdir(pred_dir))
        x=np.ndarray(img_len,dtype='float32')
        y=np.ndarray(img_len,dtype='float32')

        for i in range(252):
            pred_files=os.listdir(pred_dir)
            gt_files=os.listdir(ground_truth_dir)
            img_X=imread(ground_truth_dir+gt_files[i],as_gray=True)
            img_y=imread(pred_dir+pred_files[i],as_gray=True)
            x[i]=i
            y[i]=self.dice_coeff(img_X,img_y)
  
        plt.plot(x,y)
        plt.show
        plt.savefig(pred_dir+'accuracy_graph.png')
        


# ENTRYPOINT
if __name__ == "__main__":
    chris_app = Dice_coeff()
    chris_app.launch()
