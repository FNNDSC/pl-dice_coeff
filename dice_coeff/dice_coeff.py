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
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

sys.path.append(os.path.dirname(__file__))

# import the Chris app superclass
from chrisapp.base import ChrisApp


Gstr_title = """

Generate a title from 
http://patorjk.com/software/taag/#p=display&f=Doom&t=dice_coeff

"""

Gstr_synopsis = """


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
	    [--prediction <predictions dir>]                            \\
            [--ground_truth <ground truth dir>]                         \\
            <inputDir>                                                  \\
            <outputDir>                                                  

    BRIEF EXAMPLE

        * Bare bones execution

            mkdir in out && chmod 777 out
            python dice_coeff.py --prediction pred --ground_truth gt   \\
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

        [--prediction <predictions directory>]
        Required : The name of the folder where predictions are stored.

        [--ground_truth <ground truth directory>]
        Required : The name of the directory where ground truth are stored.

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
        self.add_argument('--prediction',dest='pred',type=str,optional=False,
                          help='Prediction directory name')
        self.add_argument('--ground_truth',dest='gt',type=str,optional=False,
                          help='Ground truth directory name')

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
        print(Gstr_synopsis)
   
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

    # For some reasons, python default sorting doesn't perform alpha numeric sorting
    # The below method performs alpha numeric sorting
    # (abc12, abc123,abc20, abc203) => (abc12,abc20,abc123,abc203)
    def sorted_alphanumeric(data):
        convert=lambda text: int(text) if text.isdigit() else text.lower()
        alphanum_key = lambda key :[convert(c) for c in re.split ('([0-9]+)',key)]
        return sorted(data,key=alphanum_key)

    def plot_accuracy(self,options):
        pred_dir=os.path.join(options.inputdir,options.pred)+'/'
        ground_truth_dir=os.path.join(options.inputdir,options.gt)+'/'
        img_len=len(os.listdir(pred_dir))
        x=np.zeros(img_len,dtype='float32')
        y=np.zeros(img_len,dtype='float32')
        nz_counter=0
        total=0.0
        avg_accuracy=0.0

        for i in range(252):
            pred_files=self.sorted_alphanumeric(os.listdir(pred_dir))
            for pf in pred_files:
                if not pf.endswith('.png'):
                    pred_files.remove(pf)
            gt_files=self.sorted_alphanumeric(os.listdir(ground_truth_dir))
            for gtf in gt_files:
                if not gtf.endswith('.png'):
                    gt_files.remove(gtf)
            img_X=imread(ground_truth_dir+gt_files[i],as_gray=True)
            img_y=imread(pred_dir+pred_files[i],as_gray=True)
            x[i]=i
            y[i]=self.dice_coeff(img_X,img_y)
            # calculate non zero avg
            if y[i]!=0.0:
                total=total+y[i]
                nz_counter=nz_counter+1
        fig= plt.figure()
        plt.title('Accuracy of the U-net model on all slices')
        plt.plot(x,y,'g--')
        plt.xlabel('no. of image slices')
        plt.ylabel('Dice Coefficient')
        plt.grid(True)
    
    
        avg=total/nz_counter
        print ("Average accuracy :" + str(avg) )
        fig.savefig(options.outputdir+"/output.png")
        


# ENTRYPOINT
if __name__ == "__main__":
    chris_app = Dice_coeff()
    chris_app.launch()
