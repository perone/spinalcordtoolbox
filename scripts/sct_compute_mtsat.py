#!/usr/bin/env python
#########################################################################################
#
# Compute MT saturation map from a PD-weigthed, a T1-weighted and a MT-weighted FLASH images
# according to Helms et al., MRM, 60:1396?1407 (2008) and equation erratum in MRM, 64:1856 (2010).
#
# ---------------------------------------------------------------------------------------
# Copyright (c) 2014 Polytechnique Montreal <www.neuro.polymtl.ca>
# Authors: Simon Levy
# Modified: 2017-01-16
#
# About the license: see the file LICENSE.TXT
#########################################################################################

from msct_image import Image
import numpy as np
import math
import sys
import sct_utils as sct
from msct_parser import Parser

class Param:
    ## The constructor
    def __init__(self):
        self.verbose = 1

# main
#=======================================================================================================================
def main():

    # Initialization
    verbose = param.verbose
    inputT1_fname = None

    # Parse input parameters
    parser = get_parser()
    arguments = parser.parse(sys.argv[1:])
    data_fname = arguments['-i']
    flipAngles = arguments['-FA']
    TRs = arguments['-TR']
    outputs_fname = arguments['-o']
    if "-inputT1" in arguments:
        inputT1_fname = arguments["-inputT1"]

    # Load data
    PDw = Image(data_fname[0])
    T1w = Image(data_fname[1])
    MTw = Image(data_fname[2])

    # Load TRs in seconds
    TR_PD = 0.001*float(TRs[0])
    TR_T1 = 0.001*float(TRs[1])
    TR_MT = 0.001*float(TRs[2])

    # Convert flip angles into radians
    alpha_PD = math.radians(float(flipAngles[0]))
    alpha_T1 = math.radians(float(flipAngles[1]))
    alpha_MT = math.radians(float(flipAngles[2]))

    # check if a T1 map was given in input; if not, compute it
    if inputT1_fname:
        inputT1 = Image(inputT1_fname)
        R1_data = 1. / inputT1.data
        sct.printv('T1 map given in input loaded.', verbose, 'info')
    else:
        # compute R1
        R1_data = 0.5 * np.divide((alpha_T1 / TR_T1) * T1w.data - (alpha_PD / TR_PD) * PDw.data, PDw.data / alpha_PD - T1w.data / alpha_T1)
        sct.printv('T1 map computed.', verbose, 'info')

    sct.printv('Compute MTsat...', verbose, 'info')
    # Compute A
    A_data = (TR_PD * alpha_T1 / alpha_PD - TR_T1 * alpha_PD / alpha_T1) * np.divide(np.multiply(PDw.data, T1w.data), TR_PD * alpha_T1 * T1w.data - TR_T1 * alpha_PD * PDw.data)

    # Compute MTsat
    MTsat = PDw
    MTsat.data = TR_MT * np.multiply((alpha_MT * np.divide(A_data, MTw.data) - 1), R1_data) - (alpha_MT ^ 2) / 2.

    # Output MTsat and T1 maps
    MTsat.setFileName(outputs_fname[0])
    MTsat.save()
    T1 = PDw
    T1.data = 1. / R1_data
    T1.setFileName(outputs_fname[1])
    T1.save()

    # To view results
    sct.printv('\nDone! To view results, type:', verbose)
    sct.printv('fslview '+MTsat.FileName+' '+T1.FileName+' &\n', verbose, 'info')


# ==========================================================================================
def get_parser():
    # Initialize the parser
    parser = Parser(__file__)
    parser.usage.set_description('Compute magnetization transfer ratio (MTR). Output is given in percentage.')
    parser.add_option(name="-i",
                      type_value="[[','],'str']",
                      description="IN THIS ORDER, the PD-weighted image, the T1-weighted image and the MT-weigthed image to compute the MTsat map.",
                      mandatory=True,
                      example='PDw.nii.gz,T1w.nii.gz,MTw.nii.gz')
    parser.add_option(name="-FA",
                      type_value="[[','],'str']",
                      description="Flip angles used for the PD-weighted image, the T1-weighted image and the MT-weigthed IN THE SAME ORDER AS FLAG -i.",
                      mandatory=True,
                      example='5,20,15')
    parser.add_option(name="-TR",
                      type_value="[[','],'str']",
                      description="Repetition times (TR) in milliseconds used for the PD-weighted image, the T1-weighted image and the MT-weigthed IN THE SAME ORDER AS FLAG -i.",
                      mandatory=True,
                      example='25,11,25')
    parser.add_option(name="-o",
                      type_value="[[','],'str']",
                      description="Output file names for the MTsat and T1 maps.",
                      mandatory=False,
                      default_value="MTsat.nii.gz,T1.nii.gz",
                      example="MTsat.nii.gz,T1.nii.gz")
    parser.add_option(name="-inputT1",
                      type_value="file",
                      description='File name of a previously computed T1 map to input in the calculation of the MTsat.',
                      mandatory=False)

    return parser


#=======================================================================================================================
# Start program
#=======================================================================================================================
if __name__ == "__main__":
    param = Param()
    main()
