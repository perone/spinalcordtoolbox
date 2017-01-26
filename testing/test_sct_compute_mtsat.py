#!/usr/bin/env python
#########################################################################################
#
# Test function sct_compute_mtsat
#
# ---------------------------------------------------------------------------------------
# Copyright (c) 2014 Polytechnique Montreal <www.neuro.polymtl.ca>
# Author(s): Simon Levy
#
# About the license: see the file LICENSE.TXT
#########################################################################################

import commands


def test(data_path):

    range_mtsat = [0.002, 0.003]
    range_t1 = [0.950, 1.150]
    output = ''
    status = 0

    # parameters
    folder_data = 'mtsat/'
    file_data = ['mt0_reg_slicereg_goldstandard.nii.gz', 'mt1.nii.gz', 'mt1_seg.nii.gz', 'seg.nii.gz']
    flipAngles = ['5', '20', '10']
    TRs = ['35', '35', '35']
    output_fname = ['mtsat.nii.gz', 't1.nii.gz']

    # define command
    cmd = 'sct_compute_mtsat -i ' + data_path + folder_data + file_data[0] \
          + ',' + data_path + folder_data + file_data[1] \
          + ',' + data_path + folder_data + file_data[2] \
          + ' -FA ' + flipAngles[0] + ',' + flipAngles[1] + ',' + flipAngles[2] \
          + ' -TR ' + TRs[0] + ',' + TRs[1] + ',' + TRs[2] \
          + ' -o ' + output_fname[0] + ',' + output_fname[1]

    s, o = commands.getstatusoutput(cmd)
    status += s
    output += o

    # if command ran without error, test integrity
    if status == 0:
        # compute mtr within mask
        from sct_average_data_within_mask import average_within_mask
        mtsat_mean, mtsat_std = average_within_mask('mtsat.nii.gz', data_path+folder_data+file_data[3], verbose=0)
        t1_mean, t1_std = average_within_mask('t1.nii.gz', data_path+folder_data+file_data[3], verbose=0)
        if not (mtsat_mean > range_mtsat[0] and mtsat_mean < range_mtsat[1]):
            status = 99
            output += '\nMean MTsat = '+str(mtsat_mean)+'\nAuthorized range: '+str(range_mtsat)
        if not (t1_mean > range_t1[0] and t1_mean < range_t1[1]):
            status = 99
            output += '\nMean T1 = '+str(t1_mean)+'\nAuthorized range: '+str(range_t1)

    return status, output


if __name__ == "__main__":
    # call main function
    test()
