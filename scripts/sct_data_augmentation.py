#!/usr/bin/env python
#########################################################################################
#
# sct_crop_image and crop image wrapper.
#
# ---------------------------------------------------------------------------------------
# Copyright (c) 2014 Polytechnique Montreal <www.neuro.polymtl.ca>
# Authors: Camille Van Assel
#
#
# About the license: see the file LICENSE.TXT
#########################################################################################

import time
from msct_parser import Parser
import sct_utils as sct
import sys
from scipy import ndimage
from msct_image import Image
import os
import numpy as np


# MAIN
# ==========================================================================================

def main():
    # Initialisation
    start_time = time.time()

    # Initialise the parser
    parser = Parser(__file__)
    parser.add_option(name="-i",
                      type_value="file",
                      description="Image source.",
                      mandatory=True,
                      example="src.nii.gz")
    parser.add_option(name="-iseg",
                      type_value="file",
                      description="Segmentation of the image source",
                      example="src_seg.nii.gz")
    parser.add_option(name="-lx",
                      type_value="int",
                      description="size of the image at x",
                      mandatory=True,
                      example="50")
    parser.add_option(name="-ly",
                      type_value="int",
                      description="size of the image at y",
                      mandatory=True,
                      example="50")
    parser.add_option(name="-r",
                      type_value="multiple_choice",
                      description="""Remove temporary files.""",
                      mandatory=False,
                      default_value='1',
                      example=['0', '1'])
    parser.add_option(name="-z",
                      type_value="int",
                      description="if you want to crop only one slice",
                      mandatory=False,
                      example="42")

    # get argument
    arguments = parser.parse(sys.argv[1:])
    fname_src = arguments['-i']
    fname_seg = arguments['-iseg']
    fname_size_x = arguments['-lx']
    fname_size_y = arguments['-ly']
    remove_temp_files = int(arguments['-r'])

    # get the size of the image
    nx, ny, nz, nt, px, py, pz, pt = Image(fname_src).dim
    sct.printv("Taille du pixel: " + str(px)+ "," + str(py) + "," + str(pz))
    # extract path file and extension
    path_src, file_src, ext_src = sct.extract_fname(fname_src)
    path_seg, file_seg, ext_seg = sct.extract_fname(fname_seg)

    # create temporary folder
    sct.printv('\nCreate temporary folder...')
    path_tmp = sct.tmp_create()

    if '-z' in arguments:
        slice_z = arguments['-z']

        # crop the image along the z dimension
        sct.run("sct_crop_image -i " + fname_src + " -dim 2 -start " + str(slice_z) + " -end " + str(slice_z) + " -o " + path_tmp + file_src + "_slice_" + str(slice_z) + ".nii.gz",
            verbose=0)
        sct.run("sct_crop_image -i " + fname_seg + " -dim 2 -start " + str(slice_z) + " -end " + str(slice_z) + " -o " + path_tmp + file_src + "_slice_" + str(slice_z) + "_seg.nii.gz",
            verbose=0)

        # if the slice has a segmentation, crop it along x and y dimension
        from numpy import asarray
        fname_slice = path_tmp + file_src + "_slice_" + str(slice_z) + ".nii.gz"
        fname_slice_seg = path_tmp + file_src + "_slice_" + str(slice_z) + "_seg.nii.gz"
        image_slice_seg = Image(fname_slice_seg)
        data_array_seg = np.squeeze(asarray(image_slice_seg.data))

        if not np.all(data_array_seg == np.zeros((nx, ny))):
            crop_x_y(fname_slice, file_src, path_src, fname_size_x, fname_size_y, nx, ny, path_tmp, slice_z)
        else:
            sct.run('rm -rf ' + fname_slice, verbose=1)
            sct.run('rm -rf ' + fname_slice_seg, verbose=1)

    else:
        # crop the image along the z dimension
        for iter in range(1,nz):
            nbre=str(iter)
            sct.run("sct_crop_image -i " + fname_src + " -dim 2 -start " + nbre + " -end " + nbre + " -o " + path_tmp + file_src + "_slice_" + nbre + ".nii.gz", verbose=0)
            sct.run("sct_crop_image -i " + fname_seg + " -dim 2 -start " + nbre + " -end " + nbre + " -o " + path_tmp + file_src + "_slice_" + nbre + "_seg.nii.gz", verbose=0)

            # crop the lines with zeros
            from numpy import asarray
            fname_slice = path_tmp + file_src + "_slice_" + nbre + ".nii.gz"
            fname_slice_seg = path_tmp + file_src + "_slice_" + nbre + "_seg.nii.gz"
            image_slice_seg = Image(fname_slice_seg)
            data_array_seg = np.squeeze(asarray(image_slice_seg.data))

            if np.all(data_array_seg == np.zeros((nx, ny))):
                sct.run('rm -rf ' + fname_slice, verbose=1)
                sct.run('rm -rf ' + fname_slice_seg, verbose=1)
            else:
                crop_x_y(fname_slice,file_src,path_src, fname_size_x, fname_size_y, nx, ny, path_tmp, nbre)

    # Delete temporary files
    if remove_temp_files:
        sct.printv('\nRemove temporary files...')
        sct.run('rm -rf ' + path_tmp, verbose=1)

    elapsed_time = time.time() - start_time
    sct.printv('\nFinished! Elapsed time: ' + str(int(round(elapsed_time))) + 's')


# crop images along x and y
# ==========================================================================================
def crop_x_y (fname_slice, file_src, path_src, fname_size_x, fname_size_y, nx, ny, path_tmp, nbre):
    import numpy as np
    nbre=str(nbre)

    # find the mass center of each slice
    fname_slice_seg = path_tmp + file_src + "_slice_" + nbre + "_seg.nii.gz"
    image_slice_seg = Image(fname_slice_seg)
    data_array_seg = np.squeeze(np.asarray(image_slice_seg.data))
    cof = ndimage.measurements.center_of_mass(data_array_seg)
    sct.printv('Centre de masse :' + str(cof))

    # crop the image along the x,y direction
    start_x = float(cof[0] - fname_size_x/2)
    start_y = float(cof[1] - fname_size_y/2)
    stop_x = float(cof[0] + fname_size_x/2)
    stop_y = float(cof[1] + fname_size_y/2)

    image_slice = Image(fname_slice)
    data_array = np.squeeze(np.asarray(image_slice.data))

    lx_start = 0
    lx_stop = 0
    ly_start = 0
    change_x_s = False
    change_x_e = False
    change_y_s = False
    change_y_e = False
    nx_s, ny_s, nz_s, nt_s, px_s, py_s, pz_s, pt_s = Image(fname_slice_seg).dim
    if start_x < 1:
        lx_start = int(abs(cof[0] - fname_size_x/2)+1)
        z = np.zeros((lx_start, ny))
        zs = np.zeros((lx_start, ny_s))
        data_array = np.append(z, data_array, axis=0)
        data_array_seg = np.append(zs, data_array_seg, axis=0)
        start_x = 0
        change_x_s = True
    if stop_x >= nx:
        lx_stop = int(cof[0] + fname_size_x/2 - nx + 1)
        z = np.zeros((lx_stop, ny))
        zs = np.zeros((lx_stop, ny_s))
        data_array = np.append(data_array, z, axis=0)
        data_array_seg = np.append(data_array_seg,zs, axis=0)
        stop_x = fname_size_x + start_x
        change_x_e = True
    if start_y < 1:
        ly_start = int(abs(cof[1] - fname_size_y/2)+1)
        z = np.zeros((nx + lx_start + lx_stop, ly_start))
        data_array = np.append(z, data_array, axis=1)
        data_array_seg = np.append(z, data_array_seg, axis=1)
        start_y = 0
        change_y_s = True
    if stop_y >= ny:
        ly_stop = int(cof[1] + fname_size_y/2 - ny +1)
        z = np.zeros((nx + lx_start + lx_stop, ly_stop))
        data_array = np.append(data_array, z, axis=1)
        data_array_seg = np.append(data_array_seg, z, axis=1)
        stop_y = fname_size_y
        change_y_e = True
    if change_x_s & (not change_x_e):
        stop_x = fname_size_x
    if change_y_s & (not change_y_e):
        stop_y = fname_size_y

    sct.printv("Taille de l'image :" + str(np.shape(data_array_seg)))
    fname_slice_out = path_tmp + file_src + "_slice_out_" + nbre + ".nii.gz"
    fname_slice_seg_out = path_tmp + file_src + "_slice_seg_out_" + nbre + ".nii.gz"
    image_slice.setFileName(fname_slice_out)
    image_slice_seg.setFileName(fname_slice_seg_out)
    image_slice.data = data_array
    image_slice_seg.data = data_array_seg
    image_slice.save()
    image_slice_seg.save()

    sct.run("sct_crop_image -i " + fname_slice_out + " -dim 0,1 -start " + str(start_x) + "," + str(start_y) + " -end " + str(stop_x-1) + "," + str(stop_y-1) + " -o " + path_tmp + file_src + "_crop_" + nbre + ".nii.gz", verbose=1)
    sct.run("sct_crop_image -i " + fname_slice_seg_out + " -dim 0,1 -start " + str(start_x) + "," + str(start_y) + " -end " + str(stop_x-1) + "," + str(stop_y-1) + " -o " + path_tmp + file_src + "_crop_seg_" + nbre + ".nii.gz", verbose=1)

    # get the curent folder
    path_out = os.getcwd()

    # out the origine at (0,0)
    im_file_src = Image(path_tmp + file_src + "_crop_" + nbre + ".nii.gz")
    im_file_src.hdr.structarr['qoffset_x'] = im_file_src.hdr.structarr['qoffset_y'] = im_file_src.hdr.structarr['qoffset_z'] = im_file_src.hdr.structarr['srow_x'][-1] = im_file_src.hdr.structarr['srow_y'][-1] = im_file_src.hdr.structarr['srow_z'][-1] = 0
    im_file_src.setFileName(path_src + file_src + "_crop_" + nbre + "_o0.nii.gz")
    im_file_src.save()

    im_file_src = Image(path_tmp + file_src + "_crop_seg_" + nbre + ".nii.gz")
    im_file_src.hdr.structarr['qoffset_x'] = im_file_src.hdr.structarr['qoffset_y'] = im_file_src.hdr.structarr['qoffset_z'] = im_file_src.hdr.structarr['srow_x'][-1] = im_file_src.hdr.structarr['srow_y'][-1] = im_file_src.hdr.structarr['srow_z'][-1] = 0
    im_file_src.setFileName(path_src + file_src + "_crop_seg_" + nbre + "_o0.nii.gz")
    im_file_src.save()

    # Generate output files
    sct.generate_output_file(path_src + file_src + "_crop_" + nbre + "_o0.nii.gz", path_out + "/" + file_src + "_crop_" + nbre + "_o0.nii.gz")
    sct.generate_output_file(path_src + file_src + "_crop_seg_" + nbre + "_o0.nii.gz",path_out + "/"+ file_src + "_crop_seg_" + nbre + "_o0.nii.gz")


# START PROGRAM
# ==========================================================================================
if __name__ == "__main__":
    # call main function
    main()
