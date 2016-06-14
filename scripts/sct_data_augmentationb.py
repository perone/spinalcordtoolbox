#!/usr/bin/env python
#########################################################################################
#
#
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
import sys
import os
import numpy as np
import sct_utils as sct
from msct_image import Image
import random

#MAIN

def main():
    # Initialisation
    start_time = time.time()
    fname_output = ''

    # Initialise the parser
    parser = Parser(__file__)
    parser.add_option(name="-f",
                      type_value="folder",
                      description="folder of input image and segmentation",
                      mandatory=False,
                      example="data_patients")
    parser.add_option(name="-i",
                      type_value="file",
                      description="Image source.",
                      mandatory=False,
                      example="src.nii.gz")
    parser.add_option(name="-d",
                      type_value="file",
                      description="Image destination.",
                      mandatory=False,
                      example="dest.nii.gz")
    parser.add_option(name="-iseg",
                      type_value="file",
                      description="Segmentation source.",
                      mandatory=False,
                      example="src_seg.nii.gz")
    parser.add_option(name="-dseg",
                      type_value="file",
                      description="Segmentation destination.",
                      mandatory=False,
                      example="dest_seg.nii.gz")
    parser.add_option(name="-t",
                      type_value="file",
                      description="image which will be modified",
                      mandatory=False,
                      example="im.nii.gz")
    parser.add_option(name="-tseg",
                      type_value="file",
                      description="Segmentation of the image wich will be modified.",
                      mandatory=False,
                      example="dest_seg.nii.gz")
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
    parser.add_option(name="-nb",
                      type_value="int",
                      description="number of random slice we apply the wrap on",
                      mandatory=True,
                      example="31")
    parser.add_option(name="-nw",
                      type_value="int",
                      description="number of random wrap we generate",
                      mandatory=True,
                      example='15')
    parser.add_option(name="-o",
                      type_value='str',
                      description="nameof the output folder",
                      mandatory=False)
    # get argument
    arguments = parser.parse(sys.argv[1:])
    fname_size_x = arguments['-lx']
    fname_size_y = arguments['-ly']
    nbre_slice = arguments['-nb']
    nbre_wrap = arguments['-nw']

    if '-f' in arguments:
        input_folder = arguments['-f']
        sct.check_folder_exist(str(input_folder), verbose=0)
        folder_path = sct.get_absolute_path(input_folder)
        sct.printv(folder_path)
    if '-o' in arguments:
        output_folder = arguments['-o']
        os.makedirs(str(output_folder))
        output_folder_path = str(output_folder) + "/"
    else:
        output_folder_path = ''

    # choose randomly a source and a destination
    list_data = []
    for root, dirs, files in os.walk(input_folder):
        for file in files:
            if file.endswith('.nii.gz') and file.find('_seg') == -1:
                pos_t2 = file.find('_t2')
                subject_name, end_name = file[0:pos_t2 + 3], file[pos_t2 + 3:]
                file_seg = subject_name + '_seg' + end_name
                list_data.append(file)
                list_data.append(file_seg)

    fname_src = []
    fname_src_seg =[]
    fname_dest = []
    fname_dest_seg = []

    if '-i' in arguments:
        fname_src[0] = arguments['-i']
        fname_src_seg[0] = arguments['-iseg']
    else:
        fname_src, fname_src_seg = random_list(folder_path, list_data, nbre_wrap)
    if '-d' in arguments:
        fname_dest[0] = arguments['-d']
        fname_dest_seg[0] = arguments['-dseg']
    else:
        fname_dest, fname_dest_seg = random_list(folder_path, list_data, nbre_wrap)

    # resample data to 1x1x1 mm^3
    for iter in range(0,nbre_wrap):
        sct.run('sct_resample -i ' + fname_src[iter] + ' -mm 1x1x1 -x nn -o ' + fname_src[iter])
        sct.run('sct_resample -i ' + fname_src_seg[iter] + ' -mm 1x1x1 -x nn -o ' + fname_src_seg[iter])
        sct.run('sct_resample -i ' + fname_dest[iter] + ' -mm 1x1x1 -x nn -o ' + fname_dest[iter])
        sct.run('sct_resample -i ' + fname_dest_seg[iter] + ' -mm 1x1x1 -x nn -o ' + fname_dest_seg[iter])

    sct.printv(str(fname_dest_seg))
    crop_segmentation(fname_src, fname_src_seg, nbre_wrap)
    crop_segmentation(fname_dest, fname_dest_seg, nbre_wrap)

    src, src_seg, nbre_src = random_slice(fname_src, fname_src_seg, fname_size_x, fname_size_y, folder_path, nbre_wrap)
    dest, dest_seg, nbre_dest = random_slice(fname_dest, fname_dest_seg, fname_size_x, fname_size_y, folder_path, nbre_wrap)

    warp = warping_field(src_seg, dest_seg, nbre_wrap)

    fname_im = []
    fname_im_seg = []

    if '-t' in arguments:
        fname_im[0] = arguments['-t']
        fname_im_seg[0] = arguments['-tseg']
    else:
        fname_im, fname_im_seg = random_list(folder_path, list_data, nbre_slice*nbre_wrap)

    im, im_seg, nbre_im = random_slice(fname_im, fname_im_seg, fname_size_x, fname_size_y, folder_path, nbre_slice*nbre_wrap)

    apply_warping_field(im, im_seg, src, dest, nbre_slice, nbre_wrap, output_folder_path, warp, nbre_im)


# Return a list of random images and segmentation of a given folder
# ==========================================================================================
def random_list(folder_path, list_data, nw):
    list=[]
    list_seg=[]

    for iter in range(0,nw):
        random_index = random.randrange(1, len(list_data) / 2 - 1)
        fname_src = folder_path + '/' + str(list_data[random_index * 2])
        fname_src_seg = folder_path + '/' + str(list_data[random_index * 2 + 1])
        list.append(fname_src)
        list_seg.append(fname_src_seg)

    return list, list_seg


# Crop the image to stay in the boundaries of the segmentation
# ==========================================================================================
def crop_segmentation(fname, fname_seg, nw):

    for iter in range(0,nw):
        nx, ny, nz, nt, px, py, pz, pt = Image(fname_seg[iter]).dim

        data_seg = Image(fname_seg[iter]).data
        i = 1
        while np.all(data_seg[:, :, i] == np.zeros((nx,ny))):
            i += 1
        i += 0.15*nz
        j = nz -1
        while np.all(data_seg[:, :, j] == np.zeros((nx,ny))):
            j -= 1
        j -= 0.1*nz

        sct.run("sct_crop_image -i " + fname[iter] + " -dim 2 -start " + str(i) + " -end " + str(j) + " -o " + fname[iter])
        sct.run("sct_crop_image -i " + fname_seg[iter] + " -dim 2 -start " + str(i) + " -end " + str(j) + " -o " + fname_seg[iter])


# Get random slices and their place in a given image
# ==========================================================================================
def random_slice(fname, fname_seg, fname_size_x, fname_size_y, folder_path, nw):

    im = []
    im_seg = []
    nbre_im = []
    for iter in range(0, nw):
        path, file, ext = sct.extract_fname(fname[iter])
        path_seg, file_seg, ext_seg = sct.extract_fname(fname_seg[iter])

        # change the orientation to RPI
        from sct_image import get_orientation_3d
        if not get_orientation_3d(Image(fname[iter])) == 'RPI':
            sct.run("sct_image -i " + fname[iter] + " -setorient RPI -o " + fname[iter], verbose=0)
            sct.run("sct_image -i " + fname_seg[iter] + " -setorient RPI -o " + fname_seg[iter], verbose=0)

        # get the number of the slice in the source image and crop it
        if not sct.check_folder_exist("Slices_" + file):
            os.makedirs("Slices_" + file)

        nx, ny, nz, nt, px, py, pz, pt = Image(fname[iter]).dim
        os.chdir("Slices_" + file)
        nbre = random.randrange(1, nz - 1)
        ima = file + "_crop_" + str(nbre) + "_o0.nii.gz"
        ima_seg = file + "_crop_seg_" + str(nbre) + "_o0.nii.gz"
        nbre_im.append(nbre)
        if not os.path.isfile(ima):
            sct.run("sct_data_augmentation.py -i " + folder_path + '/' + file + ext + " -iseg " + folder_path + '/' + file_seg + ext_seg + " -lx " + str(fname_size_x) + " -ly " + str(fname_size_y) + " -z " + str(nbre))
        os.chdir('..')
        im.append("Slices_" + file + "/" + ima)
        im_seg.append("Slices_" + file + "/" + ima_seg)

    return im, im_seg, nbre_im


# Create warping field
# ==========================================================================================
def warping_field(src_seg, dest_seg, nw):
    warp = []
    for iter in range(0, nw):

        out = "t2_output_image_transformed.nii.gz"
        out_transfo = 'transfo_' + str(iter)

        sct.run(
            'isct_antsRegistration ' + '--dimensionality 2 ' + '--transform BSplineSyN[0.5,1,3] ' + '--metric MeanSquares[' + dest_seg[iter] + ',' + src_seg[iter] + ', 1] '
            + '--convergence 2x5 ' + '--shrink-factors 2x1 ' + '--smoothing-sigmas 1x0mm ' + '--output [' + out_transfo + ','  + out +']' + ' --interpolation BSpline[3] ' + '--verbose 0')

        warp.append(out_transfo + '0Warp.nii.gz')

    return warp


# Apply warping field
# ==========================================================================================
def apply_warping_field(im, im_seg, src, dest, nbre_slice, nbre_wrap,  output_folder_path, wrap, nbre_im):

    fname_out = []
    fname_out_s = []
    j=0
    for i in range(0, nbre_wrap):
        src_path, src_file, src_ext = sct.extract_fname(src[i])
        dest_path, dest_file, dest_ext = sct.extract_fname(dest[i])
        for iter in range(0, nbre_slice):

            fname_out_im = 'transfo_' + src_file + '_' + dest_file + '_' + str(nbre_im[j]) + '.nii.gz'
            fname_out_seg = "transfo_" + src_file + '_' + dest_file + '_' + str(nbre_im[j]) + '_seg.nii.gz'
            # Apply warping field to src data
            sct.run('isct_antsApplyTransforms -d 2 -i ' + im[j] + ' -r ' + dest[i] + ' -n NearestNeighbor -t ' + wrap[i] + ' --output ' + fname_out_im)
            sct.run('isct_antsApplyTransforms -d 2 -i ' + im_seg[j] + ' -r ' + dest[i] + ' -n NearestNeighbor -t ' + wrap[i] + ' --output ' + fname_out_seg)
        fname_out.append(fname_out_im)
        fname_out_s.append(fname_out_seg)
    return fname_out, fname_out_s


# START PROGRAM
# ==========================================================================================
if __name__ == "__main__":
    # call main function
    main()