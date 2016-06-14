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
    if '-o' in arguments['-o']:
        output_folder = arguments['-o']
        os.makedirs(str(output_folder))
        output_folder_path = str(output_folder) + "/"
    else:
        output_folder_path = ''

    # choose randomly a source and a destination
    files = os.listdir("data_subjects_patients")
    for i in range(0,nbre_wrap):
        if '-i' in arguments:
            fname_src = arguments['-i']
            fname_src_seg = arguments['-iseg']
        else:
            random_index = random.randrange(1, len(files)/2)
            fname_src = "data_subjects_patients/" + files[random_index*2-1]
            fname_src_seg = "data_subjects_patients/" + files[random_index*2]

        if '-d' in arguments:
            fname_dest = arguments['-d']
            fname_dest_seg = arguments['-dseg']
        else:
            random_index = random.randrange(1, len(files)/2)
            fname_dest = "data_subjects_patients/" + files[random_index*2-1]
            fname_dest_seg = "data_subjects_patients/" + files[random_index*2]

        crop_segmentation(fname_src, fname_src_seg)
        crop_segmentation(fname_dest, fname_dest_seg)

        src, src_seg, nbre_src = random_slice(fname_src, fname_src_seg, fname_size_x, fname_size_y)
        dest, dest_seg, nbre_dest = random_slice(fname_dest, fname_dest_seg, fname_size_x, fname_size_y)

        path_src, file_src, ext_scr = sct.extract_fname(src)
        path_dest, file_dest, ext_dest = sct.extract_fname(dest)

        out = "t2_output_image_transformed.nii.gz"

        sct.run( 'isct_antsRegistration ' + '--dimensionality 2 ' + '--transform BSplineSyN[0.5,1,3] ' + '--metric MeanSquares[' + dest_seg + ',' + src_seg + ', 1] '
                         + '--convergence 2x5 ' + '--shrink-factors 2x1 ' + '--smoothing-sigmas 1x0mm ' + '--output [transfo_,' + out + '] ' + '--interpolation BSpline[3] ' + '--verbose 0')

        if '-t' in arguments:
            fname_im = arguments['-t']
            fname_im_seg = arguments['-tseg']

            for iter in range(0,nbre_slice):
                path_im, file_im, ext_im = sct.extract_fname(fname_im)

                im, im_seg, nbre_im = random_slice(fname_im, fname_im_seg, fname_size_x, fname_size_y)

                fname_out_im = output_folder_path + 'transfo_' + file_src + '_' + file_dest + '_' + str(nbre_im) + '.nii.gz'
                fname_out_seg = output_folder_path + "transfo_" + file_src + '_' + file_dest + '_' + str(
                        nbre_im) + '_seg.nii.gz'

                # Apply warping field to src data
                sct.run('isct_antsApplyTransforms -d 2 -i ' + im + ' -r ' + dest + ' -n NearestNeighbor -t ' + 'transfo_0Warp.nii.gz' + ' --output ' + fname_out_im)
                sct.run('isct_antsApplyTransforms -d 2 -i ' + im_seg + ' -r ' + dest + ' -n NearestNeighbor -t ' + 'transfo_0Warp.nii.gz' + ' --output ' + fname_out_seg)
        else:
            for iter in range(0,nbre_slice):
                random_index = random.randrange(1, len(files)/2)
                fname_im = "data_subjects_patients/" + files[random_index*2-1]
                fname_im_seg = "data_subjects_patients/" + files[random_index*2]

                path_im, file_im, ext_im = sct.extract_fname(fname_im)

                im, im_seg, nbre_im = random_slice(fname_im, fname_im_seg, fname_size_x, fname_size_y)

                fname_out_im =  output_folder_path + 'transfo_' + file_src + '_' + file_dest + '_' + str(nbre_im) + '.nii.gz'
                fname_out_seg = output_folder_path + "transfo_" + file_src + '_' + file_dest + '_' + str(nbre_im) + '_seg.nii.gz'

                # Apply warping field to src data
                sct.run('isct_antsApplyTransforms -d 2 -i ' + im + ' -r ' + dest + ' -n NearestNeighbor -t ' + 'transfo_0Warp.nii.gz' + ' --output '+ fname_out_im)
                sct.run('isct_antsApplyTransforms -d 2 -i ' + im_seg + ' -r ' + dest + ' -n NearestNeighbor -t ' + 'transfo_0Warp.nii.gz' + ' --output '+ fname_out_seg)

                # resample the final image to 1mm isotropic
                out_im = Image(fname_out_im)
                out_seg = Image(fname_out_seg)
                nx, ny, nz, nt, px, py, pz, pt = out_im.dim
                zooms=(px,py)
                new_zooms=(1,1)
                affine_im = out_im.hdr.get_qform()
                affine_seg = out_seg.hdr.get_qform()

                from scipy.ndimage import affine_transform
                R = np.diag(np.array(new_zooms)/np.array(zooms))
                new_shape=np.array(zooms)/np.array(new_zooms)*np.array(out_im.data.shape[:2])
                new_shape = np.round(new_shape).astype('i8')

                new_data = affine_transform(input=out_im.data, matrix=R, offset=np.zeros(2, ), output_shape=tuple(new_shape), order=0, mode='nearest')
                new_data_seg = affine_transform(input=out_seg.data, matrix=R, offset=np.zeros(2, ), output_shape=tuple(new_shape), order=0, mode='nearest')

                Rx = np.eye(4)
                Rx[:2, :2] = R
                new_affine = np.dot(affine_im, Rx)
                new_affine_seg = np.dot(affine_seg, Rx)

                out_im.setFileName(fname_out_im)
                out_im.data = new_data
                out_im.hdr.set_sform(new_affine)
                out_im.hdr.set_qform(new_affine)
                out_im.save()
                out_seg.setFileName(fname_out_seg)
                out_seg.data = new_data_seg
                out_seg.hdr.set_sform(new_affine_seg)
                out_seg.hdr.set_qform(new_affine_seg)
                out_seg.save()


# Crop the image to stay in the boundaries of the segmentation
# ==========================================================================================
def crop_segmentation(fname, fname_seg):

    nx, ny, nz, nt, px, py, pz, pt = Image(fname_seg).dim

    data_seg = Image(fname_seg).data
    i = 1
    while np.all(data_seg[:, :, i] == np.zeros((nx,ny))):
        i += 1
    i += 0.15*nz
    j = nz -1
    while np.all(data_seg[:, :, j] == np.zeros((nx,ny))):
        j -= 1
    j -= 0.1*nz

    sct.run("sct_crop_image -i " + fname + " -dim 2 -start " + str(i) + " -end " + str(j) + " -o " + fname)
    sct.run("sct_crop_image -i " + fname_seg + " -dim 2 -start " + str(i) + " -end " + str(j) + " -o " + fname_seg)


# Get random slices and their place in a given image
# ==========================================================================================
def random_slice(fname, fname_seg, fname_size_x, fname_size_y):

    path, file, ext = sct.extract_fname(fname)
    path_seg, file_seg, ext_seg = sct.extract_fname(fname_seg)

    # change the orientation to RPI
    from sct_image import get_orientation_3d
    if not get_orientation_3d(Image(fname)) == 'RPI':
        sct.run("sct_image -i " + fname + " -setorient RPI -o " + fname, verbose=0)
        sct.run("sct_image -i " + fname_seg + " -setorient RPI -o " + fname_seg, verbose=0)

    # get the number of the slice in the source image and crop it
    if not sct.check_folder_exist("Slices_" + file):
        os.makedirs("Slices_" + file)

    nx, ny, nz, nt, px, py, pz, pt = Image(fname).dim
    os.chdir("Slices_" + file)
    nbre = random.randrange(1, nz - 1)
    im = file + "_crop_" + str(nbre) + "_o0.nii.gz"
    im_seg = file + "_crop_seg_" + str(nbre) + "_o0.nii.gz"
    nbre_im = nbre
    while not os.path.isfile(im):
        sct.run("sct_data_augmentation -i " + "../data_subjects_patients/" + file + ext + " -iseg " + "../data_subjects_patients/" + file_seg + ext_seg + " -lx " + str(fname_size_x) + " -ly " + str(fname_size_y) + " -z " + str(nbre))
        im = file + "_crop_" + str(nbre) + "_o0.nii.gz"
        im_seg = file + "_crop_seg_" + str(nbre) + "_o0.nii.gz"
        nbre_im = nbre
        nbre = random.randrange(1, nz-1)
    os.chdir('..')
    im = "Slices_" + file + "/" + im
    im_seg = "Slices_" + file + "/" + im_seg

    return im, im_seg, nbre_im


# START PROGRAM
# ==========================================================================================
if __name__ == "__main__":
    # call main function
    main()