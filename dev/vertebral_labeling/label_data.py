# !/usr/bin/env python
#
# Label data at the posterior tip of C2-C3 disc. Need to run prepare_data first.
# Julien Cohen-Adad 2017-02-11

import sys
import os
from lxml import etree as ET
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

from PIL import Image
from matplotlib.pylab import *

# Path to SCT
path_sct = '/Users/julien/code/sct/'  # slash at the end
sys.path.append(path_sct + 'scripts/')

# Parameters
path_im = '/Users/julien/data/deep_learning/data_training/images/'
ext_im = '.png'  # extension of input images
path_annot = '/Users/julien/data/deep_learning/data_training/annotations/'
ext_annot = '.xml'
screen_resolution = 1080  # height of screen resolution for adjusting display

def onclick(event):
    """
    get click from figure
    :param event:
    :return:
    """
    global ix, iy
    ix, iy = event.xdata, event.ydata
    # print 'x = %d, y = %d' % (
    #     ix, iy)

    global coords
    coords.append((ix, iy))
    # quit after one click
    if len(coords) == 1:
        fig.canvas.mpl_disconnect(cid)
        plt.close(1)
    return coords

if __name__ == "__main__":

    # generate subject list (only keep images with specified extention)
    list_images = [x for x in os.listdir(path_im) if ext_im in x]

    # create output dir
    if not os.path.exists(path_annot):
        os.makedirs(path_annot)

    # get list of subjects with existing annotations
    list_annot = [x for x in os.listdir(path_annot) if ext_annot in x]
    # discard subjects who already have a label
    list_subjects = [x.strip(ext_im) for x in list_images if not x.strip(ext_im) in [y.strip(ext_annot) for y in list_annot] ]

    # Loop across subjects
    i = 1
    for subject in list_subjects:

        # display
        print('Image ' + str(i) + '/' + str(len(list_subjects)) + ': ' + subject)

        # Open image
        img = mpimg.imread(path_im + subject + ext_im)

        # Display image
        # ax = plt.gca(figsize=(20,10))
        fig = plt.figure(figsize=(screen_resolution/100, screen_resolution/100), dpi=100)
        ax = plt.gca()
        # fig = plt.gcf()
        imgplot = ax.imshow(img, cmap=cm.gray)
        ax.set_title('Please click at the posterior tip of the C2/C3 disk')
        # bbox_inches = 'tight'
        # plt.savefig(path_out_im + name_subject + '.jpg')
        coords = []
        cid = fig.canvas.mpl_connect('button_press_event', onclick)
        plt.show(block=True)

        # Save Annotation as xml file
        annotation = ET.Element("annotation")
        source = ET.SubElement(annotation, "source")
        ET.SubElement(source, "database").text = "NeuroPoly Lab, Polytechnique Montreal"
        ET.SubElement(source, "image").text = "NeuroPoly Lab, Polytechnique Montreal"
        ET.SubElement(source, "annotation").text = "Julien Cohen-Adad"
        size = ET.SubElement(annotation, "size")
        ET.SubElement(size, "width").text = str(img.shape[0])
        ET.SubElement(size, "height").text = str(img.shape[1])
        ET.SubElement(size, "depth").text = str(1)
        object = ET.SubElement(annotation, "object")
        ET.SubElement(object, "name").text = "C2-C3 disk"
        coord = ET.SubElement(object, "coord")
        ET.SubElement(coord, "x").text = str(coords[0][0])
        ET.SubElement(coord, "x").text = str(coords[0][1])
        tree = ET.ElementTree(annotation)
        tree.write(path_annot + subject + '.xml', pretty_print=True)
