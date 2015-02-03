#!/usr/bin/env python

__author__ = 'Simon_2'

# ======================================================================================================================

# Extract results from .txt files generated by " " and draw box-and-whisker plots comparing the absolute error of
# estimation using the different metric extraction methods with different levels of noise and tracts std.

# ======================================================================================================================
import os
import glob
import getopt
import commands
import sys
import numpy
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.legend_handler import *
# import subprocess
# path_sct = subprocess.check_output("echo %SCT_DIR%", shell=True)

class Param:
    def __init__(self):
        self.debug = 1
        self.results_folder = "data_methods_comparison"
        self.methods_to_display = 'bin,wa,wath,ml,map'


# =======================================================================================================================
# main
# =======================================================================================================================
def main():
    results_folder = param_default.results_folder
    methods_to_display = param_default.methods_to_display

    # Parameters for debug mode
    if param.debug:
        print '\n*** WARNING: DEBUG MODE ON ***\n'
        results_folder = "/Users/slevy_local/spinalcordtoolbox/dev/atlas/validate_atlas/results_20150112"#"C:/cygwin64/home/Simon_2/data_methods_comparison"
        path_sct = '/Users/slevy_local/spinalcordtoolbox' #'C:/cygwin64/home/Simon_2/spinalcordtoolbox'
        methods_to_display = 'bin,wa,wath,ml,map'
    else:
        status, path_sct = commands.getstatusoutput('echo $SCT_DIR')

        # Check input parameters
        try:
            opts, args = getopt.getopt(sys.argv[1:], 'i:m:')  # define flags
        except getopt.GetoptError as err:  # check if the arguments are defined
            print str(err)  # error
            # usage() # display usage
        # if not opts:
        #     print 'Please enter the path to the result folder. Exit program.'
        #     sys.exit(1)
        #     # usage()
        for opt, arg in opts:  # explore flags
            if opt in '-i':
                results_folder = arg
            if opt in '-m':
                methods_to_display = arg

    # Append path that contains scripts, to be able to load modules
    sys.path.append(path_sct + '/scripts')
    import sct_utils as sct

    sct.printv("Working directory: " + os.getcwd())

    results_folder_csf = results_folder + '/csf'

    sct.printv('\n\nData will be extracted from folder ' + results_folder_csf + '.', 'warning')
    sct.printv('\t\tCheck existence...')
    sct.check_folder_exist(results_folder_csf)

    # Extract methods to display
    methods_to_display = methods_to_display.strip().split(',')

    # Extract file names of the results files
    fname_results_csf = glob.glob(results_folder_csf + '/*.txt')
    # Remove doublons (due to the two folders)
    # for i_fname in range(0, len(fname_results)):
    #     for j_fname in range(0, len(fname_results)):
    #         if (i_fname != j_fname) & (os.path.basename(fname_results[i_fname]) == os.path.basename(fname_results[j_fname])):
    #             fname_results.remove(fname_results[j_fname])
    file_results = []
    for fname in fname_results_csf:
        file_results.append(os.path.basename(fname))
    for file in file_results:
        if file_results.count(file) > 1:
            ind = file_results.index(file)
            fname_results_csf.remove(fname_results_csf[ind])
            file_results.remove(file)

    nb_results_file = len(fname_results_csf)

    # 1st dim: SNR, 2nd dim: tract std, 3rd dim: mean abs error, 4th dim: std abs error
    # result_array = numpy.empty((nb_results_file, nb_results_file, 3), dtype=object)
    # SNR
    snr = numpy.zeros((nb_results_file))
    # Tracts std
    tracts_std = numpy.zeros((nb_results_file))
    # CSF value
    csf_values = numpy.zeros((nb_results_file))
    # methods' name
    methods_name = []  #numpy.empty((nb_results_file, nb_method), dtype=object)
    # labels
    error_per_label = []
    std_per_label = []
    labels_id = []
    # median
    median_results = numpy.zeros((nb_results_file, 5))
    # median std across bootstraps
    median_std = numpy.zeros((nb_results_file, 5))
    # min
    min_results = numpy.zeros((nb_results_file, 5))
    # max
    max_results = numpy.zeros((nb_results_file, 5))

    #
    for i_file in range(0, nb_results_file):

        # Open file
        f = open(fname_results_csf[i_file])  # open file
        # Extract all lines in .txt file
        lines = [line for line in f.readlines() if line.strip()]

        # extract SNR
        # find all index of lines containing the string "sigma noise"
        ind_line_noise = [lines.index(line_noise) for line_noise in lines if "sigma noise" in line_noise]
        if len(ind_line_noise) != 1:
            sct.printv("ERROR: number of lines including \"sigma noise\" is different from 1. Exit program.", 'error')
            sys.exit(1)
        else:
            # result_array[:, i_file, i_file] = int(''.join(c for c in lines[ind_line_noise[0]] if c.isdigit()))
            snr[i_file] = int(''.join(c for c in lines[ind_line_noise[0]] if c.isdigit()))

        # extract tract std
        ind_line_tract_std = [lines.index(line_tract_std) for line_tract_std in lines if
                              "range tracts" in line_tract_std]
        if len(ind_line_tract_std) != 1:
            sct.printv("ERROR: number of lines including \"range tracts\" is different from 1. Exit program.", 'error')
            sys.exit(1)
        else:
            # result_array[i_file, i_file, :] = int(''.join(c for c in lines[ind_line_tract_std[0]].split(':')[1] if c.isdigit()))
            # regex = re.compile(''('(.*)':)  # re.I permet d'ignorer la case (majuscule/minuscule)
            # match = regex.search(lines[ind_line_tract_std[0]])
            # result_array[:, i_file, :, :] = match.group(1)  # le groupe 1 correspond a '.*'
            tracts_std[i_file] = int(''.join(c for c in lines[ind_line_tract_std[0]].split(':')[1] if c.isdigit()))

        # extract CSF value
        ind_line_csf_value = [lines.index(line_csf_value) for line_csf_value in lines if
                              "# value CSF" in line_csf_value]
        if len(ind_line_csf_value) != 1:
            sct.printv("ERROR: number of lines including \"range tracts\" is different from 1. Exit program.", 'error')
            sys.exit(1)
        else:
            # result_array[i_file, i_file, :] = int(''.join(c for c in lines[ind_line_tract_std[0]].split(':')[1] if c.isdigit()))
            # regex = re.compile(''('(.*)':)  # re.I permet d'ignorer la case (majuscule/minuscule)
            # match = regex.search(lines[ind_line_tract_std[0]])
            # result_array[:, i_file, :, :] = match.group(1)  # le groupe 1 correspond a '.*'
            csf_values[i_file] = int(''.join(c for c in lines[ind_line_csf_value[0]].split(':')[1] if c.isdigit()))


        # extract method name
        ind_line_label = [lines.index(line_label) for line_label in lines if "Label" in line_label]
        if len(ind_line_label) != 1:
            sct.printv("ERROR: number of lines including \"Label\" is different from 1. Exit program.", 'error')
            sys.exit(1)
        else:
            # methods_name[i_file, :] = numpy.array(lines[ind_line_label[0]].strip().split(',')[1:])
            methods_name.append(lines[ind_line_label[0]].strip().replace(' ', '').split(',')[1:])

        # extract median
        ind_line_median = [lines.index(line_median) for line_median in lines if "median" in line_median]
        if len(ind_line_median) != 1:
            sct.printv("WARNING: number of lines including \"median\" is different from 1. Exit program.", 'warning')
            # sys.exit(1)
        else:
            median = lines[ind_line_median[0]].strip().split(',')[1:]
            # result_array[i_file, i_file, 0] = [float(m.split('(')[0]) for m in median]
            median_results[i_file, :] = numpy.array([float(m.split('(')[0]) for m in median])
            median_std[i_file, :] = numpy.array([float(m.split('(')[1][:-1]) for m in median])

        # extract min
        ind_line_min = [lines.index(line_min) for line_min in lines if "min," in line_min]
        if len(ind_line_min) != 1:
            sct.printv("WARNING: number of lines including \"min\" is different from 1. Exit program.", 'warning')
            # sys.exit(1)
        else:
            min = lines[ind_line_min[0]].strip().split(',')[1:]
            # result_array[i_file, i_file, 1] = [float(m.split('(')[0]) for m in min]
            min_results[i_file, :] = numpy.array([float(m.split('(')[0]) for m in min])

        # extract max
        ind_line_max = [lines.index(line_max) for line_max in lines if "max" in line_max]
        if len(ind_line_max) != 1:
            sct.printv("WARNING: number of lines including \"max\" is different from 1. Exit program.", 'warning')
            # sys.exit(1)
        else:
            max = lines[ind_line_max[0]].strip().split(',')[1:]
            # result_array[i_file, i_file, 1] = [float(m.split('(')[0]) for m in max]
            max_results[i_file, :] = numpy.array([float(m.split('(')[0]) for m in max])

        # extract error for each label
        error_per_label_for_file_i = []
        std_per_label_for_file_i = []
        labels_id_for_file_i = []
        # Due to 2 different kind of file structure, the number of the last label line must be adapted
        if not ind_line_median:
            ind_line_median = [len(lines) + 1]
        for i_line in range(ind_line_label[0] + 1, ind_line_median[0] - 1):
            line_label_i = lines[i_line].strip().split(',')
            error_per_label_for_file_i.append([float(error.strip().split('(')[0]) for error in line_label_i[1:]])
            std_per_label_for_file_i.append([float(error.strip().split('(')[1][:-1]) for error in line_label_i[1:]])
            labels_id_for_file_i.append(line_label_i[0])
        error_per_label.append(error_per_label_for_file_i)
        std_per_label.append(std_per_label_for_file_i)
        labels_id.append(labels_id_for_file_i)

        # close file
        f.close()

    # check if all the files in the result folder were generated with the same number of methods
    if not all(x == methods_name[0] for x in methods_name):
        sct.printv(
            'ERROR: All the generated files in folder ' + results_folder + ' have not been generated with the same number of methods. Exit program.',
            'error')
        sys.exit(1)
    # check if all the files in the result folder were generated with the same labels
    if not all(x == labels_id[0] for x in labels_id):
        sct.printv(
            'ERROR: All the generated files in folder ' + results_folder + ' have not been generated with the same labels. Exit program.',
            'error')
        sys.exit(1)

    # convert the list "error_per_label" into a numpy array to ease further manipulations
    error_per_label = numpy.array(error_per_label)
    std_per_label = numpy.array(std_per_label)
    # compute different stats
    abs_error_per_labels = numpy.absolute(error_per_label)
    max_abs_error_per_meth = numpy.amax(abs_error_per_labels, axis=1)
    min_abs_error_per_meth = numpy.amin(abs_error_per_labels, axis=1)
    mean_abs_error_per_meth = numpy.mean(abs_error_per_labels, axis=1)
    std_abs_error_per_meth = numpy.std(abs_error_per_labels, axis=1)

    nb_method = len(methods_to_display)

    sct.printv('Noise std of the ' + str(nb_results_file) + ' generated files:')
    print snr
    print '----------------------------------------------------------------------------------------------------------------'
    sct.printv('Tracts std of the ' + str(nb_results_file) + ' generated files:')
    print tracts_std
    print '----------------------------------------------------------------------------------------------------------------'
    sct.printv('CSF value of the ' + str(nb_results_file) + ' generated files:')
    print csf_values
    print '----------------------------------------------------------------------------------------------------------------'
    sct.printv('Methods used to generate results for the ' + str(nb_results_file) + ' generated files:')
    print methods_name
    print '----------------------------------------------------------------------------------------------------------------'
    sct.printv(
        'Median obtained with each method (in colons) for the ' + str(nb_results_file) + ' generated files (in lines):')
    print median_results
    print '----------------------------------------------------------------------------------------------------------------'
    sct.printv('Minimum obtained with each method (in colons) for the ' + str(
        nb_results_file) + ' generated files (in lines):')
    print min_results
    print '----------------------------------------------------------------------------------------------------------------'
    sct.printv('Maximum obtained with each method (in colons) for the ' + str(
        nb_results_file) + ' generated files (in lines):')
    print max_results
    print '----------------------------------------------------------------------------------------------------------------'
    sct.printv('Labels\' ID (in colons) for the ' + str(nb_results_file) + ' generated files (in lines):')
    print labels_id
    print '----------------------------------------------------------------------------------------------------------------'
    sct.printv(
        'Errors obtained with each method (in colons) for the ' + str(nb_results_file) + ' generated files (in lines):')
    print error_per_label


    # *********************************************** START PLOTTING HERE **********************************************

    ind_files_csf_sort = numpy.argsort(csf_values)

    matplotlib.rcParams.update({'font.size': 22, 'font.family': 'trebuchet'})
    plt.figure(figsize=(30, 15))
    width = 1.0 / (nb_method + 1)
    ind_fig = numpy.arange(len(ind_files_csf_sort)) * (1.0 + width)
    plt.ylabel('Absolute error (%)\n', fontsize=22)
    plt.xlabel('\nCSF values (in percentage of true value in tracts)', fontsize=22)
    plt.title('Absolute error within all tracts as a function of CSF values\n', fontsize=24)

    # colors = plt.get_cmap('jet')(np.linspace(0, 1.0, nb_method))
    colors = ['b', 'g', 'r', 'c', 'm', 'y', 'k']
    box_plots = []
    for meth, color in zip(methods_to_display, colors):
        i_meth = methods_name[0].index(meth)
        i_meth_to_display = methods_to_display.index(meth)

        boxprops = dict(linewidth=3, color=color)
        flierprops = dict(color=color, markeredgewidth=0.7, markersize=7, marker='.')
        whiskerprops = dict(color=color, linewidth=2)
        capprops = dict(color=color, linewidth=2)
        medianprops = dict(linewidth=3, color=color)
        meanpointprops = dict(marker='D', markeredgecolor='black', markerfacecolor='firebrick')
        meanlineprops = dict(linestyle='--', linewidth=2.5, color='purple')
        plot_i = plt.boxplot(numpy.transpose(abs_error_per_labels[ind_files_csf_sort, :, i_meth]), positions=ind_fig + i_meth_to_display * width + (float(i_meth_to_display) * width) / (nb_method + 1), widths=width, boxprops=boxprops, medianprops=medianprops, flierprops=flierprops, whiskerprops=whiskerprops, capprops=capprops)
        # plt.errorbar(ind_fig2+i_meth*width+width/2+(float(i_meth)*width)/(nb_method+1), mean_abs_error_per_meth[ind_snr_sort_tracts_std_10, i_meth], std_abs_error_per_meth[ind_snr_sort_tracts_std_10, i_meth], color=color, marker='_', linestyle='None', markersize=200*width, markeredgewidth=3)
        box_plots.append(plot_i['boxes'][0])

    # plt.legend(box_plots, methods_to_display, bbox_to_anchor=(1.01, 1), loc=2, borderaxespad=0.)
    plt.legend(box_plots, methods_to_display, loc='best', fontsize=22)
    plt.xticks(ind_fig + (numpy.floor(nb_method / 2)) * width * (1.0 + 1.0 / (nb_method + 1)), csf_values[ind_files_csf_sort])
    plt.gca().set_xlim([-width, numpy.max(ind_fig) + (nb_method + 0.5) * width])
    plt.gca().set_ylim([0, 20])
    plt.gca().yaxis.set_major_locator(plt.MultipleLocator(1.0))
    plt.gca().yaxis.set_minor_locator(plt.MultipleLocator(0.25))
    plt.grid(b=True, axis='y', which='both')

    plt.savefig(results_folder+'/absolute_error_vs_csf_values')

    plt.show()

#=======================================================================================================================
# Start program
#=======================================================================================================================
if __name__ == "__main__":
    param_default = Param()
    param = Param()
    # call main function
    main()