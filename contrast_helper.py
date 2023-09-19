'''Helper functions to create 2D line data and to evaluate the Michelson contrast of 2D line data'''

import numpy as np
import matplotlib.pyplot as plt
from os import path
from math import sqrt, log, floor, ceil
from scipy import ndimage
from . import ampd
import more_itertools as mit



def createLines(gratingNumber, size=1, Nx=1023, Ny=1023, direction=0, super_exp = 4, fillFactor=0.95, smooth_off_axis = True, crop_off_axis = True,line_broadness_factor=1, **kwargs):
    """Create 2D-array with periodic lines along one axis for contrast tests

    Parameters
    ----------
    gratingNumber : float
        number of lines/mm
    size : int, optional
        size in mm of the whole 2D-array (side length)
    Nx : int, optional
        number of pixels in x-direction
    Ny : int, optional
        number of pixels in y-drection
    direction : int, optional
        direction along which there should be the line contrast (i.e. direction = 0 leads to lines along direction = 1)
    super_exp : int, optional
        Exponent for the super gauss for the individual lines
    fillFactor : float, optional
        how much of size should be filled with lines (only full lines that fit within size*fillFactor are created)
    smooth_off_axis : bool, optional
        True leads to a 1D convolution ALONG the lines to smooth out their edges
    line_broadness_factor: float, optional
        Numbers greater than 1 lead to bigger lines with the same distance between lines (higher duty cycle)

    Returns
    -------
    ndarray
        2D-array with the lines
    """    

    def super_gauss(x, center, fwhm):
        return np.exp(-log(2)*((x-center)/(fwhm/2))**(2*super_exp))

    
    #choose which pixel numer is the "main" and "secondary" pixel number
    if direction == 0:
        pix_main = Nx
        pix_off = Ny
    else:
        pix_main = Ny
        pix_off = Nx

         
    #create 1D-coordinates for the direction with the actual lines
    lin = np.linspace(-size/2, size/2, pix_main)
    Lines = np.zeros(pix_main)
    
    #calculate width of lines (not of line pairs!)
    line_width = 1/gratingNumber/2

    #check if there are enough pixels for the given lines
    pix_per_line = pix_main/size*line_width
    if pix_per_line < 3:
        print("Resolution too small to properly represent individual lines (< 3 pixels per line)! Use data with care!")

    #number of full lines that fit within size*fillfactor
    n_lines = floor((size*fillFactor/line_width + 1)/2)
    #print(n_lines)

 
    if n_lines%2 == 0:
        offset = line_width # no line in the center
    else:
        offset = 0 # line in the center

    #create the lines
    for j in range(ceil(n_lines/2)):
        if j == 0 and offset == 0 :
            Lines += super_gauss(lin, 0, line_width*line_broadness_factor)
        else:
            offset_j = offset+2*j*line_width
            Lines += super_gauss(lin, offset_j , line_width*line_broadness_factor) + super_gauss(lin, -offset_j, line_width*line_broadness_factor)
    

    #create 2D-image
    Lines2D = np.tile(Lines,(pix_off,1)).transpose()

    #crop out edges of the lines to fulfill crop_factor
    if fillFactor < 1 and crop_off_axis:
        lin_off = np.linspace(-size/2, size/2, pix_off)
        off_axis = np.meshgrid(lin, lin_off, indexing = 'ij')[1]
        Lines2D[abs(off_axis) > (fillFactor * size/2)] = 0

    #smooth out edges in off-direction
    if smooth_off_axis and crop_off_axis:
        sigma_gauss = line_width/size*pix_off/5
        Lines2D = ndimage.gaussian_filter1d(Lines2D, sigma_gauss/5, 1)
    
    #transpose image in case direction = 1
    if direction == 1:
        return np.transpose(Lines2D)
    else:
        return Lines2D
    
     
def contrastIndependent(data, direction = 0, width_data = 1, width_lines = 1, use_ampd = True, auto_zoom = False, show_plot = False, show_peaks = True, show_average = False, **kwargs):
    """Calculates the average Michelson line contrast and its standard deviation for 2D np-array of intensity data along an axis
    Function finds lines, maxima and minima on its own and does not require knowledge about the target line data

    Parameters
    ----------
    data : ndarray
        2D array with image data
    direction : int, optional
        np-direction of the lines in 2D data
    width_data : int, optional
        relative size of whole data compared to width_lines (see below)
    width_lines : int, optional
        relative size of actual line data, use width_lines < width_data to crop of heavy noise around the lines
    use_ampd : bool, optional
        True uses the Automatic Multi-Scale Peak Detection. Otherwise groups of values > 0.5 are identified as peaks
    auto_zoom : bool, optional
        True leads to a recursive call of this function with reduced width_lines in case no contrast can be determined (usually because minima are above 0.5 after normalization)
    show_plot : bool, optional
        whether to plot the data possibly including the found peaks and valleys and the average (see below)
    show_peaks : bool, optional
        draws the found max and min values for each line if show_plot = True
    show_average : bool, optional
        draws the average of the max and min values if show_plot = True

    Returns
    -------
    (contrast, contrast_std, N_maxima)
        The contrast (max-min)/(max+min) for the averaged max and min values for all lines,
        and the standard deviation of the contrast calculated from the spread of the max and min values
        and the number if found maxima
    
    """    
    #stop recursive call of function with auto_zoom
    if width_lines < 0.4*width_data and auto_zoom:
        return (-1, 0, 0)
           
    average_1D = np.sum(data, axis = (1-direction))

    #crop to actual line data (assuming line data is centered in data)
    crop_relative = (1-width_lines/width_data)/2
    crop_absolute = int(average_1D.size*crop_relative)
    average_1D = average_1D[crop_absolute:average_1D.size-crop_absolute-1]

    #normalize 
    min_original = np.min(average_1D)
    average_1D = average_1D-min_original # background removal only for max and min idenfication
    scaled_offset = min_original/np.max(average_1D) #scale original minimum with the normalization factor of the data without offset for contrast calculation
    average_1D = average_1D/np.max(average_1D)
    
    #find high and low values
    #high = average_1D[average_1D > 0.5]
    high_pos = np.argwhere(average_1D > 0.5)
    #low = average_1D[average_1D < 0.5]
    low_pos = np.argwhere(average_1D < 0.5)

    #find groups of consecutive indices to identify individual peaks and valleys
    groups_high = [list(group) for group in mit.consecutive_groups(high_pos)]
    groups_low = [list(group) for group in mit.consecutive_groups(low_pos)]

    max_pos = []
    max = []
    min_pos = []
    min = []
  
    #find max/min within each group
    #ampd seems to be more stable to find real peaks in noisy data with noise around 0.5
    if use_ampd:
        max_pos = ampd.find_peaks_original(average_1D)
        max = average_1D[max_pos]

        #only values > 0.5 should be real lines
        wrong_peaks = np.argwhere(max < 0.5)
        if len(wrong_peaks) > 0:
            max_pos = np.delete(max_pos, wrong_peaks)
            max = np.delete(max, wrong_peaks)

        #ampd has problems with perfect periodic data and sometimes only identifies the first peak
        #use the backup method instead
        if len(max) == 1:
            max_pos = []
            max = []
            use_ampd = False
    
    if not use_ampd:
        for group in groups_high:
            if len(group) == 1:
                pass
                #max_pos.append(group[0][0])
                #max.append(average_1D[group[0][0]])
            else:
                peak = average_1D[group[0][0]:group[-1][0]+1] # get the actual values for all indices within this group
                max.append(np.max(peak))                    # save the maximum
                max_pos.append(np.argmax(peak)+group[0][0]) # index of the maximum within the group + first element in the group (which is an index of average_1D that marks the beginning of the peak)

    for group in groups_low:
        if len(group) == 1:
            pass
            #min_pos.append(group[0][0])
            #min.append(average_1D[group[0][0]])
        else:
            valley = average_1D[group[0][0]:group[-1][0]+1]
            min.append(np.min(valley))
            min_pos.append(np.argmin(valley)+group[0][0])

    #remove minima before/after first/last maxima to remove falsely identified minima (only minima between lines should count)
    # first_max = max_pos[0]
    # last_max = max_pos[-1]
    # early_minima = np.argwhere(min_pos < first_max)
    # late_minima = np.argwhere(min_pos > last_max)
    # if early_minima.size > 0 or late_minima.size > 0:
    #     min_pos = np.delete(min_pos, np.concatenate([early_minima, late_minima]))
    #     min = np.delete(min, np.concatenate([early_minima, late_minima]))

    #remove wrong minima along edge of lines and in the background before/after the first/last line
    #use that there should only be one minima between two maximas and use the lowest one
    deepest_valley_pos = []
    min_pos = np.asarray(min_pos)
    for idx, peak in enumerate(max_pos):
        if idx < (len(max_pos)-1) : # don't search behind the last peak
            possible_valleys_pos = min_pos[(peak < min_pos) & (min_pos < max_pos[idx+1])]
            if possible_valleys_pos.size > 0:
                possible_valleys = average_1D[possible_valleys_pos]
                deepest_valley_pos.append(possible_valleys_pos[np.argmin(possible_valleys)])

    min_pos = deepest_valley_pos
    min = average_1D[min_pos]


    #plot
    if show_plot:
        #plt.plot(np.linspace(-width_lines/2,width_lines/2,average_1D.size),average_1D)
        plt.plot(average_1D)
        if show_peaks:
            plt.plot(max_pos,max,'o')
            plt.plot(min_pos,min,'o')
        #if show_average:
        #    plt.hlines(max_average,0,average_1D.size -1)
        #    plt.hlines(min_average,0,average_1D.size -1)
        plt.show()

    ###calculate the contrast and its spread
    ##verify results, expected is one minima less than maxima
    #bad results should return -1 or lead to a recursive search for a senseful crop range with auto_zoom = True
    if len(max) == 0:
        return (-1,0,0)
    if len(min) < (len(max)/2):
        if auto_zoom:
            return contrastIndependent(data, direction, width_data, width_lines*0.95, use_ampd, auto_zoom, show_plot, show_peaks, show_average)
        else:
            return (-1, 0, 0)

    #calculate the average of all maxima and minima
    max_average = np.average(max)
    min_average = np.average(min)
    #consider the removal of the data offset for real contrast value
    max_average+= scaled_offset
    min_average+= scaled_offset
    
    #calculate std. deviation
    max_std = sqrt(np.var(max))
    min_std = sqrt(np.var(min))

    #print(max_average, max_std, min_average, min_std)

    #calculate contrast and its uncertainty from the uncertainties of max and min
    c_sum = max_average + min_average
    c_diff = max_average - min_average
    contrast_avg = c_diff/c_sum
    contrast_std = sqrt(max_std**2 * (1/c_sum - (c_diff/c_sum**2))**2 + min_std**2 * (1/c_sum + (c_diff/c_sum**2))**2) #verified with ufloat (not a standard package)
        
    return contrast_avg, contrast_std, len(max)


def contrastTargetData(data, target, direction = 0, width_data = 1, width_lines = 1, show_plot = False, show_peaks = True, show_average = False, **kwargs):
    """Calculates the average Michelson line contrast and its standard deviation for 2D np-array of intensity data along an axis
    Function uses target data to indentify regions in which minima or maxima should located.

    Parameters
    ----------
    data : ndarray
        2D array with image data
    target : ndarray
        2D array with target data; orientation, pixel number and extent should match image data (or vice versa)
    direction : int, optional
        np-direction of the lines in 2D data
    width_data : int, optional
        relative size of whole data compared to width_lines (see below)
    width_lines : int, optional
        relative size of actual line data, use width_lines < width_data to crop of heavy noise around the lines
    show_plot : bool, optional
        whether to plot the data possibly including the found peaks and valleys and the average (see below)
    show_peaks : bool, optional
        draws the found max and min values for each line if show_plot = True
    show_average : bool, optional
        draws the average of the max and min values if show_plot = True

    Returns
    -------
    (contrast, contrast_std, N_maxima)
        The contrast (max-min)/(max+min) for the averaged max and min values for all lines,
        and the standard deviation of the contrast calculated from the spread of the max and min values
        and the number of found maxima
    
    """    

    if data.shape != target.shape:
        raise ValueError("data and target must have the same shape")

    data_average_1D = np.sum(data, axis = (1-direction))
    target_average_1D = np.sum(target, axis = (1-direction))

    #normalize target (expected values between 0 and 1)
    target_average_1D = target_average_1D/np.max(target_average_1D)

    #identify positions of lines and valleys
    target_high_pos = np.argwhere(target_average_1D > 0.5)
    target_low_pos = np.argwhere(target_average_1D < 0.5)

    #find groups of consecutive indices to identify individual peaks and valleys
    groups_high = [list(group) for group in mit.consecutive_groups(target_high_pos)]
    groups_low = [list(group) for group in mit.consecutive_groups(target_low_pos)]

    maxima = []
    minima = []

    max_pos = []
    min_pos = []

    for group in groups_high:
        peak = data_average_1D[group[0][0]:group[-1][0]+1]
        maxima.append(np.max(peak))
        max_pos.append(np.argmax(peak)+group[0][0])

    for group in groups_low:
        valley = data_average_1D[group[0][0]:group[-1][0]+1]
        minima.append(np.min(valley))
        min_pos.append(np.argmin(valley)+group[0][0])

    #calculate the average of all maxima and minima
    max_average = np.average(maxima)
    min_average = np.average(minima)

    #plot
    if show_plot:
        #plt.plot(np.linspace(-width_lines/2,width_lines/2,average_1D.size),average_1D)
        plt.plot(data_average_1D)
        if show_peaks:
            plt.plot(max_pos,maxima,'o')
            plt.plot(min_pos,minima,'o')
        if show_average:
           plt.hlines(max_average,0,data_average_1D.size -1)
           plt.hlines(min_average,0,data_average_1D.size -1)
        plt.ylim(0, np.max(maxima))
        plt.show()

    #calculate std. deviation
    max_std = sqrt(np.var(maxima))
    min_std = sqrt(np.var(minima))

    #calculate contrast and its uncertainty from the uncertainties of max and min
    c_sum = max_average + min_average
    c_diff = max_average - min_average
    contrast_avg = c_diff/c_sum
    contrast_std = sqrt(max_std**2 * (1/c_sum - (c_diff/c_sum**2))**2 + min_std**2 * (1/c_sum + (c_diff/c_sum**2))**2) #verified with ufloat (not a standard package)
        
    return contrast_avg, contrast_std, len(maxima)


if __name__ == "__main__":

    # basepath = path.dirname(__file__)
    
    #filepath = path.abspath(path.join(basepath, "..", "***REMOVED***"))
    #print(contrast(np.load(filepath), show_plot= True, auto_zoom= True, width_lines=1, use_ampd= True))

    target = createLines(20, size = 3*1.1, Nx = 1024, Ny = 1024, fillFactor = 1/1.1, direction = 1, super_exp = 12)
    print(contrastTargetData(target, target, direction = 1, show_plot= True, show_average= True))
    

    #print(contrast(np.load(filepath), show_plot= True, auto_zoom= True, width_lines=1, direction= 1, found_maxima= True))

    #filepath = path.abspath(path.join(basepath, "..", "***REMOVED***.txt"))
    #print(contrast(np.genfromtxt(filepath), show_plot= True))

    # plt.imshow(Lines)
    # plt.colorbar()
    # plt.show()
    # print(Lines.shape)
    # print(contrast(Lines, direction=1, show_plot=True))

    # lines = createLines(5, size = 1.1, fillFactor = 1/1.1, super_exp = 12)
    # lines = np.transpose(lines)


    # plt.set_cmap('inferno')
    # plt.axis('off')
    # plt.imshow(lines)


    # plt.show()
