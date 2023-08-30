#  Copyright (c) 2022. Pokormi-Kota
"""
Functions to read files from measurement equipment & some basic functions to correct it.

See also
^^^^^^^^
Measurement equipment `Files Structure Standard <some_hyperlink>` by DynamicSystems co.
"""

import numpy as np
import pandas as pd
import os

# Public methods
__all__ = ['selectfiles', 'readsignal', 'cut_edges', 'cut_middle',
'db2val','val2db', 'sig_avg', 'surprise']


F1_1 = [1, 2, 4, 8, 16, 31.5, 63, 125, 250, 500, 1000, 2000, 4000, 8000, 16000]
F1_3 = [0.8, 1, 1.25, 1.6, 2, 2.5, 3.15, 4, 5, 6.3, 8, 10, 12.5, 16, 20, 25, 31.5, 40, 50,
        63, 80, 100, 125, 160, 200, 250, 315, 400, 500, 630, 800, 1000, 1250, 1600,
       2000, 2500, 3150, 4000, 5000, 6300, 8000, 10000, 12500, 16000, 20000]


def selectfiles ():
    """Opens Tkinter filedialog to select one or more files.
    If you need to select multiple files at once they must be located in the same directory."""
    
    import tkinter as tk
    from tkinter import filedialog

    root = tk.Tk()
    root.withdraw()
    files = filedialog.askopenfilenames(parent=root, initialdir='', title='Select record files')
    print(files)
    return files

def readsignal (device, files, unit='m/s2', axes=['X','Y','Z'], points=None, point_name=None, cut=None, 
               show=False):
    """
     Reads data from record files and returns it as {axes[0] : pandas.Series0, axes[1] : pandas.Series1}

    Parameters
    ----------
    device : int
        1 - ZetLab
        2 - SS-box
        3 - Ecofizika
        4 - GtLab
        5 - National Instruments (NI) (will be added in future versions)
        6 - SCADAS as .mat (will be added in future versions)
    files : path or list of paths
        Input file(s) to read.
    unit : {'g','m/s2','other'}, optional
        Units of data. 'g' or 'm/s2' (if 'm/s', it is defined automatically). Default is 'm/s2'.
    axes : list of str, optional
        Ordered names of signal channels. Default is ['X','Y','Z'].
    points : list of str, optional
        For data with multiple measured points in one file, points names and order in the record. 
        Used with Zetlab, NI and SCADAS devices.
    point_name : str, optional
        Name of the point. Must be in `points`, otherwise raises an error.
        
    
    Returns
    -------
    Signal by directions : dict of pandas.Series
        Keys are channels names, values are time-signals in pandas.Series format of the same length as `data`.
        Unit g is converted to m/s^2.
    """
    
    _checkformat(files, device)

    fs=None

    if device == 1:
        fs = 1000
        data = _read_zetlab(files, axes, points, cut, point_name)
    elif device == 2:
        data, fs = _read_ssbox(files, unit, axes)
    elif device == 3:
        data, fs = _read_ecofizika(files, unit, axes)
    elif device == 4:
        data, fs = _read_gtlab(files, unit, axes)
    elif device == 5:
        data = _read_NI(files, unit, axes, points)
    elif device == 6:
        data = _read_scadas(files, unit, axes, points, point_name)
    else:
        raise NotImplementedError('Unknown device')

    if show:
        _showplot(axes, data, fs, unit)

    if fs:
        return data, fs
    else:
        return data


def _checkformat (files, device):
    """ 
    Checks if files have the correct extension according to 'Files Structure Standard'.
    Raises NameError if extension mismatch the chosen device
    """
    import os.path
    import re
    ext = { 1:'.csv', 2:'.\d{3}',  3:'.csv', 4:'.txt', 5:'.lvm', 6:'.mat' } #2:'.txt',
    for file in files:
        extension = os.path.splitext(file)[-1]
        if not re.match(ext[device], extension):
            raise NameError(f'{file} extension mismatch the chosen device')
        else:
            pass


def cut_edges (data, cut):
    """Delete first and last n values

    Parameters
    ----------
    data : pandas.DataFrame or pandas.Series
        Input data
    cut : list of two int or int
        List with two values - the first refers to a number of values to delete in the beginning 
        and the second is the number of values to delete at the end. If int provided then 
        deletes equal number of values 

    Returns
    -------
    out : pandas.DataFrame or pandas.Series
        `data` without first and/or last values
        
    Examples
    --------
    >>> import pandas as pd
    >>> data = pd.Series([1,2,3,4,5,6,7,8])
    >>> new_data = vibe.read_signal.cut_edges(data, cut=[1, 2])
    >>> print(new_data)
    pandas.Series ([2, 3, 4, 5, 6])

    >>> new_data = vibe.read_signal.cut_edges(data, cut=2)
    >>> print(new_data)
    pandas.Series ([3, 4, 5, 6])

    """
    if type(cut) == int:
        cut = [cut, cut]
    data = data.iloc[cut[0] : -1*cut[1]]
    return data.reset_index(drop=True)


def cut_middle(data, cut):
    """Delete values in the middle of the record

    Parameters
    ----------
    data : pandas.DataFrame or pandas.Series
        Input data
    cut : list of two int or int
        List with two values - the first refers to the start index of interval to delete  
        and the second is the last index of interval to delete. If int provided then 
        deletes one value with index = `cut`.
        
    Returns
    -------
    out : pandas.DataFrame or pandas.Series
        `data` with some interval in the middle dropped
        
    Examples
    --------
    >>> import pandas as pd
    >>> data = pd.Series([1,2,3,4,5,6,7,8])
    >>> new_data = vibe.read_signal.cut_middle(data, cut=[3, 6])
    >>> print(new_data)
    pandas.Series ([1,2,3,7,8])
    
    >>> new_data = vibe.read_signal.cut_middle(data, cut=2)
    >>> print(new_data)
    pandas.Series ([1,2,4,5,6,7,8])
    
    """
    if type(cut) == int:
        cut = [cut, cut]
    data.drop(index=range(cut[0] , cut[1]), inplace=True)
    return data.reset_index(drop=True)


def _read_zetlab(files, axes, points, cut, point_name):
    """
    Reads data from ZetLab (aka Bochka)
    """
    if points == None:
        raise NotImplementedError("Oooops, you've forgotten about points names")
    
    fs = 1000

    vibration_df = [pd.read_csv(file, header=None) for file in files]

    # Merge all vibration files into one df
    vibration = pd.concat(vibration_df).reset_index(drop=True)
    del vibration_df
    
    if cut == None:
        cut = [0,0]
        
    # Finding the beginning of the record
    cut_start = cut[0]*fs   # to skip first seconds of the record
    for start in range(cut_start, len(vibration)):
        if vibration.iloc[start, points.index(point_name)*3] != 0:
            break
    
    # Finding the end of the record
    cut_end = cut[1]*fs
    for end in range(cut_end, len(vibration)):
        if vibration.iloc[-1*end, points.index(point_name)*3] != 0:
            break
    
    data = {}
    for ax in axes:
        data[ax] = vibration.iloc[start+30*fs:-1*(end+30*fs), points.index(point_name)*3 + axes.index(ax)].reset_index(drop=True)
        
    return data, fs


def _read_ssbox(files, unit, axes):
    """ Reads data from SanSanych Boxes. """

    # Get settings from set.txt file in the same folder as record files
    for file in os.listdir(os.path.dirname(files[0])):
        if file.startswith("set"):
            sets = os.path.join(os.path.dirname(files[0]), file)
    try:
        with open(sets) as f:
            lines = f.readlines()
            fs = int(lines[5][5:])
            data_range = int(lines[7][7])
    except UnboundLocalError:
        import warnings
        warnings.warn("File with set parameters not found, used fs=80, please, input data_range", UserWarning)
        fs = 80
        data_range = int(input('data_range = '))
    
    bits = 16;      # bits
    dt = np.dtype('int16')
    
    data = {}
    if unit == 'g':
        data1 = [np.fromfile(file, dtype=dt) for file in files]
        vibration = pd.DataFrame(np.concatenate(data1)).reset_index(drop=True)
        for ax in axes:
            data[ax] = (vibration.loc[axes.index(ax)::3,:].reset_index(drop=True)
                        /(2**bits /(2*data_range))*9.81523)

    elif unit == 'strain':
        data1 = [pd.read_table(file, header=None, sep=',', on_bad_lines='warn') for file in files]
        vibration = pd.concat(data1).reset_index(drop=True)
        for ax in axes:
            data[ax] = vibration.iloc[:,axes.index(ax)+1] / (2**bits /(2*data_range))

    return data, fs


def _read_ecofizika(files, unit, axes):
    """Reads data from Ecofizika (Octava)"""
    # vibration = pd.read_csv(files[0], sep='\t', encoding='mbcs', header=None, names = None,
    #                        skiprows=4).reset_index(drop=True)
    data = [pd.read_csv(file, sep='\t', encoding='mbcs', header=None, names = None,
                           skiprows=4).reset_index(drop=True) for file in files]
    vibration = pd.concat(data).reset_index(drop=True)
    inf = pd.read_csv(files[0], sep=' ', encoding='mbcs', header=None, names = None,
                           skiprows=2, nrows=1).reset_index(drop=True)

    fs = int(inf.iloc[0, 2])
    data = {}
    for ax in axes:
        if unit == 'm/s2':
            data[ax] = vibration.iloc[:,axes.index(ax)+1].reset_index(drop=True)
        elif unit == 'g':
            data[ax] = vibration.iloc[:,axes.index(ax)+1].reset_index(drop=True)*9.81523
    return data, fs
    
    
def _read_gtlab(files, unit, axes):
    """ Reads data from GtLab sensors (D003, D004)"""
    
    data = [pd.read_table(file, header=None, sep=' ', encoding='mbcs', skiprows=1) for file in files]
    vibration = pd.concat(data).reset_index(drop=True)
    data = {}
    for ax in axes:
        if unit == 'm/s2':
            data[ax] = vibration.iloc[:,axes.index(ax)+1]
        elif unit == 'g':
            data[ax] = vibration.iloc[:,axes.index(ax)+1] *9.81523
    fs = int(1 / (vibration.iloc[1,0] - vibration.iloc[0,0]))
    return data, fs


# def _read_NI(files, unit, axes, points):
    
#     if points == None:
#         raise NotImplementedError("Oooops, you've forgotten about points names")
    
#     if point_name not in points:
#         raise NameError('Название файла не соответствует точке измерения')
    
#     if read_once == False:
#         vibration_df = [pd.read_table(file, header=None, engine='python', skiprows = 24)
#                      for file in files]
        
#         # Merge all vibration files into one
#         vibration = pd.concat(vibration_df).reset_index(drop=True)
        
#         read_once = True
#         del vibration_df
    
#     cut_end = 1   # must be >0
    
#     data = {}
    
#     if unit == 'g':
#         for ax in axes:
#             data[ax] = vibration.iloc[:-cut_end, points.index(point_name)*3+axes.index(ax)+1].reset_index(drop=True)*9.81523
        
#     elif unit == 'm/s2':
#         for ax in axes:
#             data[ax] = vibration.iloc[:-cut_end, points.index(point_name)*3+axes.index(ax)+1].reset_index(drop=True)
        
#     return data, fs
        
    
def _read_scadas(files, unit, axes, points, point_name):
    import scipy.io as sio
    
    if points == None:
        raise NotImplementedError("Oooops, you've forgotten about points names")
    if point_name not in points:
        raise NameError('Название файла не соответствует точке измерения')
    
    matfiles = [sio.loadmat(file) for file in files]
        
    
    vibration_list = {}
    
    if unit == 'g':
#         vibration_list['X'] = pd.Series(np.concatenate(
#             [matfiles[file][list(matfiles[0])[points.index(point_name)*3+4]]['y_values'][0][0][0][0][0] 
#              for file in range(len(matfiles))], 
#             axis=0).flatten()*9.81523).drop(index=range(round(8276.1*fs),round(8276.6*fs))).iloc[10*fs:].reset_index(drop=True)
        
        vibration_list['Y'] = pd.Series(np.concatenate(
            [matfiles[file][list(matfiles[0])[points.index(point_name)*3+4]]['y_values'][0][0][0][0][0] 
             for file in range(len(matfiles))], 
            axis=0).flatten()*9.81523).drop(
            index=range(round(8379*fs),round(8379.8*fs))
        ).iloc[10*fs:].reset_index(drop=True)
        
        vibration_list['Z'] = pd.Series(np.concatenate(
            [matfiles[file][list(matfiles[0])[points.index(point_name)*3+5]]['y_values'][0][0][0][0][0] 
             for file in range(len(matfiles))], 
            axis=0).flatten()*9.81523).drop(
            index=range(round(8379*fs),round(8379.8*fs))
        ).iloc[10*fs:].reset_index(drop=True)

def _showplot (axes, data, fs, unit):
    """Show plots of input data.

    Parameters
    ----------
    axes : _type_
        _description_
    data : _type_
        _description_
    fs : _type_
        _description_
    """
    import matplotlib.pyplot as plt
    for ax in axes:
        fig, axs = plt.subplots(figsize=(10, 5), tight_layout=True)
        x = np.arange(0,  len(data[ax])) / fs
        y = data[ax]
        
        axs.plot(x, y, linewidth=0.5, label='$ Сигнал $')
        
        axs.set_title(f'$ Направление\ {ax} $', fontsize=16)
        axs.set_xlabel('$ Время,\ с $', fontsize=12)
        if unit == 'm/s':
            axs.set_ylabel('$ Ускорение,\ м/с{^2} $', fontsize=12)
        axs.legend(bbox_to_anchor=(1, 1), loc='upper left', fontsize=10, frameon=True)
        axs.grid(visible='True', which='both', axis='both', ls='--')

        plt.show()


def sig_avg (data, fs, Fv=None, time=1):
    """
    Returns the rms of an input.
    
    Parameters
    ----------
    data : array like 
        [axes[0] = rows, axes[1] = frequencies]
    fs : float or int
        sample frequency
    Fv : list of numbers
        list of frequencies or None if not applied, default is None
    time : int or float 
        Time to make averaging to (sec), e.g.:
            1 = 1 sec
            4 = 4 sec
            0.125 = 125 ms
    
    Returns
    -------
    RMS - Dataframe
        Structure: [rms, frequencies]
    """ 
    
    rms = pd.DataFrame(data=None, index=None)

    if Fv == None:
        for i in range(len(data)//(fs*time)):  
            rms.loc[i] = ( 
                (sum(data[(fs*time)*i : (fs*time)*(i+1)] **2) /(fs*time) 
                ) **(1/2))
    else:
        for j in range(len(data)):
            f = Fv[j]
            
            for i in range(int(len(data[j])//(fs*time))):
                rms.loc[i, f] = np.sqrt(np.mean(np.square(data[j][int(fs*time*i) : int(fs*time*(i+1))])))
                    # (sum(data[j][int(fs*time*i) : int(fs*time*(i+1))] **2) / (fs*time) 
                    # ) **(1/2))   # old direct approach
    return rms


def db2val (data, param='a'):
    """Convert levels (dB) to absolute values.

    Parameters
    ----------
    data : array_like
        Input data in m, m/s or m/s^2 units
    param : {'a','v','d'}, optional
        'a' - acceleration
        'v' - velocity
        'd' - displacement
        Type of input data, by default 'a'

    Returns
    -------
    array_like
        Input data in Db units
    """
    a0 = 1e-6   # m/s2
    v0 = 5e-8  # m/s
    d0 = 1e-9  # m - not in GOST!
    
    if param == 'a':
        return (10** (data/20)) * a0
    elif param == 'v':
        return (10** (data/20)) * v0
    elif param == 'd':
        return (10** (data/20)) * v0
    else:
        print('Unknown parameter')
        

def val2db (data, param='a'):
    """Convert absolute values to levels (dB)
    
    Parameters
    ----------
    data : array like
        Input data in dB units
    param : {'a','v','d'}, optional
        'a' - acceleration
        'v' - velocity
        'd' - displacement
        Type of input data, by default 'a'

    Returns
    -------
    array like
        Input data in m, m/s or m/s^2 units, according to selected `param`
    """
    a0 = 1e-6   # m/s2
    v0 = 5e-8  # m/s
    d0 = 1e-9  # m - not in GOST!
    
    if param == 'a':
        return 20 * np.log10(data / a0)
    elif param == 'v':
        return 20 * np.log10(data / v0)
    elif param == 'd':
        return 20 * np.log10(data / d0)
    else:
        print('Unknown parameter')


def surprise():
    """Call this function to see a surprise"""
    import IPython
   
    url = 'https://c.tenor.com/Z6gmDPeM6dgAAAAC/dance-moves.gif'
    gif = IPython.display.Image(url, width = 250)

    return gif
