#  Copyright (c) 2022. Pokormi-Kota
"""
Functions to prepare data from tests
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


def selectfiles():
    import tkinter as tk
    from tkinter import filedialog

    root = tk.Tk()
    root.withdraw()
    files = filedialog.askopenfilenames(parent=root, initialdir='', title='Select record files')
    print(files)
    return files

def readsignal(device, files, unit='m/s2', axes=['X','Y','Z'], data_range=None, points=None, cut=None, 
               show=False, project_name=None):
    """
     Reads data from record files and converts it to usable format.

    Parameters
    ----------
    device : int
        1 - ZetLab
        2 - SS-box
        3 - Ecofizika
        4 - National Instruments (NI)
        5 - SCADAS as .mat
    files : path, list of paths
        Initial file(s) to use.
    unit : str, optional
        Units of data. 'g' or 'm/s2'. Default is 'm/s2'.
    fraction : int, optional
        Bandwidth. 1/3-octave fraction=3, 1-octave fraction=1. Default is 3.
    axes : list of str, optional
        Axes names and order. Default is ['X','Y','Z'].
    points : list of str, optional
        For multichannel devices. Points names and order in the record.
    data_range : int, optional
        Only for SS-box. Vibration measurement range.
    
    Returns
    -------
    Signal by directions : dataframe
        Dataframe of the same length as `data` and columns as axes.
        Unit g is converted to m/s^2.
    """
    # Check if all files are right format
    _checkformat(files, device)

    fs=None

    if device == 1:
        fs = 1000
        data = _read_zetlab(files, axes, points, cut, project_name)
    elif device == 2:
        data, fs = _read_ssbox(files, axes, data_range)
    elif device == 3:
        data, fs = _read_ecofizika(files, unit, axes)
    elif device == 4:
        data = _read_NI(files, unit, axes, points)
    elif device == 5:
        data = _read_scadas(files, unit, axes, points)
    else:
        raise NotImplementedError('Unknown device')

    if show:
        _showplot(axes, data, fs)

    if fs:
        return data, fs
    else:
        return data


def _checkformat(files, device):
    """ Check for files to have wright extension """
    import os.path
    import re
    ext = { 1:'.csv', 2:'.11\d', 3:'.csv', 4:'.lvm', 5:'.mat' }
    for file in files:
        extension = os.path.splitext(file)[-1]
        if not re.match(ext[device], extension):
            raise NameError(f'{file} extension mismatch the chosen device')
        else:
            pass


def cut_edges(data, cut):
    """Cuts first and last n values"""
    if type(cut) == int:
        cut = [cut, cut]
    data = data.iloc[cut[0] : -1*cut[1]]
    return data.reset_index(drop=True)


def cut_middle(data, cut=None):
    """Delete values in the middle of the record

    Args:
        data (dataframe): _description_
        cut (list of two int, optional): _description_. Defaults to None.
    
    """
    data.drop(index=range(cut[0] , cut[1]), inplace=True)
    return data.reset_index(drop=True)


def _read_zetlab(files, axes, points, cut, project_name):
    """
    Reads data from ZetLab (Bochka)
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
        if vibration.iloc[start, points.index(project_name)*3] != 0:
            break
    
    # Finding the end of the record
    cut_end = cut[1]*fs
    for end in range(cut_end, len(vibration)):
        if vibration.iloc[-1*end, points.index(project_name)*3] != 0:
            break
    
    data = {}
    for ax in axes:
        data[ax] = vibration.iloc[start+30*fs:-1*(end+30*fs), points.index(project_name)*3 + axes.index(ax)].reset_index(drop=True)
        
    return data, fs


def _read_ssbox(files, axes, data_range):
    """ Reads data from SanSanych Boxes. """

    # Get settings from set.txt file in the same folder as record files
    for file in os.listdir(os.path.dirname(files[0])):
        if file.startswith("set"):
            sets = os.path.join(os.path.dirname(files[0]), file)
    with open(sets) as f:
        lines = f.readlines()
        fs = float(lines[6])
        data_range = int(lines[8][:-2])
    
    bits = 16;      # bits
    dt = np.dtype('int16')

    data = [np.fromfile(file, dtype=dt) for file in files]
    vibration = pd.DataFrame(np.concatenate(data)).reset_index(drop=True)
    data = pd.DataFrame()
    for ax in axes:
        data.loc[:,ax] = (vibration.loc[axes.index(ax)::3,:].reset_index(drop=True)
         /(2**bits /(2*data_range))*9.81523)

    return data, fs


def _read_ecofizika(files, unit, axes):
    vibration = pd.read_csv(files[0], sep='\t', encoding='mbcs', header=None, names = None,
                           skiprows=4).reset_index(drop=True)
    inf = pd.read_csv(files[0], sep=' ', encoding='mbcs', header=None, names = None,
                           skiprows=2, nrows=1).reset_index(drop=True)

    fs = inf.iloc[0, 2]
    data = {}
    for ax in axes:
        if unit == 'm/s2':
            data[ax] = vibration.iloc[:,axes.index(ax)+1].reset_index(drop=True)
        elif unit == 'g':
            data[ax] = vibration.iloc[:,axes.index(ax)+1].reset_index(drop=True)*9.81523
    return data, fs
    
    
# if pribor == 5:
#     points = ['T1','T2','T3','T4']
    
#     if project_name not in points:
#         raise NameError('Название файла не соответствует точке измерения')
    
#     if read_once == False:
#         vibration_df = [pd.read_table(file, header=None, engine='python', skiprows = 24)
#                      for file in vibration_files]
        
#         # Merge all vibration files into one
#         vibration = pd.concat(vibration_df).reset_index(drop=True)
        
#         read_once = True
#         del vibration_df
    
#     vibration_list = {}
    
#     cut_end = 1000   # must be >0
    
#     if unit == 'g':
#         vibration_list['X'] = detrend(vibration.iloc[:-cut_end ,points.index(project_name)*3+1].reset_index(drop=True)*9.81523)
#         vibration_list['Y'] = detrend(vibration.iloc[:-cut_end ,points.index(project_name)*3+2].reset_index(drop=True)*9.81523)
#         vibration_list['Z'] = detrend(vibration.iloc[:-cut_end ,points.index(project_name)*3+3].reset_index(drop=True)*9.81523)
        
#     elif unit == 'm/s2':
#         vibration_list['Z'] = vibration.loc[:,points.index(project_name)*3+1].reset_index(drop=True)
#         vibration_list['X'] = vibration.loc[:,points.index(project_name)*3+2].reset_index(drop=True)
#         vibration_list['Y'] = vibration.loc[:,points.index(project_name)*3+3].reset_index(drop=True)
        
    
# if pribor == 6:
#     import scipy.io as sio
    
#     points = ['T9','T10','T11']
    
#     if project_name not in points:
#         raise NameError('Название файла не соответствует точке измерения')
    
#     if read_once == False:
#         matfiles = [sio.loadmat(file) for file in vibration_files]
        
#         read_once = True
    
#     vibration_list = {}
    
#     if unit == 'g':
# #         vibration_list['X'] = pd.Series(np.concatenate(
# #             [matfiles[file][list(matfiles[0])[points.index(project_name)*3+4]]['y_values'][0][0][0][0][0] 
# #              for file in range(len(matfiles))], 
# #             axis=0).flatten()*9.81523).drop(index=range(round(8276.1*fs),round(8276.6*fs))).iloc[10*fs:].reset_index(drop=True)
        
#         vibration_list['Y'] = pd.Series(np.concatenate(
#             [matfiles[file][list(matfiles[0])[points.index(project_name)*3+4]]['y_values'][0][0][0][0][0] 
#              for file in range(len(matfiles))], 
#             axis=0).flatten()*9.81523).drop(
#             index=range(round(8379*fs),round(8379.8*fs))
#         ).iloc[10*fs:].reset_index(drop=True)
        
#         vibration_list['Z'] = pd.Series(np.concatenate(
#             [matfiles[file][list(matfiles[0])[points.index(project_name)*3+5]]['y_values'][0][0][0][0][0] 
#              for file in range(len(matfiles))], 
#             axis=0).flatten()*9.81523).drop(
#             index=range(round(8379*fs),round(8379.8*fs))
#         ).iloc[10*fs:].reset_index(drop=True)

def _showplot(axes, data, fs):
    import matplotlib.pyplot as plt
    for ax in axes:
        fig, axs = plt.subplots(figsize=(10, 5), tight_layout=True)
        x = np.arange(0,  len(data[ax])) / fs
        y = data[ax]
        
        axs.plot(x, y, linewidth=0.5, label='$ Сигнал $')
        
        axs.set_title(f'$ Направление\ {ax} $', fontsize=16)
        axs.set_xlabel('$ Время,\ с $', fontsize=12)
        axs.set_ylabel('$ Ускорение,\ м/с{^2} $', fontsize=12)
        axs.legend(bbox_to_anchor=(1, 1), loc='upper left', fontsize=10, frameon=True)
        axs.grid(visible='True', which='both', axis='both', ls='--')

        plt.show()


def sig_avg(data, fs, Fv=None, time=1):
    """
    Parameters
    ----------
    data : array like 
        [axes[0] = rows, axes[1] = frequencies]
    fs : sample frequency
    Fv : list of frequencies
        or None if not applied
    time : int or float in sec, e.g.:
            1 = 1 sec
            4 = 4 sec
            0.125 = 125 ms
    
    Returns
    -------
    RMS - Dataframe[rows, frequencies]
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
                rms.loc[i, f] = (
                    (sum(data[j][(fs*time)*i : (fs*time)*(i+1)] **2) / (fs*time) 
                    ) **(1/2))
    return rms


def db2val(data, param='a'):
    """Convert levels (dB) to absolute values"""
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
        

def val2db(data, param='a'):
    """Convert absolute values to levels (dB)"""
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
    import IPython
   
    url = 'https://c.tenor.com/Z6gmDPeM6dgAAAAC/dance-moves.gif'
    gif = IPython.display.Image(url, width = 250)

    return gif
