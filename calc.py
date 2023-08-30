#  Copyright (c) 2022. Pokormi-Kota
"""
Tools to calculate specific results from vibration measurements
"""

import numpy as np
from numpy import array, zeros, log, sqrt
import pandas as pd
from scipy.stats import norm
from math import log10, pi

# Next lines add the module directory to the system path to import its other submodules
import sys, os, inspect
SCRIPT_DIR = os.path.abspath(os.path.dirname(inspect.getfile(inspect.currentframe())))
sys.path.append(os.path.dirname(SCRIPT_DIR))

from vibe import PyOctaveBand
from vibe.read_signal import sig_avg




def octaveband(data, fs, unit, F, fraction=3, axes=['X','Y','Z'], time=1, chunksize=None):
    """Makes octave filtering of a signal acording to ANSI and averages it, by dafault to 1 sec.

    Parameters
    ----------
    data : dict (axes: array_like)
        Contains 1-d array_like with a signal.
    fs : int
        Sample frequency.
    unit : str
        Units of input data, e.g. 'm/s' or 'g'
    F : list
        Frequencies for octave filtering. The first and the last values are only required.
    fraction : int, optional
        1 for 1/1 octave, fraction=3 for 1/3 octave, by default 3
    axes : list, optional
        Names of channels of signal, by default ['X','Y','Z']
    time : int, optional
        A number of seconds to average a signal
    chunksize : int, optional
        Max length of a data piece to process at once. Use if not enough RAM, by default None

    Returns
    -------
    vel : dict (axes: dataframe)
        Dataframe (rows: velocity values (mkm/s) averaged to 1 sec, cols: frequencies)
    acc : dict (axes: dataframe)
        Dataframe (rows: acceleration values (m/s2) averaged to 1 sec, cols: frequencies)
    """    
    
    if unit == 'm/s':
        vel = {}
        acc = {}
        if chunksize == None:
            chunks = [0, len(data[axes[0]])]
        else:
            chunks = list(range(0, len(data[ax]), chunksize)) + [len(data[ax])]
        for ax in axes:
            for i in range(len(chunks)-1):
                spl, freq, xb = PyOctaveBand.octavefilter(data[ax][chunks[i]:chunks[i+1]], fs=fs, 
                                                          fraction=fraction, order=6, limits=[F[0], F[-1]], show=0, sigbands=1)
                if time == 1/fs:
                    a = pd.DataFrame(xb, index=F).T
                else:
                    a = sig_avg(xb, fs=fs, Fv=F, time=time)
                if i == 0:
                    vel[ax] = a
                else:
                    vel[ax] = pd.concat([vel[ax], a])
            vel[ax] = vel[ax].reset_index(drop=True) 
            vel[ax].drop(vel[ax].tail(1).index, inplace=True)
            del spl, freq, xb, a

            acc[ax] = pd.DataFrame(data=None, index=None)
            for f in F:
                acc[ax].loc[:, f] = vel[ax].loc[:, f] * (2*pi*f)
            vel[ax] = vel[ax] *1e6
            
            print(f'{ax} has proceeded')
            
    else:
        acc = {}
        vel = {}
        if chunksize == None:
            chunks = [0, len(data[axes[0]])]
        else:
            chunks = list(range(0, len(data[axes[0]]), chunksize)) + [len(data[axes[0]])]
        for ax in axes:
            for i in range(len(chunks)-1):
                spl, freq, xb = PyOctaveBand.octavefilter(data[ax][chunks[i]:chunks[i+1]], fs=fs, 
                                                        fraction=fraction, order=6, limits=[F[0], F[-1]], show=0, sigbands=1)
                if time == 1/fs:
                    a = pd.DataFrame(xb, index=F).T
                else:
                    a = sig_avg(xb, fs=fs, Fv=F, time=time)
                if i == 0:
                    acc[ax] = a
                else:
                    acc[ax] = pd.concat([acc[ax], a])
            acc[ax] = acc[ax].reset_index(drop=True)

            del spl, freq, xb, a

            vel[ax] = pd.DataFrame(data=None, index=None)
            for f in F:
                vel[ax].loc[:, f] = acc[ax].loc[:, f] / (2*pi*f) *1e6
            print(f'{ax} has been processed')

    return vel, acc

def statistics(data, res_param, F, stat_params):
    """Calculates statistical parameters

    Parameters
    ----------
    data : dict of dataframes or a dataframe
        Dataset for statistics calculations.
    res_param : str
        Units for results, e.g. 'a', 'v', 'La', 'Lv'. Affects only the names in some of result rows. 
    F : list
        Frequencies to include in results, must be in data columns names.
    stat_params : list of int or str
        List of numbers or names of the parameters.
        [1 : Max,
        2 : Mean,
        3 : RMS,
        4 : Mean + Sigma,
        5 : Mean + 2 Sigma,
        6 : Mean + 1.645 Sigma,
        7 : Mean + 2.33 Sigma,
        8 : L95,
        9 : L99 ]
        
    Returns
    -------
    Stat : dict {axes: dataframes}
        Dataframes structure: (rows: chosen statistical parameters, cols: frequencies)
        
    Examples
    --------
    >>> import pandas as pd
    >>> x = pd.DataFrame(data={1:[1,2,3,4], 2:[1,2,3,4], 4:[1,2,3,4], 8:[1,2,3,4]})
    >>> y = pd.DataFrame(data={1:[0,1,2,3], 2:[0,1,2,3], 4:[0,1,2,3], 8:[0,1,2,3]})
    >>> dataxy = {'X':x, 'Y':y}
    >>> res = calc.statistics(dataxy, res_param='v', F=[1,2,4,8], stat_params=[1,2])
    >>> res
    {'X':         1    2    4    8
    Max   4.0  4.0  4.0  4.0
    Mean  2.5  2.5  2.5  2.5, 
    'Y':         1    2    4    8
    Max   3.0  3.0  3.0  3.0
    Mean  1.5  1.5  1.5  1.5}
    
    >>> data = x
    >>> calc.statistics(data, res_param='v', F=[1,2,4,8], stat_params=[1,2])
        1	2	4	8
    Max	4.0	4.0	4.0	4.0
    Mean	2.5	2.5	2.5	2.5
    
    """
    if type(data) == dict:
        Stat_max = {}
        axes = list(data)

        for ax in axes:
            Stat_max[ax] = _statistic(data[ax], res_param, F, stat_params)    
    else:
        Stat_max = _statistic(data, res_param, F, stat_params)
                
    return Stat_max

def _statistic(data, res_param, F, stat_params):
    
    Stat_max = pd.DataFrame(data=None, index=None)

    try:
        for f in F:
            if (1 or 'Max') in stat_params:
                Stat_max.loc['Max',f] = data.loc[:, f].max()
            if (2 or 'Mean') in stat_params:
                Stat_max.loc['Mean',f] = data.loc[:, f].mean()
            if (3 or 'RMS') in stat_params:
                Stat_max.loc['RMS',f] = ((data.loc[:, f]**2) .sum() /len(data)) **(1/2)
            if (4 or 'Mean+Sigma') in stat_params:
                Stat_max.loc['Mean+Sigma',f] = (data.loc[:, f].mean() + 
                                                norm.fit(data.loc[:, f])[1])
            if (5 or 'Mean+2Sigma') in stat_params:
                Stat_max.loc['Mean+2Sigma',f] = (data.loc[:, f].mean() + 
                                                norm.fit(data.loc[:, f])[1] * 2)
            if (6 or 'Mean+1.645Sigma') in stat_params:
                Stat_max.loc['Mean+1.645Sigma',f] = (data.loc[:, f].mean() + 
                                                    norm.fit(data.loc[:, f])[1] * 1.645)
            if (7 or 'Mean+2.33Sigma') in stat_params:
                Stat_max.loc['Mean+2.33Sigma',f] = (data.loc[:, f].mean() + 
                                                    norm.fit(data.loc[:, f])[1] * 2.33)
            if (8 or '95') in stat_params:
                Stat_max.loc[f'{res_param[-1].upper()}95',f] = data.loc[:, f].quantile(q=0.95)
            if (9 or '99') in stat_params:
                Stat_max.loc[f'{res_param[-1].upper()}99',f] = data.loc[:, f].quantile(q=0.99)
    except KeyError:
        raise()

#         if pribor == 5:   # replace bad values on 50 Hz for NI
#             for f in [50, 100, 160]:
#                 Stat_max[ax].loc[:, f] = 0.5 * (Stat_max[ax].loc[:, F[F.index(f)-1]] + Stat_max[ax].loc[:, F[F.index(f)+1]])

#         if pribor == 6:   # replace bad values on 50 Hz for SCADAS
#             for f in [50, 100, 160]:
#                 Stat_max[ax].loc[:, f] = 0.5 * (Stat_max[ax].loc[:, F[F.index(f)-1]] + Stat_max[ax].loc[:, F[F.index(f)+1]])
                
    return Stat_max

def trains_calc(data, peak_pos, unit='abs', half_interval = 20):
    """Calculates parameters for each trains/metro pass according to SP441.

    Parameters
    ----------
    data : array_like
        _description_
    peak_pos : list
        Peaks indexes
    unit : {'dB', other}, optional
        Units of input data. Necessary for correct formula of equivalents if input is in dB or absolute units, by default 'abs'.
    half_interval : int, optional
        Number of data points to use the left from detected peak and to the right
        to search the start and the end of train influence time, by default 20

    Returns
    -------
    tuple
        (v_max, v_eq, train_speed, v_idxmax, v_back, v_train_time)
        Values are defined per train
    v_max : dict (axes: dataframe)
        Dataframe (rows: peak values, cols: frequencies). Unit is the same as input
    v_eq : dict (axes: dataframe)
        Dataframe (rows: equivalent values, cols: frequencies). Unit is the same as input
    train_speed : dict (axes: dataframe)
        Dataframe (rows: speed, cols: frequencies). 
        Defined as [length / influence time]. Unit is km/h. 
        Train length is assumed 153.6 m (standart Moscow metro train)
    v_idxmax : dict (axes: dataframe)
        Dataframe (rows: indices of defined peak values, cols: frequencies). 
    v_back : dict (axes: dataframe)

    v_train_time : dict (axes: dataframe)
        
    """
    "unit - units of data"
    axes = list(data)
    F = data[axes[0]].columns

    v_max = {}
    v_idxmax = {}
    v_train_time = {}
    v_eq = {}
    v_minL = {}
    v_minR = {}
    train_speed = {}
    v_back_eq = {}
    v_back = {}

    for ax in axes:
        v_max[ax] = pd.DataFrame(data=None, index=None)
        v_idxmax[ax] = pd.DataFrame(data=None, index=None)
        v_train_time[ax] = pd.DataFrame(data=None, index=None)
        v_eq[ax] = pd.DataFrame(data=None, index=None)
        v_minL[ax] = pd.DataFrame(data=None, index=None)
        v_minR[ax] = pd.DataFrame(data=None, index=None)
        train_speed[ax] = pd.DataFrame(data=None, index=None)
        v_back_eq[ax] = pd.DataFrame(data=None, index=None)

        for i in range(len(peak_pos)):
            try:
                for j in F:
                    minL = data[ax].loc[peak_pos[i]-half_interval:
                                       peak_pos[i]-1, j].idxmin()
                    v_minL[ax].loc[i,j] = minL
                    minR = data[ax].loc[peak_pos[i]+1:
                                       peak_pos[i]+half_interval, j].idxmin()
                    v_minR[ax].loc[i,j] = minR
                    v_train_time[ax].loc[i,j] = (minR - minL) * 0.5
                    v_max[ax].loc[i,j] = data[ax].loc[peak_pos[i]-half_interval//3 :
                                                     peak_pos[i]+half_interval//3, j].max()
                    v_idxmax[ax].loc[i,j] = data[ax].loc[peak_pos[i]-half_interval//3 :
                                                        peak_pos[i]+half_interval//3, j].idxmax()
                    
                    if unit == 'dB':
                        v_eq[ax].loc[i,j] = 10* log10(
                            ( 10**(0.1*data[ax].loc[minL:minR, j]) ).sum() 
                            / ((minR-minL))
                            )
                        if (i>1) and (v_minR[ax].loc[i-1,j] < v_minL[ax].loc[i,j]):
                            v_back_eq[ax].loc[i,j] = 10* log10(
                                ( 10**(0.1*data[ax].loc[ v_minR[ax].loc[i-1,j] 
                                                        : v_minL[ax].loc[i,j], j ]) ).sum()
                                / (v_minL[ax].loc[i,j] - v_minR[ax].loc[i-1,j])
                                )
                    else:
                        v_eq[ax].loc[i,j] = (
                            (data[ax].loc[minL:minR, j]**2).sum() 
                            / (minR - minL)
                        ) **(1/2)
                        if (i>1) and (v_minR[ax].loc[i-1,j] < v_minL[ax].loc[i,j]):
                            v_back_eq[ax].loc[i,j] = (
                                (data[ax].loc[ v_minR[ax].loc[i-1,j] 
                                             : v_minL[ax].loc[i,j], j ]**2 ).sum()
                            / (v_minL[ax].loc[i,j] - v_minR[ax].loc[i-1,j])
                            ) **(1/2)
                            
                    train_speed[ax].loc[i,j] = 552.96 / v_train_time[ax].loc[i,j]   # km/h, train length=153.6
                    
                    if v_train_time[ax].loc[i,j] < 7:
                        v_train_time[ax].loc[i,j] = 'н/д'
                        train_speed[ax].loc[i,j] = 'н/д'

            except ValueError:
                print(f'Пик {i} исключен из списка, так как он слишком близко к краю записи')
            except TypeError:
                print(f'Частота {j} Гц не существует / или неправильные названия строк в data')
                        
        # Background result
        v_back[ax] = pd.Series(data=None, index=None, dtype='float64')
        try:
            for f in F:
                t = 0
                for i in v_back_eq[ax].index:
                    if not np.isnan(v_back_eq[ax].loc[i,f]):
                        t += 1
                        if unit == 'dB':
                            v_back[ax].loc[f] = 10* log10( 1/t *
                                        ( 10**(0.1*v_back_eq[ax].loc[:,f]) ).sum()
                                                            )
                        else:
                            v_back[ax].loc[f] = ( 1/t * ( v_back_eq[ax].loc[:,f]**2 ).sum() )**(1/2)
        except KeyError:
            pass

    return (v_max, v_eq, train_speed, v_idxmax, v_back, v_train_time)

def EnterPSD(a, b, flim=None):
    """Returns cumulative rms of psd

    Parameters
    ----------
    a : 1-D array
        frequency column
    b : 1-D array
        PSD column. It is considered to be in ((m/sec^2)^2/Hz) units to obtain desired units in the result.
        Other input units are up to user's consideration.
    flim : _type_, optional
        Frequencies limits. The first value greater than `flim[1]` is uncluded if exists.
        
    Returns
    -------
    a : 1-D array
    
    b : 1-D array
    
    rms :
    
    crms :
    
    slope :
    
    """
    
    print (" ")
    print (" The input file must have two columns: \n freq(Hz) & accel((m/sec^2)^2/Hz)")
    # file dialog was used in original version
    # print (" (Find dialog box) ")

    # a,b,num =read_two_columns_from_dialog('Select PSD file')

    # print ("\n samples = %d " % num)

    
    if flim != None:
        for i in range (0, len(a)):
            if (flim[0] < a[i]):
                a1 = i
                break
        a2 = None
        for i in range (0, len(a)):
            if (flim[1] < a[i]):
                a2 = i+1   #include first value greater than flim[1]
                break
        if a2 == None:
            a2 = len(a)

        a = array(a[a1:a2])
        b = array(b[a1:a2])

    else:
        a=array(a[1:])
        b=array(b[1:])
    

    num=len(a)
    nm1=num-1
    slope =zeros(nm1,'f')

    ra=0
    carms = zeros(nm1,'f')
    for i in range (0,nm1):

        s=log(b[i+1]/b[i])/log(a[i+1]/a[i])
        
        slope[i]=s

        if s < -1.0001 or s > -0.9999:
            ra+= ( b[i+1] * a[i+1]- b[i]*a[i])/( s+1.)
        else:
            ra+= b[i]*a[i]*log( a[i+1]/a[i])
        carms[i] = sqrt(ra)

    omega=2*pi*a

    bv = zeros(num,'f') 
    bd = zeros(num,'f') 
        
    for i in range (0,num):         
        bv[i]=b[i]/omega[i]**2
     
    # bv=bv*386**2   # *386**2 to convert from g to in/s^2
    rv=0
    cvrms = zeros(nm1,'f')
    for i in range (0,nm1):
        s=log(bv[i+1]/bv[i])/log(a[i+1]/a[i])

        if s < -1.0001 or s > -0.9999:
            rv+= ( bv[i+1] * a[i+1]- bv[i]*a[i])/( s+1.)
        else:
            rv+= bv[i]*a[i]*log( a[i+1]/a[i])         
        cvrms[i] = sqrt(rv)
        
    for i in range (0,num):         
        bd[i]=bv[i]/omega[i]**2
     
    rd=0
    cdrms = zeros(nm1,'f')
    for i in range (0,nm1):
        s = log(bd[i+1]/bd[i]) / log(a[i+1]/a[i])
        
        if s < -1.0001 or s > -0.9999:
            rd += ( bd[i+1] * a[i+1]- bd[i]*a[i]) / ( s+1.)
        else:
            rd += bd[i]*a[i]*log( a[i+1]/a[i])         
        cdrms[i] = sqrt(rd)

    grms=sqrt(ra)
    vrms=sqrt(rv)
    drms=sqrt(rd)
    
    print (" ")
    print (" *** Input PSD *** ")
    print (" ")
    print (" Acceleration ")
    print ("   Overall = %10.3g m/sec^2 RMS" % grms)

    print (" ")
    print (" Velocity ") 
    print ("   Overall = %10.3g m/sec rms" % vrms)

    print (" ")
    print (" Displacement ") 
    print ("   Overall = %10.3g m rms" % drms)

    return a[:-1], (b[:-1], bv[:-1], bd[:-1]), (grms, vrms, drms), (carms[::-1], cvrms[::-1], cdrms[::-1]), slope

