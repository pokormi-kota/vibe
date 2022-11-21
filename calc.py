import numpy as np
import pandas as pd
from scipy.stats import norm
from math import log10, pi
import PyOctaveBand
from .read_signal import sig_avg

def octaveband(data, fs, unit, F, fraction=3, axes=['X','Y','Z'], chunksize=None):
    """Makes octave filtering of a signal acording to ANSI and averages it to 1 sec.

    Parameters
    ----------
    data : dict (axes: array_like)
        Contains 1-d array_like with a signal.
    fs : int
        Sample frequency.
    unit : str
        Units of input data, e.g. 'm/s' or 'g'
    F : list
        Frequencies for octave filtering. First and last values required.
    fraction : int, optional
        1 for 1/1 octave, fraction=3 for 1/3 octave, by default 3
    axes : list, optional
        Names of channels of signal, by default ['X','Y','Z']
    chunksize : int, optional
        Use if not enough RAM, by default None

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
                a = sig_avg(xb, fs=fs, Fv=F, time=1)
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
            chunks = list(range(0, len(data[ax]), chunksize)) + [len(data[ax])]
        for ax in axes:
            for i in range(len(chunks)-1):
                spl, freq, xb = PyOctaveBand.octavefilter(data[ax][chunks[i]:chunks[i+1]], fs=fs, 
                                                        fraction=fraction, order=6, limits=[F[0], F[-1]], show=0, sigbands=1)
                a = sig_avg(xb, fs=fs, Fv=F, time=1)
                if i == 0:
                    acc[ax] = a
                else:
                    acc[ax] = pd.concat([acc[ax], a])
            acc[ax] = acc[ax].reset_index(drop=True)

            del spl, freq, xb, a

            vel[ax] = pd.DataFrame(data=None, index=None)
            for f in F:
                vel[ax].loc[:, f] = acc[ax].loc[:, f] / (2*pi*f) *1e6
            print(f'{ax} has proceeded')

    return vel, acc

def statistics(data, res_param, F, stat_params, axes=['X','Y','Z']):
    """Calculates statistical parameters

    Parameters
    ----------
    data : dict of dfs
        Dataset for statistics calculations.
    res_param : str
        Units for results, e.g. 'a', 'v', 'La', 'Lv'. Affects only the names in some of resulting rows. 
    F : list
        Frequencies to include in results, must be in data columns names.
    stat_params : list
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
    axes : list, optional
        Channels of data, by default ['X','Y','Z']
    """
    Stat_max = {}

    for ax in axes:
        Stat_max[ax] = pd.DataFrame(data=None, index=None)

        for f in F:
            if (1 or 'Max') in stat_params:
                Stat_max[ax].loc['Max',f] = data[ax].loc[:, f].max()
            if (2 or 'Mean') in stat_params:
                Stat_max[ax].loc['Mean',f] = data[ax].loc[:, f].mean()
            if (3 or 'RMS') in stat_params:
                Stat_max[ax].loc['RMS',f] = ((data[ax].loc[:, f]**2) .sum() /len(data[ax])) **(1/2)
            if (4 or 'Mean+Sigma') in stat_params:
                Stat_max[ax].loc['Mean+Sigma',f] = (data[ax].loc[:, f].mean() + 
                                                    norm.fit(data[ax].loc[:, f])[1])
            if (5 or 'Mean+2Sigma') in stat_params:
                Stat_max[ax].loc['Mean+2Sigma',f] = (data[ax].loc[:, f].mean() + 
                                                     norm.fit(data[ax].loc[:, f])[1] * 2)
            if (6 or 'Mean+1.645Sigma') in stat_params:
                Stat_max[ax].loc['Mean+1.645Sigma',f] = (data[ax].loc[:, f].mean() + 
                                                         norm.fit(data[ax].loc[:, f])[1] * 1.645)
            if (7 or 'Mean+2.33Sigma') in stat_params:
                Stat_max[ax].loc['Mean+2.33Sigma',f] = (data[ax].loc[:, f].mean() + 
                                                        norm.fit(data[ax].loc[:, f])[1] * 2.33)
            if (8 or '95') in stat_params:
                Stat_max[ax].loc[f'{res_param[-1].upper()}95',f] = data[ax].loc[:, f].quantile(q=0.95)
            if (9 or '99') in stat_params:
                Stat_max[ax].loc[f'{res_param[-1].upper()}99',f] = data[ax].loc[:, f].quantile(q=0.99)

#         if pribor == 5:   # replace bad values on 50 Hz for NI
#             for f in [50, 100, 160]:
#                 Stat_max[ax].loc[:, f] = 0.5 * (Stat_max[ax].loc[:, F[F.index(f)-1]] + Stat_max[ax].loc[:, F[F.index(f)+1]])

#         if pribor == 6:   # replace bad values on 50 Hz for SCADAS
#             for f in [50, 100, 160]:
#                 Stat_max[ax].loc[:, f] = 0.5 * (Stat_max[ax].loc[:, F[F.index(f)-1]] + Stat_max[ax].loc[:, F[F.index(f)+1]])
                
    return Stat_max

def trains_calc(data, peak_pos, unit='abs', half_interval = 20):
    """Calculates parameters according to SP441.

    Parameters
    ----------
    data : _type_
        _description_
    peak_pos : _type_
        _description_
    unit : str, optional
        _description_, by default 'abs'
    half_interval : int, optional
        _description_, by default 20

    Returns
    -------
    _type_
        _description_
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
                            / ((minR-minL) * 0.5)
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
                            / v_train_time[ax].loc[i,j]
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

    return (v_max, v_eq, train_speed, v_idxmax, v_back, v_train_time)

