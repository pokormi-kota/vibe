import matplotlib.pyplot as plt
from matplotlib import rcParams
plt.style.use('seaborn-whitegrid')
rcParams['figure.facecolor'] = 'white'
rcParams['savefig.facecolor'] = 'white'
rcParams['font.family'] = 'serif'
rcParams['font.sans-serif'] = ['Times New Roman']
rcParams['font.style'] = 'normal'
plt.rcParams['mathtext.default'] = 'regular'
import numpy as np

from .read_signal import val2db


def statpic(stat, res_param, F, fraction='1/3', showvc=True, vclines=['C','D','E','F','G'], name=f'statvc', table=True, save=False):
    """_summary_

    Parameters
    ----------
    stat : dict (axes: dataframes)
        Statistics that is calculated
    res_param : _type_
        _description_
    F : _type_
        _description_
    fraction : str, optional
        _description_, by default '1/3'
    showvc : bool, optional
        _description_, by default True
    vclines : list, optional
        _description_, by default ['C','D','E','F','G']
    name : _type_, optional
        _description_, by default f'statvc'
    table : bool, optional
        _description_, by default True
    save : bool, optional
        _description_, by default False
    """
    vc = {'A':50, 'B':25, 'C':12.5, 'D':6.25, 'E':3.12, 'F':1.56, 'G':0.78}
    
    axes = list(stat)
    stat_ = {}

    for ax in axes:
        if table:
            fig, axs = plt.subplots(1,1, figsize=(10, 4), constrained_layout=True) #gridspec_kw={'height_ratios': [8, 1], 'hspace': 0.05}
        else:
            fig, axs = plt.subplots(1,1, figsize=(10, 5), constrained_layout=True)
        axs.set_xscale('log')
        if 'L' not in res_param:
            axs.set_yscale('log')
        axs.set_xticks(ticks=F, labels=F)

        if res_param == 'Lv':
            stat_[ax] = val2db(stat[ax] * 1e-6, param='v')
        elif res_param == 'La':
            stat_[ax] = val2db(stat[ax], param='a')
        else:
            stat_[ax] = stat[ax]

        x = F
        labels = {'Max': f'$ {res_param[-1].upper()}$' '$ _{max} $',
                  'Mean': f'$ {res_param[-1].upper()}$' '$ _{mean} $',
                  'RMS': f'$ {res_param[-1].upper()}$' '$ _{rms} $',
                  'Mean+Sigma': f'$ {res_param[-1].upper()}$' '$ _{mean}+ \sigma $',
                  'Mean+2Sigma': f'$ {res_param[-1].upper()}$' '$ _{mean}+2 \sigma $',
                  'Mean+1.645Sigma': f'$ {res_param[-1].upper()}$' '$ _{mean}+1.645 \sigma $',
                  'Mean+2.33Sigma': f'$ {res_param[-1].upper()}$' '$ _{mean}+2.33 \sigma $',
                  'RMS_background': '$ RMS_{background} $',
                  f'{res_param[-1].upper()}50': f'$ {res_param[-1].upper()}$' '$ _{50} $', 
                  f'{res_param[-1].upper()}95': f'$ {res_param[-1].upper()}$' '$ _{95} $',
                  f'{res_param[-1].upper()}99': f'$ {res_param[-1].upper()}$' '$ _{99} $'
                 }

        rowLabels=['Измеренная\n величина']
        for param in stat_[ax].index:
            axs.plot(x, stat_[ax].loc[param], linewidth=1, label=labels[param])
            rowLabels.append(labels[param])
        if showvc:
            for line in vclines:
                axs.hlines(vc[line], F[0], F[-1], ls='--', linewidth=1, 
                           colors=[f'C{vclines.index(line)+stat_[ax].shape[0]}'])
                axs.text(F[-1], vc[line], f'$ VC-{line}\ ({vc[line]} \ мкм/с) $', fontsize=10,
                         horizontalalignment='right', verticalalignment='bottom')

        axs.legend(loc="lower left", ncol=6, fontsize=10, frameon=True)
        axs.grid(visible='True', which='both', axis='both', ls='--')
        axs.set_title(f'$ Направление\ {ax} $', fontsize=14)
        axs.set_xlabel(f'$ Среднегеометрическая\ частота\ {fraction}\ октавной\ полосы,\ Гц $', fontsize=14)

        if res_param == 'v':
            axs.set_ylabel('$ Виброскорость,\ мкм/с $', fontsize=14)
        elif res_param == 'a':
            axs.set_ylabel('$ Виброускорение,\ м/с^{2} $', fontsize=14)
        elif res_param == 'Lv':
            axs.set_ylabel('$ Уровень\ виброскорости,\ дБ $', fontsize=14)
        elif res_param == 'La':
            axs.set_ylabel('$ Уровень\ виброускорения ,\ дБ $', fontsize=14)
        
        if table:
            spec = fig.add_gridspec(ncols=1, nrows=2, height_ratios=[8,1])
            axs1 = fig.add_subplot(spec[1])
            l, b, w, h = axs1.get_position().bounds
            axs1.set_position([l,-h,w,h], which='both')
            axs1.axis('off')

            the_table = axs1.table(cellText=np.vstack([stat_[ax].columns, stat_[ax].applymap('{:,.3f}'.format).values]),
                                    rowLabels = np.array(rowLabels),
                                    colWidths = [(1 + 0.2)/stat_[ax].shape[1]] * stat_[ax].shape[1],
                                    in_layout=True, loc='bottom'
                                    )
            the_table.auto_set_font_size(False)
            the_table.set_fontsize(6)
            if res_param == 'v':
                axs1.set_title(f'$ Виброскорость\ (мкм/с)\ в\ {fraction}\ октавной\ полосе\ со\ среднегеометрической\ частотой\ (Гц) $', 
                                fontsize=12, y=0)
            elif res_param == 'a':
                axs1.set_title(f'$ Виброускорение\ (м/с^{2})\ в\ {fraction}\ октавной\ полосе\ со\ среднегеометрической\ частотой\ (Гц) $', 
                                fontsize=12, y=0)
            elif res_param == 'Lv':
                axs1.set_title(f'$ Уровень\ виброскорости\ (дБ)\ в\ {fraction}\ октавной\ полосе\ со\ среднегеометрической\ частотой\ (Гц) $', 
                                fontsize=12, y=0)
            elif res_param == 'La':
                axs1.set_title(f'$ Уровень\ виброускорения\ (дБ)\ в\ {fraction}\ октавной\ полосе\ со\ среднегеометрической\ частотой\ (Гц) $', 
                                fontsize=12, y=0)

        if save:
            plt.savefig(f'{name}_{ax}.png', dpi=300, bbox_inches='tight')

def bandspic(acc=None, vel=None, res_param='v', F=[8, 16, 31.5, 63, 125, 250], name=f'bands', save=False):

    if acc != None:
        axes = list(acc)
        x = np.arange(0, len(acc[axes[0]][0:]))   # assume len(acc) == len(vel)
    else:
        axes = list(vel)
        x = np.arange(0, len(vel[axes[0]][0:]))

    for ax in axes:
        fig, axs = plt.subplots(figsize=(10, 5), tight_layout=True)
        
        if res_param == 'La':
            for i in F:
                y = val2db(acc[ax].loc[0:, i], 'a')
                axs.plot(x, y, linewidth=0.5, label=f'$ Частота\ {i}\ Гц $')
            axs.set_ylabel('$ Уровень\ виброускорения,\ дБ $', fontsize=12)
            
        elif res_param == 'a':
            for i in F:
                y = acc[ax].loc[0:, i]
                axs.plot(x, y, linewidth=0.5, label=f'$ Частота\ {i}\ Гц $')
            axs.set_ylabel('$ Виброускорение,\ м/с^{2} $', fontsize=12)
            
        elif res_param == 'Lv':
            for i in F:
                y = val2db(vel[ax].loc[0:, i] / 1e6 , 'v')
                axs.plot(x, y, linewidth=0.5, label=f'$ Частота\ {i}\ Гц $')
            axs.set_ylabel('$ Уровень\ виброскорости,\ дБ $', fontsize=12)
            
        elif res_param == 'v':
            for i in F:
                y = vel[ax].loc[:, i]
                axs.plot(x, y, linewidth=0.5, label=f'$ Частота\ {i}\ Гц $')
            axs.set_ylabel('$ Виброскорость,\ мкм/с $', fontsize=12)
            
            
        axs.set_title(f'$ Направление\ {ax} $', fontsize=12)
        axs.set_xlabel('$ Время,\ с $', fontsize=12)
        axs.legend(bbox_to_anchor=(1, 1), loc='upper left', fontsize=10, frameon=True)
        axs.grid(visible='True', which='both', axis='both', ls='--')
        
        if save:
            plt.savefig(f'{name}_{ax}_{res_param}.png', dpi=300)

def bandshist(acc=None, vel=None, res_param='v', F=None, name=f'bands', save=False):
    if F == None:
        F = [1, 2, 4, 8, 16, 31.5, 63, 125, 250]

    if acc != None:
        axes = list(acc)
        x = np.arange(0, len(acc[axes[0]][0:]))
    else:
        axes = list(vel)
        x = np.arange(0, len(vel[axes[0]][0:]))

    for ax in axes:
        fig = plt.figure(constrained_layout=True, figsize=(10, 5))
        subfigs = fig.subfigures(1, 2, wspace=0.07)
        
        axsLeft = subfigs[0].subplots()
        
    #     axsLeft.vlines(dinner_time, 0, vel1[ax].max().max(), ls='--', linewidth=1, colors='k')
    #     axsLeft.text(dinner_time, vel1[ax].max().max(), f'$ Приостановка\ СМР $', fontsize=10, rotation='vertical',
    #                  horizontalalignment='left', verticalalignment='top')

        axsLeft.set_title(f'$ Направление\ {ax} $', fontsize=12)
        axsLeft.set_xlabel('$ Время,\ с $', fontsize=12)
        if res_param == 'La':
            for i in F:
                y = val2db(acc[ax].loc[:, i])
                axsLeft.plot(x, y, linewidth=0.5, label=f'$ {i}\ Гц $')
            axsLeft.set_ylabel('$ Уровень\ виброускорения,\ дБ $', fontsize=12)

        elif res_param == 'a':
            for i in F:
                y = acc[ax].loc[:, i]
                axsLeft.plot(x, y, linewidth=0.5, label=f'$ {i}\ Гц $')
            axsLeft.set_ylabel('$ Виброускорение,\ м/с^{2} $', fontsize=12)

        elif res_param == 'Lv':
            for i in F:
                y = val2db(vel[ax].loc[:, i] * 1e-6, param='v')
                axsLeft.plot(x, y, linewidth=0.5, label=f'$ {i}\ Гц $')
            axsLeft.set_ylabel('$ Уровень\ виброскорости,\ дБ $', fontsize=12)

        elif res_param == 'v':
            for i in F:
                y = vel[ax].loc[:, i]
                axsLeft.plot(x, y, linewidth=0.5, label=f'$ {i}\ Гц $')
            axsLeft.set_ylabel('$ Виброскорость,\ мкм/с $', fontsize=12)
        
        leg = axsLeft.legend(loc='best', fontsize=8, frameon=True)
        for legobj in leg.legendHandles:
            legobj.set_linewidth(1.0)
        axsLeft.grid(visible='True', which='both', axis='both', ls='--')
        
        axsRight = subfigs[1].subplots(3,3)
        
        for axs, f in zip(axsRight.flat, F):

            axs.set_title(f'$ {f}\ Гц $', fontsize=10)
            if res_param == 'La':
                y = val2db(acc[ax].loc[:, i], param='a')
                axs.set_xlabel('$ Уровень\ виброускорения,\ дБ $', fontsize=8)
            elif res_param == 'a':
                y = acc[ax].loc[:, f]
                axs.set_xlabel('$ Виброускорение,\ м/с^{2} $', fontsize=8)
            elif res_param == 'Lv':
                y = val2db(vel[ax].loc[:, i] * 1e-6, param='v')
                axs.set_xlabel('$ Уровень\ виброскорости,\ дБ $', fontsize=8)
            elif res_param == 'v':
                y = vel[ax].loc[:, f]
                axs.set_xlabel('$ Виброскорость,\ мкм/с $', fontsize=8)

            axs.hist(y, bins=50, histtype='bar', density=True, log=True,
                    color=[f'C{F.index(f)}'])

            axs.grid(visible='True', which='both', axis='both', ls='--')
            axs.tick_params(axis='both', which='major', labelsize=6)
            axs.set_ylabel('$ Плотность\ вероятности $', fontsize=8)
        
        if save:
            plt.savefig(f'{name}_{ax}_{res_param}.png', dpi=300)

def signalhist(data, fs, unit, axes=['X','Y','Z'], name='signalhist', save=False):
    quantiles = [0.5, 1, 5, 10, 20, 40, 60, 80, 90, 95, 99, 99.9]

    fig = plt.figure(constrained_layout=True, figsize=(20, 12))
    subfigs = fig.subfigures(1, 2, wspace=0.07)
    axsLeft = subfigs[0].subplots(1, len(axes))
    axsRight = subfigs[1].subplots(2, len(axes), gridspec_kw={'height_ratios': [2, 1], 'hspace': 0.05})

    for ax in axes:
        i = axes.index(ax)
        
        # Plot signals
        x = np.arange(0,  len(data[ax])) / fs
        y = data[ax]
        
        axsLeft[i].plot(y, x, linewidth=0.5, label='$ Сигнал $')
    #     axsLeft[i].hlines(dinner_time, y.min(), y.max(), ls='--', linewidth=1, colors='r')
    #     axsLeft[i].text(y.max(), dinner_time, f'$ Приостановка\ СМР $', fontsize=10,
    #                      horizontalalignment='right', verticalalignment='top')
        axsLeft[i].invert_yaxis()
        axsLeft[i].set_title(f'$ Направление\ {ax} $', fontsize=14)
        axsLeft[i].xaxis.tick_top()
        axsLeft[i].xaxis.set_label_position('top')
        if unit == 'm/s':   # ZetLab
            axsLeft[i].set_xlabel('$ Виброскорость,\ м/с $', fontsize=14)
        else:
            axsLeft[i].set_xlabel('$ Виброускорение,\ м/с{^2} $', fontsize=14)
        axsLeft[i].grid(visible='True', which='both', axis='both', ls='--')
        axsLeft[0].set_ylabel('$ Время,\ с $', fontsize=14)
        
        # Plot histograms
        axsRight[0,i].hist(abs(data[ax]), bins=25, histtype='bar', density=True, stacked=True, log=True, label=f'$ {ax} $')

        axsRight[0,i].set_title(f'$ Направление\ {ax} $', fontsize=14)
        if unit == 'm/s':
            axsRight[0,i].set_xlabel('$ Виброскорость,\ м/с $', fontsize=14)
        else:
            axsRight[0,i].set_xlabel('$ Виброускорение,\ м/с^{2} $', fontsize=14)
        axsRight[0,i].grid(visible='True', which='both', axis='both', ls='--')
        axsRight[0,0].set_ylabel('$ Плостность\ вероятности $', fontsize=14)
        
        # Plot tables with quantiles
        content = []
        for q in quantiles:
            content.append(abs(data[ax]).quantile(q=q/100))
        axsRight[1,i].axis('off')
        axsRight[1,i].table(cellText = [[f'{c:.2e}'] for c in content],
                            rowLabels = list(map(lambda x: str(x)+' %', quantiles)), 
                            colLabels = ['Значение']*len(content),
                            colWidths=[0.5], loc='center', fontsize=14
                        )
        axsRight[1,i].set_title(f'$ Процентиль $', fontsize=14, y=0.9)
    
    if save:
        plt.savefig(f'{name}.png', dpi=300)

def signalpic(data, fs, unit, axes=['X','Y','Z'], name='signalhist', save=False):
    for ax in axes:
        fig, axs = plt.subplots(figsize=(10, 5), tight_layout=True)
        x = np.arange(0,  len(data[ax])) / fs
        y = data[ax]

        axs.plot(x, y, linewidth=0.5, label='$ Сигнал $')

        axs.set_title(f'$ Направление\ {ax} $', fontsize=14)
        axs.set_xlabel('$ Время,\ с $', fontsize=14)
        if unit == 'm/s':
            axs.set_ylabel('$ Виброскорость,\ м/с $', fontsize=14)
        else:
            axs.set_ylabel('$ Виброускорение,\ м/с{^2} $', fontsize=14)
        # axs.legend(bbox_to_anchor=(1, 1), loc='upper left', fontsize=10, frameon=True)
        axs.grid(visible='True', which='both', axis='both', ls='--')
    
        if save:
            plt.savefig(f'{name}_{ax}.png', dpi=300)

