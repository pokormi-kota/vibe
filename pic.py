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
    vc = {'A':50, 'B':25, 'C':12.5, 'D':6.25, 'E':3.12, 'F':1.56, 'G':0.78}
    
    axes = list(stat)
    stat_ = {}

    for ax in axes:
        fig, axs = plt.subplots(1,1, figsize=(10, 6), constrained_layout=True)   #gridspec_kw={'height_ratios': [8, 1], 'hspace': 0.05}
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
            axs1 = fig.add_subplot()
            axs1.axis('off')

            the_table = axs1.table(cellText=np.vstack([stat_[ax].columns,stat_[ax].applymap('{:,.3f}'.format).values]),
                                    rowLabels = np.array(rowLabels),
                                    in_layout=True, loc='bottom'
                                    )
            the_table.auto_set_font_size(False)
            the_table.set_fontsize(6)
            axs1.set_title(f'$ Виброскорость\ (мкм/с)\ в\ {fraction}\ октавной\ полосе\ со\ среднегеометрической\ частотой\ (Гц) $', 
                            fontsize=12, y=0)

        if save:
            plt.savefig(f'{name}_{ax}.png', dpi=300)