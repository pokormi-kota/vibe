
import numpy as np
import pandas as pd
from scipy.stats import norm

def statistics(data, res_param, F, stat_params, axes=['X','Y','Z']):
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
            if (6 or 'Mean+1.96Sigma') in stat_params:
                Stat_max[ax].loc['Mean+1.645Sigma',f] = (data[ax].loc[:, f].mean() + 
                                                norm.fit(data[ax].loc[:, f])[1] * 1.645)
            if (7 or 'Mean+2.576Sigma') in stat_params:
                Stat_max[ax].loc['Mean+2.33Sigma',f] = (data[ax].loc[:, f].mean() + 
                                                norm.fit(data[ax].loc[:, f])[1] * 2.33)
            if (8 or '95') in stat_params:
                Stat_max[ax].loc[f'{res_param.upper()}95',f] = data[ax].loc[:, f].quantile(q=0.95)
            if (9 or '99') in stat_params:
                Stat_max[ax].loc[f'{res_param.upper()}99',f] = data[ax].loc[:, f].quantile(q=0.99)

#         if pribor == 5:   # replace bad values on 50 Hz for NI
#             for f in [50, 100, 160]:
#                 Stat_max[ax].loc[:, f] = 0.5 * (Stat_max[ax].loc[:, Fv[Fv.index(f)-1]] + Stat_max[ax].loc[:, Fv[Fv.index(f)+1]])

#         if pribor == 6:   # replace bad values on 50 Hz for SCADAS
#             for f in [50, 100, 160]:
#                 Stat_max[ax].loc[:, f] = 0.5 * (Stat_max[ax].loc[:, Fv[Fv.index(f)-1]] + Stat_max[ax].loc[:, Fv[Fv.index(f)+1]])
                
    return Stat_max