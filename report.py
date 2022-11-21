import math
import os
import pandas as pd
import xlsxwriter

from .read_signal import val2db

__all__ = ['stat_report']

def stat_report(stats, res_param='v', subs=['max','eq'], spec=None, specLabels=None, fraction='1/3', init_rows=3, init_cols=1, name=f'stat_results', width=21):    
    """Saves (statistics) report for protocol in .xlsx format that fits a standard (A4, vertical) page.

    Parameters
    ----------
    stats : list
        contains dictionaries (keys = axes) of dataframes (rows = stat params, cols = Freqs)
    res_param : str
        'v' - velocity (mkm/s)
        'Lv' - velocity levels (dB)
        'a' - acceleration (m/s^2)
        'La' - acceleration levels (dB)
    subs : list, optional
        subscripts for each stat heading, by default ['max','eq']
    spec : list
        Contains dicionaries with characteristics to be written in the initial columns, e.g. [train_daytime.loc[:,63], train_time.loc[:,63]]
    specLabels : list
        Labels for initial columns, e.g. ['№ п/п', 'Время прохода', 'Время воздействия, с']
    init_rows : int, optional
        Number of rows with some headings, needed if one wants to leave blank rows at the top, by default 3
    init_cols : int, optional
        Number of columns with some parameters except stat, needed if one wants to leave blank columns at the left, by default 1 or len(spec)+1
    name : str, optional
        Name of the file to be saved, by default f'stat_results'
    width : int, optional
        Number of columns to fit on a page, by default 20
    """
    if spec != None:
        init_cols = len(spec) + 1
    n_stats = len(stats)   # number of statistics sets
    stats_ = [0]*n_stats
    axes = list(stats[0])
    F = next(iter(stats[0].values())).columns

    workbook = xlsxwriter.Workbook(f'{name}.xlsx', {'strings_to_numbers':  True})

    if 'L' in res_param:
        res = res_param
        num_format = workbook.add_format({
            'num_format': '0.0',
            'font_name': 'Times New Roman',
            'font_size': 9})
    else:
        res = res_param.upper()
        num_format = workbook.add_format({
            'num_format': '0.00',
            'font_name': 'Times New Roman',
            'font_size': 9})
    cell_format = workbook.add_format({
        'font_name': 'Times New Roman',
        'font_size': 9,
        'align': 'center',
        'text_wrap': True})
    subscript = workbook.add_format({
        'font_name': 'Times New Roman',
        'font_size': 9,
        'font_script': 2})
    merge_format = workbook.add_format({
        'font_name': 'Times New Roman',
        'font_size': 9,
        'align': 'center',
        'valign': 'vcenter',
        'text_wrap': True})


    worksheet = workbook.add_worksheet()

    # Recalculate the width of table
    # and make splits by frequencies if length is more then {width} columns
    split = 1
    width_ = len(F)*n_stats + init_cols
    while width_ > width:   # 20...22 is a reasonable number of columns to fit on a page
        split += 1
        width_ -= width + init_cols

    cols_byline_num = math.ceil(len(F)/split)   # will be actual number of columns from stats in one line 

    #  Write headings
#     worksheet.merge_range(0, 0, init_rows-1, 0, 'Показатель', cell_format)

    if res_param == 'v':
        worksheet.merge_range(0, init_cols, 0, init_cols-1 +  cols_byline_num *n_stats, 
                          f'Значения виброскоростей, мкм/с в {fraction} октавной полосе со среднегеометрической частотой, Гц',
                          merge_format)
    elif res_param == 'La':
        worksheet.merge_range(0, init_cols, 0, init_cols-1+cols_byline_num*n_stats,
                              f'Уровни виброускорений, дБ в {fraction} октавной полосе со среднегеометрической частотой, Гц',
                              merge_format)
    elif res_param == 'Lv':
        worksheet.merge_range(0, init_cols, 0, init_cols-1+cols_byline_num*n_stats,
                              f'Уровни виброскоростей, дБ в {fraction} октавной полосе со среднегеометрической частотой, Гц',
                              merge_format)
    if n_stats == 2:
        for f in range(0, cols_byline_num):
                worksheet.write_rich_string(1,
                                            f*n_stats+init_cols,
                                            cell_format, f'{res}',
                                            subscript, f'{subs[0]}', cell_format)
                worksheet.write_rich_string(1,
                                            f*n_stats+init_cols+1,
                                            cell_format, f'{res}',
                                            subscript, f'{subs[1]}', cell_format)
    for ax in axes:
        a = axes.index(ax)
        # Convert to required units
        for i in range(n_stats):
            stats_[i] = {}
            if res_param == 'Lv':
                stats_[i][ax] = val2db(stats[i][ax] * 1e-6, param='v')
            elif res_param == 'La':
                stats_[i][ax] = val2db(stats[i][ax], param='a')
            else:
                stats_[i][ax] = stats[i][ax]

        worksheet.merge_range(init_rows-1+a + a*(stats[0][ax].shape[0]+1)*split,
                              init_cols, 
                              init_rows-1+a + a*(stats[0][ax].shape[0]+1)*split,
                              init_cols+cols_byline_num*n_stats-1, 
                              f'Ось {ax}', merge_format)
        
        for i in range(split):
            # left (init) columns headings
            worksheet.write((init_rows+a + i*(stats[0][ax].shape[0]+1) +
                              a*(stats[0][ax].shape[0]+1)*split),
                             0,
                             'Показатель', cell_format)
            if spec != None:
                for s in range(len(specLabels)):
                    worksheet.write((init_rows+a + i*(stats[0][ax].shape[0]+1) +
                              a*(stats[0][ax].shape[0]+1)*split),
                             s,
                             f'{specLabels[s]}', cell_format)
            
                            
            # fill in left (init) columns
            for param in stats[0][ax].index:
                index_loc = stats[0][ax].index.get_loc(param)   # row number by its name
                worksheet.write(((init_rows+1+a + i*(stats[0][ax].shape[0]+1)
                                + a*(stats[0][ax].shape[0]+1)*split)+index_loc),
                                0,
                                f'{param}', cell_format)
                if spec != None:
                    for s in range(len(spec)):
                        worksheet.write(((init_rows+1+a + i*(stats[0][ax].shape[0]+1)
                                        + a*(stats[0][ax].shape[0]+1)*split)+index_loc),
                                        s+1,
                                        f'{spec[s][ax].loc[index_loc,63]}', num_format)

            for f in range(i*(cols_byline_num), (i+1)*(cols_byline_num)):
                if f < len(F):
                    # write frequencies
                    if n_stats == 1:
                        worksheet.write((init_rows+a + i*(stats[0][ax].shape[0]+1) +
                                         a*(stats[0][ax].shape[0]+1)*split),

                                        init_cols+f-i*(cols_byline_num), 

                                        f'{F[f]}', cell_format)
                    else:
                        worksheet.merge_range((init_rows+a + i*(stats[0][ax].shape[0]+1) + 
                                               a*(stats[0][ax].shape[0]+1)*split),

                                              init_cols+f*n_stats-i*(cols_byline_num)*n_stats,

                                              (init_rows+a + i*(stats[0][ax].shape[0]+1) + 
                                               a*(stats[0][ax].shape[0]+1)*split),

                                              init_cols+n_stats-1+f*n_stats-i*(cols_byline_num)*n_stats,

                                              f'{F[f]}', merge_format)

                    # Write data
                    for param in stats[0][ax].index:
                        index_loc = stats[0][ax].index.get_loc(param)

                        st = 0
                        for stat in stats_:
                            worksheet.write(((init_rows+1+a + i*(stat[ax].shape[0]+1) +
                                              a*(stat[ax].shape[0]+1)*split)+index_loc),

                                            init_cols+f*n_stats-i*(cols_byline_num)*n_stats + st,

                                            f'{stat[ax].loc[param,F[f]]}', num_format)
                            st += 1

    workbook.close()

    print(f'Report {name} is saved to: {os.getcwd()}')

def trains_report(params, res_param, train_daytime, stats=[], fraction='1/3', init_rows=3-1, init_cols=3, name='trains'):
    """Saves report with trains for protocol in .xlsx format

    Parameters
    ----------
    res_param : str
        'v' - velocity (mkm/s)
        'Lv' - velocity levels (dB)
        'a' - acceleration (m/s^2)
        'La' - acceleration levels (dB)
    params : list
        [v_max, v_eq, v_train_time], train_time is the last
    train_daytime : dict
        _description_
     stats : list
        contains dictionaries (keys = axes) of dataframes (rows = stat params, cols = Freqs)
    fraction : str
        '1/1' or '1/3' octave
    init_rows : int, optional
        number of rows with some headings, by default 3
    init_cols : int, optional
        number of columns with some parameters except stat, by default 1
    name : str, optional
        name of the file to be created, by default 'trains'
    """

    workbook = xlsxwriter.Workbook(f'{name}.xlsx', {'strings_to_numbers':  True})

    if 'L' in res_param:
        num_format = workbook.add_format({
            'num_format': '0.0',
            'font_name': 'Times New Roman',
            'font_size': 9})
    else:
        num_format = workbook.add_format({
            'num_format': '0.00',  #'General'
            'font_name': 'Times New Roman',
            'font_size': 9})
    cell_format = workbook.add_format({
        'font_name': 'Times New Roman',
        'font_size': 9,
        'align': 'center',
        'text_wrap': True})
    subscript = workbook.add_format({
        'font_name': 'Times New Roman',
        'font_size': 9,
        'font_script': 2})
    merge_format = workbook.add_format({
        'font_name': 'Times New Roman',
        'font_size': 9,
        'align': 'center',
        'valign': 'vcenter',
        'text_wrap': True})
    rotated_format = workbook.add_format({
        'font_name': 'Times New Roman',
        'font_size': 9,
        'align': 'center',
        'valign': 'vcenter',
        'text_wrap': True})

    n_stats = len(stats)   # number of statistics sets
    v_train_time = params[-1]
    stats_ = [0]*n_stats
    axes = list(params[0])
    if type(params[0][axes[0]]) == pd.core.frame.DataFrame:
        F = params[0][axes[0]].columns
    else:
        F = [1]

    for ax in axes:
        # Convert to required units
        for i in range(n_stats):
            stats_[i] = {}
            if res_param == 'Lv':
                stats_[i][ax] = val2db(stats[i][ax] * 1e-6, param='v')
            elif res_param == 'La':
                stats_[i][ax] = val2db(stats[i][ax], param='a')
            else:
                stats_[i][ax] = stats[i][ax]
        params_ = []
        if res_param == 'Lv':
            params_.append(val2db(params[0][ax] * 1e-6, param='v'))
            params_.append(val2db(params[1][ax] * 1e-6, param='v'))
        elif res_param == 'La':
            params_.append(val2db(params[0][ax], param='a'))
            params_.append(val2db(params[1][ax], param='a'))
        else:
            params_.append(params[0][ax])
            params_.append(params[1][ax])

        worksheet = workbook.add_worksheet(f'Ось {ax}')

        rotated_format.set_rotation(90)
        worksheet.merge_range(0, 0, 2, 0, '№ п.п', cell_format)
        worksheet.merge_range(0, 1, 2, 1, 'Время прохождения', cell_format)
        worksheet.merge_range(0, 2, 2, 2, 'Время воздействия, с', cell_format)

        merge_format.set_rotation(0)
        if res_param == 'Lv':
            worksheet.merge_range(0, init_cols, 0, init_cols-1+len(F)*n_stats,
                                  f'Уровни виброскоростей, дБ в {fraction} октавной полосе со среднегеометрической частотой, Гц',
                                  merge_format)
        elif res_param == 'v':
            worksheet.merge_range(0, init_cols, 0, init_cols-1+len(F)*n_stats,
                                  (f'Значения виброскоростей, мкм/с в {fraction} октавной полосе со среднегеометрической частотой, Гц'),
                                  merge_format)
        elif res_param == 'La':
            worksheet.merge_range(0, init_cols, 0, init_cols-1+len(F)*n_stats,
                                  f'Уровни виброускорений, дБ в {fraction} октавной полосе со среднегеометрической частотой, Гц',
                                  merge_format)

        worksheet.merge_range(init_rows+1, 0,
                              init_rows+1, init_cols-1+len(F)*n_stats, 
                              f'Ось {ax}', merge_format)

        for j in range(len(F)):
            worksheet.merge_range(1, init_cols+j*n_stats, 1, init_cols+n_stats-1 +j*n_stats, f'{F[j]}', merge_format)
            if 'L' in res_param:
                worksheet.write_rich_string(2, init_cols+j*n_stats, 
                                            cell_format, 'L',
                                            subscript, res_param[-1],
                                            cell_format, 'max', cell_format)
                worksheet.write_rich_string(2, init_cols+1 +j*n_stats, 
                                            cell_format, 'L',
                                            subscript, res_param[-1],
                                            cell_format, 'eq', cell_format)
            else:
                worksheet.write_rich_string(2, init_cols+j*n_stats, 
                                            cell_format, res_param[-1],
                                            subscript, 'max', cell_format)
                worksheet.write_rich_string(2, init_cols+1 +j*n_stats, 
                                            cell_format, res_param[-1],
                                            subscript, 'eq', cell_format)
            
        num=0
        for i in params[0][ax].index:                
            num += 1
            worksheet.write((3+num), 0,
                           f'{num}', cell_format)
            worksheet.write((3+num), 1,
                            f'{train_daytime[ax].loc[i,63]}', cell_format)
            worksheet.write((3+num), 2,
                            f'{v_train_time[ax].loc[i,63]}', cell_format)

            for j in range(len(F)):
                worksheet.write((3+num), init_cols+ j*n_stats, 
                                f'{params_[0].loc[i,F[j]]}', num_format)
                worksheet.write((3+num), init_cols+1 +j*n_stats, 
                                f'{params_[1].loc[i,F[j]]}', num_format)
            
        if n_stats>0:
            for param in stats_[0][ax].index:
                index_loc = stats_[0][ax].index.get_loc(param)   # row number by its name

                worksheet.write((6+len(params[0][ax])+index_loc), 0, f'{param}', num_format)

                for j in range(len(F)):

                    worksheet.write((6+len(params[0][ax])+index_loc), init_cols +j*n_stats, 
                                    f'{stats_[0][ax].loc[param,F[j]]}', num_format)
                    worksheet.write((6+len(params[0][ax])+index_loc), init_cols+1 +j*n_stats, 
                                    f'{stats_[1][ax].loc[param,F[j]]}', num_format)
            
    workbook.close()

    print(f'Report {name} is saved to: {os.getcwd()}')

def trains_report1(stats, res_param, params, time, train_daytime, fraction='1/3', init_rows=3-1, init_cols=6, name=f'trains'):
    """Saves report with trains for protocol in .xlsx format.
    Includes train type, railway number and speed.

    Parameters
    ----------
    stats : list
        contains dictionaries (keys = axes) of dataframes (rows = stat params, cols = Freqs)
    res_param : str
        'v' - velocity (mkm/s)
        'Lv' - velocity levels (dB)
        'a' - acceleration (m/s^2)
        'La' - acceleration levels (dB)
    params : _type_
        [v_max, v_eq, v_train_time]
    time : dict
        _description_
    train_daytime : dict
        _description_
    init_rows : int, optional
        number of rows with some headings, by default 3
    init_cols : int, optional
        number of columns with some parameters except stat, by default 1
    name : str, optional
        '', by default f'trains'
    """

    workbook = xlsxwriter.Workbook(f'{name}.xlsx', {'strings_to_numbers':  True})
    
    if 'L' in res_param:
        num_format = workbook.add_format({
            'num_format': '0.0',
            'font_name': 'Times New Roman',
            'font_size': 9})
    else:
        num_format = workbook.add_format({
            'num_format': '0.00',
            'font_name': 'Times New Roman',
            'font_size': 9})
    cell_format = workbook.add_format({
        'font_name': 'Times New Roman',
        'font_size': 9,
        'align': 'center',
        'text_wrap': True})
    subscript = workbook.add_format({
        'font_name': 'Times New Roman',
        'font_size': 9,
        'font_script': 2})
    merge_format = workbook.add_format({
        'font_name': 'Times New Roman',
        'font_size': 9,
        'align': 'center',
        'valign': 'vcenter',
        'text_wrap': True})
    rotated_format = workbook.add_format({
        'font_name': 'Times New Roman',
        'font_size': 9,
        'align': 'center',
        'valign': 'vcenter',
        'text_wrap': True})

    n_stats = len(stats)   # number of statistics sets
    v_train_time = params[-1]
    stats_ = [0]*n_stats
    axes = list(stats[0])
    F = next(iter(stats[0].values())).columns

    for ax in axes:
        # Convert to required units
        for i in range(n_stats):
            stats_[i] = {}
            if res_param == 'Lv':
                stats_[i][ax] = val2db(stats[i][ax] * 1e-6, param='v')
            elif res_param == 'La':
                stats_[i][ax] = val2db(stats[i][ax], param='a')
            else:
                stats_[i][ax] = stats[i][ax]
        params_ = []
        if res_param == 'Lv':
            params_.append(val2db(params[0][ax] * 1e-6, param='v'))
            params_.append(val2db(params[4][ax] * 1e-6, param='v'))
        elif res_param == 'La':
            params_.append(val2db(params[0][ax], param='a'))
            params_.append(val2db(params[4][ax], param='a'))
        else:
            params_.append(params[0][ax])
            params_.append(params[4][ax])

        worksheet = workbook.add_worksheet(f'Ось {ax}')

        rotated_format.set_rotation(90)
        worksheet.merge_range(0, 0, init_rows, 0, '№ п.п', cell_format)
        worksheet.merge_range(0, 1, init_rows, 1, 'Время прохождения', cell_format)
        worksheet.merge_range(0, 2, init_rows, 2, 'Тип поезда', cell_format)
        worksheet.merge_range(0, 3, init_rows, 3, 'Номер пути', cell_format)
        worksheet.merge_range(0, 4, init_rows, 4, 'Скорость, км/ч', cell_format)
        worksheet.merge_range(0, 5, init_rows, 5, 'Время воздействия, с', cell_format)

        merge_format.set_rotation(0)
        if res_param == 'Lv':
            worksheet.merge_range(0, init_cols, 0, init_cols+len(F)*n_stats,
                                  f'Уровни виброскоростей, дБ в {fraction} октавной полосе со среднегеометрической частотой, Гц',
                                  merge_format)
        elif res_param == 'v':
            worksheet.merge_range(0, init_cols, 0, init_cols+len(F)*n_stats,
                                  (f'Значения виброскоростей, мкм/с в {fraction} октавной полосе со среднегеометрической частотой, Гц'),
                                  merge_format)
        elif res_param == 'La':
            worksheet.merge_range(0, init_cols, 0, init_cols+len(F)*n_stats,
                                  f'Уровни виброускорений, дБ в {fraction} октавной полосе со среднегеометрической частотой, Гц',
                                  merge_format)

        worksheet.merge_range(init_rows+1, 0,
                              init_rows+1, init_cols+len(F)*n_stats, 
                              f'Ось {ax}', merge_format)

        for j in range(len(F)):
            worksheet.merge_range(1, init_cols+j*n_stats, 1, init_cols+n_stats-1 +j*n_stats, f'{F[j]}', merge_format)
            if 'L' in res_param:
                worksheet.write_rich_string(2, init_cols+j*n_stats, 
                                            cell_format, 'L',
                                            subscript, res_param[-1],
                                            cell_format, 'max', cell_format)
                worksheet.write_rich_string(2, init_cols+1 +j*n_stats, 
                                            cell_format, 'L',
                                            subscript, res_param[-1],
                                            cell_format, 'eq', cell_format)
            else:
                worksheet.write_rich_string(2, init_cols+j*n_stats, 
                                            cell_format, res_param[-1],
                                            subscript, 'max', cell_format)
                worksheet.write_rich_string(2, init_cols+1 +j*n_stats, 
                                            cell_format, res_param[-1],
                                            subscript, 'eq', cell_format)
            
        num=0
        for i in params[0][ax].index:
            if time.loc[i,2] == 'электричка':
                traintype = time.loc[i,3]
                railway = time.loc[i,4]
            else:
                traintype = time.loc[i,2]
                railway = 'н/д'
                
            if time.loc[i,2] == 'электричка':
                train_speed = time.loc[i,6]
            else:
                train_speed = params[2][ax].loc[i,63]
                
            num += 1
            worksheet.write((3+num), 0,
                           f'{num}', cell_format)
            worksheet.write((3+num), 1,
                            f'{train_daytime[ax].loc[i,63]}', cell_format)
            worksheet.write((3+num), 2,
                            f'{traintype}', cell_format)
            worksheet.write((3+num), 3,
                            f'{railway}', cell_format)
            worksheet.write((3+num), 4,
                            f'{train_speed}', num_format)
            worksheet.write((3+num), 5,
                            f'{v_train_time[ax].loc[i,63]}', cell_format)

            for j in range(len(F)):
                worksheet.write((3+num), init_cols+ j*n_stats, 
                                f'{params_[0].loc[i,F[j]]}', num_format)
                worksheet.write((3+num), init_cols+1 +j*n_stats, 
                                f'{params_[1].loc[i,F[j]]}', num_format)

        for param in stats_[0][ax].index:
            index_loc = stats_[0][ax].index.get_loc(param)   # row number by its name

            worksheet.write((6+len(params[0][ax])+index_loc), 0, f'{param}', num_format)

            for j in range(len(F)):

                worksheet.write((6+len(params[0][ax])+index_loc), init_cols +j*n_stats, 
                                f'{stats_[0][ax].loc[param,F[j]]}', num_format)
                worksheet.write((6+len(params[0][ax])+index_loc), init_cols+1 +j*n_stats, 
                                f'{stats_[1][ax].loc[param,F[j]]}', num_format)

    workbook.close()

    print(f'Report {name} is saved to: {os.getcwd()}')

