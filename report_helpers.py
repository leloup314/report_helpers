"""
This file contains easy to use helper functions for the daily struggles of report writing
"""
import os
import inspect
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.figure import Figure
from collections import defaultdict, OrderedDict  # Needed for dict_to_table
from copy import deepcopy

# fit function that returns covariance matrix and fit parameters; sqrt of diagonal elements of
# cov_matrix are the std deviation or gaussian error
from scipy.optimize import curve_fit

# get current path
current_path = os.getcwd()


def red_chisquare(observed, expected, observed_error, popt):
    return np.sum(((observed - expected) / observed_error)**2 / (len(observed_error) - len(popt) - 1))


def plot(data, x_label=None, y_label=None, title=None, output_pdf=None,
         fit=None, x_scale='linear', y_scale='linear',
         legend_outside=False, **kwargs):
    """
    Function to plot a given data set. Optionallly, fit data to a given function and plot fit and data.
    
    data: dict
        data should contain a dataset for each key it has, where dataset is list of input data e.g. data={'data': [x, y, x_err, y_err], 'label': 'Data'}:
            x: list 
                list with data in x dimension
            y: list
                list with data in y dimension
            x_err: list
                list with errors on x
            y_err: list
                list with errors on x
            label: str
                label to be displayed in plots legend for input data; default is "Data"
        for multiple plots: data={'data': [[x, y, x_err, y_err], [x1, y1, x1_err, y1_err]], 'label': ['Data', 'Data1']}
    
    x_label: str
        x-axis label
    
    y_label: str
        y-axis label
    
    title: str
        tile of plot
    
    output_pdf: str
        output pdf file; if None plot.pdf
    
    fit: dict 
        fit should contain fitting-related stuff e.g. fit={'func': func, 'data': [x_fit, y_fit, y_err_fit], 'start_parameters': None, 'label': 'fit_label', 'units': ['mV', 'A', None]}
            x_fit: list 
                list with data in x dimension
            y_fit: list
                list with data in y dimension
            y_err_fit: list
                list with errors on x
            label: str
                label to be displayed in plots legend for fit; default is "Fit"
        if None, only plot data. For multiple fits: fit={'func': [func, func1], 'data': [[x_fit, y_fit, y_err_fit], [x1_fit, y1_fit, y1_err_fit]] 'start_parameters': None or [p0, p1], 'label': ['fit_label','fit_label1']}
        
    x_scale: str
        x-axis scale, either "linear" or "log"
    y_scale: str
        y-axis scale "linear" or "log"
    legend_outside: bool
        if True, legend is outside plot
    """

    # Initialize input parameters if they're None
    if output_pdf is None:
        output_pdf = os.path.join(current_path, 'plot.pdf')
    if x_label is None:
        x_label = 'x'
    if y_label is None:
        y_label = 'y'
    if title is None:
        title = 'No title'

    # See https://matplotlib.org/api/_as_gen/matplotlib.axes.Axes.errorbar.html for explanation
    plot_kwargs = {'ls': 'none', 'lw': 1, 'marker': 'o', 'ms': 5,
                   'mfc': 'w', 'mew': 1, 'capsize': 2, 'elinewidth': 1,
                   'alpha': 1.0}

    legend_kwargs = {'loc': 'upper left', 'fontsize': 'medium'}

    # Add keyword args
    if kwargs:
        for key in kwargs.keys():
            if key in legend_kwargs.keys():
                legend_kwargs[key] = kwargs[key]
            else:
                plot_kwargs[key] = kwargs[key]

    # make temporary data to not corrupt original input data
    tmp_data = deepcopy(data)

    # make every value list if not already list
    for key in tmp_data.keys():
        if not isinstance(tmp_data[key], list):
            tmp_data[key] = [tmp_data[key]]
        else:
            pass

    plot = None
    # Open output file as out
    with PdfPages(output_pdf) as out:
        plt.figure()
        plt.xlabel(x_label)
        plt.ylabel(y_label)
        plt.grid()
        plt.xscale(x_scale)
        plt.yscale(y_scale)
        plt.title(title)

        # get input data
        input_data = np.array(tmp_data['data'])

        if not len(input_data.shape) == 1:
            data_sets = 1 if len(input_data.shape) == 2 and input_data.dtype != np.dtype('O') else input_data.shape[0]
        else:
            raise ValueError('Dimension of input data corrupt')

        data_labels = tmp_data['label'] if 'label' in tmp_data.keys() else ['Data' if i == 0 else 'Data %i' % (i + 1) for i in range(data_sets)]

        for n_dsets in range(data_sets):

            if data_sets == 1:
                x = np.array(input_data[0])
                y = np.array(input_data[1])
                x_err = np.array(input_data[2])
                y_err = np.array(input_data[3])

            else:
                x = np.array(input_data[n_dsets][0])
                y = np.array(input_data[n_dsets][1])
                x_err = np.array(input_data[n_dsets][2])
                y_err = np.array(input_data[n_dsets][3])

            if 'marker' in tmp_data.keys():
                plot_kwargs['marker'] = tmp_data['marker'][n_dsets]

            if 'mfc' in tmp_data.keys():
                plot_kwargs['mfc'] = tmp_data['mfc'][n_dsets]

            plt.errorbar(x, y, xerr=x_err, yerr=y_err, label=data_labels[n_dsets], **plot_kwargs)

        if fit is not None:

            # make fit data to not corrupt original input data
            tmp_fit = deepcopy(fit)

            # make every value list if not already list
            for key in tmp_fit.keys():
                if not isinstance(tmp_fit[key], list):
                    tmp_fit[key] = [tmp_fit[key]]
                else:
                    pass

            global_fit_parameters, global_fit_param_errors = np.array([]), np.array([])

            fit_data = np.array(tmp_fit['data'])
            fit_funcs = tmp_fit['func']

            if not len(fit_data.shape) == 1:
                fit_sets = 1 if len(fit_data.shape) == 2 and fit_data.dtype != np.dtype('O') else fit_data.shape[0]
            else:
                raise ValueError('Dimensions of fit data corrupt')

            p0 = tmp_fit['start_parameters'] if 'start_parameters' in tmp_fit.keys() else [None] * fit_sets

            fit_labels = tmp_fit['label'] if 'label' in tmp_fit.keys() else ['Fit_%i' % (i + 1) for i in
                                                                             range(fit_sets)]

            precision = [2] * fit_sets if 'precision' not in tmp_fit.keys() else tmp_fit['precision']

            zorder = tmp_fit['zorder'] if 'zorder' in tmp_fit.keys() else None

            fit_range = tmp_fit['fit_range'] if 'fit_range' in tmp_fit.keys() else None
            
            chisquare = tmp_fit['chisquare'] if 'chisquare' in tmp_fit.keys() else True
            
            rsquare = tmp_fit['rsquare'] if 'rsquare' in tmp_fit.keys() else False

            if len(precision) < fit_sets:
                while len(precision) < fit_sets:
                    precision.append(precision[0])

            for n_fits in range(fit_sets):

                fit_func = fit_funcs[n_fits]
                fit_func_label = fit_labels[n_fits]

                if fit_sets == 1:
                    x_fit = np.array(fit_data[0])
                    y_fit = np.array(fit_data[1])
                    y_err_fit = np.array(fit_data[2])

                else:
                    x_fit = np.array(fit_data[n_fits][0])
                    y_fit = np.array(fit_data[n_fits][1])
                    y_err_fit = np.array(fit_data[n_fits][2])

                # Fit to data 
                fit_parameters, fit_cov = curve_fit(fit_func, x_fit, y_fit, p0[n_fits], y_err_fit, absolute_sigma=True)

                # Deduce std. deviation from covariance matrix
                fit_param_errors = np.sqrt(np.diag(fit_cov))

                # Make residuals for r2 value
                residuals = y_fit - fit_func(x_fit, *fit_parameters)
                ss_res = np.sum(np.power(residuals, 2))
                ss_tot = np.sum(np.power((y_fit - np.mean(y_fit)), 2))
                r_2 = 1. - (ss_res / ss_tot)

                # Make reduced chisquare
                red_chi2 = red_chisquare(y_fit, fit_func(x_fit, *fit_parameters), y_err_fit, fit_parameters)

                # Get arguments of fit function; dismiss 0th element since it represents variable
                fit_args = inspect.getargspec(fit_func)[0][1:]

                # Make legend entry
                spacer = '  '
                fit_label = 'Fit function:\n' + spacer + '%s\nFit parameters:\n' % fit_func_label
                for i, arg in enumerate(fit_args):
                    con_fit = '%.' + str(precision[n_fits]) + 'f' if np.absolute(
                        fit_parameters[i]) >= 1e-3 and np.absolute(fit_parameters[i]) <= 1e1 else '%.' + str(
                        precision[n_fits]) + 'E'
                    con_err = '%.' + str(precision[n_fits]) + 'f' if np.absolute(
                        fit_param_errors[i]) >= 1e-3 and np.absolute(fit_param_errors[i]) <= 1e1 else '%.' + str(
                        precision[n_fits]) + 'E'
                    if 'units' in tmp_fit.keys():
                        if tmp_fit['units'][i] is not None:
                            add_str = spacer + r'%s = ' % arg + '(' + con_fit % fit_parameters[i] + '$\pm$' + con_err % fit_param_errors[i] + ') %s' % fit['units'][i] + '\n'
                        else:
                            add_str = spacer + r'%s = ' % arg + con_fit % fit_parameters[i] + '$\pm$' + con_err % fit_param_errors[i] + '\n'
                    else:
                        add_str = spacer + r'%s = ' % arg + con_fit % fit_parameters[i] + '$\pm$' + con_err % fit_param_errors[i] + '\n'
                    fit_label += add_str
                
                if chisquare:
                    con_chi2 = '%.5f' if np.absolute(red_chi2) >= 1e-5 else '%.5E'
                    fit_label += r'$\mathrm{Red. \chi^2 = ' + con_chi2 % red_chi2 + '}$' + '\n'
                if rsquare:
                    fit_label += r'$\mathrm{R^2 = %.5f}$' % r_2 + '\n'
                    
                # See https://matplotlib.org/api/_as_gen/matplotlib.axes.Axes.errorbar.html for explanaition
                fit_kwargs = {}

                fit_kwargs['label'] = fit_label
                if zorder:
                    fit_kwargs['zorder'] = zorder

                if fit_range:
                    x = fit_range[n_fits]
                else:
                    x = x_fit

                plt.plot(x, fit_func(x, *fit_parameters), **fit_kwargs)

                global_fit_parameters = np.append(global_fit_parameters, fit_parameters)
                global_fit_param_errors = np.append(global_fit_param_errors, fit_param_errors)

        if legend_outside:
            legend_kwargs['bbox_to_anchor'] = (1, 1)
        plt.legend(**legend_kwargs)
        plot = (plt.gcf(), plt.gca())
        try:
            out.savefig(bbox_inches='tight')
        except TypeError:
            out.savefig()
        plt.show()

    if fit is not None:
        return global_fit_parameters, global_fit_param_errors, plot

    return plot


def dict_to_table(table_dict, output_tex=None, col_sep='5mm', row_sep='0.33cm', placeholder='', alignment=None,
                  headers=None, caption=None, label=None, lists_only=True):
    """
    Function to translate a Python dict into a tex table
    
    table_dict: dict
        dictionary with list as values to be placed in table; best choice is OrderedDict from collections since it respects key order
    output_tex: file
        output tex-file, if None: table.tex
    col_sep: str 
        numerical value and allowed tex length unit e.g. mm, cm, pt to set column separation
    row_sep: str
        numerical value and allowed tex length unit e.g. mm, cm, pt to set row separation
    headers: list or tuple
        strings to be used as table headers; can be text or TeX-code, if None, dicts keys are used as headers; then best choice OrderedDict
    caption: str
        caption to be used as table caption; text or TeX-code
    alignment: str:
        string of characters to determine alignment of columns e.g. "c|c|c" for 3 centered columns with separator lines.
        If None, use separators in between columns and center
    label: str
        string to be used as citation label in tex document
    lists_only: bool
        if True then only they keys of table_dict whose values are lists are put in the table
    """

    input_dict = deepcopy(table_dict)
    tmp = defaultdict(list)  # Temporary dict with lists as default values; see collections.defaultdict
    sub_columns = []
    multicol_list = []

    if lists_only:
        for key in input_dict.keys():
            if isinstance(input_dict[key], OrderedDict) or isinstance(input_dict[key], dict):
                for sub_key in input_dict[key].keys():
                    if not isinstance(input_dict[key][sub_key], list):
                        del input_dict[key][sub_key]
            elif not isinstance(input_dict[key], list):
                del input_dict[key]

    # Dict is empty
    for key in input_dict.keys():
        if not input_dict[key]:
            del input_dict[key]

    if not input_dict:
        return

    # make sub columns
    for i, key in enumerate(input_dict.keys()):
        if isinstance(input_dict[key], OrderedDict) or isinstance(input_dict[key], dict):
            if i == 0 or i == len(input_dict.keys()) - 1:
                al = 'c|' if i == 0 else '|c'
            else:
                al = '|c|'
            mc = '\\multicolumn{%i}{%s}{%s}' % (len(input_dict[key]), al, str(key))
            multicol_list.append(mc)
            for sub_key in input_dict[key].keys():
                sub_columns.append(sub_key)

        else:
            multicol_list.append(key)
            sub_columns.append('')

    column_strings = input_dict.keys()  # First take keys of dict as columns; later check if separate headers are provided; if so: overwrite

    if headers is not None:
        if len(headers) == len(input_dict.keys()):
            column_strings = headers
        else:
            raise ValueError('Tables columns and amount of headers not matching')

    if output_tex is None:
        output_tex = os.path.join(current_path, 'table.tex')  # Make output file in current directory

    if alignment is None:
        if not multicol_list:
            alignment = "".join(["c|" * len(column_strings)])[:-1]  # Ugly hack
        else:
            alignment = "".join(["c|" * len(sub_columns)])[:-1]  # Ugly hack

    else:
        # Count amount of letters in alignment to check if it matches columns
        count = 0
        for c in alignment:
            if c.isalpha():
                if c not in ['l', 'c', 'r']:
                    raise ValueError('Alignment can only be left "l", centered "c" or right "r".')
                else:
                    count += 1
            else:
                pass
        if count != len(column_strings):
            msg = 'Alignment dimensions (%i) and number of columns (%i) do not match' % (count, len(column_strings))
            raise ValueError(msg)

    # Declare strings for table environment; double backslash needed for interpreter to write backslash: "\\" == "\"
    # Correct latex formatting
    newline = '\n'
    tab = '\t'
    begin_table = '\\begin{table}[h]' + newline
    end_table = '\\end{table}' + newline
    begin_center = tab + '\\begin{center}' + newline
    end_center = tab + '\\end{center}' + newline
    begin_tabular = 2 * tab + '\\begin{tabular}{%s}' % alignment + newline
    end_tabular = 2 * tab + '\\end{tabular}' + newline
    toprule = 3 * tab + '\\toprule' + newline
    midrule = 3 * tab + '\\hline' + newline
    bottomrule = 3 * tab + '\\bottomrule' + newline
    separator = ' & '
    endline = '\\\[%s]' % row_sep + newline
    columnsep = 2 * tab + '\\setlength{\\tabcolsep}{%s}' % col_sep + newline
    cap = tab + '\\caption{%s}' % str(caption) + newline
    lab = tab + '\\label{%s}' % str(label) + newline

    with open(output_tex,
              'w') as f:  # open file as f; with environment offers automated file handling and closes file after being out of scope

        # Writing LaTex environment/table variables related to file
        f.write(begin_table)
        f.write(begin_center)
        f.write(columnsep)
        f.write(begin_tabular)
        f.write(toprule)

        # Store lengths of all data; needed to determine maximum # of entris in data to fill table with blank spaces
        lengths = []

        # Get lengths
        for key in input_dict.keys():
            if not isinstance(input_dict[key], OrderedDict):
                lengths.append(len(input_dict[key]))
            else:
                for sub_key in input_dict[key].keys():
                    lengths.append(len(input_dict[key][sub_key]))

        # Get maximum of lengths
        max_entries = max(lengths)

        # Loop over headers
        for head in column_strings:
            for i in range(max_entries):

                if not isinstance(input_dict[head], OrderedDict):
                    # Loop over maximum range an fill temporary dict; keys are i
                    try:
                        tmp[i].append(str(input_dict[head][i]))  # try append input data at i as string
                    except IndexError:
                        tmp[i].append(
                            placeholder)  # if there is no dat at i because current heads dat is shorter, fill in blank space
                else:
                    for sub_head in input_dict[head].keys():
                        try:
                            tmp[i].append(str(input_dict[head][sub_head][i]))  # try append input data at i as string
                        except IndexError:
                            tmp[i].append(
                                placeholder)  # if there is no dat at i because current heads dat is shorter, fill in blank space

        # Write headers to file
        if multicol_list == column_strings:
            f.write(3 * tab + separator.join(column_strings) + endline)
            f.write(midrule)
        else:
            f.write(3 * tab + separator.join(multicol_list) + endline)
            f.write(midrule)
            f.write(3 * tab + separator.join(sub_columns) + endline)
            f.write(midrule)

        # Write actual table entries to file
        for i in tmp.keys():
            to_write = separator.join(tmp[i]) + endline
            f.write(3 * tab + to_write)

        # Writing LaTex environment/table variables related to file
        f.write(bottomrule)
        f.write(end_tabular)
        f.write(end_center)
        if caption is not None:
            f.write(cap)
        if label is not None:
            f.write(lab)
        f.write(end_table)
