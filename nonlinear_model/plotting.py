import random
from random import uniform as random_uniform

import numpy as np
import pandas as pd
import pyfiglet
from joblib import Parallel, delayed
from matplotlib import pyplot as plt
from scipy import stats


def figsize_square():
    plt.rcParams['figure.figsize'] = (6, 6)
def figsize_horizontal():
    plt.rcParams['figure.figsize'] = (9, 4)
def figsize_vertical():
    plt.rcParams['figure.figsize'] = (4, 9)

plt.rc('font', size=11)
SUBS = [1, 2, 3, 4, 5, 6, 7, 8, 9]
LINTHRESH = 10 ** -4


def make_grid():
    plt.grid(False)
    plt.grid(which='both', alpha=0.33, linewidth=1, zorder=0)

def pvalue_to_asterisks(pvalue):
    if pvalue <= 0.0001:
        return "****"
    elif pvalue <= 0.001:
        return "***"
    elif pvalue <= 0.01:
        return "**"
    elif pvalue <= 0.05:
        return "*"
    return ""


def param_to_latex(p):
    try:
        prefix, p = p.split('___')
    except ValueError:
        prefix = False

    l,f, t = p.split('_')
    l = l.replace('a', '\\alpha').replace('b', '\\beta')
    if not prefix:
        return '$%s^{%s}_{%s}$' % (l,f,t)
    else:
        return '%s: $%s^{%s}_{%s}$' % (prefix, l, f, t)

def print_title(msg, banner=None):
    if banner:
        pyfiglet.print_figlet(banner)
    print(msg)


def plot_population(model, population, y0, t0, T, plot_target=True, linthresh=LINTHRESH, max_models_in_plot=5):
    figsize_square()
    boxplot_figure = plt.figure()

    boxplot_population_targets(population, linthresh=linthresh, figure=boxplot_figure, scatterplot=True, color='grey', scatterplot_color='grey')
    boxplot_population_last_value(population, figure=boxplot_figure, linthresh=linthresh, t0=t0, T=T)

    plot_figure = plt.figure()
    make_grid()

    step = max(1, int(len(population) / max_models_in_plot))

    for m in population[::step]:
        plot_model(m, y0 or m.target_as_y0(), t0, T, figure=plot_figure, plot_target=plot_target, linthresh=linthresh)

    return (boxplot_figure, plot_figure)


def plot_model(model, y0, t0, T, figure=None, plot_target=True, linthresh=LINTHRESH):
    figsize_square()
    if figure:
        plt.figure(figure)
    else:
        figure = plt.figure()
        make_grid()

    solution = model.simulate(y0, t0, T)
    cmap = plt.get_cmap('Paired')
    # neqs = float(len(model['equations']))

    for i, eq in enumerate(model['equations']):
        plt.plot(solution['t'], solution['y'][i], color=cmap(i))

    plt.legend(model['equations'])

    if plot_target:
        for i, eq in enumerate(model['equations']):
            target = model['target'][eq]
            if not callable(target):
                f = lambda t: np.ones(len(solution['t'])) * target
            else:
                f = target
            plt.plot(solution['t'], f(solution['t']), '--', color=cmap(i))
    plt.title(''.join(model['name'].split(' ')[1:]) + ' T=%s' % T)
    plt.xlabel('time (s)')
    plt.ylabel('average frequency (Hz)')

    if linthresh:
        plt.yscale('symlog', linthresh=linthresh, subs=SUBS)
        # plt.xscale('symlog', linthresh=linthresh)

    return figure


def solutions_last_values(equations: list, solutions: list[dict]):
    a = np.zeros((len(solutions), len(equations)))
    for i, s in enumerate(solutions):
        a[i, :] = s['y'].transpose()[-1]
    return a


def plot_parameters(models: list, t=0, figure=None, linthresh=LINTHRESH):
    figsize_vertical()
    if figure:
        plt.figure(figure)
    else:
        figure = plt.figure()
    make_grid()

    colorsP = [x['color'] for x in plt.cycler(color=plt.cm.get_cmap('Set1').colors)]
    colorsC = [x['color'] for x in plt.cycler(color=plt.cm.get_cmap('Accent').colors)]

    for i, model in enumerate(models):
        columns = sorted(list(model['constants'].keys()))
        columns = [str(c) for c in columns]
        values = [model['constants'][k] for k in columns]
        values = [v(t) if callable(v) else v for v in values]
        columns = [param_to_latex(p) for p in columns]
        plt.plot(values, columns, 'o', label=model['name'] + ' Const', color=colorsC[i])

        columns = sorted(list(model['parameters'].keys()))
        columns = [str(c) for c in columns]
        values = [model['parameters'][k] for k in columns]
        values = [v(t) if callable(v) else v for v in values]
        columns = [param_to_latex(p) for p in columns]
        plt.plot(values, columns, 'v', label=model['name'] + ' Params', color=colorsP[i])

    if linthresh:
        plt.xscale('symlog', linthresh=linthresh, subs=SUBS)

    if len(models) < 5:
        plt.legend()

    return figure


def boxplot_population_targets(population: list, figure=None, linthresh=LINTHRESH, scatterplot=False, color='grey', scatterplot_color='grey'):
    figsize_square()
    if figure:
        plt.figure(figure)
    else:
        figure = plt.figure()
    make_grid()
    equations = population[0]['equations']
    targets = np.array([[m['target'][x] for m in population] for x in equations]).transpose()
    targets_df = pd.DataFrame(columns=equations, data=targets)

    if scatterplot:
        targets_df.boxplot(color=color)
        for i, col in enumerate(equations):
            random.seed(1)
            points = targets_df[col]
            x = i + 1
            width = 0.08
            L = x - width
            R = x + width
            plt.plot([random_uniform(L, R) for _ in range(len(points))], points, 'o', alpha=0.25, color=scatterplot_color,
                     zorder=0)
    else:
        targets_df.boxplot(color=color)

    if linthresh:
        plt.yscale('symlog', linthresh=linthresh, subs=SUBS)
    plt.title('%s Population target values' % ''.join(population[0]['name'].split(' ')[1:]))
    plt.ylabel('average frequency (Hz)')
    return figure


def boxplot_population_last_value(population: list, figure=None, linthresh=LINTHRESH, t0=0, T=1):
    figsize_square()
    if figure:
        plt.figure(figure)
    else:
        figure = plt.figure()
    make_grid()
    equations = population[0]['equations']

    data = Parallel(n_jobs=-1)(delayed(m.simulate)(m.target_as_y0(), t0, T) for m in population)

    targets = [m['y'].transpose()[-1] for m in data if m['t'][-1] >= T]
    if len(targets):
        targets_df = pd.DataFrame(columns=equations, data=np.array(targets))
        targets_df.boxplot()

        for i, col in enumerate(equations):
            random.seed(1)
            points = targets_df[col]
            x = i + 1
            width = 0.08
            L = x - width
            R = x + width
            plt.plot([random_uniform(L, R) for _ in range(len(points))], points, 'o', alpha=0.25, color='orange',
                     zorder=0)

    missing_targets = [m['y'].transpose()[-1] for m in data if m['t'][-1] < T]
    if len(missing_targets):
        targets_df = pd.DataFrame(columns=equations, data=np.array(missing_targets))
        targets_df.boxplot(color='orange', )

    if linthresh:
        plt.yscale('symlog', linthresh=linthresh, subs=SUBS)

    plt.title('%s Values at T=%s (%s samples)' % (
        str(population[0]['name'].split(' ')[1:]), T, len(targets)))
    plt.ylabel('average frequency (Hz)')
    return figure


def extract_column(data, column):
    pops = [list() for pop in data]
    for pindex, pop in enumerate(data):
        for ind in pop:
            pops[pindex].append(ind[column])
    df = pd.DataFrame(pops)
    return df.T


def boxplot_populations_last_value_by_equation(populations: list[list], linthresh=LINTHRESH, t0=0, T=1,
                                               title_postfix=''):
    figsize_square()
    equations = populations[0][0]['equations']
    populations_names = [''.join(p[0]['name'].split(' ')[1:]) or 'SHAM' for p in populations]

    figures = dict()

    sim_data = [Parallel(n_jobs=-1)(delayed(m.simulate)(m.target_as_y0(), t0, T) for m in p) for p in populations]
    data = [[m['y'].transpose()[-1] for m in pop if m['t'][-1] >= T] for pop in sim_data]

    for idx, eq in enumerate(equations):
        figure = plt.figure()
        make_grid()

        df = extract_column(data, idx)
        df.columns = populations_names
        df = df.dropna()
        df.boxplot()

        for i, col in enumerate(populations_names):
            random.seed(1)
            points = df[col]
            x = i + 1
            width = 0.08
            L = x - width
            R = x + width
            plt.plot([random_uniform(L, R) for _ in range(len(points))], points, 'o', alpha=0.25, color='orange',
                     zorder=0)

        if linthresh:
            plt.yscale('symlog', linthresh=linthresh, subs=SUBS)

        ok_count = min([len(x) for x in data])
        plt.title(eq + ' at T=%s ' % T + '(%s samples)' % ok_count + title_postfix)

        plt.setp(figure.axes[0].get_xticklabels(), rotation=45, horizontalalignment='right')
        plt.ylabel('average frequency (Hz)')
        figures[str(eq)] = figure
    return figures


def histplot_populations_last_value_by_equation(populations: list[list], linthresh=LINTHRESH, t0=0, T=1,
                                                title_postfix=''):
    figsize_square()
    equations = populations[0][0]['equations']
    populations_names = [''.join(p[0]['name'].split(' ')[1:]) or 'SHAM' for p in populations]

    figures = dict()

    sim_data = [Parallel(n_jobs=-1)(delayed(m.simulate)(m.target_as_y0(), t0, T) for m in p) for p in populations]
    data = [[m['y'].transpose()[-1] for m in pop if m['t'][-1] >= T] for pop in sim_data]

    for idx, eq in enumerate(equations):
        figure = plt.figure()
        # make_grid()

        df = extract_column(data, idx)
        df.columns = populations_names
        df = df.dropna()

        means = df.mean()
        sem = df.sem()
        means[np.isnan(means)] = 0
        sem[np.isnan(sem)] = 0

        stat = stats.tukey_hsd(*np.array(df).transpose())
        annotations = [pvalue_to_asterisks(v) for v in stat.pvalue[0]]

        # container = plt.bar(means.index, means, yerr=sem, capsize=12, edgecolor='black', error_kw={'zorder': 0},
        #                     color=plt.cm.binary(range(0, 255, int(255 / len(means))), alpha=100), zorder=10)
        
        bars_content = plt.bar(means.index, means, color='teal', edgecolor='black',
                            #color=plt.cm.binary(range(0, 128, int(128 / len(equations))), alpha=0), 
                            alpha=1,
                            zorder=1)
        container = plt.bar(means.index, means, yerr=sem, capsize=12, edgecolor='black',
                            #color=plt.cm.binary(range(0, 128, int(128 / len(equations))), alpha=0), 
                            alpha=0,
                            zorder=3)
        plt.bar_label(container, annotations, size=18)

        for idx, bar in enumerate(container):
            random.seed(1)
            x = bar.get_x()
            width = bar.get_width() / 2.
            L = x + width - width / 3.
            R = x + width + width / 3.
            points = df[populations_names[idx]]
            plt.plot([random_uniform(L, R) for _ in range(len(points))], points, 'o', alpha=0.2, color='orange',
                     zorder=2)

        if linthresh:
            plt.yscale('symlog', linthresh=linthresh, subs=SUBS)

        ok_count = min([len(x) for x in data])
        plt.title(eq + ' at T=%s ' % T + ' (%s samples) ' % ok_count + title_postfix)
        plt.setp(figure.axes[0].get_xticklabels(), rotation=45, horizontalalignment='right')
        plt.ylabel('average frequency (Hz)')
        figures[str(eq)] = figure
    return figures


def boxplot_population_parameters(population: list, linthresh=LINTHRESH, figure=None, color=None, alt_title=None):
    figsize_horizontal()
    if figure:
        plt.figure(figure)
    else:
        figure = plt.figure()

    make_grid()
    columns = population[0]._optimize_get_state_keys()
    columns = [param_to_latex(p) for p in columns]
    data = np.array([m._optimize_get_state() for m in population])
    df = pd.DataFrame(columns=columns, data=data)
    fp = {'markeredgecolor': color}

    if data.sum() > 0:
        df.boxplot(vert=True, color=color, rot=90, flierprops=fp, )
    if linthresh:
        plt.yscale('symlog', linthresh=linthresh, subs=SUBS)

    if alt_title is None:
        plt.title('%s Parameters distribution' % str(population[0]['name'].split(' ')[1:]))
    else:
        plt.title(alt_title)

    return figure


def histplot_population_parameters(populations: list[list], linthresh=LINTHRESH, title_postfix=''):
    figsize_square()
    figures = dict()
    columns = populations[0][0]._optimize_get_state_keys()
    populations_names = [''.join(p[0]['name'].split(' ')[1:]) or 'SHAM' for p in populations]
    for column in columns:
        figure = plt.figure()
        # make_grid()
        try:
            data = [[x.P[column] for x in p] for p in populations]
        except KeyError:
            for p in populations:
                for x in p:
                    try:
                        x.P[column]
                    except:
                        print(x)
                        raise
            raise
        df = pd.DataFrame(data)
        df = df.transpose()
        df.columns = populations_names
        df = df.dropna()

        means = df.mean()
        sem = df.sem()
        means[np.isnan(means)] = 0
        sem[np.isnan(sem)] = 0

        stat = stats.tukey_hsd(*np.array(df).transpose())
        annotations = [pvalue_to_asterisks(v) for v in stat.pvalue[0]]
        
        bars_container = plt.bar(means.index, means, color='teal', edgecolor='black',
                            #color=plt.cm.binary(range(0, 128, int(128 / len(populations))), alpha=0), 
                            alpha=1,
                            zorder=1)
        container = plt.bar(means.index, means, yerr=sem, capsize=12, edgecolor='black',
                            #color=plt.cm.binary(range(0, 128, int(128 / len(populations))), alpha=0), 
                            alpha=0,
                            zorder=3)
        plt.bar_label(container, annotations, size=18)

        for idx, bar in enumerate(container):
            random.seed(1)
            x = bar.get_x()
            width = bar.get_width() / 2.
            L = x + width - width / 3.
            R = x + width + width / 3.
            points = df[populations_names[idx]]
            plt.plot([random_uniform(L, R) for _ in range(len(points))], points, 'o', alpha=0.2, color='orange',
                     zorder=2)

        if linthresh:
            plt.yscale('symlog', linthresh=linthresh, subs=SUBS)

        ok_count = min([len(x) for x in data])
        plt.title(param_to_latex(column) + ' (%s samples) ' % ok_count + title_postfix)
        plt.xticks(rotation='vertical')

        figures[str(column)] = figure
    return figures


def plot_max_eigenvalue_distribution(population: list, figure=None, title_label=''):
    figsize_square()
    if figure:
        plt.figure(figure)
    else:
        figure = plt.figure()
    eigs = [e for pop in population for e in pop._eigenvalues_real_part()]

    make_grid()
    plt.hist(eigs, bins=100)  # , range=[0, 1])
    plt.title('%sPopulation eigenvalues distribution (%s)' % (title_label, str(len(population))))
    plt.ylabel('count')
    plt.xlabel('$Re(\lambda)$')
    plt.yscale('symlog')
    return figure


def plot_population_fitness_distribution(population: list, figure=None, title_label=''):
    figsize_square()
    if figure:
        plt.figure(figure)
    else:
        figure = plt.figure()

    make_grid()
    fits = -np.log10(1 - np.array([i['fitness_history'][-1][1] for i in population]))
    plt.hist(fits, bins=int((max(fits) + 2) * 10))  # , range=[0, 1])
    plt.title('%sPopulation fitness distribution (%s)' % (title_label, str(len(population))))
    plt.ylabel('count')
    plt.xlabel('x')
    return figure


def plot_population_fitness_delta_distribution(population: list, figure=None, title_label=''):
    figsize_square()
    if figure:
        plt.figure(figure)
    else:
        figure = plt.figure()

    make_grid()
    fits = np.array([i['fitness_history'][-1][1] for i in population]) - np.array(
            [i['fitness_history'][0][1] for i in population])
    plt.hist(fits, bins=int((max(fits) + 2) * 10))  # , range=[0, 1])
    plt.title('%sPopulation $\Delta$fitness distribution (%s)' % (title_label, str(len(population))))
    plt.ylabel('count')
    plt.ylabel('$\Delta$fitness')
    return figure


def plot_population_fitness(population: list, figure=None, color='blue'):
    figsize_square()
    if figure:
        plt.figure(figure)
    else:
        figure = plt.figure()

    fits = -np.log10(1 - np.array([i['fitness_history'][-1][1] for i in population]))
    plt.barh([i['name'] for i in population], fits, color=color)  # , range=[0, 1])
    plt.setp(figure.axes[0].get_xticklabels(), rotation=90, horizontalalignment='right')
    # plt.setp(figure.axes[0].get_yticklabels(), rotation=90, horizontalalignment='right')
    plt.title('Population fitness by individual')
    make_grid()
    return figure


def plot_fitness(fitness_history: list, model_name, figure=None, base='generations'):
    figsize_square()
    if figure:
        plt.figure(figure)
        if len(figure.axes) < 2:
            plt.twinx()
    else:
        figure = plt.figure()
        plt.twinx()

    history = np.array(fitness_history).transpose()
    if not len(history):
        history = np.array([[0], [0]])

    if base == 'generations':
        x = list(range(len(history[0])))
    else:
        x = history[0]
        x -= x[0]

    plt.sca(figure.axes[0])
    plt.plot(x, history[1], label='Fitness', color='blue')

    plt.sca(figure.axes[1])
    plt.plot(x, -np.log10(1 - history[1]), label='9s', color='red')
    # plt.yscale('log', subs=SUBS,)
    make_grid()
    plt.title(model_name + ' Fitness (blue) =  $1-10^{-y}$ (red) ' + '[over ' + base + ']')

    return figure
