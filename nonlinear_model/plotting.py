import random
from random import uniform as random_uniform

import numpy as np
import pandas as pd
import pyfiglet
from joblib import Parallel, delayed
from matplotlib import pyplot as plt
from scipy import stats

plt.rcParams['figure.figsize'] = (7, 7)
plt.rc('font', size=12)
SUBS = [1, 2, 3, 4, 5, 6, 7, 8, 9]
LINTHRESH = 10 ** -4


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


def print_title(msg, banner=None):
    if banner:
        pyfiglet.print_figlet(banner)
    print(msg)


def plot_population(model, population, y0, t0, T, plot_target=True, linthresh=LINTHRESH, max_models_in_plot=5):
    boxplot_figure = plt.figure()

    boxplot_population_targets(population, linthresh=linthresh, figure=boxplot_figure)
    boxplot_population_last_value(population, figure=boxplot_figure, linthresh=linthresh, t0=t0, T=T)

    plot_figure = plt.figure()
    plt.grid(which='both')

    step = max(1, int(len(population) / max_models_in_plot))

    for m in population[::step]:
        plot_model(m, y0 or m.target_as_y0(), t0, T, figure=plot_figure, plot_target=plot_target, linthresh=linthresh)

    return (boxplot_figure, plot_figure)


def plot_model(model, y0, t0, T, figure=None, plot_target=True, linthresh=LINTHRESH):
    if figure:
        plt.figure(figure)
    else:
        figure = plt.figure()
        plt.grid(which='both')

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
    if figure:
        plt.figure(figure)
    else:
        figure = plt.figure()
    plt.grid(which='both')

    colorsP = [x['color'] for x in plt.cycler(color=plt.cm.get_cmap('Set1').colors)]
    colorsC = [x['color'] for x in plt.cycler(color=plt.cm.get_cmap('Accent').colors)]

    for i, model in enumerate(models):
        columns = sorted(list(model['constants'].keys()))
        columns = [str(c) for c in columns]
        values = [model['constants'][k] for k in columns]
        values = [v(t) if callable(v) else v for v in values]
        plt.plot(values, columns, 'o', label=model['name'] + ' Const', color=colorsC[i])

        columns = sorted(list(model['parameters'].keys()))
        columns = [str(c) for c in columns]
        values = [model['parameters'][k] for k in columns]
        values = [v(t) if callable(v) else v for v in values]
        plt.plot(values, columns, 'v', label=model['name'] + ' Params', color=colorsP[i])

    if linthresh:
        plt.xscale('symlog', linthresh=linthresh, subs=SUBS)

    if len(models) < 5:
        plt.legend()

    return figure


def boxplot_population_targets(population: list, figure=None, linthresh=LINTHRESH, scatterplot=False, color='grey'):
    if figure:
        plt.figure(figure)
    else:
        figure = plt.figure()
    plt.grid(which='both')
    equations = population[0]['equations']
    targets = np.array([[m['target'][x] for m in population] for x in equations]).transpose()
    targets_df = pd.DataFrame(columns=equations, data=targets)

    if scatterplot:
        targets_df.boxplot()
        for i, col in enumerate(equations):
            random.seed(1)
            points = targets_df[col]
            x = i + 1
            width = 0.125
            L = x - width
            R = x + width
            plt.plot([random_uniform(L, R) for _ in range(len(points))], points, 'o', alpha=0.25, color='orange',
                     zorder=0)
    else:
        targets_df.boxplot(color=color)

    if linthresh:
        plt.yscale('symlog', linthresh=linthresh, subs=SUBS)
    plt.title('%s Population target values' % ''.join(population[0]['name'].split(' ')[1:]))
    plt.ylabel('average frequency (Hz)')
    return figure


def boxplot_population_last_value(population: list, figure=None, linthresh=LINTHRESH, t0=0, T=1):
    if figure:
        plt.figure(figure)
    else:
        figure = plt.figure()
    plt.grid(which='both')
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
            width = 0.125
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

    plt.title('%s Values at T=%s (%s OK, %s NF)' % (
        str(population[0]['name'].split(' ')[1:]), T, len(targets), len(missing_targets)))
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
    equations = populations[0][0]['equations']
    populations_names = [''.join(p[0]['name'].split(' ')[1:]) or 'SHAM' for p in populations]

    figures = dict()

    sim_data = [Parallel(n_jobs=-1)(delayed(m.simulate)(m.target_as_y0(), t0, T) for m in p) for p in populations]
    data = [[m['y'].transpose()[-1] for m in pop if m['t'][-1] >= T] for pop in sim_data]

    for idx, eq in enumerate(equations):
        figure = plt.figure()
        plt.grid(which='both')

        df = extract_column(data, idx)
        df.columns = populations_names
        df = df.dropna()
        df.boxplot()

        for i, col in enumerate(populations_names):
            random.seed(1)
            points = df[col]
            x = i + 1
            width = 0.125
            L = x - width
            R = x + width
            plt.plot([random_uniform(L, R) for _ in range(len(points))], points, 'o', alpha=0.25, color='orange',
                     zorder=0)

        if linthresh:
            plt.yscale('symlog', linthresh=linthresh, subs=SUBS)

        ok_count = min([len(x) for x in data])
        plt.title(eq + ' at T=%s ' % T + '($\geq$%s OK)' % ok_count + title_postfix)

        plt.setp(figure.axes[0].get_xticklabels(), rotation=45, horizontalalignment='right')
        plt.ylabel('average frequency (Hz)')
        figures[str(eq)] = figure
    return figures


def histplot_populations_last_value_by_equation(populations: list[list], linthresh=LINTHRESH, t0=0, T=1,
                                                title_postfix=''):
    equations = populations[0][0]['equations']
    populations_names = [''.join(p[0]['name'].split(' ')[1:]) or 'SHAM' for p in populations]

    figures = dict()

    sim_data = [Parallel(n_jobs=-1)(delayed(m.simulate)(m.target_as_y0(), t0, T) for m in p) for p in populations]
    data = [[m['y'].transpose()[-1] for m in pop if m['t'][-1] >= T] for pop in sim_data]

    for idx, eq in enumerate(equations):
        figure = plt.figure()
        plt.grid(which='both')

        df = extract_column(data, idx)
        df.columns = populations_names
        df = df.dropna()

        means = df.mean()
        sem = df.sem()
        means[np.isnan(means)] = 0
        sem[np.isnan(sem)] = 0

        stat = stats.tukey_hsd(*np.array(df).transpose())
        annotations = [pvalue_to_asterisks(v) for v in stat.pvalue[0]]

        container = plt.bar(means.index, means, yerr=sem, capsize=12, edgecolor='black',
                            color=plt.cm.binary(range(0, 128, int(128 / len(equations))), alpha=0), zorder=2)
        plt.bar_label(container, annotations, size=18)

        for idx, bar in enumerate(container):
            random.seed(1)
            x = bar.get_x()
            width = bar.get_width() / 2.
            L = x + width - width / 2
            R = x + width + width / 2
            points = df[populations_names[idx]]
            plt.plot([random_uniform(L, R) for _ in range(len(points))], points, 'o', alpha=0.25, color='orange',
                     zorder=1)

        if linthresh:
            plt.yscale('symlog', linthresh=linthresh, subs=SUBS)

        ok_count = min([len(x) for x in data])
        plt.title(eq + ' at T=%s ' % T + ' ($\geq$%s OK) ' % ok_count + title_postfix)
        plt.setp(figure.axes[0].get_xticklabels(), rotation=45, horizontalalignment='right')
        plt.ylabel('average frequency (Hz)')
        figures[str(eq)] = figure
    return figures


def boxplot_population_parameters(population: list, linthresh=LINTHRESH, figure=None, color=None, alt_title=None):
    if figure:
        plt.figure(figure)
    else:
        figure = plt.figure()

    plt.grid(which='both')
    columns = population[0]._optimize_get_state_keys()
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
    figures = dict()
    columns = populations[0][0]._optimize_get_state_keys()
    populations_names = [''.join(p[0]['name'].split(' ')[1:]) or 'SHAM' for p in populations]
    for column in columns:
        figure = plt.figure()
        plt.grid(which='both')
        data = [[x.P[column] for x in p] for p in populations]
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

        container = plt.bar(means.index, means, yerr=sem, capsize=12, edgecolor='black',
                            color=plt.cm.binary(range(0, 128, int(128 / len(populations))), alpha=0), zorder=2)
        plt.bar_label(container, annotations, size=18)

        for idx, bar in enumerate(container):
            random.seed(1)
            x = bar.get_x()
            width = bar.get_width() / 2.
            L = x + width - width / 2
            R = x + width + width / 2
            points = df[populations_names[idx]]
            plt.plot([random_uniform(L, R) for _ in range(len(points))], points, 'o', alpha=0.25, color='orange',
                     zorder=1)

        if linthresh:
            plt.yscale('symlog', linthresh=linthresh, subs=SUBS)

        ok_count = min([len(x) for x in data])
        plt.title(column + ' ($\geq$%s OK) ' % ok_count + title_postfix)
        plt.xticks(rotation='vertical')

        figures[str(column)] = figure
    return figures


def plot_max_eigenvalue_distribution(population: list, figure=None, title_label=''):
    if figure:
        plt.figure(figure)
    else:
        figure = plt.figure()
    eigs = [e for pop in population for e in pop._eigenvalues_real_part()]

    plt.grid(which='both')
    plt.hist(eigs, bins=100)  # , range=[0, 1])
    plt.title('%sPopulation eigenvalues distribution (%s)' % (title_label, str(len(population))))
    plt.ylabel('count')
    plt.xlabel('$Re(\lambda)$')
    plt.yscale('symlog')
    return figure


def plot_population_fitness_distribution(population: list, figure=None, title_label=''):
    if figure:
        plt.figure(figure)
    else:
        figure = plt.figure()

    plt.grid(which='both')
    fits = -np.log10(1 - np.array([i['fitness_history'][-1][1] for i in population]))
    plt.hist(fits, bins=int((max(fits) + 2) * 10))  # , range=[0, 1])
    plt.title('%sPopulation fitness distribution (%s)' % (title_label, str(len(population))))
    plt.ylabel('count')
    plt.xlabel('x')
    return figure


def plot_population_fitness_delta_distribution(population: list, figure=None, title_label=''):
    if figure:
        plt.figure(figure)
    else:
        figure = plt.figure()

    plt.grid(which='both')
    fits = np.array([i['fitness_history'][-1][1] for i in population]) - np.array(
            [i['fitness_history'][0][1] for i in population])
    plt.hist(fits, bins=int((max(fits) + 2) * 10))  # , range=[0, 1])
    plt.title('%sPopulation $\Delta$fitness distribution (%s)' % (title_label, str(len(population))))
    plt.ylabel('count')
    plt.ylabel('$\Delta$fitness')
    return figure


def plot_population_fitness(population: list, figure=None, color='blue'):
    if figure:
        plt.figure(figure)
    else:
        figure = plt.figure()

    fits = -np.log10(1 - np.array([i['fitness_history'][-1][1] for i in population]))
    plt.barh([i['name'] for i in population], fits, color=color)  # , range=[0, 1])
    plt.setp(figure.axes[0].get_xticklabels(), rotation=90, horizontalalignment='right')
    # plt.setp(figure.axes[0].get_yticklabels(), rotation=90, horizontalalignment='right')
    plt.title('Population fitness by individual')
    plt.grid(which='both')
    return figure


def plot_fitness(fitness_history: list, model_name, figure=None, base='generations'):
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
    plt.grid(which='both')
    plt.title(model_name + ' Fitness (blue) =  $1-10^{-y}$ (red) ' + '[over ' + base + ']')

    return figure
