import os
from collections import defaultdict

import matplotlib.cm as cm
import seaborn as sns
from matplotlib.colors import Normalize

from CONF import *
from models import *
from plotting import *

plt.rcParams['figure.figsize'] = (7, 7)

N_JOBS = -1

PLOT_LINTHRESH = False  # 1e-4
BOXPLOT_LINTHRESH = False  # 1e-4


def mutation_state(model, mutation_scale, mutations_number, parameter_index, y0, t0, T, label='N/A'):
    model_state = model._optimize_get_state()
    value = model_state[parameter_index]
    value_range = np.linspace(value * (1 - mutation_scale), value * (1 + mutation_scale), mutations_number)
    targets = list()
    for v in value_range:
        mutated_model = model.copy()
        new_state = model_state.copy()
        new_state[parameter_index] = v
        mutated_model._optimize_set_state(new_state)
        res = mutated_model.simulate(y0, 0, T)
        targets.append([v, res['y'].transpose()[-1], res['t'][-1]])
    return label, targets


def main(fit=True, plot=False):
    files = sorted(filter(lambda x: x.startswith('S_'), os.listdir(POPULATION_BASE_PATH + '')))
    individuals = [Healthy_combined_fit.load(POPULATION_BASE_PATH + '%s' % f) for f in files]
    reject_threshold = 1 - 2e-8
    individuals = [ind for ind in individuals if ind['fitness_history'][-1][1] > reject_threshold]
    for i in individuals:
        i['parameters'].update(i.lesion_LDA()['parameters'])
    individuals = [i.lesion_SHAM() for i in individuals]

    t0 = 0
    T = SIMULATION_TIME
    T_mutation = T

    mutation_scale = 0.5
    mutations = 100
    mutations_plot_one_every = 1

    available_params = individuals[0]._optimize_get_state_keys()

    if fit:

        states = defaultdict(list)
        for model in individuals:
            res = model.simulate(model.target_as_y0(), 0, T)
            states['HEALTHY'] += [[0, res['y'].transpose()[-1], res['t'][-1]]]

        mutation_states = Parallel(n_jobs=N_JOBS)(delayed(mutation_state)(model,
                                                                          mutation_scale,
                                                                          mutations,
                                                                          parameter_index,
                                                                          model.target_as_y0(),
                                                                          t0,
                                                                          T_mutation,
                                                                          parameter_name) for
                                                  parameter_index, parameter_name in enumerate(available_params)
                                                  for model in individuals)

        with open('./fitted_models/S_000_SENSITIVITY_STATES', 'bw') as f:
            pickle.dump(mutation_states, f)

        for parameter, results in mutation_states:
            states[parameter] += results

        for key, val in states.items():
            states[key] = sorted(val, key=lambda x: x[0])

        with open('./fitted_models/S_000_SENSITIVITY_STATES', 'bw') as f:
            pickle.dump(states, f)
    else:

        with open('./fitted_models/S_000_SENSITIVITY_STATES', 'br') as f:
            states = pickle.load(f)

    if plot:
        print_title("STEP 03: parameter sensitivity analysis", 'STEP 03')
        model = Healthy_combined_fit().lesion_SHAM()

        healthy_reference = None

        for param, data in list(states.items()):
            continue

            figure = plt.figure()

            values = np.array([x[0] for x in data])
            states_data = np.array([x[1] for x in data])
            end_times = np.array([x[2] for x in data])

            color_normalizer = Normalize(vmin=min(values), vmax=max(values))
            color_map = cm.ScalarMappable(norm=color_normalizer, cmap=cm.spring)
            colors = [color_map.to_rgba(x) for x in values]

            df = pd.DataFrame(columns=model["equations"], data=states_data)

            if param == 'HEALTHY':
                healthy_reference = df
                df.boxplot(color='red')
            elif healthy_reference is not None:
                healthy_reference.boxplot(color='red')
                df.boxplot(color='green')

            for i, c in enumerate(df.columns):
                y = df[c]
                plt.scatter([i + 1] * len(y), y, alpha=0.3, s=25, c=colors)  # , c=y, s=10)
                for n in range(len(y)):
                    if end_times[n] < T:
                        plt.scatter([i + 1], y[n], alpha=1, s=200, c='red', marker='x')  # , c=y, s=10)

            if PLOT_LINTHRESH:
                plt.yscale('symlog', linthresh=PLOT_LINTHRESH, subs=[1, 2, 3, 4, 5, 6, 7, 8, 9])
            plt.title(str(param) + '   %s - %s' % (min(values), max(values)))

        # Sensitivity matrix
        no_param_states = states.pop('HEALTHY')

        sensitivity_matrix = np.zeros(shape=(len(states.keys()), len(no_param_states[0][1])))

        for i, (param, results) in enumerate(states.items()):
            values, good_results = list(zip(*[(x[0], x[1]) for x in results if x[2] >= T]))
            values = pd.DataFrame(values)
            good_results = pd.DataFrame(good_results)
            G = (good_results / good_results.median()).std()
            V = (values / values.median()).std()
            sensitivity_index = G / float(V)
            sensitivity_matrix[i] = sensitivity_index
        sensitivity_matrix /= sensitivity_matrix.max()
        sensitivity_df = pd.DataFrame(sensitivity_matrix, columns=model['equations'], index=states.keys())

        figure = plt.figure()
        sns.heatmap(sensitivity_df, annot=True, cmap='YlOrBr')
        figure.savefig(os.path.join(POPULATION_BASE_PATH, 'sensitivity_analysis_matrix_LDA.png'), dpi=FIG_DPI,
                       bbox_inches='tight')


if __name__ == '__main__':
    main(fit=True, plot=True)
