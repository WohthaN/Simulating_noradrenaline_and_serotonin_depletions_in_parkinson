import math
import os

from CONF import FIG_DPI, SIMULATION_TIME, POPULATION_BASE_PATH
from models import *
from plotting import *

N_JOBS = 1
PLOT_LINTHRESH = False  # 10 ** -4
BOXPLOT_LINTHRESH = 1e-3


def slice_populations(populations):
    groups = len(populations)
    size = len(populations[0])
    per_group = math.floor(size / groups)
    sliced = list()
    for i, group in enumerate(populations):
        sliced.append(group[i * per_group:(i + 1) * per_group])
    return sliced


def main(fit=True, plot=False):
    t0 = 0
    T = SIMULATION_TIME
    people_in_population = 240
    mutation_scale = 0.5 / 4

    if fit:
        base = Healthy_combined_fit()
        base['name'] = 'S_xxx'
        np.random.seed(1984)
        individuals = [base] + [base.new_mutated_target_model(scale=mutation_scale) for i in
                                range(people_in_population)]
        for i, ind in enumerate(individuals):
            ind['name'] = ind['name'].split('_')[0] + '_' + '%03i' % (i)
            ind.apply()

        for idx, ind in enumerate(individuals[::1]):
            try:
                ind = Healthy_combined_fit.load(POPULATION_BASE_PATH + '%s' % ind['name'])
                individuals[idx] = ind
            except FileNotFoundError:
                filename = POPULATION_BASE_PATH + '%s' % ind['name']
                ind.apply()
                ind = ind.optimize(ind.target_as_y0(), t0, T, save_checkpoint_name=filename)[0]
                individuals[idx] = ind
                ind.save(filename)

    else:
        files = sorted(filter(lambda x: x.startswith('S_'), os.listdir(POPULATION_BASE_PATH + '')))
        individuals = [Healthy_combined_fit.load(POPULATION_BASE_PATH + '%s' % f) for f in files]
        people_in_population = len(individuals)

    if plot:
        print_title("STEP 02: fitted population (%s) on target distribution with mutation scale %s" %
                    (people_in_population, mutation_scale), 'STEP 02')

        figure = plt.figure()
        make_grid()
        for i in individuals:
            plot_fitness(i['fitness_history'], 'Combined', figure=figure, base='generations')
        figure.savefig(os.path.join(POPULATION_BASE_PATH, 'population_fitness_generations.png'), dpi=FIG_DPI,
                       bbox_inches='tight')

        figure = plt.figure()
        make_grid()
        for i in individuals:
            plot_fitness(i['fitness_history'], 'Combined', figure=figure, base='time')
        figure.savefig(os.path.join(POPULATION_BASE_PATH, 'population_fitness_time.png'), dpi=FIG_DPI,
                       bbox_inches='tight')

        figure = plot_population_fitness_distribution(individuals)
        figure.savefig(os.path.join(POPULATION_BASE_PATH, 'population_fitness_distribution.png'), dpi=FIG_DPI,
                       bbox_inches='tight')

        figure = boxplot_population_targets(individuals, linthresh=False, scatterplot=True)
        figure.savefig(os.path.join(POPULATION_BASE_PATH, 'population_targets_before_fit.png'), dpi=FIG_DPI,
                       bbox_inches='tight')

        TP = [i.lesion_LDA() for i in individuals]
        figure = boxplot_population_targets(TP, linthresh=False, color='red')
        TP = [i.lesion_L5HT() for i in individuals]
        figure = boxplot_population_targets(TP, figure=figure, linthresh=False, color='orange')
        TP = [i.lesion_LNE() for i in individuals]
        figure = boxplot_population_targets(TP, figure=figure, linthresh=False, color='blue')
        figure = boxplot_population_targets(individuals, figure=figure, linthresh=False, color='black')
        figure.savefig(os.path.join(POPULATION_BASE_PATH, 'population_targets_before_fit_lesions.png'), dpi=FIG_DPI,
                       bbox_inches='tight')

        reject_threshold = 1 - 2e-8
        rejects = [ind for ind in individuals if ind['fitness_history'][-1][1] <= reject_threshold]
        individuals = [ind for ind in individuals if ind['fitness_history'][-1][1] > reject_threshold]

        figure = plt.figure()
        make_grid()
        # plot_population_fitness(individuals, figure=figure, color='blue')
        plot_population_fitness(rejects, figure=figure, color='red')

        figure.savefig(os.path.join(POPULATION_BASE_PATH, 'population_fitness_individual.png'), dpi=FIG_DPI,
                       bbox_inches='tight')

        figure = boxplot_population_parameters(individuals, linthresh=BOXPLOT_LINTHRESH)
        figure.savefig(os.path.join(POPULATION_BASE_PATH, 'population_parameters_distribution.png'), dpi=FIG_DPI,
                       bbox_inches='tight')

        populations = list()
        for kind in ['lesion_SHAM', 'lesion_LDA', 'lesion_L5HT', 'lesion_LNE',
                     'lesion_LDA_L5HT',
                     'lesion_LDA_LNE']:
            # Healthy
            current_individuals = [i.__getattribute__(kind)() for i in individuals]
            populations.append(current_individuals)
            boxplot_population_parameters(current_individuals, linthresh=BOXPLOT_LINTHRESH)
            (boxplot, plot) = plot_population(individuals[0], current_individuals, None, t0, T,
                                              linthresh=PLOT_LINTHRESH, plot_target=False)
            boxplot.savefig(os.path.join(POPULATION_BASE_PATH, 'population_target_%s.png' % kind), dpi=FIG_DPI,
                            bbox_inches='tight')

            stability_plot = plot_max_eigenvalue_distribution(
                    current_individuals,
                    title_label=str(current_individuals[0]['name'].split(' ')[
                                    1:]) + ' ')
            stability_plot.savefig(os.path.join(POPULATION_BASE_PATH, 'population_max_eigenvalues_%s.png' % kind),
                                   dpi=FIG_DPI,
                                   bbox_inches='tight')

        sliced_populations = slice_populations(populations)
        sliced_populations_desc = ''# + str([len(x) for x in sliced_populations])

        figures_dict = boxplot_populations_last_value_by_equation(populations, linthresh=PLOT_LINTHRESH, t0=t0, T=T,
                                                                  title_postfix='')
        for eq_name, figure in figures_dict.items():
            figure.savefig(os.path.join(POPULATION_BASE_PATH, 'population_by_equation_%s.png' % eq_name), dpi=FIG_DPI,
                           bbox_inches='tight')

        figures_dict = boxplot_populations_last_value_by_equation(sliced_populations, linthresh=PLOT_LINTHRESH, t0=t0,
                                                                  T=T, title_postfix=sliced_populations_desc)
        for eq_name, figure in figures_dict.items():
            figure.savefig(os.path.join(POPULATION_BASE_PATH, 'population_by_equation_%s_sliced.png' % eq_name),
                           dpi=FIG_DPI,
                           bbox_inches='tight')

        figures_dict = histplot_populations_last_value_by_equation(populations, linthresh=PLOT_LINTHRESH, t0=t0, T=T,
                                                                   title_postfix='')
        for eq_name, figure in figures_dict.items():
            figure.savefig(os.path.join(POPULATION_BASE_PATH, 'population_by_equation_hist_%s.png' % eq_name),
                           dpi=FIG_DPI,
                           bbox_inches='tight')

        figures_dict = histplot_populations_last_value_by_equation(sliced_populations, linthresh=PLOT_LINTHRESH, t0=t0,
                                                                   T=T, title_postfix=sliced_populations_desc)
        for eq_name, figure in figures_dict.items():
            figure.savefig(os.path.join(POPULATION_BASE_PATH, 'population_by_equation_hist_%s_sliced.png' % eq_name),
                           dpi=FIG_DPI,
                           bbox_inches='tight')

        # Additional stuff generated for documentation

        # GP plots for comparison
        LNE_pop = [populations[0], populations[1], populations[3], populations[5]]

        figures_dict = histplot_populations_last_value_by_equation(LNE_pop, linthresh=PLOT_LINTHRESH, t0=t0, T=T,
                                                                   title_postfix='')
        figures_dict['GP'].savefig(os.path.join(POPULATION_BASE_PATH, 'population_by_equation_hist_LNE_GP.png'),
                                   dpi=FIG_DPI,
                                   bbox_inches='tight')
        L5HT_pop = [populations[0], populations[1], populations[2], populations[4]]

        figures_dict = histplot_populations_last_value_by_equation(L5HT_pop, linthresh=PLOT_LINTHRESH, t0=t0, T=T,
                                                                   title_postfix='')
        figures_dict['GP'].savefig(os.path.join(POPULATION_BASE_PATH, 'population_by_equation_hist_L5HT_GP.png'),
                                   dpi=FIG_DPI,
                                   bbox_inches='tight')

        LNE_pop = [sliced_populations[0], sliced_populations[1], sliced_populations[3], sliced_populations[5]]

        figures_dict = histplot_populations_last_value_by_equation(LNE_pop, linthresh=PLOT_LINTHRESH, t0=t0, T=T,
                                                                   title_postfix=' ')# + str([len(x) for x in LNE_pop]))
        figures_dict['GP'].savefig(os.path.join(POPULATION_BASE_PATH, 'population_by_equation_hist_LNE_GP_sliced.png'),
                                   dpi=FIG_DPI,
                                   bbox_inches='tight')
        L5HT_pop = [sliced_populations[0], sliced_populations[1], sliced_populations[2], sliced_populations[4]]

        figures_dict = histplot_populations_last_value_by_equation(L5HT_pop, linthresh=PLOT_LINTHRESH, t0=t0, T=T,
                                                                   title_postfix=' ')# + str([len(x) for x in LNE_pop]))
        figures_dict['GP'].savefig(os.path.join(POPULATION_BASE_PATH, 'population_by_equation_hist_L5HT_GP_sliced.png'),
                                   dpi=FIG_DPI,
                                   bbox_inches='tight')

        # STATS

        with open(os.path.join(POPULATION_BASE_PATH, 'stats_individuals'), 'w') as f:
            f.write(str(len(individuals)))
        with open(os.path.join(POPULATION_BASE_PATH, 'stats_rejects'), 'w') as f:
            f.write(str(len(rejects)))
        with open(os.path.join(POPULATION_BASE_PATH, 'stats_population'), 'w') as f:
            f.write(str(len(rejects) + len(individuals)))

        average_generations = np.average([len(i['fitness_history']) for i in individuals])
        with open(os.path.join(POPULATION_BASE_PATH, 'stats_average_generations'), 'w') as f:
            f.write(str(int(average_generations)))
        average_time = np.average([i['fitness_history'][-1][0] - i['fitness_history'][0][0] for i in individuals])
        with open(os.path.join(POPULATION_BASE_PATH, 'stats_average_time'), 'w') as f:
            f.write('%.2f' % (average_time / 3600.))

        figure = plt.figure()
        make_grid()
        for i in individuals[::int(len(individuals) / 10)]:
            plot_fitness(i['fitness_history'], 'Combined', figure=figure, base='generations')
        figure.savefig(os.path.join(POPULATION_BASE_PATH, 'population_fitness_generations_subsample.png'), dpi=FIG_DPI,
                       bbox_inches='tight')

        figure = plt.figure()
        make_grid()
        for i in individuals[::int(len(individuals) / 10)]:
            plot_fitness(i['fitness_history'], 'Combined', figure=figure, base='time')
        figure.savefig(os.path.join(POPULATION_BASE_PATH, 'population_fitness_time_subsample.png'), dpi=FIG_DPI,
                       bbox_inches='tight')

        # Transient plots
        m = individuals[5]

        figure = plot_model(m, m.target_as_y0(), 0, 0.1, figure=None, linthresh=False)
        mLDA = m.lesion_LDA()
        mLDA['target'] = m['target']
        figure = plot_model(mLDA, m.target_as_y0(), 0.1, 0.2, figure=figure, linthresh=False)
        figure.savefig(os.path.join(POPULATION_BASE_PATH, 'transient_example_LDA.png'), dpi=FIG_DPI,
                       bbox_inches='tight')

        figure = plot_model(m, m.target_as_y0(), 0, 0.1, figure=None, linthresh=False)
        mL5HT = m.lesion_L5HT()
        mL5HT['target'] = m['target']
        figure = plot_model(mL5HT, m.target_as_y0(), 0.1, 0.2, figure=figure, linthresh=False)
        figure.savefig(os.path.join(POPULATION_BASE_PATH, 'transient_example_L5HT.png'), dpi=FIG_DPI,
                       bbox_inches='tight')

        figure = plot_model(m, m.target_as_y0(), 0, 0.1, figure=None, linthresh=False)
        mLNE = m.lesion_LNE()
        mLNE['target'] = m['target']
        figure = plot_model(mLNE, m.target_as_y0(), 0.1, 0.2, figure=figure, linthresh=False)
        figure.savefig(os.path.join(POPULATION_BASE_PATH, 'transient_example_LNE.png'), dpi=FIG_DPI,
                       bbox_inches='tight')

        figure = plot_model(m, m.target_as_y0(), 0, 0.1, figure=None, linthresh=False)
        mLDA = m.lesion_LDA()
        mLDA['target'] = m['target']
        figure = plot_model(mLDA, m.target_as_y0(), 0.1, 0.2, figure=figure, linthresh=False)
        y0 = m.lesion_LDA().simulate(m.target_as_y0(), 0, 0.1)['y'].transpose()[-1]
        mLDALNE = m.lesion_LDA_LNE()
        mLDALNE['target'] = m['target']
        figure = plot_model(mLDALNE, y0, 0.2, 0.3, figure=figure, linthresh=False)
        figure.savefig(os.path.join(POPULATION_BASE_PATH, 'transient_example_LDA+LNE.png'), dpi=FIG_DPI,
                       bbox_inches='tight')

        figure = plot_model(m, m.target_as_y0(), 0, 0.1, figure=None, linthresh=False)
        mLDA = m.lesion_LDA()
        mLDA['target'] = m['target']
        figure = plot_model(mLDA, m.target_as_y0(), 0.1, 0.2, figure=figure, linthresh=False)
        y0 = m.lesion_LDA().simulate(m.target_as_y0(), 0, 0.1)['y'].transpose()[-1]
        mLDAL5HT = m.lesion_LDA_L5HT()
        mLDAL5HT['target'] = m['target']
        figure = plot_model(mLDAL5HT, y0, 0.2, 0.3, figure=figure, linthresh=False)
        figure.savefig(os.path.join(POPULATION_BASE_PATH, 'transient_example_LDA+L5HT.png'), dpi=FIG_DPI,
                       bbox_inches='tight')

        figure = plot_model(m, m.target_as_y0(), 0, 0.1, figure=None, linthresh=False)
        mL5HT = m.lesion_LDA_L5HT()
        mL5HT['target'] = m['target']
        figure = plot_model(mL5HT, m.target_as_y0(), 0.1, 0.2, figure=figure, linthresh=False)
        figure.savefig(os.path.join(POPULATION_BASE_PATH, 'transient_example_direct_LDA+L5HT.png'), dpi=FIG_DPI,
                       bbox_inches='tight')

        figure = plot_model(m, m.target_as_y0(), 0, 0.1, figure=None, linthresh=False)
        mLNE = m.lesion_LDA_LNE()
        mLNE['target'] = m['target']
        figure = plot_model(mLNE, m.target_as_y0(), 0.1, 0.2, figure=figure, linthresh=False)
        figure.savefig(os.path.join(POPULATION_BASE_PATH, 'transient_example_direct_LDA+LNE.png'), dpi=FIG_DPI,
                       bbox_inches='tight')

        figures_dict = histplot_population_parameters(populations, title_postfix='', linthresh=False)
        for param, figure in figures_dict.items():
            figure.savefig(os.path.join(POPULATION_BASE_PATH,
                                        'populations_by_parameter_%s.png' % (param)),
                           dpi=FIG_DPI,
                           bbox_inches='tight')
        figures_dict = histplot_population_parameters(sliced_populations, title_postfix=sliced_populations_desc,
                                                      linthresh=False)
        for param, figure in figures_dict.items():
            figure.savefig(os.path.join(POPULATION_BASE_PATH,
                                        'populations_by_parameter_%s_sliced.png' % (param)),
                           dpi=FIG_DPI,
                           bbox_inches='tight')


if __name__ == '__main__':
    main(fit=True, plot=False)
