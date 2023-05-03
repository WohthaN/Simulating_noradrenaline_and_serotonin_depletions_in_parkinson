from CONF import FIG_DPI, SIMULATION_TIME, POPULATION_BASE_PATH
from models import *
from plotting import *
from s02_population import slice_populations
import os

PLOT_DERIVATIVES = False
PLOT_LINTHRESH = False  # 10 ** -4
BOXPLOT_LINTHRESH = 1e-3

BL = {'lesion_SHAM'        : '+SHAM',
      'lesion_L6OHDA'      : '+6OHDA',
      'lesion_LpCPA'       : '+pCPA',
      'lesion_LDSP4'       : '+DSP4',
      'lesion_L6OHDA_LpCPA': '+6OHDA+pCPA',
      'lesion_L6OHDA_LDSP4': '+6OHDA+DSP4'}

POPULATION_BASE_PATH___CURE_DRN = os.path.join(POPULATION_BASE_PATH, 'CURE_DRN')
POPULATION_BASE_PATH___CURE_LC = os.path.join(POPULATION_BASE_PATH, 'CURE_LC')
POPULATION_BASE_PATH___CURE_COMBINED = os.path.join(POPULATION_BASE_PATH, 'CURE_DRN_LC')

if not os.path.exists(POPULATION_BASE_PATH___CURE_DRN):
    os.mkdir(POPULATION_BASE_PATH___CURE_DRN)
if not os.path.exists(POPULATION_BASE_PATH___CURE_LC):
    os.mkdir(POPULATION_BASE_PATH___CURE_LC)
if not os.path.exists(POPULATION_BASE_PATH___CURE_COMBINED):
    os.mkdir(POPULATION_BASE_PATH___CURE_COMBINED)


def is_already_optimized(filename):
    try:
        Healthy_combined_fit.load(filename)
        return True
    except:
        return False

def main(fit=True, plot=False):
    t0 = 0
    T = SIMULATION_TIME
    if fit:
        files = sorted(filter(lambda x: x.startswith('S_'), os.listdir(POPULATION_BASE_PATH + '')))
        individuals = [Healthy_combined_fit.load(POPULATION_BASE_PATH + '%s' % f) for f in files]
        reject_threshold = 1 - 2e-8
        individuals = [ind for ind in individuals if ind['fitness_history'][-1][1] > reject_threshold]
        individuals_6OHDA = [ind._impose_target(ind.lesion_L6OHDA()) for ind in individuals]

        individuals_cure_DRN = list()
        individuals_cure_LC = list()
        individuals_cure_combined = list()

        for i in individuals_6OHDA:
            cure_individual = Cure_DRN()
            cure_individual.update(i.copy(keep_fitness_history=False))
            cure_individual.apply()
            individuals_cure_DRN.append(cure_individual)

            cure_individual = Cure_LC()
            cure_individual.update(i.copy(keep_fitness_history=False))
            cure_individual.apply()
            individuals_cure_LC.append(cure_individual)

            cure_individual = Cure_combined()
            cure_individual.update(i.copy(keep_fitness_history=False))
            cure_individual.apply()
            individuals_cure_combined.append(cure_individual)

        optimizer_popsize = 60
        tol = 1e-6
        for idx, individual in enumerate(individuals_cure_DRN):
            individual.apply()
            filename = (POPULATION_BASE_PATH___CURE_DRN + '/%s' % individual['name']).replace(' ', '_')
            if not is_already_optimized(filename):
                fitted_individual = \
                    individual.optimize(individual.target_as_y0(), t0, T, seed=1, popsize=optimizer_popsize, tol=tol)[0]
                fh = fitted_individual['fitness_history']
                fitted_individual['fitness_history'] = fh
                fitted_individual.save(filename)
                individuals_cure_DRN[idx] = fitted_individual

        for idx, individual in enumerate(individuals_cure_LC):
            individual.apply()
            filename = (POPULATION_BASE_PATH___CURE_LC + '/%s' % individual['name']).replace(' ', '_')
            if not is_already_optimized(filename):
                fitted_individual = \
                    individual.optimize(individual.target_as_y0(), t0, T, seed=1, popsize=optimizer_popsize, tol=tol)[0]
                fh = fitted_individual['fitness_history']
                fitted_individual['fitness_history'] = fh
                fitted_individual.save(filename)
                individuals_cure_LC[idx] = fitted_individual

        for idx, individual in enumerate(individuals_cure_combined):
            individual.apply()
            filename = (POPULATION_BASE_PATH___CURE_COMBINED + '/%s' % individual['name']).replace(' ', '_')
            if not is_already_optimized(filename):
                fitted_individual = \
                    individual.optimize(individual.target_as_y0(), t0, T, seed=1, popsize=int(optimizer_popsize / 2),
                                        tol=tol)[0]
                fh = fitted_individual['fitness_history']
                fitted_individual['fitness_history'] = fh
                fitted_individual.save(filename)
                individuals_cure_combined[idx] = fitted_individual


    files = sorted(filter(lambda x: x.startswith('S_'), os.listdir(POPULATION_BASE_PATH + '')))
    individuals = [Healthy_combined_fit.load(POPULATION_BASE_PATH + '%s' % f) for f in files]
    reject_threshold = 1 - 2e-8
    individuals = [ind for ind in individuals if ind['fitness_history'][-1][1] > reject_threshold]
    individuals_6OHDA = [ind._impose_target(ind.lesion_L6OHDA()) for ind in individuals]

    files = sorted(filter(lambda x: x.startswith('S_'), os.listdir(POPULATION_BASE_PATH___CURE_DRN + '')))
    individuals_cure_DRN = [Cure_DRN.load(POPULATION_BASE_PATH___CURE_DRN + '/%s' % f) for f in files]
    for i in individuals_cure_DRN:
        i.apply()

    files = sorted(filter(lambda x: x.startswith('S_'), os.listdir(POPULATION_BASE_PATH___CURE_LC + '')))
    individuals_cure_LC = [Cure_LC.load(POPULATION_BASE_PATH___CURE_LC + '/%s' % f) for f in files]
    for i in individuals_cure_LC:
        i.apply()

    files = sorted(filter(lambda x: x.startswith('S_'), os.listdir(POPULATION_BASE_PATH___CURE_COMBINED + '')))
    individuals_cure_combined = [Cure_combined.load(POPULATION_BASE_PATH___CURE_COMBINED + '/%s' % f) for f in
                                 files]
    for i in individuals_cure_combined:
        i.apply()

    if plot:
        print_title("STEP 04: treatment", 'STEP 04')

        # FITNESS DISTRIBUTIONS
        figure = plot_population_fitness_distribution(individuals, title_label='SHAM ')
        figure.savefig(os.path.join(POPULATION_BASE_PATH, 'population_cure_fitness_distribution.png'), dpi=FIG_DPI,
                       bbox_inches='tight')

        figure = plot_population_fitness_distribution(individuals_cure_DRN, title_label='cure_DRN ')
        figure.savefig(os.path.join(POPULATION_BASE_PATH, 'population_cure_DRN_fitness_distribution.png'), dpi=FIG_DPI,
                       bbox_inches='tight')

        figure = plot_population_fitness_distribution(individuals_cure_LC, title_label='cure_LC ')
        figure.savefig(os.path.join(POPULATION_BASE_PATH, 'population_cure_LC_fitness_distribution.png'), dpi=FIG_DPI,
                       bbox_inches='tight')

        figure = plot_population_fitness_distribution(individuals_cure_combined, title_label='cure_combined ')
        figure.savefig(os.path.join(POPULATION_BASE_PATH, 'population_cure_combined_fitness_distribution.png'),
                       dpi=FIG_DPI,
                       bbox_inches='tight')

        # DELTA FITNESS
        figure = plot_population_fitness_delta_distribution(individuals, title_label='SHAM ')
        figure.savefig(os.path.join(POPULATION_BASE_PATH, 'population_cure_fitness_delta_distribution.png'),
                       dpi=FIG_DPI,
                       bbox_inches='tight')

        figure = plot_population_fitness_delta_distribution(individuals_cure_DRN, title_label='cure_DRN ')
        figure.savefig(os.path.join(POPULATION_BASE_PATH, 'population_cure_DRN_fitness_delta_distribution.png'),
                       dpi=FIG_DPI,
                       bbox_inches='tight')

        figure = plot_population_fitness_delta_distribution(individuals_cure_LC, title_label='cure_LC ')
        figure.savefig(os.path.join(POPULATION_BASE_PATH, 'population_cure_LC_fitness_delta_distribution.png'),
                       dpi=FIG_DPI,
                       bbox_inches='tight')

        figure = plot_population_fitness_delta_distribution(individuals_cure_combined, title_label='cure_combined ')
        figure.savefig(os.path.join(POPULATION_BASE_PATH, 'population_cure_combined_fitness_delta_distribution.png'),
                       dpi=FIG_DPI,
                       bbox_inches='tight')

        # STABILITY
        stability_plot = plot_max_eigenvalue_distribution(individuals, title_label='SHAM ')
        stability_plot.savefig(os.path.join(POPULATION_BASE_PATH, 'cure_population_max_eigenvalues_%s.png' % "SHAM"),
                               dpi=FIG_DPI,
                               bbox_inches='tight')

        stability_plot = plot_max_eigenvalue_distribution(individuals_cure_DRN, title_label='cure_DRN ')
        stability_plot.savefig(
                os.path.join(POPULATION_BASE_PATH, 'cure_population_max_eigenvalues_%s.png' % "cure_DRN"),
                dpi=FIG_DPI,
                bbox_inches='tight')

        stability_plot = plot_max_eigenvalue_distribution(individuals_cure_LC, title_label='cure_LC ')
        stability_plot.savefig(os.path.join(POPULATION_BASE_PATH, 'cure_population_max_eigenvalues_%s.png' % "cure_LC"),
                               dpi=FIG_DPI,
                               bbox_inches='tight')

        stability_plot = plot_max_eigenvalue_distribution(individuals_cure_combined, title_label='cure_combined ')
        stability_plot.savefig(
                os.path.join(POPULATION_BASE_PATH, 'cure_population_max_eigenvalues_%s.png' % "cure_combined"),
                dpi=FIG_DPI,
                bbox_inches='tight')

        # PARAMETERS
        figure = boxplot_population_parameters(individuals, linthresh=BOXPLOT_LINTHRESH)
        figure.savefig(os.path.join(POPULATION_BASE_PATH, 'population_cure_sham_parameters_distribution.png'),
                       dpi=FIG_DPI,
                       bbox_inches='tight')

        for i in individuals_cure_DRN:
            i['parameters']['a_EXT_DRN'] = i.P['a_EXT_DRN']
        figure = boxplot_population_parameters(individuals_cure_DRN, linthresh=BOXPLOT_LINTHRESH)
        figure.savefig(os.path.join(POPULATION_BASE_PATH, 'population_cure_DRN_parameters_distribution.png'),
                       dpi=FIG_DPI,
                       bbox_inches='tight')

        for i in individuals_cure_LC:
            i['parameters']['a_EXT_LC'] = i.P['a_EXT_LC']
        figure = boxplot_population_parameters(individuals_cure_LC, linthresh=BOXPLOT_LINTHRESH)
        figure.savefig(os.path.join(POPULATION_BASE_PATH, 'population_cure_LC_parameters_distribution.png'),
                       dpi=FIG_DPI,
                       bbox_inches='tight')

        for i in individuals_cure_combined:
            i['parameters']['a_EXT_DRN'] = i.P['a_EXT_DRN']
            i['parameters']['a_EXT_LC'] = i.P['a_EXT_LC']
        figure = boxplot_population_parameters(individuals_cure_combined, linthresh=BOXPLOT_LINTHRESH)
        figure.savefig(os.path.join(POPULATION_BASE_PATH, 'population_cure_combined_parameters_distribution.png'),
                       dpi=FIG_DPI,
                       bbox_inches='tight')

        populations = [individuals, individuals_6OHDA,
                       [i.cure_DRN() for i in individuals_cure_DRN],
                       [i.cure_LC() for i in individuals_cure_LC],
                       [i.cure_DRN_LC() for i in individuals_cure_combined],
                       ]

        sliced_populations = slice_populations(populations)
        sliced_populations_desc = ' ' + str([len(x) for x in sliced_populations])

        figures_dict = boxplot_populations_last_value_by_equation(populations, linthresh=PLOT_LINTHRESH, t0=t0, T=T,
                                                                  title_postfix=' whole population')

        for eq_name, figure in figures_dict.items():
            figure.savefig(os.path.join(POPULATION_BASE_PATH,
                                        'cure_population_6OHDA_by_equation_%s.png' % (eq_name)),
                           dpi=FIG_DPI,
                           bbox_inches='tight')

        figures_dict = boxplot_populations_last_value_by_equation(sliced_populations, linthresh=PLOT_LINTHRESH,
                                                                  t0=t0, T=T,
                                                                  title_postfix=sliced_populations_desc)
        for eq_name, figure in figures_dict.items():
            figure.savefig(os.path.join(POPULATION_BASE_PATH,
                                        'cure_population_6OHDA_by_equation_%s_sliced.png' % (eq_name)),
                           dpi=FIG_DPI,
                           bbox_inches='tight')

        figures_dict = histplot_populations_last_value_by_equation(populations, linthresh=PLOT_LINTHRESH, t0=t0,
                                                                   T=T, title_postfix=' whole population')
        for eq_name, figure in figures_dict.items():
            figure.savefig(os.path.join(POPULATION_BASE_PATH,
                                        'cure_population_6OHDA_by_equation_hist_%s.png' % (eq_name)),
                           dpi=FIG_DPI,
                           bbox_inches='tight')

        figures_dict = histplot_populations_last_value_by_equation(sliced_populations, linthresh=PLOT_LINTHRESH,
                                                                   t0=t0,
                                                                   T=T, title_postfix=sliced_populations_desc)

        for eq_name, figure in figures_dict.items():
            figure.savefig(os.path.join(POPULATION_BASE_PATH,
                                        'cure_population_6OHDA_by_equation_hist_%s_sliced.png' % (eq_name)),
                           dpi=FIG_DPI,
                           bbox_inches='tight')

        base_lesion = 'lesion_L6OHDA'
        kind = ['SHAM', BL[base_lesion], '+EXT_DRN', '+EXT_LC', '+EXT_DRN+EXT_LC']
        for fitted_individual, pop in enumerate(populations):
            (boxplot, plot) = plot_population(pop[0], pop, None, t0, T, linthresh=PLOT_LINTHRESH, plot_target=False)
            boxplot.savefig(
                    os.path.join(POPULATION_BASE_PATH,
                                 'cure_population_%s_target_%s.png' % (base_lesion, kind[fitted_individual])),
                    dpi=FIG_DPI,
                    bbox_inches='tight')
            plot.savefig(os.path.join(POPULATION_BASE_PATH,
                                      'cure_population_%s_target_%s_plot.png' % (
                                          base_lesion, kind[fitted_individual])),
                         dpi=FIG_DPI,
                         bbox_inches='tight')

        for fitted_individual, pop in enumerate(sliced_populations):
            (boxplot, plot) = plot_population(pop[0], pop, None, t0, T, linthresh=PLOT_LINTHRESH, plot_target=False)
            boxplot.savefig(
                    os.path.join(POPULATION_BASE_PATH,
                                 'cure_population_%s_target_%s_sliced.png' % (
                                     base_lesion, kind[fitted_individual])),
                    dpi=FIG_DPI,
                    bbox_inches='tight')
            plot.savefig(os.path.join(POPULATION_BASE_PATH,
                                      'cure_population_%s_target_%s_plot_sliced.png' % (
                                          base_lesion, kind[fitted_individual])),
                         dpi=FIG_DPI,
                         bbox_inches='tight')

        # CURABLE VS UNCURABLE
        stats_individuals_cure_combined = list()
        stats_individuals = list()
        stats_individuals_L6OHDA = list()

        for i in individuals:
            sham = i.lesion_SHAM()
            l6ohda = Healthy_combined_fit()
            l6ohda.update(i.lesion_L6OHDA().copy())
            l6ohda = l6ohda.lesion_SHAM()
            sham['fitness_history'] = i['fitness_history']
            l6ohda['fitness_history'] = i['fitness_history']
            stats_individuals.append(sham)
            stats_individuals_L6OHDA.append(l6ohda)

        for i in individuals_cure_combined:
            cured = i.cure_DRN_LC().lesion_SHAM()
            cured['fitness_history'] = i['fitness_history']
            stats_individuals_cure_combined.append(cured)

        cured_threshold = 5

        cured = [i for i in stats_individuals_cure_combined if
                 -np.log10(1 - i['fitness_history'][-1][1]) >= cured_threshold]
        not_cured = [i for i in stats_individuals_cure_combined if
                     -np.log10(1 - i['fitness_history'][-1][1]) < cured_threshold]

        figure = boxplot_population_parameters(stats_individuals, linthresh=BOXPLOT_LINTHRESH, color='black')
        figure = boxplot_population_parameters(stats_individuals_L6OHDA, linthresh=BOXPLOT_LINTHRESH, color='orange',
                                               figure=figure, alt_title='SHAM vs 6OHDA (%s - %s)' % (
                len(stats_individuals), len(stats_individuals_L6OHDA)))
        figure.savefig(os.path.join(POPULATION_BASE_PATH,
                                    'cure_population_parameters_SHAM_vs_6OHDA.png'),
                       dpi=FIG_DPI,
                       bbox_inches='tight')

        figure = boxplot_population_parameters(stats_individuals, linthresh=BOXPLOT_LINTHRESH, color='black')
        figure = boxplot_population_parameters(stats_individuals_cure_combined, linthresh=BOXPLOT_LINTHRESH,
                                               color='orange',
                                               figure=figure, alt_title='SHAM vs TREATED (%s - %s)' % (
                len(stats_individuals), len(stats_individuals_cure_combined)))
        figure.savefig(os.path.join(POPULATION_BASE_PATH,
                                    'cure_population_parameters_SHAM_vs_TREATED.png'),
                       dpi=FIG_DPI,
                       bbox_inches='tight')

        figure = boxplot_population_parameters(stats_individuals_L6OHDA, linthresh=BOXPLOT_LINTHRESH, color='black')
        figure = boxplot_population_parameters(stats_individuals_cure_combined, linthresh=BOXPLOT_LINTHRESH,
                                               color='orange',
                                               figure=figure, alt_title='6OHDA vs TREATED (%s - %s)' % (
                len(stats_individuals_L6OHDA), len(stats_individuals_cure_combined)))
        figure.savefig(os.path.join(POPULATION_BASE_PATH,
                                    'cure_population_parameters_6OHDA_vs_TREATED.png'),
                       dpi=FIG_DPI,
                       bbox_inches='tight')

        figure = boxplot_population_parameters(cured, linthresh=BOXPLOT_LINTHRESH, color='green')
        figure = boxplot_population_parameters(not_cured, linthresh=BOXPLOT_LINTHRESH, color='red', figure=figure,
                                               alt_title='Cured vs Not-Cured (%s - %s)' % (
                                                   len(cured), len(not_cured)))
        figure.savefig(os.path.join(POPULATION_BASE_PATH,
                                    'cure_population_parameters_Cured_vs_NotCured.png'),
                       dpi=FIG_DPI,
                       bbox_inches='tight')

        figure = boxplot_population_parameters(stats_individuals_L6OHDA, linthresh=BOXPLOT_LINTHRESH, color='black')
        figure = boxplot_population_parameters(cured, linthresh=BOXPLOT_LINTHRESH, color='orange',
                                               figure=figure, alt_title='6OHDA vs Cured (%s - %s)' % (
                len(stats_individuals_L6OHDA), len(cured)))
        figure.savefig(os.path.join(POPULATION_BASE_PATH,
                                    'cure_population_parameters_6OHDA_vs_CURED.png'),
                       dpi=FIG_DPI,
                       bbox_inches='tight')

        figure = boxplot_population_parameters(stats_individuals_L6OHDA, linthresh=BOXPLOT_LINTHRESH, color='black')
        figure = boxplot_population_parameters(not_cured, linthresh=BOXPLOT_LINTHRESH, color='orange',
                                               figure=figure, alt_title='6OHDA vs Not-Cured (%s - %s)' % (
                len(stats_individuals_L6OHDA), len(not_cured)))
        figure.savefig(os.path.join(POPULATION_BASE_PATH,
                                    'cure_population_parameters_6OHDA_vs_NotCured.png'),
                       dpi=FIG_DPI,
                       bbox_inches='tight')

        a_EXT_DRN = list()
        a_EXT_LC = list()
        CDRN___a_EXT_DRN = list()
        CLC___a_EXT_LC = list()
        FITS = list()
        for i in [i for i in individuals_cure_combined if
                  -np.log10(1 - i['fitness_history'][-1][1]) >= cured_threshold]:
            a_EXT_DRN.append(i.P['a_EXT_DRN'])
            a_EXT_LC.append(i.P['a_EXT_LC'])
            CDRN___a_EXT_DRN.append(i.P['CDRN___a_EXT_DRN'])
            CLC___a_EXT_LC.append(i.P['CLC___a_EXT_LC'])
            FITS.append(-np.log10(1 - i['fitness_history'][-1][1]))

        a_EXT_DRN = np.array(a_EXT_DRN)
        a_EXT_LC = np.array(a_EXT_LC)
        CDRN___a_EXT_DRN = np.array(CDRN___a_EXT_DRN)
        CLC___a_EXT_LC = np.array(CLC___a_EXT_LC)

        D_DRN = (CDRN___a_EXT_DRN - a_EXT_DRN) / a_EXT_DRN
        D_LC = (CLC___a_EXT_LC - a_EXT_LC) / a_EXT_LC

        FITS = np.array(FITS)
        FITS = 255 * (FITS - min(FITS)) / (max(FITS) - min(FITS))
        FITS = [plt.cm.plasma(int(f)) for f in FITS]
        figure = plt.figure()

        plt.scatter(D_LC, D_DRN, c=FITS)
        plt.title('Treatment Combined Relative $\Delta$a_EXT_LC vs $\Delta$a_EXT_DRN (%s)' % len(D_LC))
        plt.xlabel('$\Delta$a_EXT_LC')
        plt.ylabel('$\Delta$a_EXT_DRN')
        plt.xscale('symlog', linthresh=1e-1, subs=SUBS)
        plt.yscale('symlog', linthresh=1e-8, subs=SUBS)
        plt.grid(which='both')
        figure.savefig(os.path.join(POPULATION_BASE_PATH,
                                    'cure_combined_relativedelta_parameters.png'),
                       dpi=FIG_DPI,
                       bbox_inches='tight')

        figures_dict = histplot_population_parameters(populations, title_postfix=' whole population', linthresh=False)
        for param, figure in figures_dict.items():
            figure.savefig(os.path.join(POPULATION_BASE_PATH,
                                        'cure_populations_by_parameter_%s.png' % (param)),
                           dpi=FIG_DPI,
                           bbox_inches='tight')

        figures_dict = histplot_population_parameters(sliced_populations, title_postfix=sliced_populations_desc,
                                                      linthresh=False)
        for param, figure in figures_dict.items():
            figure.savefig(os.path.join(POPULATION_BASE_PATH,
                                        'cure_populations_by_parameter_%s_sliced.png' % (param)),
                           dpi=FIG_DPI,
                           bbox_inches='tight')

        for i in cured:
            i['name'] = i['name'].split(' ')[0] + ' CURED'
        for i in not_cured:
            i['name'] = i['name'].split(' ')[0] + ' NOT_CURED'

        populations = [individuals, individuals_6OHDA, cured, not_cured]
        figures_dict = histplot_population_parameters(populations, title_postfix=' whole population', linthresh=False)
        for param, figure in figures_dict.items():
            figure.savefig(os.path.join(POPULATION_BASE_PATH,
                                        'cure_vs_not_cured_populations_by_parameter_%s.png' % (param)),
                           dpi=FIG_DPI,
                           bbox_inches='tight')


if __name__ == '__main__':
    # main(fit=False, plot=True)
    main(fit=True, plot=False)
