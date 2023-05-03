from CONF import SIMULATION_TIME, POPULATION_BASE_PATH
from models import *
from plotting import *
from s02_population import slice_populations
from s04_treatment import BL, POPULATION_BASE_PATH___CURE_COMBINED, POPULATION_BASE_PATH___CURE_DRN, \
    POPULATION_BASE_PATH___CURE_LC
import os

PLOT_DERIVATIVES = False
PLOT_LINTHRESH = False  # 10 ** -4
BOXPLOT_LINTHRESH = 1e-4


def tukey__str__(tukey_stat):
    # Note: `__str__` prints the confidence intervals from the most
    # recent call to `confidence_interval`. If it has not been called,
    # it will be called with the default CL of .95.
    if tukey_stat._ci is None:
        tukey_stat.confidence_interval(confidence_level=.95)
    s = ("Tukey's HSD Pairwise Group Comparisons"
         f" ({tukey_stat._ci_cl * 100:.1f}% Confidence Interval)\n")
    s += "Comparison  Statistic   p-value     Lower  CI   Upper CI\n"
    for i in range(tukey_stat.pvalue.shape[0]):
        for j in range(i, tukey_stat.pvalue.shape[0]):
            if i != j:
                s += (f" ({i} - {j}) {tukey_stat.statistic[i, j]:>12.3e}"
                      f"{tukey_stat.pvalue[i, j]:>12.3e}"
                      f"{tukey_stat._ci.low[i, j]:>12.3e}"
                      f"{tukey_stat._ci.high[i, j]:>12.3e}\n")
    return s


def main(fit=True, plot=False):
    t0 = 0
    T = SIMULATION_TIME
    if fit:
        pass
    else:
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
        print_title("STEP 05: statistics", 'STEP 05')
        reject_threshold = 1 - 2e-8
        rejects = [ind for ind in individuals if ind['fitness_history'][-1][1] <= reject_threshold]
        individuals = [ind for ind in individuals if ind['fitness_history'][-1][1] > reject_threshold]

        populations = list()

        lesions_list = ['lesion_SHAM', 'lesion_L6OHDA', 'lesion_LpCPA', 'lesion_LDSP4',
                        'lesion_L6OHDA_LpCPA',
                        'lesion_L6OHDA_LDSP4']
        for lesion in lesions_list:
            populations.append([i.__getattribute__(lesion)() for i in individuals])

        populations = slice_populations(populations)
        populations_pCPA = [populations[0], populations[1], populations[2], populations[4]]
        ll_pCPA = ['lesion_SHAM', 'lesion_L6OHDA', 'lesion_LpCPA',
                   'lesion_L6OHDA_LpCPA',
                   ]
        ll_DSP4 = ['lesion_SHAM', 'lesion_L6OHDA', 'lesion_LDSP4',
                   'lesion_L6OHDA_LpCPA',
                   ]
        populations_DSP4 = [populations[0], populations[1], populations[3], populations[5]]

        for pop, ll in [(populations, lesions_list), (populations_pCPA, ll_pCPA), (populations_DSP4, ll_DSP4)]:
            for eq in pop[0][0]['equations']:
                eq_data = np.array(
                        [np.array(
                                [m.simulate(m.target_as_y0(), t0, T)['y'].transpose()[-1][m['equations'].index(eq)] for
                                 m
                                 in
                                 p]) for p in pop])

                DFB = len(eq_data) - 1
                DFW = len(eq_data.flatten()) - len(eq_data)
                f, p = stats.f_oneway(*eq_data)
                text = ["Groups: %s" % ' '.join(str(i) + ':' + BL[g] for i, g in enumerate(ll))]
                text.append("ANOVA: F=%.3e, p=%.3e, dofB=%s, dofW=%s" % (f, p, DFB, DFW))
                text.append(tukey__str__(stats.tukey_hsd(*eq_data)))
                text = '\n'.join(text)
                with open(os.path.join(POPULATION_BASE_PATH, 'ANOVA_lesions_%s___%s.txt' % (eq, '-'.join(ll))),
                          'w') as f:
                    f.write(text)
                print(eq + ':\n')
                print(text)

        populations = slice_populations([individuals, individuals_6OHDA,
                                         [i.cure_DRN() for i in individuals_cure_DRN],
                                         [i.cure_LC() for i in individuals_cure_LC],
                                         [i.cure_DRN_LC() for i in individuals_cure_combined],
                                         ])

        groups = ['SHAM', BL['lesion_L6OHDA'], '+cure_DRN',
                  '+cure_LC', '+cure_DRN+cure_LC']

        sim_data = [Parallel(n_jobs=-1)(delayed(m.simulate)(m.target_as_y0(), t0, T) for m in p) for p in
                    populations]

        eqs_index = populations[0][0]['equations'].index

        for eq in populations[0][0]['equations']:
            eq_data = np.array(
                    [np.array(
                            [m['y'].transpose()[-1][eqs_index(eq)] for m
                             in
                             p]) for p in sim_data])

            DFB = len(eq_data) - 1
            DFW = len(eq_data.flatten()) - len(eq_data)
            f, p = stats.f_oneway(*eq_data)
            text = ["Groups: %s" % ' '.join(str(i) + ':' + g for i, g in enumerate(groups))]
            text.append("ANOVA: F=%s, p=%s, dofB=%s, dofW=%s" % (f, p, DFB, DFW))
            text.append(tukey__str__(stats.tukey_hsd(*eq_data)))
            text = '\n'.join(text)
            with open(os.path.join(POPULATION_BASE_PATH, 'ANOVA_cure_%s.txt' % eq), 'w') as f:
                f.write(text)
            print(eq + ':\n')
            print(text)


if __name__ == '__main__':
    main(fit=False, plot=True)
