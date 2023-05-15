from CONF import POPULATION_BASE_PATH, SIMULATION_TIME
from models import *
from plotting import *

PLOT_DERIVATIVES = False
PLOT_LINTHRESH = False  # 10 ** -4
BOXPLOT_LINTHRESH = 1e-4


def main(fit=True, plot=False):

    checkpoint_filename = False  # './HEALTHY_CHECKPOINT'
    model = Healthy_combined_fit()
    model.apply()
    y0 = model.target_as_y0()
    t0 = 0
    T = SIMULATION_TIME

    if fit:
        model['name'] = 'S_000'
        model, fitness_history = model.optimize(y0, t0, T, save_checkpoint_name=checkpoint_filename)
        model.save(POPULATION_BASE_PATH + 'S_000')
    else:
        model = model.load(POPULATION_BASE_PATH + 'S_000')

    if plot:
        print_title("STEP 01: healthy fit on average target data", 'STEP 01')
        plot_fitness(model['fitness_history'], model['name'], base='generations')
        plot_fitness(model['fitness_history'], model['name'], base='time')
        model.apply()
        plot_parameters([model])
        plot_model(model, model.target_as_y0(), t0, T,
                   linthresh=PLOT_LINTHRESH)

        lesions = [model.lesion_LDA(), model.lesion_LNE(), model.lesion_L5HT(),
                   model.lesion_LDA_LNE(), model.lesion_LDA_L5HT()]

        for model in lesions:
            plot_parameters([model, model], linthresh=BOXPLOT_LINTHRESH)
            plot_model(model, model.target_as_y0(), t0, T, linthresh=PLOT_LINTHRESH)

        plt.show()


if __name__ == '__main__':
    main(fit=True, plot=False)
