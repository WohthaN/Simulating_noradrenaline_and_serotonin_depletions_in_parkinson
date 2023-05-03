import copy
import functools
import gc
import math
import pickle
import pprint
import time
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import scipy
import scipy.optimize
import tqdm
from numpy import array as nparray, diagflat, linalg
from numpy.linalg import inv as invert_matrix
from scipy.integrate import solve_ivp
import signal

from plotting import *

np.random.seed(5)

# np.seterr(over='raise', divide='raise', invalid='raise', under='ignore')
# np.seterr(all='raise')

PARAMETERS_SEARCH_HISTORY_FILE_PATH = '/ramtmp/PSH'

nano = 1.  # 10**-9
milli = 10 ** -3

from datetime import datetime

PLOT_OPTIMIZE = False
TIME_LIMIT_SECONDS = 60 * 60 * 12
IVP_TIME_LIMIT_SECONDS = 3


def sieve_pass_positive(x):
    return np.maximum(0, x)


def sieve_pass_negative(x):
    return np.minimum(0, x)


def sieve_pass_all(x):
    return x


class TimeOutException(Exception):
    pass


class Model(dict):

    def __init__(self):
        self['name'] = 'Model'
        self['equations'] = list()
        self['parameters'] = dict()
        self['parameters_constraints'] = dict()
        self['constants'] = dict()
        self['target'] = dict()
        self['target_constraints'] = dict()
        self['applied_lesions'] = list()
        self['fitness_history'] = list()
        self.save_parameters_search_history = False
        self.default_min_param = 0
        self.default_max_param = 1e5

    def apply(self):
        pass

    def copy(self, keep_fitness_history=False):
        fh = self.pop('fitness_history')
        c = copy.deepcopy(self)
        self['fitness_history'] = fh
        if not keep_fitness_history:
            c['fitness_history'] = list()
        else:
            c['fitness_history'] = copy.deepcopy(fh)
        return c

    def _impose_target(self, other):
        other['target'] = copy.deepcopy(self['target'])
        return other

    def print(self):
        m = self.copy()
        m.pop('fitness_history')
        pprint.pprint(m, sort_dicts=True, width=100)

    def save(self, filename):
        self['timestamp'] = datetime.now().isoformat()
        with open(filename, 'bw') as f:
            pickle.dump(self.copy(keep_fitness_history=True), f)

    def _invalidate_caches(self):
        for cp in ['P', 'parameters_and_constants', 'E', 'y_prime_functions']:
            if cp in self.__dict__:
                del self.__dict__[cp]

    def _clean_constants(self):
        for k in self['parameters'].keys():
            try:
                self['constants'].pop(k)
            except KeyError:
                pass

    def __setitem__(self, key, value):
        super(Model, self).__setitem__(key, value)
        self._invalidate_caches()

    @classmethod
    def load(self, filename):
        with open(filename, 'br') as f:
            new = self()
            new.update(pickle.load(f))
            return new

    @functools.cached_property
    def parameters_and_constants(self):
        # parameters can overwrite constants!
        return self['constants'] | self['parameters']

    @functools.cached_property
    def P(self):
        """
        combined dictionary with all constants and parameters
        :return:
        """
        return self.parameters_and_constants

    @functools.cached_property
    def E(self):
        """
        equation index dictionary
        :return:
        """
        return dict((k, i) for (i, k) in enumerate(self['equations']))

    def with_constants_only(self):
        self['constants'] = self.P
        self['parameters'] = {}
        self._clean_constants()
        self._invalidate_caches()
        return self

    @functools.cached_property
    def y_prime_functions(self):
        return tuple(self.__getattribute__(f)() for f in self['equations'])

    def y_prime(self, t, y):
        return nparray([f(t, y) for f in self.y_prime_functions])

    def simulate(self, y0: np.array, t0: float, T: float) -> dict:

        def event_negative(t, y):
            return min(y)

        event_negative.terminal = True
        event_negative.direction = 1

        def event_too_large(t, y):
            return max(y) - 100

        event_negative.terminal = True
        event_negative.direction = -1

        def timeout_handler(num, stack):
            raise TimeOutException()

        signal.signal(signal.SIGALRM, timeout_handler)
        signal.alarm(IVP_TIME_LIMIT_SECONDS)
        try:
            sim = solve_ivp(self.y_prime, (t0, T), y0,
                            method='BDF',
                            vectorized=True,
                            max_step=(T - t0) / 25,
                            events=[event_negative, event_too_large]
                            )
            signal.alarm(0)
        except TimeOutException:
            # The simulation is taking too long, it must be taking too small steps.
            # Return a fake 'null' solution.
            sim = {
                't': np.zeros(25),
                'y': np.zeros((len(self['equations']), 25))
            }
            signal.alarm(0)
        return sim

    def target_as_y0(self):
        return np.array([self['target'][eq] if eq in self['target'] else 0.5 for eq in self['equations']])

    def _fitness_simulation_mse(self, y0, t0, T, limit_to_equations=None,
                                simulation=None,
                                ignore_before_t=None,
                                sieve=sieve_pass_all):

        # for e in limit_to_equations:
        #     if e not in self['equations']:
        #         raise Exception('WRONG limit_to_equation EQ %s!' % e)

        if ignore_before_t is None:
            ignore_before_t = t0

        errors = list()

        if not simulation:
            res = self.simulate(y0, t0, T)
        else:
            res = simulation

        solution = res['y']
        time = res['t']

        # Ignore the ignore_before_t if the simulation didn't go long enough, or if there aren't enough values
        # after.
        if time.max() <= ignore_before_t or (time >= ignore_before_t).sum() <= 3:
            ignore_before_t = t0

        solution = solution[:, time >= ignore_before_t]
        time = time[time >= ignore_before_t]

        t = time[1:]
        dt = t - time[:-1]

        for k, v in self['target'].items():
            if limit_to_equations is None or (limit_to_equations is not None and k in limit_to_equations):
                if not callable(v):
                    f = lambda t: np.ones(len(t)) * v
                else:
                    f = v

                e = sieve(np.array(solution[self.E[k]][1:] - f(t))) ** 2 * dt
                errors.append(e)

        errors = np.array(errors)

        return errors.sum()

    def _fitness_time_score(self, y0, t0, T, simulation=None, ignore_before_t=None):

        if not simulation:
            simulation = self.simulate(y0, t0, T)

        # 0 if no simulation, 1 if whole interval integrated (account for early stopping)
        t = simulation['t']
        if ignore_before_t:
            t0 = ignore_before_t
            if t[-1] < t0:
                return 0
        return ((t[-1] - t0) / (T - t0))

    def _fitness_simulation(self, y0, t0, T, limit_to_equations=None, simulation=None, ignore_before_t=None,
                            sieve=sieve_pass_all):

        if not simulation:
            res = self.simulate(y0, t0, T)
        else:
            res = simulation

        # if not res['success']:
        #     return 0

        t = res['t']

        mse = self._fitness_simulation_mse(y0, t0, T,
                                           limit_to_equations=limit_to_equations,
                                           simulation=res,
                                           ignore_before_t=ignore_before_t,
                                           sieve=sieve).flatten()

        return float(self._fitness_time_score(y0, t0, T, res) / (1 + mse))

    def fitness(self, y0, t0, T):
        return self._fitness_simulation(y0, t0, T)

    def new_mutated_target_model(self, scale=1.):
        new_model = self.copy()

        def mutate(v, check_constraints=(-np.inf, np.inf)):
            # If we don't find an acceptable value after some trials, give up mutating and keep what's there.
            for _ in range(100):
                new_v = np.random.normal(loc=v, scale=scale * v)
                if check_constraints[0] <= new_v <= check_constraints[1]:
                    return new_v
            raise Exception("This should never really happen... (%s <= (%s, %s) <= %s)" % (
                check_constraints[0], new_v, v, check_constraints[1]))
            return v

        for k, v in self['target'].items():
            new_model['target'][k] = mutate(v, self['target_constraints'].get(k, (
                self.default_min_param, self.default_max_param)))

        new_model._invalidate_caches()
        return new_model

    def _optimize_get_state(self):
        keys = sorted(self['parameters'].keys())
        return [self['parameters'][k] for k in keys]

    def _optimize_get_state_keys(self):
        keys = sorted(self['parameters'].keys())
        return keys

    def _optimize_get_bounds(self):
        keys = sorted(self['parameters'].keys())
        bounds = [self['parameters_constraints'].get(k, (self.default_min_param, self.default_max_param)) for k in keys]
        return scipy.optimize.Bounds(*zip(*bounds))

    def _optimize_get_bounds_as_list(self):
        keys = sorted(self['parameters'].keys())
        bounds = [self['parameters_constraints'].get(k, (self.default_min_param, self.default_max_param)) for k in keys]
        return bounds

    def _optimize_set_state(self, state):
        keys = sorted(self['parameters'].keys())
        for i, k in enumerate(keys):
            self['parameters'][k] = state[i]
        self._invalidate_caches()

    def optimize_local(self,
                       y0: np.array,
                       t0: float,
                       T: float,
                       N_JOBS=-1,
                       save_checkpoint_name=False):

        fitness_history = list()

        def error(x):
            m = self.copy()
            m._optimize_set_state(x)
            fitness = m.fitness(y0, t0, T)
            if self.save_parameters_search_history:
                with open(PARAMETERS_SEARCH_HISTORY_FILE_PATH, 'ba') as f:
                    f.write(pickle.dumps((m.copy(), fitness)))
            return (1. - fitness)

        x0 = self._optimize_get_state()

        with tqdm.tqdm() as progressbar:
            def callback(x):
                f = 1. - error(x)
                if fitness_history:
                    conv = (f - fitness_history[-1][1])
                else:
                    conv = 0
                fitness_history.append((time.time(), f))
                progressbar.update()
                progressbar.set_postfix({'fitness': '%e (%s)' % (f, -np.log10(1 - f)),
                                         'conv'   : '%e' % conv, 'name': self['name']})

                m = self.copy()
                m._optimize_set_state(x)
                m['fitness_history'] = fitness_history
                if save_checkpoint_name:
                    m.save(save_checkpoint_name)

            res = scipy.optimize.minimize(error, x0,
                                          options={
                                              'maxfev'  : 1000000,
                                              'maxiter' : 2000,
                                              'adaptive': True,
                                              'xatol'   : 1e-6,
                                              'fatol'   : 1e-6,
                                          },
                                          callback=callback,
                                          bounds=self._optimize_get_bounds(),
                                          method='Nelder-Mead'
                                          )

        best = self.copy(keep_fitness_history=True)
        best._optimize_set_state(res.x)
        best['fitness_history'] += fitness_history
        # best.print()
        # print('Target ', str(np.array([self['target'][k] for k in self['equations']])))
        print('Fitness ', best.fitness(y0, t0, T))
        return best, fitness_history

    def _optimize_global_error(self, x):
        m = self.copy()
        m._optimize_set_state(x)
        fitness = m.fitness(self._og_y0, self._og_t0, self._og_T)
        if self.save_parameters_search_history:
            with open(PARAMETERS_SEARCH_HISTORY_FILE_PATH, 'ba') as f:
                f.write(pickle.dumps((m.copy(), fitness)))
        # del m
        return (1 - fitness)

    def optimize_global_DE(self,
                           y0: np.array,
                           t0: float,
                           T: float,
                           N_JOBS=-1,
                           save_checkpoint_name=False,
                           seed=1984,
                           popsize=2,
                           tol=1e-3):

        self._og_y0 = y0
        self._og_t0 = t0
        self._og_T = T

        x0 = self._optimize_get_state()

        start_time = time.time()

        fitness_history = self['fitness_history']
        if not len(fitness_history):
            fitness_history.append((start_time, 1. - self._optimize_global_error(x0)))

        if PLOT_OPTIMIZE:
            fig = plt.figure()
            plt.ion()
            plot_parameters([self], figure=fig)
            plt.show()
            plt.draw()
            plt.pause(0.00001)

        with tqdm.tqdm() as progressbar:
            def callback(x, convergence=0):
                f = 1. - self._optimize_global_error(x)

                m = self.copy()
                m._optimize_set_state(x)
                m['fitness_history'] = fitness_history
                if save_checkpoint_name:
                    m.save(save_checkpoint_name)

                if fitness_history:
                    conv = (f - fitness_history[-1][1])
                else:
                    conv = 0

                fitness_history.append((time.time(), f))
                progressbar.update()
                progressbar.set_postfix(
                        {'fitness' : '%e (%s)' % (f, -np.log10(1 - f)),
                         'alg_conv': '%e' % convergence,
                         'conv'    : '%e' % conv,
                         'name'    : self['name']})

                if conv != 0 and PLOT_OPTIMIZE:
                    fig.clear()
                    plot_parameters([m], figure=fig)
                    plt.draw()
                    plt.pause(0.00001)

                if time.time() - start_time > TIME_LIMIT_SECONDS:
                    return True

                if f > 1 - 1e-8:
                    return True
                else:
                    return False

            # import ray
            # from ray.util.multiprocessing import Pool as RayPool
            # runtime_env = {"working_dir": "./"}
            # ray.init(runtime_env=runtime_env)
            # ray_remote_args = {"scheduling_strategy": "SPREAD", 'num_cpus': 1}
            # MAP = RayPool(ray_remote_args=ray_remote_args).map

            executor = ProcessPoolExecutor()
            MAP = executor.map

            res = scipy.optimize.differential_evolution(
                    func=self._optimize_global_error,
                    bounds=tuple(self._optimize_get_bounds_as_list()),
                    callback=callback,
                    x0=x0,
                    maxiter=100000,
                    strategy='best1exp',
                    workers=MAP,
                    updating='deferred',
                    polish=False,
                    mutation=0.95,
                    recombination=0.95,
                    init='halton',
                    popsize=popsize,
                    tol=tol,
                    seed=seed,
            )

        best = self.copy(keep_fitness_history=True)
        best._optimize_set_state(res.x)
        best['fitness_history'] = fitness_history
        # best.print()
        # print('Target ', str(np.array([self['target'][k] for k in self['equations']])))
        final_fitness = best.fitness(y0, t0, T)
        print('Fitness ', final_fitness)
        gc.collect()  # When looping optimizations, if fast, gc may not run frequently enough
        # print(res)
        return best, fitness_history, final_fitness

    def optimize(self,
                 y0: np.array,
                 t0: float,
                 T: float,
                 N_JOBS=-1,
                 save_checkpoint_name=False,
                 seed=False,
                 popsize=4,
                 tol=1e-3
                 ):

        best = self

        for i, seed in enumerate([42, 1984, 69, 2013, 126, 500, 86, 31, 71546, 978456]):

            fresh_start = best.__class__()
            fresh_start.apply()
            best._optimize_set_state(fresh_start._optimize_get_state())

            best, fitness_history, final_fitness = best.optimize_global_DE(
                    y0, t0, T, N_JOBS=N_JOBS,
                    save_checkpoint_name=save_checkpoint_name,
                    seed=seed,
                    popsize=popsize,
                    tol=tol)

            if final_fitness >= 1 - 1e-8:
                break

        return best, fitness_history


class Healthy(Model):

    def __init__(self):
        super(Healthy, self).__init__()
        self['name'] = 'S00'
        self['equations'] = ['GP', 'StrD1', 'StrD2', 'SNc', 'DRN', 'LC']

        self['parameters'] = {
        }

        self['constants'] = {
            'a_GP_GP'      : 18 * milli,
            'a_EXT_GP'     : 100.,  # 22 / (18 * milli),
            'a_StrD1_GP'   : 100.,
            'a_StrD2_GP'   : 100.,
            'a_DRN_GP'     : 100.,

            'a_StrD1_StrD1': 2 * milli,
            'a_EXT_StrD1'  : 100.,  # 8 / (2 * milli),
            'a_SNc_StrD1'  : 100.,
            'a_DRN_StrD1'  : 100.,

            'a_StrD2_StrD2': 2 * milli,
            'a_EXT_StrD2'  : 100.,  # 9 / (2 * milli),
            'a_SNc_StrD2'  : 100.,
            'a_DRN_StrD2'  : 100.,

            'a_SNc_SNc'    : 1.5 * milli,
            'a_EXT_SNc'    : 100.,  # 4.5 / (1.5 * milli),
            'b_LC_SNc'     : 100,
            'a_DRN_SNc'    : 100.,
            'a_LC_SNc'     : 100.,

            'a_DRN_DRN'    : 3.3 * milli,
            'a_EXT_DRN'    : 100.,  # 1.2 / (3.3 * milli),
            'a_SNc_DRN'    : 100.,
            'a_LC_DRN'     : 100.,

            'a_LC_LC'      : 0.8 * milli,
            'a_EXT_LC'     : 100.,  # 2.5 / (0.8 * milli),
            'a_DRN_LC'     : 100.,
            'a_SNc_LC'     : 100.,

        }

        self['parameters_signs'] = {
            'a_GP_GP'      : -1,
            'a_EXT_GP'     : 1,
            'a_StrD1_GP'   : -1,
            'a_StrD2_GP'   : -1,
            'a_DRN_GP'     : 1,

            'a_StrD1_StrD1': -1,
            'a_EXT_StrD1'  : 1,
            'a_SNc_StrD1'  : 1,
            'a_DRN_StrD1'  : 1,

            'a_StrD2_StrD2': -1,
            'a_EXT_StrD2'  : 1,
            'a_SNc_StrD2'  : -1,
            'a_DRN_StrD2'  : 1,

            'a_SNc_SNc'    : -1,
            'a_EXT_SNc'    : 1,
            'b_LC_SNc'     : 1,
            'a_DRN_SNc'    : -1,
            'a_LC_SNc'     : -1,

            'a_DRN_DRN'    : -1,
            'a_EXT_DRN'    : 1,
            'a_SNc_DRN'    : -1,
            'a_LC_DRN'     : 1,

            'a_LC_LC'      : -1,
            'a_EXT_LC'     : 1,
            'a_DRN_LC'     : -1,
            'a_SNc_LC'     : 1,

        }

        self['parameters_constraints'] = {
            # constraints are [self.default_min_param, self.default_max_param] by default
        }

        self['target'] = {
            'GP'   : 22,  # Hz
            'StrD1': 10,  # Hz
            'StrD2': 9,  # Hz
            'SNc'  : 4.47,  # Hz
            'DRN'  : 1.41,  # Hz
            'LC'   : 2.3,  # Hz
        }

        self['target_constraints'] = {
            'GP'   : [18, 26],  # Hz
            'StrD1': [8, 12],  # Hz
            'StrD2': [7, 11],  # Hz
            'SNc'  : [3.5, 5.5],  # Hz
            'DRN'  : [1, 2],  # Hz
            'LC'   : [1.9, 3],  # Hz
        }

    def apply(self):

        if 'SHAM' not in self['applied_lesions']:
            self['applied_lesions'].append('SHAM')
            self['name'] += ' +SHAM'

        self._invalidate_caches()
        self['constants'] = self.P

        self['parameters'] = {

            'a_StrD1_GP' : self.P['a_StrD1_GP'],
            'a_StrD2_GP' : self.P['a_StrD2_GP'],
            'a_DRN_GP'   : self.P['a_DRN_GP'],
            'a_EXT_GP'   : self.P['a_EXT_GP'],
            #
            'a_SNc_StrD1': self.P['a_SNc_StrD1'],
            'a_DRN_StrD1': self.P['a_DRN_StrD1'],
            'a_EXT_StrD1': self.P['a_EXT_StrD1'],
            #
            'a_SNc_StrD2': self.P['a_SNc_StrD2'],
            'a_DRN_StrD2': self.P['a_DRN_StrD2'],
            'a_EXT_StrD2': self.P['a_EXT_StrD2'],
            #
            'a_DRN_SNc'  : self.P['a_DRN_SNc'],
            'a_LC_SNc'   : self.P['a_LC_SNc'],
            'b_LC_SNc'   : self.P['b_LC_SNc'],
            'a_EXT_SNc'  : self.P['a_EXT_SNc'],
            #
            'a_SNc_DRN'  : self.P['a_SNc_DRN'],
            'a_LC_DRN'   : self.P['a_LC_DRN'],
            'a_EXT_DRN'  : self.P['a_EXT_DRN'],
            #
            'a_DRN_LC'   : self.P['a_DRN_LC'],
            'a_SNc_LC'   : self.P['a_SNc_LC'],
            'a_EXT_LC'   : self.P['a_EXT_LC'],
        }
        self._clean_constants()
        self['parameters_constraints'] = {
            # constraints are [self.default_min_param, self.default_max_param] by default
        }

    def GP(self):
        gp_idx = self.E['GP']
        strd1_idx = self.E['StrD1']
        strd2_idx = self.E['StrD2']
        drn_idx = self.E['DRN']
        T_GP = self.P['a_GP_GP']
        a_StrD1_GP = self.P['a_StrD1_GP']
        a_StrD2_GP = self.P['a_StrD2_GP']
        a_DRN_GP = self.P['a_DRN_GP']
        a_EXT_GP = self.P['a_EXT_GP']

        def _GP(t, y):
            return -(1. / T_GP) * y[gp_idx] - a_StrD1_GP * y[strd1_idx] - a_StrD2_GP * y[strd2_idx] + \
                a_DRN_GP * y[drn_idx] + a_EXT_GP

        return _GP

    def StrD1(self):
        strd1_idx = self.E['StrD1']
        drn_idx = self.E['DRN']
        snc_idx = self.E['SNc']
        t_strd1 = self.P['a_StrD1_StrD1']
        a_drn_strd1 = self.P['a_DRN_StrD1']
        a_snc_strd1 = self.P['a_SNc_StrD1']
        a_ext_strd1 = self.P['a_EXT_StrD1']

        def _StrD1(t, y):
            return -(1. / t_strd1) * y[strd1_idx] + a_drn_strd1 * y[drn_idx] + a_snc_strd1 * y[snc_idx] + a_ext_strd1

        return _StrD1

    def StrD2(self):
        strd2_idx = self.E['StrD2']
        drn_idx = self.E['DRN']
        snc_idx = self.E['SNc']
        t_strd2 = self.P['a_StrD2_StrD2']
        a_drn_strd2 = self.P['a_DRN_StrD2']
        a_snc_strd2 = self.P['a_SNc_StrD2']
        a_ext_strd2 = self.P['a_EXT_StrD2']

        def _StrD2(t, y):
            return -(1. / t_strd2) * y[strd2_idx] + a_drn_strd2 * y[drn_idx] - a_snc_strd2 * y[snc_idx] + a_ext_strd2

        return _StrD2

    def SNc(self):
        snc_idx = self.E['SNc']
        drn_idx = self.E['DRN']
        lc_idx = self.E['LC']
        t_snc = self.P['a_SNc_SNc']
        a_drn_snc = self.P['a_DRN_SNc']
        a_lc_snc = self.P['a_LC_SNc']
        b_lc_snc = self.P['b_LC_SNc']
        a_ext_snc = self.P['a_EXT_SNc']

        def _SNc(t, y):
            return - (1. / t_snc) * y[snc_idx] \
                - a_drn_snc * y[drn_idx] - a_lc_snc * y[lc_idx] + (b_lc_snc * y[lc_idx] ** 2) + a_ext_snc

        return _SNc

    def DRN(self):
        drn_idx = self.E['DRN']
        snc_idx = self.E['SNc']
        lc_idx = self.E['LC']
        t_drn = self.P['a_DRN_DRN']
        a_snc_drn = self.P['a_SNc_DRN']
        a_lc_drn = self.P['a_LC_DRN']
        a_ext_drn = self.P['a_EXT_DRN']

        def _DRN(t, y):
            return - (1. / t_drn) * y[drn_idx] + a_lc_drn * y[lc_idx] - a_snc_drn * y[snc_idx] + a_ext_drn

        return _DRN

    def LC(self):
        lc_idx = self.E['LC']
        drn_idx = self.E['DRN']
        snc_idx = self.E['DRN']
        t_lc = self.P['a_LC_LC']
        a_drn_lc = self.P['a_DRN_LC']
        a_snc_lc = self.P['a_SNc_LC']
        a_ext_lc = self.P['a_EXT_LC']

        def _LC(t, y):
            return - (1. / t_lc) * y[lc_idx] - a_drn_lc * y[drn_idx] + a_snc_lc * y[snc_idx] + a_ext_lc

        return _LC

    def _invalidate_caches(self):
        super(Healthy, self)._invalidate_caches()
        if '_matrices' in self.__dict__:
            del self.__dict__['_matrices']

    @functools.cached_property
    def _matrices(self):
        eqs = self['equations']
        signs = self['parameters_signs']
        len_eqs = len(eqs)
        A = np.zeros((len_eqs, len_eqs))
        C = np.zeros((len_eqs, len_eqs))
        b = np.zeros((len_eqs, 1))

        equation_index = dict((e, i) for i, e in enumerate(eqs))
        for n, v in self.P.items():
            if n.startswith('a') or n.startswith('b'):
                dest, eq_from, eq_to = n.split('_')
                if dest == 'a':
                    if eq_from == eq_to:
                        v = 1 / v
                    if eq_from == 'EXT':
                        b[equation_index[eq_to]] = signs[n] * v
                    else:
                        A[equation_index[eq_to], equation_index[eq_from]] = signs[n] * v
                elif dest == 'b':
                    C[equation_index[eq_to], equation_index[eq_from]] = signs[n] * v

        return A, C, b

    def y_prime(self, t, y):
        A, C, b = self._matrices
        return A.dot(y) + C.dot(y ** 2) + b

    def _eigenvalues_real_part(self):
        A, C, b = self._matrices
        real_part_of_eigs = np.real(np.linalg.eig(A)[0])

        def f(y):
            return A.dot(y) + C.dot(y * y) + b

        def fprime(y):
            return A + 2 * C.dot(diagflat(y))

        eigsum = sum((max(0, e) for e in real_part_of_eigs))
        if eigsum <= 0:
            tol = 1e-9
            y = -invert_matrix(A).dot(b)
            itercount = 0
            while itercount <= 25:
                itercount += 1
                y_last = y
                y = y_last - invert_matrix(fprime(y_last)).dot(f(y_last))
                if abs(y - y_last).max() <= tol:
                    break
                if y.max() > 100 or y.min() < 0:
                    break

            A_tilde = fprime(y)
            real_part_of_eigs = np.real(np.linalg.eig(A_tilde)[0])

        return real_part_of_eigs

    def asymptotic_stability_score(self):
        eigsum = sum((max(0, e) for e in self._eigenvalues_real_part()))
        return 1. / (1 + eigsum)

    def _split_fitness(self, y0, t0, T):
        res = self.simulate(y0, t0, T)
        fits = list()
        for eq in self['equations']:
            fits.append(self._fitness_simulation(y0, t0, T, simulation=res,
                                                 limit_to_equations=[eq],
                                                 ignore_before_t=None))
        return fits

    def _combine_split_fitnesses(self, fits):
        # return np.min(fits)
        # return np.average(fits)
        # return min(fits) / np.average(fits)
        return math.sqrt(min(fits) * np.average(fits))
        # return np.prod(fits)**(1/len(fits))

    def fitness(self, y0, t0, T):
        return self._combine_split_fitnesses(self._split_fitness(y0, t0, T))


class L6OHDA(Healthy):

    def __init__(self):
        super(L6OHDA, self).__init__()

    def apply(self):
        if '6OHDA' not in self['applied_lesions']:
            self['applied_lesions'].append('6OHDA')
            self['name'] += ' +6OHDA'

            # self['target']['GP'] *= 0.90
            self['target']['SNc'] *= 0.1
            self['target']['LC'] *= 0.8

        self._invalidate_caches()
        self['constants'] = self.P
        self['parameters'] = {
            'b_LC_SNc' : self.P.get('b_LC_SNc', False) or 1.,
            'a_LC_SNc' : self.P.get('a_LC_SNc', False) or 1.,
            'a_DRN_SNc': self.P.get('a_DRN_SNc', False) or 1.,
            'a_EXT_SNc': self.P.get('a_EXT_SNc', False) or 1.,
        }
        self['parameters_constraints']['b_LC_SNc'] = [0, self.default_max_param]
        self['parameters_constraints']['a_EXT_SNc'] = (
            self.default_min_param, self.P.get('a_EXT_SNc', False) or self.default_max_param)

        self._clean_constants()

    def _split_fitness(self, y0, t0, T):
        res = self.simulate(y0, t0, T)

        return [
            self._fitness_simulation(y0, t0, T, limit_to_equations=['GP'],
                                     simulation=res,
                                     ignore_before_t=(T - t0) / 2),
            self._fitness_simulation(y0, t0, T, limit_to_equations=['SNc'],
                                     simulation=res,
                                     ignore_before_t=(T - t0) / 2,
                                     sieve=sieve_pass_positive
                                     ),
            self._fitness_simulation(y0, t0, T, limit_to_equations=['LC'],
                                     simulation=res,
                                     ignore_before_t=(T - t0) / 2,
                                     sieve=sieve_pass_positive
                                     )

        ]

    def fitness(self, y0, t0, T):
        return self._combine_split_fitnesses(self._split_fitness(y0, t0, T))


class LpCPA(Healthy):

    def __init__(self):
        super(LpCPA, self).__init__()

    def apply(self):
        if 'pCPA' not in self['applied_lesions']:
            self['applied_lesions'].append('pCPA')
            self['name'] += ' +pCPA'

            self['target']['GP'] *= 0.65
            self['target']['DRN'] *= 0.3

        self._invalidate_caches()
        self['constants'] = self.P
        self['parameters'] = {
            'a_EXT_DRN': self.P['a_EXT_DRN'],
            'a_LC_DRN' : self.P['a_LC_DRN'],
            'a_SNc_DRN': self.P['a_SNc_DRN'],
        }
        self['parameters_constraints']['a_EXT_DRN'] = (
            self.default_min_param, self.P.get('a_EXT_DRN', False) or self.default_max_param)
        self._clean_constants()

    def _split_fitness(self, y0, t0, T):
        res = self.simulate(y0, t0, T)

        return [
            self._fitness_simulation(y0, t0, T, limit_to_equations=['GP'],
                                     simulation=res,
                                     ignore_before_t=(T - t0) / 2),

            self._fitness_simulation(y0, t0, T, limit_to_equations=['DRN'],
                                     simulation=res,
                                     ignore_before_t=(T - t0) / 2,
                                     sieve=sieve_pass_positive),
        ]

    def fitness(self, y0, t0, T):
        return self._combine_split_fitnesses(self._split_fitness(y0, t0, T))


class LDSP4(Healthy):

    def __init__(self):
        super(LDSP4, self).__init__()

    def apply(self):
        if 'DSP4' not in self['applied_lesions']:
            self['applied_lesions'].append('DSP4')
            self['name'] += ' +DSP4'

            self['target']['LC'] *= 0.2

        self._invalidate_caches()
        self['constants'] = self.P
        self['parameters'] = {
            'a_EXT_LC': self.P['a_EXT_LC'],
            'a_DRN_LC': self.P['a_DRN_LC']
        }
        self['parameters_constraints']['a_EXT_LC'] = (
            self.default_min_param, self.P.get('a_EXT_LC', False) or self.default_max_param)
        self._clean_constants()

    def _split_fitness(self, y0, t0, T):
        res = self.simulate(y0, t0, T)

        return [
            self._fitness_simulation(y0, t0, T, limit_to_equations=['GP'],
                                     simulation=res,
                                     ignore_before_t=(T - t0) / 2),
            self._fitness_simulation(y0, t0, T, limit_to_equations=['LC'],
                                     simulation=res,
                                     ignore_before_t=(T - t0) / 2,
                                     sieve=sieve_pass_positive),
        ]

    def fitness(self, y0, t0, T):
        return self._combine_split_fitnesses(self._split_fitness(y0, t0, T))


class L6OHDA_LDSP4(Healthy):

    def __init__(self):
        super(L6OHDA_LDSP4, self).__init__()

    def apply(self):
        if '6OHDA+DSP4' not in self['applied_lesions']:
            self['applied_lesions'].append('6OHDA+DSP4')
            self['name'] += ' +6OHDA+DSP4'
            self.original_GP = self['target']['GP']
            self.max_GP = self.original_GP
            self.min_GP = self.original_GP * 0.65

        self._invalidate_caches()
        self['constants'] = self.P
        self['parameters'] = {}
        self._clean_constants()

    def _split_fitness(self, y0, t0, T):
        res = self.simulate(y0, t0, T)

        time_score = self._fitness_time_score(y0, t0, T,
                                              simulation=res)

        self['target']['GP'] = self.min_GP
        min_GP_score = self._fitness_simulation(y0, t0, T, limit_to_equations=['GP'],
                                                simulation=res,
                                                ignore_before_t=(T - t0) / 2,
                                                sieve=sieve_pass_negative
                                                )
        self['target']['GP'] = self.max_GP
        max_GP_score = self._fitness_simulation(y0, t0, T, limit_to_equations=['GP'],
                                                simulation=res,
                                                ignore_before_t=(T - t0) / 2,
                                                sieve=sieve_pass_positive
                                                )
        self['target']['GP'] = self.original_GP

        return [
            time_score,
            min_GP_score,
            max_GP_score
        ]

    def fitness(self, y0, t0, T):
        return self._combine_split_fitnesses(self._split_fitness(y0, t0, T))


class L6OHDA_LpCPA(Healthy):

    def __init__(self):
        super(L6OHDA_LpCPA, self).__init__()

    def apply(self):
        if '6OHDA+pCPA' not in self['applied_lesions']:
            self['applied_lesions'].append('6OHDA+pCPA')
            self['name'] += ' +6OHDA+pCPA'

        self.original_GP = self['target']['GP']
        self.max_GP = self.original_GP * 0.75
        self.min_GP = self.original_GP * 0.55

        self._invalidate_caches()
        self['constants'] = self.P
        self['parameters'] = {}
        self._clean_constants()

    def _split_fitness(self, y0, t0, T):
        res = self.simulate(y0, t0, T)

        time_score = self._fitness_time_score(y0, t0, T,
                                              simulation=res)

        self['target']['GP'] = self.min_GP
        min_GP_score = self._fitness_simulation(y0, t0, T, limit_to_equations=['GP'],
                                                simulation=res,
                                                ignore_before_t=(T - t0) / 2,
                                                sieve=sieve_pass_negative
                                                )
        self['target']['GP'] = self.max_GP
        max_GP_score = self._fitness_simulation(y0, t0, T, limit_to_equations=['GP'],
                                                simulation=res,
                                                ignore_before_t=(T - t0) / 2,
                                                sieve=sieve_pass_positive
                                                )
        self['target']['GP'] = self.original_GP

        return [
            time_score,
            min_GP_score,
            max_GP_score
        ]

    def fitness(self, y0, t0, T):
        return self._combine_split_fitnesses(self._split_fitness(y0, t0, T))


class Healthy_combined_fit(Healthy):

    def __init__(self):
        super(Healthy_combined_fit, self).__init__()

    def apply(self):
        self['parameters_constraints']['L6OHDA___b_LC_SNc'] = [0, self.default_max_param]

        self._invalidate_caches()
        self['constants'] = self.P

        # self['constants'].update(
        #         {
        #         }
        # )

        self['parameters'] = {

            'a_StrD1_GP'        : self.P['a_StrD1_GP'],
            'a_StrD2_GP'        : self.P['a_StrD2_GP'],
            'a_DRN_GP'          : self.P['a_DRN_GP'],
            'a_EXT_GP'          : self.P['a_EXT_GP'],

            'a_SNc_StrD1'       : self.P['a_SNc_StrD1'],
            'a_DRN_StrD1'       : self.P['a_DRN_StrD1'],
            'a_EXT_StrD1'       : self.P['a_EXT_StrD1'],

            'a_SNc_StrD2'       : self.P['a_SNc_StrD2'],
            'a_DRN_StrD2'       : self.P['a_DRN_StrD2'],
            'a_EXT_StrD2'       : self.P['a_EXT_StrD2'],

            'a_DRN_SNc'         : self.P['a_DRN_SNc'],
            'a_LC_SNc'          : self.P['a_LC_SNc'],
            'b_LC_SNc'          : self.P['b_LC_SNc'],
            'a_EXT_SNc'         : self.P['a_EXT_SNc'],

            'a_SNc_DRN'         : self.P['a_SNc_DRN'],
            'a_LC_DRN'          : self.P['a_LC_DRN'],
            'a_EXT_DRN'         : self.P['a_EXT_DRN'],

            'a_DRN_LC'          : self.P['a_DRN_LC'],
            'a_EXT_LC'          : self.P['a_EXT_LC'],
            'a_SNc_LC'          : self.P['a_SNc_LC'],

            'L6OHDA___a_LC_SNc' : self.P.get('L6OHDA___a_LC_SNc', False) or self.P['a_LC_SNc'],
            'L6OHDA___b_LC_SNc' : self.P.get('L6OHDA___b_LC_SNc', False) or self.P['b_LC_SNc'],
            'L6OHDA___a_DRN_SNc': self.P.get('L6OHDA___a_DRN_SNc', False) or self.P['a_DRN_SNc'],
            'L6OHDA___a_EXT_SNc': self.P.get('L6OHDA___a_EXT_SNc', False) or self.P['a_EXT_SNc'],

            'LpCPA___a_LC_DRN'  : self.P.get('LpCPA___a_LC_DRN', False) or self.P['a_LC_DRN'],
            'LpCPA___a_SNc_DRN' : self.P.get('LpCPA___a_SNc_DRN', False) or self.P['a_SNc_DRN'],
            'LpCPA___a_EXT_DRN' : self.P.get('LpCPA___a_EXT_DRN', False) or self.P['a_EXT_DRN'],

            'LDSP4___a_DRN_LC'  : self.P.get('LDSP4___a_DRN_LC', False) or self.P['a_DRN_LC'],
            'LDSP4___a_EXT_LC'  : self.P.get('LDSP4___a_EXT_LC', False) or self.P['a_EXT_LC'],

        }
        self['parameters_constraints']['b_LC_SNc'] = [0, self.default_max_param]

        self._clean_constants()
        self._invalidate_caches()

    def lesion_SHAM(self):
        healthy = Healthy()
        healthy.update(self.copy())
        healthy.apply()

        healthy._clean_constants()
        healthy._invalidate_caches()
        return healthy

    def lesion_L6OHDA(self):
        l6ohda = L6OHDA()
        l6ohda.update(self.copy())
        l6ohda.apply()
        l6ohda['parameters']['a_LC_SNc'] = self.P['L6OHDA___a_LC_SNc']
        l6ohda['parameters']['b_LC_SNc'] = self.P['L6OHDA___b_LC_SNc']
        l6ohda['parameters']['a_DRN_SNc'] = self.P['L6OHDA___a_DRN_SNc']
        l6ohda['parameters']['a_EXT_SNc'] = self.P['L6OHDA___a_EXT_SNc']

        l6ohda._clean_constants()
        l6ohda._invalidate_caches()
        return l6ohda

    def lesion_LpCPA(self):
        lpcpa = LpCPA()
        lpcpa.update(self.copy())
        lpcpa.apply()
        lpcpa['parameters']['a_LC_DRN'] = self.P['LpCPA___a_LC_DRN']
        lpcpa['parameters']['a_SNc_DRN'] = self.P['LpCPA___a_SNc_DRN']
        lpcpa['parameters']['a_EXT_DRN'] = self.P['LpCPA___a_EXT_DRN']

        lpcpa._clean_constants()
        lpcpa._invalidate_caches()
        return lpcpa

    def lesion_LDSP4(self):
        ldsp4 = LDSP4()
        ldsp4.update(self.copy())
        ldsp4.apply()
        ldsp4['parameters']['a_DRN_LC'] = self.P['LDSP4___a_DRN_LC']
        ldsp4['parameters']['a_EXT_LC'] = self.P['LDSP4___a_EXT_LC']

        ldsp4._clean_constants()
        ldsp4._invalidate_caches()
        return ldsp4

    def lesion_L6OHDA_LpCPA(self):
        lesioned = L6OHDA_LpCPA()
        lesioned.update(self.copy())
        lesioned.apply()

        lesioned['constants'] = lesioned.P
        lesioned['parameters'] = dict()

        lesioned['constants']['a_LC_SNc'] = self.P['L6OHDA___a_LC_SNc']
        lesioned['constants']['b_LC_SNc'] = self.P['L6OHDA___b_LC_SNc']
        lesioned['constants']['a_DRN_SNc'] = self.P['L6OHDA___a_DRN_SNc']
        lesioned['constants']['a_EXT_SNc'] = self.P['L6OHDA___a_EXT_SNc']

        lesioned['constants']['a_LC_DRN'] = self.P['LpCPA___a_LC_DRN']
        lesioned['constants']['a_SNc_DRN'] = self.P['LpCPA___a_SNc_DRN']
        lesioned['constants']['a_EXT_DRN'] = self.P['LpCPA___a_EXT_DRN']

        lesioned._clean_constants()
        lesioned._invalidate_caches()

        return lesioned

    def lesion_L6OHDA_LDSP4(self):
        lesioned = L6OHDA_LDSP4()
        lesioned.update(self.copy())
        lesioned.apply()

        lesioned['constants'] = lesioned.P
        lesioned['parameters'] = dict()

        lesioned['constants']['a_LC_SNc'] = self.P['L6OHDA___a_LC_SNc']
        lesioned['constants']['b_LC_SNc'] = self.P['L6OHDA___b_LC_SNc']
        lesioned['constants']['a_DRN_SNc'] = self.P['L6OHDA___a_DRN_SNc']
        lesioned['constants']['a_EXT_SNc'] = self.P['L6OHDA___a_EXT_SNc']

        lesioned['constants']['a_DRN_LC'] = self.P['LDSP4___a_DRN_LC']
        lesioned['constants']['a_EXT_LC'] = self.P['LDSP4___a_EXT_LC']

        lesioned._clean_constants()
        lesioned._invalidate_caches()

        return lesioned

    def _split_fitness_parameters_limits(self):
        # Lesioned EXT must be smaller than healthy EXT
        f_6OHDA = 1 / (1 + max(0, self.P['L6OHDA___a_EXT_SNc'] - self.P['a_EXT_SNc']))
        f_pCPA = 1 / (1 + max(0, self.P['LpCPA___a_EXT_DRN'] - self.P['a_EXT_DRN']))
        f_DSP4 = 1 / (1 + max(0, self.P['LDSP4___a_EXT_LC'] - self.P['a_EXT_LC']))
        return [f_6OHDA, f_pCPA, f_DSP4]

    def fitness(self, y0, t0, T):
        healthy = self.lesion_SHAM()
        l6ohda = self.lesion_L6OHDA()
        lpcpa = self.lesion_LpCPA()
        ldsp4 = self.lesion_LDSP4()

        l6ohdapcpa = self.lesion_L6OHDA_LpCPA()
        l6ohdadsp4 = self.lesion_L6OHDA_LDSP4()

        fits = \
            healthy._split_fitness(healthy.target_as_y0(), t0, T) + \
            l6ohda._split_fitness(healthy.target_as_y0(), t0, T) + \
            l6ohda._split_fitness(l6ohda.target_as_y0(), t0, T) + \
            lpcpa._split_fitness(healthy.target_as_y0(), t0, T) + \
            lpcpa._split_fitness(lpcpa.target_as_y0(), t0, T) + \
            ldsp4._split_fitness(healthy.target_as_y0(), t0, T) + \
            ldsp4._split_fitness(ldsp4.target_as_y0(), t0, T) + \
            l6ohdapcpa._split_fitness(healthy.target_as_y0(), t0, T) + \
            l6ohdapcpa._split_fitness(l6ohda.target_as_y0(), t0, T) + \
            l6ohdadsp4._split_fitness(healthy.target_as_y0(), t0, T) + \
            l6ohdadsp4._split_fitness(l6ohda.target_as_y0(), t0, T) + \
            [healthy.asymptotic_stability_score(),
             l6ohda.asymptotic_stability_score(),
             lpcpa.asymptotic_stability_score(),
             ldsp4.asymptotic_stability_score(),
             l6ohdapcpa.asymptotic_stability_score(),
             l6ohdadsp4.asymptotic_stability_score(),
             ] + \
            self._split_fitness_parameters_limits()

        return self._combine_split_fitnesses(fits)


class Cure(Healthy_combined_fit):

    def apply(self):
        self._invalidate_caches()
        self['constants'] = self.P
        self['parameters'] = {}
        self._clean_constants()
        self._invalidate_caches()
        self._param_fitness_penalty = 1

    def lesion_SHAM(self):
        lesioned = self.__class__()
        lesioned.update(self._impose_target(super(Cure, self).lesion_SHAM()))
        return lesioned

    def lesion_L6OHDA(self):
        lesioned = self.__class__()
        lesioned.update(self._impose_target(super(Cure, self).lesion_L6OHDA()))
        return lesioned

    def _split_fitness(self, y0, t0, T):
        res = self.simulate(y0, t0, T)
        fits = list()
        equations = self['equations']
        for eq in equations:
            fits.append(self._fitness_simulation(y0, t0, T, simulation=res,
                                                 limit_to_equations=[eq],
                                                 ignore_before_t=(T - t0) / 2))
        return fits

    def fitness(self, y0, t0, T):
        return self._combine_split_fitnesses(
                self._split_fitness(y0, t0, T) +
                [self.asymptotic_stability_score()]
        )


class Cure_DRN(Cure):

    def apply(self):
        super(Cure_DRN, self).apply()

        if 'cure_DRN' not in self['applied_lesions']:
            self['applied_lesions'].append('cure_DRN')
            self['name'] += ' +cure_DRN'

        self['parameters'] = {
            'CDRN___a_EXT_DRN': self.P.get('CDRN___a_EXT_DRN', False) or self.P['a_EXT_DRN']
        }
        self._clean_constants()
        self._invalidate_caches()

    def cure_DRN(self):
        cure = Cure_DRN()
        cure.update(self.copy())
        cure.apply()
        cure['parameters']['a_EXT_DRN'] = self.P['CDRN___a_EXT_DRN']
        cure._clean_constants()
        cure._invalidate_caches()
        return cure

    def _split_fitness_parameters_limits(self):
        f = 1 / (1 + self._param_fitness_penalty * max(0, self.P['a_EXT_DRN'] - self.P['CDRN___a_EXT_DRN']))
        return [f]

    def _split_fitness(self, y0, t0, T):
        res = self.simulate(y0, t0, T)
        fits = list()
        equations = self['equations'].copy()
        # equations.pop(equations.index('SNc'))
        equations.pop(equations.index('DRN'))
        # equations.pop(equations.index('LC'))
        for eq in equations:
            fits.append(self._fitness_simulation(y0, t0, T, simulation=res,
                                                 limit_to_equations=[eq],
                                                 ignore_before_t=(T - t0) / 2))
        return fits + [self.asymptotic_stability_score()]

    def fitness(self, y0, t0, T):
        limits = self._split_fitness_parameters_limits()
        cured = self.cure_DRN()
        return cured._combine_split_fitnesses(cured._split_fitness(y0, t0, T) + limits)


class Cure_LC(Cure):
    def apply(self):
        super(Cure_LC, self).apply()

        if 'cure_LC' not in self['applied_lesions']:
            self['applied_lesions'].append('cure_LC')
            self['name'] += ' +cure_LC'

        self['parameters'] = {
            'CLC___a_EXT_LC': self.P.get('CLC___a_EXT_LC', False) or self.P['a_EXT_LC']
        }
        self._clean_constants()
        self._invalidate_caches()

    def cure_LC(self):
        cure = Cure_LC()
        cure.update(self.copy())
        cure.apply()
        cure['parameters']['a_EXT_LC'] = self.P['CLC___a_EXT_LC']
        cure._clean_constants()
        cure._invalidate_caches()
        return cure

    def _split_fitness_parameters_limits(self):
        f = 1 / (1 + self._param_fitness_penalty * max(0, self.P['a_EXT_LC'] - self.P['CLC___a_EXT_LC']))
        return [f]

    def _split_fitness(self, y0, t0, T):
        res = self.simulate(y0, t0, T)
        fits = list()
        equations = self['equations'].copy()
        # equations.pop(equations.index('SNc'))
        # equations.pop(equations.index('DRN'))
        equations.pop(equations.index('LC'))
        for eq in equations:
            fits.append(self._fitness_simulation(y0, t0, T, simulation=res,
                                                 limit_to_equations=[eq],
                                                 ignore_before_t=(T - t0) / 2))
        return fits + [self.asymptotic_stability_score()]

    def fitness(self, y0, t0, T):
        limits = self._split_fitness_parameters_limits()
        cured = self.cure_LC()
        return cured._combine_split_fitnesses(cured._split_fitness(y0, t0, T) + limits)


class Cure_combined(Cure):
    def apply(self):
        super(Cure_combined, self).apply()

        if 'cure_DRN' not in self['applied_lesions']:
            self['applied_lesions'].append('cure_DRN')
            self['name'] += ' +cure_DRN'
        if 'cure_LC' not in self['applied_lesions']:
            self['applied_lesions'].append('cure_LC')
            self['name'] += ' +cure_LC'

        self['parameters'] = {
            'CDRN___a_EXT_DRN': self.P.get('CDRN___a_EXT_DRN', False) or self.P['a_EXT_DRN'],
            'CLC___a_EXT_LC'  : self.P.get('CLC___a_EXT_LC', False) or self.P['a_EXT_LC']
        }
        self._clean_constants()
        self._invalidate_caches()

    def cure_DRN_LC(self):
        cure = Cure_combined()
        cure.update(self.copy())
        cure.apply()
        cure['parameters']['a_EXT_DRN'] = self.P['CDRN___a_EXT_DRN']
        cure['parameters']['a_EXT_LC'] = self.P['CLC___a_EXT_LC']
        cure._clean_constants()
        cure._invalidate_caches()
        return cure

    def _split_fitness_parameters_limits(self):
        fdrn = 1 / (1 + self._param_fitness_penalty * max(0, self.P['a_EXT_LC'] - self.P['CLC___a_EXT_LC']))
        flc = 1 / (1 + self._param_fitness_penalty * max(0, self.P['a_EXT_DRN'] - self.P['CDRN___a_EXT_DRN']))
        return [fdrn, flc]

    def _split_fitness(self, y0, t0, T):
        res = self.simulate(y0, t0, T)
        fits = list()
        equations = self['equations'].copy()
        # equations.pop(equations.index('SNc'))
        equations.pop(equations.index('DRN'))
        equations.pop(equations.index('LC'))
        for eq in equations:
            fits.append(self._fitness_simulation(y0, t0, T, simulation=res,
                                                 limit_to_equations=[eq],
                                                 ignore_before_t=(T - t0) / 2))
        return fits

    def fitness(self, y0, t0, T):
        limits = self._split_fitness_parameters_limits()
        cured = self.cure_DRN_LC()
        return cured._combine_split_fitnesses(
                cured._split_fitness(y0, t0, T) + \
                limits + \
                [self.asymptotic_stability_score()])
