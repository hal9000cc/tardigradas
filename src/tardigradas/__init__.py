import gc
from abc import ABC, abstractmethod
import numpy as np
#import random
import datetime as dt
from enum import Enum
import pickle
from .gen_types import *


class CrossoverBitType(Enum):
    uniform = 0
    one_point = 1
    two_point = 2


class CrossoverFloatType(Enum):
    uniform = 0
    arithmetic = 1
    BLX = 2


class TardigradasException(Exception):
    def __init__(self, *args):
        if args:
            self.message = args[0]
        else:
            self.message = None

    def __str__(self):
        if self.message:
            return f'{self.__class__.__name__}: {self.message}'
        else:
            return self.__class__.__name__


TradigradasException = TardigradasException


class Resolve(ABC):

    def __init__(self, tardigradas, chromo=None, use_defaults=False):

        self.tardigradas = tardigradas

        if chromo is None:
            self.chromo_new(use_defaults=use_defaults)
        else:
            self.chromo_new(chromo)

    def chromo_new_random(self, use_default=False):

        ixb_bits = self.tardigradas.gen_types == GenType.bit.value
        ixb_int = self.tardigradas.gen_types == GenType.int.value
        ixb_float = self.tardigradas.gen_types == GenType.float.value

        new_chromo = np.zeros(self.tardigradas.chromo_size, dtype=float)

        n_bits = ixb_bits.sum()
        new_chromo[ixb_bits] = np.random.random(n_bits) > 0.5

        n_float = ixb_float.sum()
        new_chromo[ixb_float] = self.tardigradas.chromo_bounds_min[ixb_float] + np.random.random(n_float) * (self.tardigradas.chromo_bounds_max[ixb_float] - self.tardigradas.chromo_bounds_min[ixb_float])

        new_chromo[ixb_int] = np.random.randint(self.tardigradas.chromo_bounds_min[ixb_int], self.tardigradas.chromo_bounds_max[ixb_int] + 1)

        if use_default:
            ixb_defaults = ~np.isnan(self.tardigradas.chromo_defaults)
            ixb_defaults[np.random.random(len(ixb_defaults)) > self.tardigradas.chromo_defaults_probability] = False
            new_chromo[ixb_defaults] = self.tardigradas.chromo_defaults[ixb_defaults]

        self.chromo = new_chromo

    def chromo_new(self, chromo=None, use_defaults=False):

        if chromo is not None:
            self.chromo = chromo
        else:
            self.chromo_new_random(use_defaults)

    def __getitem__(self, item):

        assert type(item) == int
        i_gen = item

        gen_value = self.chromo[i_gen]
        gen_type = GenType(self.tardigradas.gen_types[i_gen])

        if gen_type == GenType.bit or gen_type == GenType.int:
            gen_value = int(gen_value)

        return gen_value

    def chromo_valid(self):
        return True

    @staticmethod
    @abstractmethod
    def init_environment(tardigradas):
        pass

    @staticmethod
    def is_equal(tardigradas, chromo1, chromo2):
        if hasattr(chromo1, 'chromo'):
            chromo1 = chromo1.chromo
        if hasattr(chromo2, 'chromo'):
            chromo2 = chromo2.chromo
        return np.array_equal(chromo1, chromo2)

    @staticmethod
    @abstractmethod
    def gen_info(tardigradas):
        chromo_gen_types = None
        chromo_bounds_min = None
        chromo_bounds_max = None
        chromo_gen_comments = None
        gen_groups = None
        gen_default = None
        gen_default_probability = None
        return chromo_gen_types, (chromo_bounds_min, chromo_bounds_max), chromo_gen_comments, gen_groups, gen_default, gen_default_probability

    @abstractmethod
    def fitness(self):
        pass


class Tardigradas:

    def __init__(self, resolve_class, population_size, crossover_fraction=0.5, fresh_blood_fraction=0.0, gen_mutation_fraction=0.1, fitness_environment=None, n_elits=None):

        self.resolve_class = resolve_class
        self.environment = fitness_environment

        if n_elits is None:
            n_elits = 1

        self.n_elits = int(n_elits)
        self.population_size = population_size
        self.crossover_fraction = crossover_fraction
        self.fresh_blood_fraction = fresh_blood_fraction
        self.gen_mutation_fraction = gen_mutation_fraction

        resolve_class.init_environment(self)

        chromo_gen_types, chromo_gen_bounds, chromo_gen_comments, chromo_gen_groups, chromo_gen_defaults, chromo_gen_defaults_probability = resolve_class.gen_info(self)

        self.chromo_size = len(chromo_gen_types)
        self.gen_comments = chromo_gen_comments

        self.gen_types = np.array([gen_type.value for gen_type in chromo_gen_types], dtype=int)

        self.chromo_bounds_min = np.array(chromo_gen_bounds[0], dtype=float)
        self.chromo_bounds_max = np.array(chromo_gen_bounds[1], dtype=float)
        self.chromo_gen_groups = np.array(chromo_gen_groups, dtype=int)
        self.chromo_defaults = np.array(chromo_gen_defaults, dtype=float)
        self.chromo_defaults_probability = np.array(chromo_gen_defaults_probability, dtype=float)

    def show_progress(self, *_):
        time = dt.datetime.now()
        print(f'{time}: {self.iterations=}, {self.step_score=}')
        return False

    def new_valid_resolve(self, use_defaults=False):
        n_attempts = 200
        for i in range(n_attempts):
            random_resolve = self.resolve_class(self, use_defaults=use_defaults)
            if random_resolve.chromo_valid(): return random_resolve

        raise TardigradasException(f"can't create a new random chromosome in {n_attempts} attempts")

    def population_init(self):

        population = [self.new_valid_resolve(True) for i in range(self.population_size)]

        self.population = np.array(population)

        self.iterations = 0
        self.scores_epoch = []
        self.custom_scores_epoch = []
        self.best_score = None
        self.best_iteration = 0
        self.best_resolve = None
        self.step_best_resolve = None
        self.step_score = None
        self.step_custom_score = None
        self.fitness_progress_fun = None

    def crossover_uniform(self, kid, parent1, parent2, ixb_gens):

        kid[ixb_gens] = parent1[ixb_gens]

        ix_no_group = self.chromo_gen_groups == 0
        ixb_move_no_group = np.random.random(ix_no_group.sum()) > 0.5

        groups = set(self.chromo_gen_groups[self.chromo_gen_groups > 0])
        ixb_move_group = np.random.random(len(groups)) > 0.5

        ixb_move2 = np.zeros(self.chromo_size, dtype=bool)
        ixb_move2[ix_no_group] = ixb_move_no_group

        for i_group, group in enumerate(groups):
            ixb_move2[self.chromo_gen_groups == group] = ixb_move_group[i_group]

        kid[ixb_gens & ixb_move2] = parent2[ixb_gens & ixb_move2]
        return kid

    def crossover_blx(self, kid, parent1, parent2, ixb_gens, alpha=0.5):
        parents = np.vstack((parent1[ixb_gens], parent2[ixb_gens]))
        I = np.abs(parents[0] - parents[1]) * alpha
        lows = np.clip(parents.min(0) - I, self.chromo_bounds_min[ixb_gens], self.chromo_bounds_max[ixb_gens])
        highs = np.clip(parents.max(0) + I, self.chromo_bounds_min[ixb_gens], self.chromo_bounds_max[ixb_gens])
        kid[ixb_gens] = lows + np.random.random(len(lows)) * (highs - lows)

    def mutation_gauss(self, value, value_min, value_max, std=0.5):
        assert value_min != value_max
        new_value = np.random.normal(value, max(value - value_min, value_max - value) * std)
        return np.clip(new_value, value_min, value_max)

    def crossover(self, ix):

        kids = []
        n_kids = len(ix) // 2
        for i in range(n_kids):

            i_parent1 = ix[i]
            i_parent2 = ix[i + n_kids]

            if self.n_elits > 0 and self.population[i_parent1].is_equal(self, self.population[i_parent1].chromo, self.population[i_parent2].chromo):
                # if dad and mom are the same, do mutation
                kids += self.mutation([i_parent1])
                continue

            # bx_crossover_uniform = (self.gen_types == GenType.bit.value)
            # bx_crossover_blx = (self.gen_types == GenType.int.value) | (self.gen_types == GenType.float.value)
            bx_crossover_uniform = np.ones(len(self.gen_types), dtype=bool)

            kid_chromo = np.zeros(self.chromo_size, dtype=float)
            self.crossover_uniform(kid_chromo, self.population[i_parent1].chromo, self.population[i_parent2].chromo, bx_crossover_uniform)
            # self.crossover_blx(kid_chromo, self.population[i_parent1].chromo, self.population[i_parent2].chromo, bx_crossover_blx)

            for i_int_gen in (self.gen_types == GenType.int.value).nonzero()[0]:
                kid_chromo[i_int_gen] = round(kid_chromo[i_int_gen])

            kids.append(self.resolve_class(self, kid_chromo))

        return kids

    def get_mutation_kid(self, parent, n_mutation):

        kid_chromo = np.array(parent.chromo)
        ix_available_mutation = (self.chromo_bounds_min != self.chromo_bounds_max).nonzero()[0]
        points = np.random.choice(ix_available_mutation, n_mutation)

        for point in points:

            gen_type = GenType(self.gen_types[point])
            gen = kid_chromo[point]

            if gen_type == GenType.bit:
                new_gen = (gen + 1) % 2

            elif gen_type == GenType.int:
                new_gen = int(self.mutation_gauss(gen, self.chromo_bounds_min[point], self.chromo_bounds_max[point]))
                while new_gen == gen and self.chromo_bounds_min[point] < self.chromo_bounds_max[point]:
                    new_gen = np.random.randint(self.chromo_bounds_min[point], self.chromo_bounds_max[point] + 1)

            elif gen_type == GenType.float:
                new_gen = self.mutation_gauss(gen, self.chromo_bounds_min[point], self.chromo_bounds_max[point])

            kid_chromo[point] = new_gen

        kid = self.resolve_class(self, kid_chromo)

        return kid

    def mutation(self, ix):

        n_mutation = int(round(abs(np.random.normal(0, self.chromo_size * self.gen_mutation_fraction))))
        n_mutation = np.clip(n_mutation, 1, self.chromo_size)

        kids = []
        for i_parent in ix:

            n_attempts = 200
            for _ in range(n_attempts):
                kid = self.get_mutation_kid(self.population[i_parent], n_mutation)
                if kid.chromo_valid() and not kid.is_equal(self, kid.chromo, self.population[i_parent].chromo):
                    break
            else:
                raise TardigradasException(f"can't create a mutated chromosome in {n_attempts} attempts")

            kids.append(kid)

        return kids

    def estimate_population(self):

        scores = []
        for i, resolve in enumerate(self.population):

            #score, custom_score = resolve.fitness()
            score = resolve.fitness()
            scores.append(np.array(score, dtype=float))

            if self.fitness_progress_fun:
                self.fitness_progress_fun(self, i / self.population_size)

        scores = np.vstack(scores)
        self.scores = scores[:, 0]
        self.full_scores = scores

    @staticmethod
    def select_parents(expectation, count):

        whell = expectation.cumsum()
        parents = np.zeros(count, int)
        step_size = 1 / count
        position = np.random.random() * step_size
        lowest = 0

        for i in range(count):
            for j in range(lowest, len(whell)):
                if position < whell[j]:
                    parents[i] = j
                    lowest = j
                    break
            position += step_size

        return parents

    @staticmethod
    def rank(estimates):
        expectation = np.zeros(len(estimates), dtype=float)
        ix = (-np.array(estimates)).argsort()
        expectation[ix] = 1.0 / pow(np.arange(1, len(ix) + 1), 0.5)
        return expectation / np.sum(expectation)

    def kill_doubles(self):

        self.n_killed_doubles = 0
        for i_test_resolve, test_resolve in enumerate(self.population[:-1]):
            for i_resolve, resolve in enumerate(self.population[i_test_resolve + 1:]):
                if test_resolve.is_equal(self, test_resolve, resolve):
                    self.population[i_test_resolve + 1 + i_resolve] = self.new_valid_resolve()
                    self.n_killed_doubles += 1

    def state_dict(self):
        return {
            'epoch_score': self.step_score,
            'step_score': self.step_score,
            'step_custom_score': self.step_custom_score,
            'iterations': self.iterations,
            'scores_epoch': self.scores_epoch,
            'custom_scores_epoch': self.custom_scores_epoch,
            'crossover_fraction': self.crossover_fraction,
            'fresh_blood_fraction': self.fresh_blood_fraction,
            'gen_mutation_fraction': self.gen_mutation_fraction,
            'chromo_bounds_max': self.chromo_bounds_max,
            'chromo_bounds_min': self.chromo_bounds_min,
            'chromo_gen_groups': self.chromo_gen_groups,
            'chromo_defaults': self.chromo_defaults,
            'chromo_defaults_probability': self.chromo_defaults_probability,
            'gen_types': self.gen_types,
            'chromo_size': self.chromo_size,
            'gen_comments': self.gen_comments,
            'n_elits': self.n_elits,
            'population_size': self.population_size,
            'scores': self.scores,
            'best_score': self.best_score,
            'best_iteration': self.best_iteration,
            'best_resolve': None if self.best_resolve is None else self.best_resolve.chromo,
            'population': [resolve.chromo for resolve in self.population]
        }

    def restore_from_dict(self, state):

        for key, data in state.items():
            self.__setattr__(key, data)

        if 'step_score' not in state and 'epoch_score' in state:
            self.step_score = state['epoch_score']
        if 'step_custom_score' not in state:
            self.step_custom_score = None

        self.population = np.array([self.resolve_class(self, chromo) for chromo in state['population']])
        if state.get('best_resolve') is not None:
            self.best_resolve = self.resolve_class(self, state['best_resolve'])
        else:
            self.best_resolve = None
        self.step_best_resolve = self.best_resolve
        self.fitness_progress_fun = None

    def save_to_file(self, file_name):
        with open(file_name, 'wb') as file:
            pickle.dump(self.state_dict(), file)

    def restore_from_file(self, file_name):

        with open(file_name, 'rb') as file:
            state = pickle.load(file)

        self.restore_from_dict(state)

    def step(self):

        self.estimate_population()

        n_crossover = int(np.round((self.population_size - self.n_elits) * self.crossover_fraction))
        n_fresh_blood = int(np.round((self.population_size - self.n_elits) * self.fresh_blood_fraction))
        n_mutation = self.population_size - self.n_elits - n_crossover - n_fresh_blood

        n_parents = n_crossover * 2 + n_mutation

        expectation = self.rank(self.scores)

        ix_parents = self.select_parents(expectation, n_parents)

        ix_parents = np.random.choice(ix_parents, len(ix_parents), replace=False)

        ix_best = (-np.array(self.scores)).argsort()
        kids_elit = self.population[ix_best[: self.n_elits]]
        self.step_score = self.scores[ix_best[0]]
        self.step_custom_score = self.full_scores[ix_best[0]]
        self.step_best_resolve = self.population[ix_best[0]]

        if self.best_score is None or self.step_score > self.best_score:
            self.best_score = self.step_score
            self.best_iteration = self.iterations
            self.best_resolve = self.population[ix_best[0]]
            print(f'{self.best_iteration=}')

        self.scores_epoch.append(self.step_score)
        self.custom_scores_epoch.append(self.step_custom_score)

        kids_crossover = self.crossover(ix_parents[0: (2 * n_crossover)])
        kids_mutation = self.mutation(ix_parents[(2 * n_crossover):])
        kids_new = [self.new_valid_resolve(True) for i in range(n_fresh_blood)]

        self.population = np.hstack([kids_elit, np.array(kids_crossover), np.array(kids_mutation), np.array(kids_new)])

        #replace the bad ones with new ones
        for i_resolve, resolve in enumerate(self.population):
            if not resolve.chromo_valid():
                self.population[i_resolve] = self.new_valid_resolve()

        self.kill_doubles()

        self.iterations += 1

    def loop(self, max_iterations=None, epoch_without_improve=50, loop_fun=None, fitness_progress_fun=None):

        if not hasattr(self, 'best_score'):
            self.best_score = None
        if not hasattr(self, 'best_iteration'):
            self.best_iteration = 0
        self.fitness_progress_fun = fitness_progress_fun

        if loop_fun is None:
            loop_fun = self.show_progress

        while True:

            self.step()
            gc.collect()

            if loop_fun is not None and loop_fun(self): break
            if max_iterations is not None and self.iterations > max_iterations: break
            if self.iterations - self.best_iteration > epoch_without_improve: break
