from bundle import Bundle
from random import uniform
from numpy.random import randint
from item import Item
import random
import numpy as np


class Population:
    items = None
    constraints = None

    @classmethod
    def set_items(cls, items):
        cls.items = items

    @classmethod
    def set_constraints(cls, constraints):
        cls.constraints = constraints

    def __init__(self, population_size):
        self.population_size = population_size
        self.bundles = [Bundle(items=self.items, constraints=self.constraints) for _ in range(population_size)]

    def get_fittest(self) -> Bundle:
        return max(self.bundles, key=lambda bundle: bundle.get_fitness())

    def get_weakest(self) -> Bundle:
        return min(self.bundles, key=lambda bundle: bundle.get_fitness())

    def get_fittest_elitism(self, n) -> list[Bundle]:
        self.bundles.sort(key=lambda bundle: bundle.get_fitness(), reverse=True)
        unique_bundles = list(set(self.bundles))
        unique_bundles.sort(key=lambda bundle: bundle.get_fitness(), reverse=True)
        return unique_bundles[:n]

    def get_size(self) -> int:
        return self.population_size

    def get_bundle(self, index) -> Bundle:
        return self.bundles[index]

    def save_bundle(self, index, bundle) -> None:
        self.bundles[index] = bundle

class GeneticAlgorithm:

    def __init__(self, constraints=None, items=None, population_size=100, crossover_rate=0.9,
                mutation_rate=0.1, elitism_param=10, cross_method="random_uniform",
                child_count= 1, max_gen=100,
                target_fitness=None, global_best_score=None):
        self.population_size = population_size
        self.crossover_rate = crossover_rate
        self.mutation_rate = mutation_rate
        self.elitism_param = max(elitism_param, 1)
        self.cross_method = cross_method
        self.child_count = max(child_count, 1)
        self.max_gen = max_gen
        self.target_fitness = target_fitness
        self.global_best_score = global_best_score
        self.items = items
        self.constraints = constraints
        Population.set_constraints(self.constraints)
        Population.set_items(self.items)

    def check_attrs(self):
        if not self.constraints \
        or not self.items:
            raise TypeError("Constraints and Items are required.")

    def check_test_attrs(self):
        if not self.global_best_score:
            raise ValueError("Global best score is required")

    def run_with_tests(self):
        self.check_test_attrs()
        pop = Population(self.population_size)
        generation_counter = 0
        fittest = None
        convergence, mins, average_fitness = [], [], []
        stds, ranges, variances, unqs, valids = [], [], [], [], []
        prox_avg_distances, avg_distances = [], []
        top_v_score, top_s_score, top_e_score, top_d_score = [], [], [], []
        avg_v_score, avg_s_score, avg_e_score, avg_d_score = [], [], [], []

        while True:
            generation_counter += 1
            pop = self.evolve_population(pop)
            fittest = pop.get_fittest()

            convergence.append(fittest.get_fitness())
            mins.append(pop.get_weakest().get_fitness())
            fitnesses = [bundle.get_fitness() for bundle in pop.bundles]
            average_fitness.append(np.mean(fitnesses))

            stds.append(np.std(fitnesses))
            ranges.append(np.ptp(fitnesses))
            variances.append(np.var(fitnesses))

            valid_sols = [bundle.get_hash() for bundle in pop.bundles if bundle.get_fitness() > 0.9]
            valids.append(len(valid_sols)/len(pop.bundles))
            unqs.append(measure_uniqueness(valid_sols, total_items=len(pop.bundles)))

            avg_distance = [abs(self.global_best_score - bundle.get_fitness()) for bundle in pop.bundles]
            avg_distances.append(np.mean(avg_distance))
            prox_avg_distances.append(np.mean([(dist/self.global_best_score) for dist in avg_distance]))

            avg_v_score.append(np.mean([bundle.get_value_score() for bundle in pop.bundles]))
            avg_s_score.append(np.mean([bundle.get_similarity() for bundle in pop.bundles]))
            avg_e_score.append(np.mean([bundle.get_epsilon_score() for bundle in pop.bundles]))
            avg_d_score.append(np.mean([bundle.get_delta_score() for bundle in pop.bundles]))

            top_v_score.append(fittest.get_value_score())
            top_s_score.append(fittest.get_similarity())
            top_e_score.append(fittest.get_epsilon_score())
            top_d_score.append(fittest.get_delta_score())


            if generation_counter == self.max_gen:
                break
            if (self.target_fitness is not None) and fittest.get_fitness() >= self.target_fitness:
                break

      # Calculate income
        candidates = pop.get_fittest_elitism(self.elitism_param)
        best = max(candidates, key=lambda bundle: bundle.get_fitness())

        results = {
            "best_solution": best,
            "convergence": convergence,
            "generations": generation_counter,
            "average_fitness": average_fitness,
            "mins": mins,
            "stds": stds,
            "ranges": ranges,
            "variances": variances,
            "unqs": unqs,
            "valids": valids,
            "avg_distances":avg_distances,
            "prox_avg_distances": prox_avg_distances,
            "candidates": candidates,
            "avg_v_score": avg_v_score,
            "avg_d_score": avg_d_score,
            "avg_e_score": avg_e_score,
            "avg_s_score": avg_s_score,
            "top_s_score": top_s_score,
            "top_e_score": top_e_score,
            "top_d_score": top_d_score,
            "top_v_score": top_v_score,
        }

        return results

    def run(self):
        pop = Population(self.population_size)
        generation_counter = 0

        while True:
            if generation_counter == self.max_gen:
                break
            if (self.target_fitness is not None) and fittest.get_fitness() == self.target_fitness:
                break
            fittest = pop.get_fittest()
            generation_counter += 1
            pop = self.evolve_population(pop)

        candidates = pop.get_fittest_elitism(self.elitism_param)
        best = max(candidates, key=lambda bundle: bundle.get_fitness())
        return best, candidates

    def evolve_population(self, population) -> Population:
        next_population = Population(self.population_size)

        # elitism: the top fittest individuals from previous population survive
        # so we copy the top 10 individuals to the next iteration (next population)
        # in this case the population fitness can not decrease during the iterations
        elites = population.get_fittest_elitism(self.elitism_param)
        next_population.bundles = elites + next_population.bundles
        parents = self.random_selection(population)
        # crossover
        for index in range(self.elitism_param, next_population.get_size()):
            if self.crossover_rate <= 0:
                break
            if uniform(0, 1) <= self.crossover_rate:
                first = np.random.choice(parents)
                second = np.random.choice(parents)
                next_population.save_bundle(index, self.crossover(first, second))

        # mutation
        for index in range(self.elitism_param, next_population.get_size()):
            if self.mutation_rate <= 0:
                break
            self.mutate(next_population.bundles[index])

        return next_population

    def minmax_parents(self, parent1, parent2):
        if len(parent1.contents) == len(parent2.contents):
            shorter_parent = parent1
            longer_parent = parent2
            max_len = len(parent1.contents)
            min_len = len(parent1.contents)
        else:
            shorter_parent = min([parent1, parent2], key = lambda x: len(x.contents))
            longer_parent = max([parent1, parent2], key = lambda x: len(x.contents))
            max_len = max(len(parent1.contents), len(parent2.contents))
            min_len = min(len(parent1.contents), len(parent2.contents))

        return longer_parent, shorter_parent, max_len, min_len

    def cross_two_point(self, parent1, parent2, child_count):
        longer_parent, shorter_parent, max_len, min_len = self.minmax_parents(parent1, parent2)

        children = []
        for _ in range(child_count):
            valid_cross_points = random.sample(range(0, min_len), k=2)
            short_k1 = min(valid_cross_points)
            short_k2 = max(valid_cross_points)
            cross_window = short_k2 - short_k1
            cross_bundle_generated = False
            long_k1 = random.choice(range(0, max_len - cross_window))
            long_k2 = long_k1 + cross_window

            cross_bundle = Bundle(empty=True, items=self.items, constraints=self.constraints)

            inherit_len = random.choice([shorter_parent, longer_parent])
            if inherit_len == shorter_parent:
                cross_bundle.contents.extend(shorter_parent.contents[:short_k1])
                cross_bundle.contents.extend(longer_parent.contents[long_k1:long_k2])
                cross_bundle.contents.extend(shorter_parent.contents[short_k2:])
                cross_bundle_generated = True

            if inherit_len == longer_parent and not cross_bundle_generated:
                cross_bundle.contents.extend(longer_parent.contents[:long_k1])
                cross_bundle.contents.extend(shorter_parent.contents[short_k1:short_k2])
                cross_bundle.contents.extend(longer_parent.contents[long_k2:])

            children.append(cross_bundle)

        children.sort(key=lambda x: x.get_fitness(), reverse=True)
        return children[0]

    def cross_single_point(self,parent1, parent2, child_count):
        children = []
        for _ in range(child_count):
            start = randint(len(parent1.contents) - 1)
            end = len(parent1.contents)

            if random.choice([0,1]) == 0:
                parent1, parent2 = parent2, parent1
            cross_bundle = Bundle(empty=True, items=self.items, constraints=self.constraints)

            cross_bundle.contents.extend(parent1.contents[:start])
            cross_bundle.contents.extend(parent2.contents[start:end])
            cross_bundle.contents.extend(parent1.contents[end:])
            children.append(cross_bundle)

        children.sort(key=lambda x: x.get_fitness(), reverse=True)
        return children[0]

    def cross_uniform(self, parent1, parent2, child_count):
        longer_parent, shorter_parent, max_len, min_len = self.minmax_parents(parent1, parent2)
        children = []
        extended_shorter_parent = shorter_parent.contents[:]
        extended_shorter_parent.extend(longer_parent.contents[min_len:max_len])
        for i in range(child_count):
            cross_bundle = Bundle(empty=True, items=self.items, constraints=self.constraints)
            cross_length = random.choice(range(min_len, max_len+1)) if max_len != min_len else max_len
            for i in range(cross_length):
                item_to_add = random.choice([longer_parent.contents[i], extended_shorter_parent[i]])
                cross_bundle.contents.append(item_to_add)
            children.append(cross_bundle)

        children.sort(key=lambda x: x.get_fitness(), reverse=True)
        return children[0]

    def random_cross_uniform(self,parent1, parent2, child_count):
        _, _, max_len, min_len = self.minmax_parents(parent1, parent2)

        gene_pool = list(parent1.contents)
        gene_pool.extend(list(parent2.contents))
        children = []

        for _ in range(child_count):
            size = random.choice(range(min_len, max_len+1)) if max_len != min_len else max_len
            new_bundle = random.sample(gene_pool, k=size)
            cross_bundle = Bundle(bundle=new_bundle, items=self.items, constraints=self.constraints)
            children.append(cross_bundle)


        children.sort(key=lambda x: x.get_fitness(), reverse=True)
        return children[0]

    def crossover(self, parent1, parent2) -> Item:
        child_count = self.child_count
        method = self.cross_method
        # print(method)
        if method == "two_point":
            child = self.cross_two_point(parent1, parent2, child_count)

        elif method == "single_point":
            if (len(parent1.contents) != len(parent2.contents)):
                child = self.cross_two_point(parent1, parent2, child_count)
            else:
                child = self.cross_single_point(parent1, parent2, child_count)

        elif method == "uniform":
            child = self.cross_uniform(parent1, parent2, child_count)

        elif method == "random_uniform":
            child = self.random_cross_uniform(parent1,parent2, child_count)

        else:
            raise NotImplementedError("use either 'two_point', 'single_point', 'uniform', 'random_uniform'")
        return child

    def mutate(self, individual) -> None:
        for index in range(len(individual.contents)):
            if uniform(0, 1) <= self.mutation_rate:
                individual.contents[index] = random.choice(self.items)


    # this is called tournament selection
    def random_selection(self, actual_population: Population) -> Bundle:
        parents = []
        while len(parents) < len(actual_population.bundles):
            size = int(0.1*len(actual_population.bundles))
            tournament = random.sample(actual_population.bundles, k=size)

            champion = max(tournament, key= lambda b: b.get_fitness())
            parents.append(champion)

        return parents
# utils
def measure_uniqueness(bundles, total_items):
    unique_hashes = set(bundles)
    return len(unique_hashes) / total_items