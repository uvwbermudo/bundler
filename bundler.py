from item import Item
from itemDB import ItemDB
from constraints import Constraints
from ga import GeneticAlgorithm
from bundle import Bundle
import random
import numpy as np

class Bundler:

    def __init__(self,):
        self.algo = None
        self.perfect_bundle = None
        self.perfect_score = 0
        self.constraints = None
        self.itemDB = None
        self.best_score = 1

    def is_initialized(self):
        return (self.algo != None and self.constraints != None and self.itemDB != None)


    def init_constraints(self, weight_limit=0, price_limit=0, repetition_limit=-1,
                        bundle_size=0, csv_file=None, db_size=None,
                        get_mode='constrained', adjust=False):

        init_bundle_size = bundle_size

        if weight_limit == 0 or price_limit == 0 or bundle_size == 0:
            if not adjust:
                raise ValueError("price_limit, weight_limit, bundle_size required")
        print("No constraints set, default constraints is price=0, weight=0, bundle_size=3")
        bundle_size = 3 if bundle_size == 0 else bundle_size
        init_bundle_size = bundle_size

        self.constraints = Constraints()
        self.constraints.set_constraints(weight_limit=weight_limit, price_limit=price_limit, bundle_size=bundle_size, repetition_limit=repetition_limit)

        self.itemDB = ItemDB(constraints=self.constraints)
        if csv_file:
            self.itemDB.initialize_products_df(csv_file)

        self.itemDB.set_product_get_mode(get_mode)

        if db_size:
            self.itemDB.set_max_items(db_size)

        if adjust and (init_bundle_size == 0 or weight_limit == 0  or price_limit == 0):
            print('Adjusting Constraints weight limit and price limit to 75th percentile of the items')
        items = self.itemDB.get_items()
        d_prices = [item.discounted_price for item in items]
        o_prices = [item.original_price for item in items]
        weights = [item.weight for item in items]
        dp_75 = np.percentile(d_prices, 75)
        op_75 = np.percentile(o_prices, 75)
        w_75 = np.percentile(weights, 75)
        price_limit = (op_75 + dp_75)/2 if price_limit == 0 else price_limit
        weight_limit = w_75 if weight_limit == 0 else weight_limit
        self.constraints = Constraints()
        self.constraints.set_constraints(weight_limit=weight_limit, price_limit=price_limit, bundle_size=bundle_size)
        self.itemDB = ItemDB(constraints=self.constraints)

        if csv_file:
            self.itemDB.initialize_products_df(csv_file)

        self.itemDB.set_product_get_mode(get_mode)

        if db_size:
            self.itemDB.set_max_items(db_size)

    def get_best_score(self):
        return self.best_score

    def get_perfect_score(self):
        if not self.perfect_bundle:
            self.include_perfects(1)
            self.perfect_score = self.perfect_bundle.get_fitness()
            self.get_best_score()
        return self.perfect_score 

    def include_perfects(self, count):
        perfects = []

        if self.itemDB.empty():
            self.itemDB.build()

        bundle_size = self.constraints.get_bundle_size()

        for i in range(count):
            perfect_item = Item(
                asin = f'Perfect{i}',
                title = f'Perfect{i}',
                original_price = 1000,
                discounted_price = self.constraints.get_price_limit()/bundle_size -0.0000000001,
                rating = 5,
                weight = 0.000001,
                brand = 'Perfect',
                categories = 'Perfect',
                )
        self.itemDB.cache_insert_product(perfect_item)
        perfects.append(perfect_item)

        perfect_bundle = random.choices(perfects, k=bundle_size)
        perfect_bundle = Bundle(bundle=perfect_bundle, constraints=self.constraints, items=self.itemDB.get_items())
        self.perfect_bundle = perfect_bundle

    def init_GA(self, population_size=100, crossover_rate=0.9, mutation_rate=0.1,
                elitism_param=10, cross_method="random_uniform", child_count= 1, max_gen=500,
                target_fitness=None, test=False, targetted=False):

        if self.constraints is None:
            raise ValueError('Initialize Constraints First')


        global_best = self.get_best_score() 
        if target_fitness:
            global_best = target_fitness
            self.best_score = target_fitness

        if test:
            if not targetted: 
                target_fitness = None

            self.algo = GeneticAlgorithm(
                            constraints=self.constraints, items=self.itemDB.get_items(),
                            population_size=population_size, crossover_rate=crossover_rate,
                            mutation_rate=mutation_rate, elitism_param=elitism_param,
                            cross_method=cross_method, child_count=child_count,
                            max_gen=max_gen, global_best_score=global_best, target_fitness=target_fitness)
        else:
            self.algo = GeneticAlgorithm(
                        constraints=self.constraints, items=self.itemDB.get_items(),
                        population_size=population_size, crossover_rate=crossover_rate,
                        mutation_rate=mutation_rate, elitism_param=elitism_param,
                        cross_method=cross_method, child_count=child_count,
                        max_gen=max_gen, target_fitness=target_fitness)

    def run(self, test=False):
        if not self.is_initialized():
            raise RuntimeError("Bundler is not properly initialized")
        if test:
            return self.algo.run_with_tests()
        else:
            return self.algo.run()

