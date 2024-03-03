from item import Item
import random
import hashlib
from collections import Counter

class Bundle:

    def __init__(self, items, constraints, empty=False, bundle = []):
        self.constraints = constraints
        self.items = items
        self.bundle_size = self.constraints.get_bundle_size()
        if empty:
            self.contents = []
        elif not bundle:
            self.contents = self.generate_bundle(self.bundle_size) if not empty else []
        else:
            self.contents = bundle

    def generate_bundle(self, size) -> list[Item]:
        bundle = random.choices(self.items, k=size)
        return bundle

    def get_weight(self) -> float:
        weight = 0
        for item in self.contents:
            weight += item.weight
        return weight

    def get_original_price(self) -> float:
        price = 0
        for item in self.contents:
            price += item.original_price
        return price

    def get_discounted_price(self) -> float:
        min_price = 0
        for item in self.contents:
            min_price += item.discounted_price
        return min_price

    def jaccard_similarity(self, item1, item2) -> float:
        item1_attrs = set(item1.categories)
        item2_attrs = set(item2.categories)
        item1_attrs.add(item1.brand)
        item2_attrs.add(item2.brand)

        intersection = len(item1_attrs.intersection(item2_attrs))
        union = len(item1_attrs.union(item2_attrs))
        similarity = intersection / union
        return similarity

    def get_similarity(self) -> float:
        total_similarity = 0
        num_pairs = 0

        for i in range(len(self.contents)):
            for j in range(i + 1, len(self.contents)):
                total_similarity += self.jaccard_similarity(self.contents[i], self.contents[j])
                num_pairs += 1

        if num_pairs == 0:
            return 0  # Avoid division by zero if the list is empty

        average_similarity = total_similarity / num_pairs
        return average_similarity

    # original price must be greater than the price limit
    def get_delta_score(self) -> float:
        original_price_sum = 0
        for item in self.contents:
            original_price_sum += item.original_price

        if original_price_sum <= self.constraints.get_price_limit():
            return 0

        return (original_price_sum - self.constraints.get_price_limit())/original_price_sum

    # discount must be lower than the price limit and as close to the price limit as possible
    def get_epsilon_score(self) -> float:
        discounted_price_sum = 0
        for item in self.contents:
            discounted_price_sum += item.discounted_price

        if discounted_price_sum > self.constraints.get_price_limit():
            return 0

        return discounted_price_sum/self.constraints.get_price_limit()

    def get_weight_score(self) -> float:
        total_weight = 0
        for item in self.contents:
            total_weight += item.weight

        if total_weight > self.constraints.get_weight_limit():
            return 0
        return 1

    def get_value_score(self) -> float:
        v_score = 0
        for item in self.contents:
            v_score += item.value

        return v_score/len(self.contents)

    def get_repetition_score(self):
        r_score = 1
        if self.constraints.get_repetition_limit() < 0:
            return r_score

        allowed_length = 1+ self.constraints.get_repetition_limit()

        counter = Counter(self.contents)
        for item, count in counter.items():
            if count > allowed_length:
                r_score = 0
                break
        return r_score


    def get_fitness(self) -> float:
        if len(self.contents) > self.bundle_size:
            return 0

        constraints_score = 4
        similarity_score = self.get_similarity()
        delta_score = self.get_delta_score()
        epsilon_score = self.get_epsilon_score()
        value_score = self.get_value_score()
        weight_score = self.get_weight_score()
        repetition_score = self.get_repetition_score()

        for score in [delta_score, epsilon_score, weight_score, repetition_score]:
            if score == 0:
                constraints_score -= 1

        score_sum = ((90/4) * constraints_score) + ((10/4)* (similarity_score + delta_score + epsilon_score + value_score))
        normalized_score = score_sum / 100
        return normalized_score

    def get_hash(self):
        sorted_items = sorted(self.contents, key=lambda x: x.asin)
        combined_names = "".join(item.asin for item in sorted_items)
        return hashlib.sha256(combined_names.encode()).hexdigest()

    def __repr__(self):
        return str({
            'Weight': self.get_weight(),
            'Min_price': self.get_discounted_price(),
            'Price': self.get_original_price(),
            'Fitness': self.get_fitness()
            # 'Items': self.contents
            })

    def __str__(self):
        return self.__repr__()

    def __hash__(self):
        return hash(frozenset(self.contents))

    def __eq__(self, other):
        return set(self.contents) == set(other.contents)