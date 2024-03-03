import random

class Constraints:

    def __init__ (self):
        self.weight_limit = 0
        self.price_limit = 0
        self.bundle_size = 0

    def set_constraints(self, weight_limit = None, price_limit = None, bundle_size = None, repetition_limit=-1):
        self.weight_limit = weight_limit if weight_limit else self.weight_limit
        self.price_limit = price_limit if price_limit else self.price_limit
        self.bundle_size = self.set_bundle_size(bundle_size)
        self.repetition_limit = repetition_limit

    def get_repetition_limit(self):
        return self.repetition_limit

    def set_bundle_size(self, size):
        if isinstance(size, int):
            self.bundle_size = size
            return self.bundle_size
        if isinstance(size, list) and len(size) == 2:
            self.bundle_size = range(size[0],size[1]+1)
            return self.bundle_size
        raise ValueError("Size must be an int or a list containing 2 values specifying the range")

    def get_weight_limit(self,) -> float: return self.weight_limit

    def get_price_limit(self,) -> float: return self.price_limit

    def get_bundle_size(self,) -> int:
        if isinstance(self.bundle_size, range):
            return random.choice(self.bundle_size)
        return self.bundle_size

    def complete_constraints(self,) -> bool:
        return self.price_limit != 0 and self.weight_limit != 0 and self.bundle_size != 0

    def __str__(self):
        return f"Price {self.get_price_limit()}, Weight {self.get_weight_limit()}, Size {self.bundle_size}, Repeats {self.get_repetition_limit()}"