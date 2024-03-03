from item import Item
import pandas as pd

class ItemDB:

    def __init__ (self, constraints=None):
        self.products_df = pd.DataFrame()
        self.queried_products = []
        self.price_limit = 0
        self.weight_limit = 0
        self.bundle_size = 0
        self.bundle_size_param = 0
        self.max_items = 0
        self.product_get_mode = 'constrained'
        self.curr_product_get_mode = None
        self.constraints = constraints
    
    def initialize_products_df(self, csv_path):
        self.products_df = pd.read_csv(csv_path)
        """
            This is only implemeneted for testing using a dataframe,
            this class should be the one to communicate with the products database
        """

    def cache_insert_product(self, product):
        self.queried_products.append(product)

    def empty(self):
        return len(self.queried_products) == 0

    def set_product_get_mode(self, mode='constrained'):
        if mode in ['all', 'constrained']:
            self.product_get_mode = mode
        else:
            raise NotImplementedError(f"Mode {mode} does not exist, try 'all' or 'constrained' ")

    def build(self):
        self.get_items()

    def set_max_items(self, n):
        self.max_items = n

    def get_items(self):
        if self.products_df.empty:
            return self.get_items_from_db()
        else:
            return self.get_items_from_df()

    def get_items_from_db(self):
        raise NotImplementedError('Please implement getting items from DB, use a Pandas Dataframe instead and use initialize_products_df')
    
    def constraints_unchanged(self):
        return self.price_limit == self.constraints.get_price_limit() \
                and self.weight_limit == self.constraints.get_weight_limit() \
                and self.bundle_size_param == self.constraints.bundle_size

    def get_items_from_df(self):
        if self.constraints_unchanged() and self.product_get_mode == self.curr_product_get_mode:
            return self.queried_products

        if self.constraints is None or (self.constraints is not None and not self.constraints.complete_constraints()):
            self.set_product_get_mode('all')

        print('Building product list from df...')
        self.price_limit = self.constraints.get_price_limit()
        self.weight_limit = self.constraints.get_weight_limit()
        self.bundle_size = self.constraints.get_bundle_size()
        self.bundle_size_param = self.constraints.bundle_size
        print(f'Constraints | Price:{self.price_limit} | Weight: {self.weight_limit}| Bundle: {self.bundle_size_param}|')
        new_products_df = self.products_df
        if self.product_get_mode == 'constrained':
            new_products_df = new_products_df[
            (self.products_df['weight'] < self.constraints.get_weight_limit())
            & (self.products_df['discounted_price'] < self.constraints.get_price_limit())
            ]

        itemized_products = new_products_df.apply(lambda row: Item(
            asin=row['asin'],
            title=row['title'],
            original_price=row['original_price'],
            discounted_price=row['discounted_price'],
            rating=row['rating'],
            weight=row['weight'],
            brand=row['brand'],
            categories=row['categories'],
        ), axis=1)
        
        itemized_products = itemized_products.values.tolist()
        self.queried_products = sorted(itemized_products, key= lambda x: (x.discounted_price, x.weight, x.original_price))
        if self.max_items > 0:
            self.queried_products = self.queried_products[:self.max_items]

        self.curr_product_get_mode = self.product_get_mode
        return self.queried_products
# actually gitubag naman nako to ganina pero for clarity kay para sa ako what's best for me kay if u take responsibility of how i feel and how u made me feel, for us to be something in the future. that way i don't need to go back to square one. dili nako siya maexplain right now pero it's kind of a big deal for me ang kani man gud. it's something na related pud sa ako childhood years. but ha, i am not asking you this. and honestly, like you, mahadlok pud ko makaguba ug relasyon sa mga tao