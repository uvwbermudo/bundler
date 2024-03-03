class Item:

    def __init__(self, asin, title, original_price, discounted_price,
                rating, weight, brand, categories, img_url=None):
        self.asin = asin
        self.title = title
        self.original_price = original_price
        self.discounted_price = discounted_price
        self.rating = rating
        self.weight = weight
        self.brand = brand
        self.categories = set(self.get_clean_categories(categories))
        self.value = self.get_value()

    def get_value(self) -> float:
        return (1 + (abs(self.rating - 2.5)/2.5))/2

    def get_clean_categories(self, input) -> str:
        categories = input.strip().replace(' ',"")
        category_split = categories.split('>')
        return category_split if category_split else categories

    def __repr__(self):
        return str({
            "ASIN": self.asin,
            # "Title": self.title,
            # "PRICE": self.price,
            "D_PRICE": self.discounted_price,
            "O_PRICE": self.original_price,
            "RATING": self.rating,
            "WEIGHT": self.weight,
            "BRAND": self.brand,
            "CATEGORIES": self.categories,
            # "VALUE": self.value
        })

    def __hash__(self):
        return hash(self.asin)

    def __eq__(self, other):
        return isinstance(other, Item) and self.name == other.name