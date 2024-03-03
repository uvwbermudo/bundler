from bundle import Bundle
from itemDB import ItemDB
import time
from constraints import Constraints
import sys
sys.setrecursionlimit(1000000000)


def dfs_helper(i, item_list, weight_limit, price_limit, bundle_size, curr_bundle):
    if i == len(item_list) or len(curr_bundle) > bundle_size:
        return 0, curr_bundle

    max_score, max_bundle = dfs_helper(i+1, item_list, weight_limit, price_limit, bundle_size, curr_bundle)

    item = item_list[i]
    new_weight_cap = weight_limit - item.weight
    new_price_cap = price_limit - item.discounted_price

    if new_weight_cap >= 0 and new_price_cap >= 0:
        curr_bundle = curr_bundle + [item]
        if len(curr_bundle) == bundle_size:
            bundle = Bundle(bundle=curr_bundle, items=items, constraints=constraints)
            bundle_fitness = bundle.get_fitness()
            max_score = max(max_score, bundle_fitness)
            if max_score == bundle_fitness:
                max_bundle = curr_bundle

        score, updated_bundle = dfs_helper(i, item_list, new_weight_cap, new_price_cap, bundle_size, curr_bundle)
        max_score = max(max_score, score)

        if max_score == score:
            max_bundle = updated_bundle

    return max_score, max_bundle

def dfs(item_list, weight_limit, price_limit, bundle_size):
    return dfs_helper(0, item_list, weight_limit, price_limit, bundle_size, [])

if __name__ == '__main__':
    weight_limit = float(input('Enter weight constraint: '))
    price_limit = float(input('Enter price constraint: '))
    bundle_size = int(input('Enter bundle size: '))
    repetition_limit = int(input('Enter repetition limit: '))
    constraints = Constraints()
    constraints.set_constraints(
        weight_limit=weight_limit,
        price_limit=price_limit,
        bundle_size=bundle_size,
        repetition_limit=repetition_limit
        )
    CSV_FILEPATH='amazon_dataset_v4.csv'
    print(constraints) 

    itemDB = ItemDB(constraints=constraints)
    itemDB.initialize_products_df(CSV_FILEPATH)
    itemDB.set_product_get_mode('constrained')
    print('bundle_size', bundle_size, 'weight', weight_limit, 'price', price_limit, f'items repeat {repetition_limit} times')

    items = itemDB.get_items()
    weight_limit = constraints.get_weight_limit()
    price_limit = constraints.get_price_limit()
    bundle_size = constraints.get_bundle_size()
    print('Total Items', len(items))

    start = time.time()
    res = dfs(items, weight_limit, price_limit, bundle_size)
    end = time.time() - start

    print(res[0])
    weight, price, min_price = 0,0,0
    print('\nRESULT===================')
    print('Time', end)
    ctr = 0
    for item in res[1]:
        ctr+=1
        print(item)
        print(item.title)
        weight += item.weight
        price += item.original_price
        min_price += item.discounted_price
    print(bundle_size, ctr, 'COUTNERS')
    print('Weight', weight, 'Constraint:', constraints.get_weight_limit())
    print('Original Price', price, 'Constraint:',  constraints.get_price_limit())
    print('Discounted Price', min_price, 'Constraint:', constraints.get_price_limit())
    print('=========================\n')

    result = Bundle(bundle=res[1], constraints=constraints, items=items)
    print(result.get_value_score())
    print(result.get_delta_score())
    print(result.get_epsilon_score())
    print(result.get_repetition_score())
    print(result.get_similarity())