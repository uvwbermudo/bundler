import os
import matplotlib.pyplot as plt
import numpy as np
import csv
import time 
from bundler import Bundler

def plot_line(line, title, xlabel, ylabel, folder, name):
    folderpath = os.path.join(folder, 'graphs')
    os.makedirs(folderpath, exist_ok=True)
    if '.png' not in name:
        name = f"{name}.png"
    filepath = os.path.join(folder, 'graphs', name)
    fig, ax = plt.subplots()
    ax.plot(line)
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.grid(True)
    plt.savefig(filepath, bbox_inches='tight')
    plt.close()
    return fig, ax

def double_plot(frontline, backline, title, xlabel, ylabel, frontlabel, backlabel, folder, name):
    folderpath = os.path.join(folder, 'graphs')
    os.makedirs(folderpath, exist_ok=True)
    fig, ax = plt.subplots()  # Create a new figure and axis object
    ax.plot(frontline, label=frontlabel)
    ax.plot(backline, '--', label=backlabel)
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.grid(True)
    ax.legend()  # Show legend for labels
    if '.png' not in name:  # Add '.png' extension if missing
        name += '.png'
    filepath = os.path.join(folder, 'graphs', name)
    plt.savefig(filepath, bbox_inches='tight')  # Save the plot as an image with the provided filename
    plt.close()

def measure_convergence_rate(cvg):
    improvements = 0
    for i in range(1, len(cvg)):
        prev = cvg[i-1]
        curr = cvg[i]
        improvement = abs((curr - prev)/prev)
        improvements += improvement
    return improvements/len(cvg)

def measure_uniqueness(bundles, total_items):
    unique_hashes = set(bundles)
    return len(unique_hashes) / total_items

def generate_folder():
    folder = 'results'
    ctr = 0
    while folder in os.listdir(os.getcwd()):
        ctr += 1
    folder = f"results{ctr}"
    return folder

def save_arrays(res_dict, folder, subfolder):
    folder_path = os.path.join(folder, subfolder)
    os.makedirs(folder_path, exist_ok=True)
    for key in res_dict.keys():
        path = os.path.join(folder_path, key)
        np.save(path, res_dict[key])

def save_as_txt(res_dict, folder, filename):
    if 'txt' not in filename:
        txt_filename = f"{filename}.txt"
    else:
        txt_filename = filename

    csv_filename = f"{filename}.csv"

    os.makedirs(folder, exist_ok=True)
    txt_path = os.path.join(folder, txt_filename)
    csv_path = os.path.join(folder, csv_filename)

    # Write to text file
    with open(txt_path, 'a') as txt_file:
        for key in res_dict.keys():
            res = f"{key} = {res_dict[key]} \n"
            txt_file.write(res)

    # Write to CSV file
    with open(csv_path, 'w', newline='') as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=res_dict.keys())
        writer.writeheader()
        writer.writerow(res_dict)

def run_tests(trials, bundler, target=None):
    convergences, prox_convergences, average_fitnesses, best_solutions_fitnesses = [], [], [], []
    generations, stds, ranges, variances, mins, unqs, times, valids = [], [], [], [], [], [], [], []
    prox_best_solutions_fitnesses, convergence_rates = [], []
    prox_avg_distances, avg_distances, candidates = [], [], []
    avg_s, avg_v, avg_d, avg_e = [], [], [], []
    top_s, top_v, top_d, top_e = [], [], [], []
    bundler_results, target_hits = [], []

    folder = generate_folder()
    targetted = True

    if target is None:
        target = bundler.get_best_score()
        targetted = False

    for i in range(trials):
        print(f"Trial {i+1} out of {trials}")
        start_time = time.time()
        results = bundler.run(test=True)
        bundler_results.append([results['best_solution'], results['candidates']])
        times.append(time.time() - start_time)

        convergences.append(results['convergence'])
        convergence_rates.append(measure_convergence_rate(results['convergence']))
        prox_convergences.append([(target - x)/target for x in results['convergence']])
        average_fitnesses.append(results['average_fitness'])
        best_sol_fitness = results['best_solution'].get_fitness()
        best_solutions_fitnesses.append(best_sol_fitness)
        if best_sol_fitness >= target:
            target_hits.append(1)
        else:
            target_hits.append(0)
        prox_best_solutions_fitnesses.append(abs(target - results['best_solution'].get_fitness()))
        generations.append(results['generations'])
        stds.append(results['stds'])
        mins.append(results['mins'])
        variances.append(results['variances'])
        ranges.append(results['ranges'])
        unqs.append(results['unqs'])
        valids.append(results['valids'])
        prox_avg_distances.append(results['prox_avg_distances'])
        avg_distances.append(results['avg_distances'])
        candidates.append(results['candidates'])
        top_v.append(results['top_v_score'])
        top_s.append(results['top_s_score'])
        top_d.append(results['top_d_score'])
        top_e.append(results['top_e_score'])
        avg_v.append(results['avg_v_score'])
        avg_s.append(results['avg_s_score'])
        avg_d.append(results['avg_d_score'])
        avg_e.append(results['avg_e_score'])

    mean_generations = np.mean(generations)
    mean_best_solution = np.mean(best_solutions_fitnesses)
    mean_prox_best_solution = np.mean(prox_best_solutions_fitnesses)


    if not targetted:
        prox_avg_distances = np.array(prox_avg_distances)
        final_prox_avg_distances = [ val[-1] for val in prox_avg_distances]
    else:
        prox_avg_distances = 0
        final_prox_avg_distances = 'None'

    print('Saving results...')
    final_stds = [ val[-1] for val in stds]
    final_avg_fitness = [ val[-1] for val in average_fitnesses]
    final_mins = [ val[-1] for val in mins]
    final_vars = [ val[-1] for val in variances]
    final_ranges = [ val[-1] for val in ranges]
    final_unqs = [ val[-1] for val in unqs]
    final_valids = [ val[-1] for val in valids]
    final_top_s = [ val[-1] for val in top_s]
    final_top_e = [ val[-1] for val in top_e]
    final_top_d = [ val[-1] for val in top_d]
    final_top_v = [ val[-1] for val in top_v]
    final_avg_s = [ val[-1] for val in avg_s]
    final_avg_e = [ val[-1] for val in avg_e]
    final_avg_d = [ val[-1] for val in avg_d]
    final_avg_v = [ val[-1] for val in avg_v]

    mean_final_std = np.mean(final_stds)
    mean_final_mins = np.mean(final_mins)
    mean_final_vars = np.mean(final_vars)
    mean_final_ranges = np.mean(final_ranges)
    mean_final_unqs = np.mean(final_unqs)
    mean_final_valids = np.mean(final_valids)
    mean_exec_time = np.mean(times)
    mean_convergence_rate = np.mean(convergence_rates)
    mean_final_top_s = np.mean(final_top_s)
    mean_final_top_e = np.mean(final_top_e)
    mean_final_top_d = np.mean(final_top_d)
    mean_final_top_v = np.mean(final_top_v)
    mean_final_avg_s = np.mean(final_avg_s)
    mean_final_avg_e = np.mean(final_avg_e)
    mean_final_avg_d = np.mean(final_avg_d)
    mean_final_avg_v = np.mean(final_avg_v)
    mean_final_avg_fitness = np.mean(final_avg_fitness)
    mean_final_avg_prox_distances = np.mean(final_prox_avg_distances) if not targetted else 'None'

    candidates = np.array(candidates)
    candidates_fitness = [[bundle.get_fitness() for bundle in arr ] for arr in candidates]
    candidates_ave_min = np.mean([min(arr) for arr in candidates_fitness])
    candidates_ave_max = np.mean([max(arr) for arr in candidates_fitness])
    candidates_hash = [[bundle.get_hash() for bundle in arr] for arr in candidates]
    candidates_unq_ratio = np.mean([[measure_uniqueness(arr, total_items=len(arr))] for arr in candidates_hash])
    candidates_ave_fitness = np.mean([np.mean(arr) for arr in candidates_fitness])

    candidates_similarity = [[bundle.get_similarity() for bundle in arr ] for arr in candidates]
    candidates_epsilon = [[bundle.get_epsilon_score() for bundle in arr ] for arr in candidates]
    candidates_delta = [[bundle.get_delta_score() for bundle in arr ] for arr in candidates]
    candidates_value = [[bundle.get_value_score() for bundle in arr ] for arr in candidates]
    candidates_avg_similarity = np.mean([np.mean(arr) for arr in candidates_similarity])
    candidates_avg_epsilon = np.mean([np.mean(arr) for arr in candidates_epsilon])
    candidates_avg_delta = np.mean([np.mean(arr) for arr in candidates_delta])
    candidates_avg_value = np.mean([np.mean(arr) for arr in candidates_value])
    target_hit_rate = sum(target_hits)/len(target_hits)

    val_metrics = {
        "target_hit_rate": target_hit_rate,
        "target": target,
        "mean_best_solution": mean_best_solution,
        "mean_final_avg_fitness": mean_final_avg_fitness,
        "mean_prox_best_solution": mean_prox_best_solution,
        "mean_prox_avg_solution": mean_final_avg_prox_distances,
        "mean_convergence_rate": mean_convergence_rate,
        "mean_generations": mean_generations,
        "mean_exec_time": mean_exec_time,
        "mean_final_std": mean_final_std,
        "mean_final_mins": mean_final_mins,
        "mean_final_vars": mean_final_vars,
        "mean_final_ranges": mean_final_ranges,
        "mean_final_unqs": mean_final_unqs,
        "mean_final_valids": mean_final_valids,
        "mean_final_top_similarity": mean_final_top_s,
        "mean_final_top_epsilon": mean_final_top_s,
        "mean_final_top_delta": mean_final_top_s,
        "mean_final_top_value": mean_final_top_s,
        "mean_final_avg_similarity": mean_final_top_s,
        "mean_final_avg_epsilon": mean_final_top_s,
        "mean_final_avg_delta": mean_final_top_s,
        "mean_final_avg_value": mean_final_top_s,
        "candidates_ave_min": candidates_ave_min,
        "candidates_ave_max": candidates_ave_max,
        "candidates_unq_ratio": candidates_unq_ratio,
        "candidates_ave_fitness": candidates_ave_fitness,
        "candidates_avg_similarity": candidates_avg_similarity,
        "candidates_avg_epsilon": candidates_avg_epsilon,
        "candidates_avg_delta": candidates_avg_delta,
        "candidates_avg_value": candidates_avg_value,
    }

    save_as_txt(val_metrics, folder, 'val metrics')

    if targetted:
        print(f'Results saved to {os.path.join(os.getcwd(),folder)}')
        return

    mean_avg_fitness = np.mean(average_fitnesses, axis=0)
    prox_convergences = np.array(prox_convergences)
    mean_prox_convergence = np.mean(prox_convergences, axis=0)
    mean_prox_avg_distances = np.mean(prox_avg_distances, axis=0)
    # Graphs
    mean_stds = np.mean(stds, axis=0)
    mean_mins = np.mean(mins, axis=0)
    mean_vars = np.mean(variances, axis=0)
    mean_ranges = np.mean(ranges, axis=0)
    mean_unqs = np.mean(unqs, axis=0)
    mean_valids = np.mean(valids, axis=0)
    mean_top_s = np.mean(top_s, axis=0)
    mean_top_e = np.mean(top_e, axis=0)
    mean_top_d = np.mean(top_d, axis=0)
    mean_top_v = np.mean(top_v, axis=0)
    mean_avg_s = np.mean(avg_s, axis=0)
    mean_avg_e = np.mean(avg_e, axis=0)
    mean_avg_d = np.mean(avg_d, axis=0)
    mean_avg_v = np.mean(avg_v, axis=0)

    convergences = np.array(convergences)
    mean_convergence = np.mean(convergences, axis=0)
    mean_average_fitness = np.mean(average_fitnesses, axis=0)


    arrays = {
        "convergences": convergences,
        "prox_convergences": prox_convergences,
        "convergence_rates": convergence_rates,
        "average_fitnesses": average_fitnesses,
        "best_solution_fitnesses": best_solutions_fitnesses,
        "generations": generations,
        "stds": stds,
        "variances": variances,
        "ranges": ranges,
        "unqs": unqs,
        "valids": valids,
    }

    mean_arrays = {
        "mean_convergence":mean_convergence,
        "mean_average_fitness":mean_average_fitness,
        "mean_prox_convergence":mean_prox_convergence,
        "mean_stds":mean_stds,
        "mean_mins":mean_mins,
        "mean_vars":mean_vars,
        "mean_ranges":mean_ranges,
        "mean_unqs":mean_unqs,
        "mean_valids":mean_valids,
        "mean_top_s":mean_top_s,
        "mean_top_e":mean_top_e,
        "mean_top_d":mean_top_d,
        "mean_top_v":mean_top_v,
        "mean_avg_s":mean_avg_s,
        "mean_avg_e":mean_avg_e,
        "mean_avg_d":mean_avg_d,
        "mean_avg_v":mean_avg_v,
    }

    mean_top_s = np.mean(top_s, axis=0)
    mean_top_e = np.mean(top_e, axis=0)
    mean_top_d = np.mean(top_d, axis=0)
    mean_top_v = np.mean(top_v, axis=0)
    mean_avg_s = np.mean(avg_s, axis=0)
    mean_avg_e = np.mean(avg_e, axis=0)
    mean_avg_d = np.mean(avg_d, axis=0)
    mean_avg_v = np.mean(avg_v, axis=0)

    mean_top_s_plt = double_plot(frontline=mean_top_s, backline=mean_avg_s, title="Similarity Score Over Generations",
                                xlabel='Generations', ylabel='Similarity Score', folder=folder, name='mean_top_s',
                                frontlabel='Best', backlabel='Average')

    mean_top_e_plt = double_plot(frontline=mean_top_e, backline=mean_avg_e, title="Epsilon Score Over Generations",
                                xlabel='Generations', ylabel='Epsilon Score', folder=folder, name='mean_top_e',
                                frontlabel='Best', backlabel='Average')

    mean_top_d_plt = double_plot(frontline=mean_top_d, backline=mean_avg_d, title="Delta Score Over Generations",
                                xlabel='Generations', ylabel='Delta Score', folder=folder, name='mean_top_d',
                                frontlabel='Best', backlabel='Average')

    mean_top_v_plt = double_plot(frontline=mean_top_v, backline=mean_avg_v, title="Value Score Over Generations",
                                xlabel='Generations', ylabel='Value Score', folder=folder, name='mean_top_v',
                                frontlabel='Best', backlabel='Average')

    mean_convergence_plt = double_plot(frontline=mean_convergence, backline=mean_average_fitness, title='Average Convergence',
                                        xlabel='Generations', ylabel='Fitness Score',folder=folder, name='mean_convergence',
                                        frontlabel='Best', backlabel='Average')

    mean_valids_unq_plt = double_plot(frontline=mean_unqs, backline=mean_valids, title="Valid Solutions Over Generations %",
                                        xlabel='Generations', ylabel='Amount in %', folder=folder, name='mean_valids_unqs',
                                        frontlabel='Unique', backlabel='Valid')

    mean_unqs_plt = plot_line(line=mean_unqs, title='Valid Unique Solutions Over Generations %', xlabel='Generations', ylabel='%',folder=folder, name='mean_unqs')

    mean_valids_plt = plot_line(line=mean_valids, title='Valid Solutions Over Generations %', xlabel='Generations', ylabel='%',folder=folder, name='mean_valids')

    mean_ranges_plt = plot_line(line=mean_ranges, title='Ranges Over Generations', xlabel='Generations', ylabel='Range',folder=folder, name='mean_ranges')

    mean_vars_plt = plot_line(line=mean_vars, title='Variance Over Generations', xlabel='Generations', ylabel='Variance',folder=folder, name='mean_vars')

    mean_stds_plt = plot_line(line=mean_stds, title='STD Over Generations', xlabel='Generations', ylabel='Standard Deviation',folder=folder, name='mean_stds')

    mean_prox_convergence_plt = plot_line(line=mean_prox_convergence, title='Average Convergence', xlabel='Generations', ylabel='Distance to Global Optima', name='mean_prox_convergence_plt', folder=folder)

    mean_prox_convergence_with_avg_plt = double_plot(frontline=mean_prox_convergence, backline=mean_prox_avg_distances,
                                                title='Average Convergence', xlabel='Generations', ylabel='Distance to Global Optima',folder=folder, name='mean_prox_convergence_with_avg_plt',
                                                frontlabel='Best', backlabel='Average')


    save_arrays(arrays, folder, 'arrays')
    save_arrays(mean_arrays, folder, 'mean_arrays')
    # results = {
    #     'convergence':convergences,
    #     'average_fitnesses': average_fitnesses,
    #     'mean_best_solution': mean_best_solution,
    #     'mean_convergence': mean_convergence,
    #     'mean_fitness': mean_average_fitness,
    #     'mean_stds': mean_stds,
    #     'mean_final_std': mean_final_std,
    #     'mean_mins': mean_mins,
    #     'mean_final_min': mean_final_mins,
    #     'mean_vars': mean_vars,
    #     'mean_final_var': mean_final_vars,
    #     'mean_ranges': mean_ranges,x
    #     'mean_final_range': mean_final_ranges,
    #     'mean_unqs': mean_unqs,
    #     'mean_final_unqs': mean_final_unqs,
    #     'mean_exec_time': mean_exec_time,
    #     'mean_valids': mean_valids,
    #     'mean_final_valids': mean_final_valids,
    #     'mean_convergence_rate':  mean_convergence_rate
    # }
    print(f'Results saved to {os.path.join(os.getcwd(),folder)}')

    return bundler_results
if __name__ == '__main__':
    CSV_FILEPATH='amazon_dataset_v4.csv'
    bundler = Bundler()

    bundler.init_constraints(
        weight_limit=2, 
        price_limit=25, 
        bundle_size= [3,6], 
        csv_file=CSV_FILEPATH,
        get_mode='all',
        db_size=500,
        adjust=True)
    
    target = None
    bundler.init_GA(
        elitism_param=20,
        max_gen=100, 
        mutation_rate=0, 
        crossover_rate=0.4, 
        population_size=300, 
        test=True,  
        cross_method='random_uniform', 
        child_count=1,
        target_fitness=target)
    
    results = run_tests(trials=5, bundler=bundler, target=target)