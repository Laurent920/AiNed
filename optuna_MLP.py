from mpi4py import MPI
import logging
import sys
import os

import jax
import mpi4jax
import joblib
import matplotlib.pyplot as plt
from pathlib import Path
import json

import optuna
from optuna.visualization.matplotlib import plot_contour
from optuna.visualization.matplotlib import plot_edf
from optuna.visualization.matplotlib import plot_intermediate_values
from optuna.visualization.matplotlib import plot_optimization_history
from optuna.visualization.matplotlib import plot_parallel_coordinate
from optuna.visualization.matplotlib import plot_param_importances
from optuna.visualization.matplotlib import plot_rank
from optuna.visualization.matplotlib import plot_slice
from optuna.visualization.matplotlib import plot_timeline

# from optuna.integration.wandb import WeightsAndBiasesCallback
# import wandb

from other_helpers.helpers import Params
from async_MPI import batch_predict, train, main

comm = MPI.COMM_WORLD
rank = comm.Get_rank()      # Real rank
size = comm.Get_size()

name = "mnist_mlp"
# name = "shd_mlp"
# name = "nmnist_mlp"

random_seed = 42
key = jax.random.key(random_seed)

PARAM_RANGES = {
    'num_hidden_layers':{'type': 'int', 'low': 1, 'high': 2, 'step': 1},
    'n_units':          {'type': 'categorical', 'choices': [64, 128, 256]},
    'restrict':         {'type': 'float', 'low': 1.0, 'high': 3.0, 'step': 0.1},
    'learning_rate':    {'type': 'float', 'low': 1e-4, 'high': 1e-4, 'log': True},
    'firing_nb':        {'type': 'int', 'low': 1, 'high': 128, 'log': True},
    'init_thresholds':  {'type': 'float', 'low': 0.0, 'high': 0.1},
    'threshold_lr':     {'type': 'float', 'low': 0.0, 'high': 1e-1},
    'sparsity_impact':  {'type': 'float', 'low': 0.0, 'high': 1e-3},
}

def objective(trial):
    if rank == 0:
        optimizer = trial.suggest_categorical("optimizer", ["adam"])

        num_hidden_layers = trial.suggest_int("num_hidden_layers", **{k:v for k,v in PARAM_RANGES['num_hidden_layers'].items() if k != 'type'})
        dataset = name.split("_")[0]
        match dataset:
            case "mnist":
                layer_sizes = [28*28]
            case "shd":
                layer_sizes = [700]
            case _:
                print("Study name need to start with the dataset name")
                sys.exit(1) 
        for i in range(num_hidden_layers):
            hidden_l_size = trial.suggest_categorical("n_units_l{}".format(i+1), PARAM_RANGES['n_units']['choices'])
            layer_sizes.append(hidden_l_size)
        match dataset:
            case "mnist":
                layer_sizes.append(10)
            case "shd":
                layer_sizes.append(20)

        
        restrict = trial.suggest_float("restrict", **{k:v for k,v in PARAM_RANGES['restrict'].items() if k != 'type'})
        learning_rate = trial.suggest_float("learning_rate", **{k:v for k,v in PARAM_RANGES['learning_rate'].items() if k != 'type'})
        firing_nb = trial.suggest_int("firing_nb", **{k:v for k,v in PARAM_RANGES['firing_nb'].items() if k != 'type'})
        init_thresholds = trial.suggest_float("init_thresholds", **{k:v for k,v in PARAM_RANGES['init_thresholds'].items() if k != 'type'})
        threshold_lr = trial.suggest_float("threshold_lr", **{k:v for k,v in PARAM_RANGES['threshold_lr'].items() if k != 'type'})
        sparsity_impact = trial.suggest_float("sparsity_impact", **{k:v for k,v in PARAM_RANGES['sparsity_impact'].items() if k != 'type'})
    else:
        dataset = num_hidden_layers = layer_sizes = None
        restrict = learning_rate = firing_nb = init_thresholds = threshold_lr = sparsity_impact = None

    dataset             = comm.bcast(dataset, root=0)
    num_hidden_layers   = comm.bcast(num_hidden_layers, root=0)
    layer_sizes         = comm.bcast(layer_sizes, root=0)
    restrict            = comm.bcast(restrict, root=0)
    learning_rate       = comm.bcast(learning_rate, root=0)
    firing_nb           = comm.bcast(firing_nb, root=0)
    init_thresholds     = comm.bcast(init_thresholds, root=0)
    threshold_lr        = comm.bcast(threshold_lr, root=0)
    sparsity_impact     = comm.bcast(sparsity_impact, root=0)

    params = Params(
                dataset=dataset,
                random_seed=random_seed,
                layer_sizes=tuple(layer_sizes), 
                init_thresholds=init_thresholds, 
                num_epochs=5, 
                learning_rate=learning_rate, 
                batch_size=36,
                load_file=False,
                shuffle_activations=False,
                restrict=tuple((restrict,) * len(layer_sizes)),
                firing_nb=firing_nb,
                sync_rate=1,
                max_nonzero=0,
                shuffle_input=False,
                threshold_lr=threshold_lr, 
                sparsity_impact=tuple((sparsity_impact,) * len(layer_sizes)), # Beta sparse
                rerun="",
                async_layer=-1,
                history_size=0
            )
    # print(params)
    mpi4jax.barrier(comm=comm)

    # trial layer_size, epoch_nb, learning_rate, batch_size, restrict, firing_nb
    # trial threshold_lr, sparsity_impact

    val_acc, max_it_normalized = main(random_seed, key, rank, size, comm, trial, params)
    trial_output = (val_acc*100)-max_it_normalized
    if rank == 0: print(f"Validation acc: {val_acc*100}%, normalized max iterations: {max_it_normalized}, trial output val: {trial_output}")
    return trial_output


def main_optuna():
    n_trials = 20
    read_data = False

    if rank == 0:
        # wandb.init(
        #     project="AiNed",  # choose your project name
        #     entity="Laurent", 
        #     config={},           # will be updated automatically by Optuna
        # )
        # wandb_callback = WeightsAndBiasesCallback(metric_name="loss")

        optuna.logging.get_logger("optuna").addHandler(logging.StreamHandler(sys.stdout))
        db_dir = "optuna/db/"         
        os.makedirs(db_dir, exist_ok=True)

        study = optuna.create_study(study_name=name, 
                                    directions=["maximize"], 
                                    pruner=optuna.pruners.MedianPruner(),
                                    storage=f"sqlite:///{db_dir}{name}.db",
                                    load_if_exists=True) #sampler=optuna.samplers.TPESampler()
        print(f"Sampler is {study.sampler.__class__.__name__}")

        # study.enqueue_trial({   "bagging_fraction": 0.75,
        #                         "bagging_freq": 5,
        #                         "min_child_samples": 20,
        #                     })
        add_trial = 1
        if add_trial:
            base_path = f"network_results/{name.split('_')[0]}/training/MLP"
            trials_data = parse_json_files(base_path, name.split('_')[0])
            added, skipped = add_trials_to_study(study, trials_data)
            print(f"Added {added} existing trials, skipped {skipped}")
        
        # study.optimize(objective, n_trials=n_trials, callbacks=[wandb_callback],) # , timeout=300
        if not read_data:
            study.optimize(objective, n_trials=n_trials) # , timeout=300
        
        best_params = study.best_params
        print(best_params)
        print(study.best_value)
        print(study.best_trial)

        # Visualize the optimization history.
        fig = plot_optimization_history(study)
        plt.savefig("optuna/plots/history.png") 

        # Visualize the learning curves of the trials. 
        plot_intermediate_values(study)
        plt.savefig("optuna/plots/intermediate_vals.png") 

        # Visualize high-dimensional parameter relationships.
        plot_parallel_coordinate(study)
        plt.savefig("optuna/plots/parallel_coords_all.png") 

        # Select parameters to visualize.
        plot_parallel_coordinate(study, params=["sparsity_impact", "n_units_l1"])
        plt.savefig("optuna/plots/parallel_coords.png") 

        # Visualize hyperparameter relationships.
        plot_contour(study)
        plt.savefig("optuna/plots/contour_all.png") 

        # Select parameters to visualize.
        plot_contour(study, params=["sparsity_impact", "n_units_l1"])
        plt.savefig("optuna/plots/contour.png") 
        
        # Visualize individual hyperparameters as slice plot.
        plot_slice(study)
        plt.savefig("optuna/plots/slice_all.png") 

        # Select parameters to visualize.
        plot_slice(study, params=["sparsity_impact", "n_units_l1"])
        plt.savefig("optuna/plots/slice.png") 

        # Visualize parameter importances.
        plot_param_importances(study)
        plt.savefig("optuna/plots/param_importances.png") 
        
        # Learn which hyperparameters are affecting the trial duration with hyperparameter importance.
        optuna.visualization.plot_param_importances(
            study, target=lambda t: t.duration.total_seconds(), target_name="duration"
        )

        # Visualize empirical distribution function. 
        plot_edf(study)
        plt.savefig("optuna/plots/edf.png") 

        # Visualize parameter relations with scatter plots colored by objective values.
        plot_rank(study)
        plt.savefig("optuna/plots/rank.png") 

        # Visualize the optimization timeline of performed trials. 
        plot_timeline(study)
        plt.savefig("optuna/plots/timeline.png") 

        
        # Customize generated figures
        # ---------------------------
        # In :mod:`optuna.visualization` and :mod:`optuna.visualization.matplotlib`, a function returns an editable figure object:
        # :class:`plotly.graph_objects.Figure` or :class:`matplotlib.axes.Axes` depending on the module.
        # This allows users to modify the generated figure for their demand by using API of the visualization library.
        # The following example replaces figure titles drawn by Plotly-based :func:`~optuna.visualization.plot_intermediate_values` manually.
        # fig = plot_intermediate_values(study)

        # fig.update_layout(
        #     title="Hyperparameter optimization for FashionMNIST classification",
        #     xaxis_title="Epoch",
        #     yaxis_title="Validation Accuracy",
        # )

        # joblib.dump(study, "study.pkl")
    else:
        if not read_data:
            for i in range(n_trials):
                objective(1)


def parse_json_files(base_path, dataset_name):
    """
    Parse JSON files from the directory structure and extract trial parameters.
    
    Args:
        base_path: Base directory path (e.g., 'mnist/training/MLP')
        dataset_name: Name of dataset for filtering (e.g., 'mnist')
    
    Returns:
        List of dictionaries containing trial parameters and values
    """
    print(f"Parsing {base_path} on {dataset_name} dataset")
    trials_data = []
    
    # Walk through all directories and subdirectories
    for root, dirs, files in os.walk(base_path):
        for file in files:
            # Only process files ending with _adam_.json
            if file.endswith('_adam_.json'):
                file_path = os.path.join(root, file)
                
                try:
                    with open(file_path, 'r') as f:
                        data = json.load(f)
                    
                    # Extract parameters
                    layer_sizes = data.get('layer_sizes', [])
                    validation_accuracy = data.get('validation accuracy', [])
                    it_mean = data.get('iterations mean', [])
                    
                    # Skip if essential data is missing
                    if not layer_sizes or not validation_accuracy or not it_mean:
                        print(f"Skipping {file_path}: Missing essential data")
                        continue
                    
                    epoch_eval = 5
                    # Calculate trial value
                    last_it_mean = [it[epoch_eval-1] for it in it_mean]
                    if len(last_it_mean) <= 1:
                        print(f"Skipping {file_path}: Insufficient iteration data")
                        continue
                    
                    normalized_it = max(last_it_mean[1:]) / last_it_mean[0]
                    last_val_acc = validation_accuracy[epoch_eval-1]
                    trial_value = (last_val_acc * 100) - normalized_it
                    
                    intermediate_values = {}
                    # Calculte intermediate values
                    for epoch in range(epoch_eval-2):
                        intermediate_it_mean = [it[epoch] for it in it_mean]
                        normalized_intermediate_it = max(intermediate_it_mean[1:]) / intermediate_it_mean[0]
                        intermediate_val_acc = validation_accuracy[epoch]
                        intermediate_values[epoch] = (intermediate_val_acc * 100) - normalized_intermediate_it
                    # Extract parameters for Optuna
                    num_hidden_layers = len(layer_sizes) - 2  # Exclude input and output layers
                    
                    # Extract restrict and threshold impact
                    restrict_raw = data.get('restrict', [1.0])
                    threshold_impact_raw = data.get('threshold impact', [0.0])

                    # Check if restrict is a single value or uniform list
                    if isinstance(restrict_raw, list):
                        if len(set(restrict_raw)) > 1:  # Different values in list
                            print(f"Skipping {file_path}: restrict has non-uniform values {restrict_raw}")
                            continue
                        restrict = restrict_raw[0]
                    else:
                        restrict = restrict_raw
                    if restrict == 0 or restrict == -1:
                        restrict = 1.0

                    # Check if threshold impact is a single value or uniform list
                    if isinstance(threshold_impact_raw, list):
                        if len(set(threshold_impact_raw)) > 1:  # Different values in list
                            print(f"Skipping {file_path}: threshold impact has non-uniform values {threshold_impact_raw}")
                            continue
                        threshold_impact = threshold_impact_raw[0]
                    else:
                        threshold_impact = threshold_impact_raw

                    # Build parameters dict
                    params = {
                        'optimizer': 'adam',
                        'num_hidden_layers': num_hidden_layers,
                        'restrict': restrict,
                        'learning_rate': data.get('learning rate', 0.0001),
                        'firing_nb': data.get('firing number', 0),
                        'init_thresholds': 0.0,  # Not in JSON, use default
                        'threshold_lr': data.get('threshold lr', 0.0),
                        'sparsity_impact': threshold_impact,
                    }
                    
                    # Add hidden layer sizes
                    for i in range(num_hidden_layers):
                        params[f'n_units_l{i+1}'] = layer_sizes[i+1]
                    
                    trials_data.append({
                        'params': params,
                        'value': trial_value,
                        'file': file_path,
                        'intermediate': intermediate_values
                    })
                    
                    print(f"Parsed {file_path}: value={trial_value:.4f}")
                    
                except json.JSONDecodeError:
                    print(f"Error decoding JSON: {file_path}")
                except Exception as e:
                    print(f"Error processing {file_path}: {str(e)}")
    
    return trials_data

def check_params_in_range(params, study):
    """
    Check if parameters fall within the study's search space.
    """
    out_of_range = []
    
    for param_name, param_value in params.items():
        # Handle hidden layer units
        if param_name.startswith('n_units_l'):
            if param_value not in PARAM_RANGES['n_units']['choices']:
                out_of_range.append(f"{param_name}={param_value} not in {PARAM_RANGES['n_units']['choices']}")
            continue
        
        if param_name not in PARAM_RANGES:
            continue
        
        param_config = PARAM_RANGES[param_name]
        
        if param_config['type'] == 'categorical':
            if param_value not in param_config['choices']:
                out_of_range.append(f"{param_name}={param_value} not in {param_config['choices']}")
        else:  # int or float
            # Convert to float for comparison if needed
            try:
                value_to_check = float(param_value)
                if not (param_config['low'] <= value_to_check <= param_config['high']):
                    out_of_range.append(f"{param_name}={param_value} not in [{param_config['low']}, {param_config['high']}]")
            except (ValueError, TypeError):
                out_of_range.append(f"{param_name}={param_value} is not a valid number")
    
    return len(out_of_range) == 0, out_of_range


def add_trials_to_study(study, trials_data):
    """
    Add parsed trials to Optuna study.
    
    Args:
        study: Optuna study object
        trials_data: List of trial dictionaries from parse_json_files
    
    Returns:
        Tuple of (added_count, skipped_count)
    """
    added_count = 0
    skipped_count = 0
    
    for trial_data in trials_data:
        params = trial_data['params']
        value = trial_data['value']
        file_path = trial_data['file']
        intermediate_values = trial_data['intermediate']

        # Check if parameters are in range
        in_range, out_of_range = check_params_in_range(params, study)
        
        if not in_range:
            print(f"Skipping {file_path}: Parameters out of range")
            for msg in out_of_range:
                print(f"  - {msg}")
            skipped_count += 1
            continue
        
        # Create distributions for the parameters
        distributions = {
            'optimizer': optuna.distributions.CategoricalDistribution(['adam']),
            'num_hidden_layers': optuna.distributions.IntDistribution(
                PARAM_RANGES['num_hidden_layers']['low'], 
                PARAM_RANGES['num_hidden_layers']['high'], 
                step=PARAM_RANGES['num_hidden_layers']['step']
            ),
            'restrict': optuna.distributions.FloatDistribution(
                PARAM_RANGES['restrict']['low'], 
                PARAM_RANGES['restrict']['high'], 
                step=PARAM_RANGES['restrict']['step']
            ),
            'learning_rate': optuna.distributions.FloatDistribution(
                PARAM_RANGES['learning_rate']['low'], 
                PARAM_RANGES['learning_rate']['high'], 
                log=PARAM_RANGES['learning_rate']['log']
            ),
            'firing_nb': optuna.distributions.IntDistribution(
                PARAM_RANGES['firing_nb']['low'], 
                PARAM_RANGES['firing_nb']['high'], 
                log=PARAM_RANGES['firing_nb']['log']
            ),
            'init_thresholds': optuna.distributions.FloatDistribution(
                PARAM_RANGES['init_thresholds']['low'], 
                PARAM_RANGES['init_thresholds']['high']
            ),
            'threshold_lr': optuna.distributions.FloatDistribution(
                PARAM_RANGES['threshold_lr']['low'], 
                PARAM_RANGES['threshold_lr']['high']
            ),
            'sparsity_impact': optuna.distributions.FloatDistribution(
                PARAM_RANGES['sparsity_impact']['low'], 
                PARAM_RANGES['sparsity_impact']['high']
            ),
        }

        # Add distributions for hidden layers
        for i in range(params['num_hidden_layers']):
            distributions[f'n_units_l{i+1}'] = optuna.distributions.CategoricalDistribution(
                PARAM_RANGES['n_units']['choices']
            )
        try:
            study.add_trial(
                optuna.trial.create_trial(
                    params=params,
                    distributions=distributions,
                    values=[value],  # Use list for multi-objective
                    intermediate_values=intermediate_values, 
                )
            )
            added_count += 1
            print(f"Added trial from {file_path}: value={value:.4f}")
        except Exception as e:
            print(f"Error adding trial from {file_path}: {str(e)}")
            skipped_count += 1
    
    return added_count, skipped_count


if __name__ == "__main__":
    main_optuna()