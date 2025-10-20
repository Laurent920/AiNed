from async_MPI import batch_predict, train, main
import jax
from mpi4py import MPI
import optuna

import mpi4jax

comm = MPI.COMM_WORLD
rank = comm.Get_rank()      # Real rank
size = comm.Get_size()

def objective(trial):
    optimizer = trial.suggest_categorical("optimizer", ["MomentumSGD", "Adam"])
    num_layers = trial.suggest_int("num_layers", 1, 3) # log=True, step=5
    dropout_rate = trial.suggest_float("dropout_rate", 0.0, 1.0) # log=True, step=0.1

    # trial layer_size, epoch_nb, learning_rate, batch_size, restrict, firing_nb
    # trial threshold_lr, sparsity_impact

    time, accuracy = None, None
    return time, accuracy


if __name__ == "__main__":
    random_seed = 42
    key = jax.random.key(random_seed)
    
    main(random_seed, key, rank, size, comm)

    study = optuna.create_study(directions=["minimize", "maximize"]) #sampler=optuna.samplers.TPESampler()
    print(f"Sampler is {study.sampler.__class__.__name__}")

    study.optimize(objective, n_trials=20) # , timeout=300
    
    best_params = study.best_params
    study.best_value
    study.best_trial

# WITH PRUNING NEED TO ADD TO THE TRAIN FUNCTION 
# def objective(trial):
#     iris = sklearn.datasets.load_iris()
#     classes = list(set(iris.target))
#     train_x, valid_x, train_y, valid_y = sklearn.model_selection.train_test_split(
#         iris.data, iris.target, test_size=0.25, random_state=0
#     )

#     alpha = trial.suggest_float("alpha", 1e-5, 1e-1, log=True)
#     clf = sklearn.linear_model.SGDClassifier(alpha=alpha)

#     for step in range(100):
#         clf.partial_fit(train_x, train_y, classes=classes)

#         # Report intermediate objective value.
#         intermediate_value = 1.0 - clf.score(valid_x, valid_y)
#         trial.report(intermediate_value, step)

#         # Handle pruning based on the intermediate value.
#         if trial.should_prune():
#             raise optuna.TrialPruned()

#     return 1.0 - clf.score(valid_x, valid_y)


# ###################################################################################################
# # Set up the median stopping rule as the pruning condition.

# # Add stream handler of stdout to show the messages
# optuna.logging.get_logger("optuna").addHandler(logging.StreamHandler(sys.stdout))
# study = optuna.create_study(pruner=optuna.pruners.MedianPruner())
# study.optimize(objective, n_trials=20)