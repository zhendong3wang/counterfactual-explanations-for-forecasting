import csv
import os
import random as python_random
import numpy as np
import tensorflow as tf


class ResultWriter:
    def __init__(self, file_name, dataset_name):
        self.file_name = file_name
        self.dataset_name = dataset_name

    def write_head(self):
        # write the head in csv file
        with open(self.file_name, "w") as f:
            writer = csv.writer(f)
            writer.writerow(
                [
                    "dataset",
                    "random_seed",
                    "forecast_model",
                    "cf_model",
                    "horizon",
                    "desired_change",
                    "fraction_std",
                    "forecast_smape",
                    "forecast_mase",
                    "validity_ratio",
                    "proximity",
                    "compactness",
                    "step_validity_auc",
                    # "step_valid_counts",
                    # "slope_difference",
                    # "slope_difference_preds",
                ]
            )

    def write_result(
        self,
        random_seed,
        method_name,
        cf_method_name,
        horizon,
        desired_change,
        fraction_std,
        forecast_smape,
        forecast_mase,
        validity_ratio,
        proximity,
        compactness,
        step_validity_auc,
        # step_valid_counts,
        # slope_difference,
        # slope_difference_preds,
    ):
        with open(self.file_name, "a") as f:
            writer = csv.writer(f)
            writer.writerow(
                [
                    self.dataset_name,
                    random_seed,
                    method_name,
                    cf_method_name,
                    horizon,
                    desired_change,
                    fraction_std,
                    forecast_smape,
                    forecast_mase,
                    validity_ratio,
                    proximity,
                    compactness,
                    step_validity_auc,
                    # step_valid_counts,
                    # slope_difference,
                    # slope_difference_preds,
                ]
            )


# Method: Fix the random seeds to get consistent models
def reset_seeds(seed_value=39):
    # ref: https://keras.io/getting_started/faq/#how-can-i-obtain-reproducible-results-using-keras-during-development
    os.environ["PYTHONHASHSEED"] = str(seed_value)
    # necessary for starting Numpy generated random numbers in a well-defined initial state.
    np.random.seed(seed_value)
    # necessary for starting core Python generated random numbers in a well-defined state.
    python_random.seed(seed_value)
    # set_seed() will make random number generation
    tf.random.set_seed(seed_value)
