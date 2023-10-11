import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler


class DataLoader:
    """
    Load data into desired formats for training/validation/testing, including preprocessing.
    """

    def __init__(self, horizon, back_horizon):
        self.horizon = horizon
        self.back_horizon = back_horizon
        self.scaler = list()

    def preprocessing(
        self,
        y,
        train_size=0.6,
        val_size=0.2,
        normalize=False,
        sequence_stride=1,
        ablation_horizon=None,
    ):
        self.sequence_stride = sequence_stride
        y = y.copy().astype("float")

        # count valid timesteps for each individual series
        self.valid_steps = np.isfinite(y).sum(axis=1)
        train_lst, val_lst, test_lst = list(), list(), list()

        for idx in range(y.shape[0]):
            y_sample = y[idx, : self.valid_steps[idx]]
            valid_steps_sample = self.valid_steps[idx]

            train = y_sample[: int(train_size * valid_steps_sample)]
            val = y_sample[
                (int(train_size * valid_steps_sample) - self.horizon) : int(
                    (train_size + val_size) * valid_steps_sample
                )
            ]
            test = y_sample[
                (int((train_size + val_size) * valid_steps_sample) - self.horizon) :
            ]

            if normalize:
                scaler = MinMaxScaler(feature_range=(0, 1), clip=False)
                train = remove_extra_dim(scaler.fit_transform(add_extra_dim(train)))
                val = remove_extra_dim(scaler.transform(add_extra_dim(val)))
                test = remove_extra_dim(scaler.transform(add_extra_dim(test)))
                self.scaler.append(scaler)

            train_lst.append(train)
            val_lst.append(val)
            test_lst.append(test)

        (
            self.X_train,
            self.Y_train,
            self.train_idxs,
        ) = self.create_sequences(
            train_lst,
            self.horizon,
            self.back_horizon,
            self.sequence_stride,
        )
        (self.X_val, self.Y_val, self.val_idxs) = self.create_sequences(
            val_lst,
            self.horizon,
            self.back_horizon,
            self.sequence_stride,
        )
        (
            self.X_test,
            self.Y_test,
            self.test_idxs,
        ) = self.create_sequences(
            test_lst,
            self.horizon,
            self.back_horizon,
            self.sequence_stride,
        )

        self.X_train, self.Y_train = add_extra_dim(self.X_train), add_extra_dim(
            self.Y_train
        )
        self.X_val, self.Y_val = add_extra_dim(self.X_val), add_extra_dim(self.Y_val)
        self.X_test, self.Y_test = add_extra_dim(self.X_test), add_extra_dim(
            self.Y_test
        )

        #         # for training autoencoders
        #         self.X_train_padded, self.padding_size = conditional_padding(self.X_train)
        #         self.X_val_padded, _ = conditional_padding(self.X_val)
        #         self.X_test_padded, _ = conditional_padding(self.X_test)

        # for ablation study of the forecasting horizon
        if ablation_horizon is not None:
            self.Y_train = self.Y_train[:, :ablation_horizon, :]
            self.Y_val = self.Y_val[:, :ablation_horizon, :]
            self.Y_test = self.Y_test[:, :ablation_horizon, :]

    @staticmethod
    def create_sequences(series_lst, horizon, back_horizon, sequence_stride):
        Xs, Ys, sample_idxs = list(), list(), list()

        # TODO: val_size * total_steps (should not) <= back_horizon + horizon
        for idx, series in enumerate(series_lst):
            len_series = series.shape[0]
            if len_series < (horizon + back_horizon):
                print(
                    f"Warning: not enough timesteps to split for sample {idx}, len: {len_series}, horizon: {horizon}, back: {back_horizon}."
                )

            for i in range(0, len_series - back_horizon - horizon, sequence_stride):
                Xs.append(series[i : (i + back_horizon)])
                Ys.append(series[(i + back_horizon) : (i + back_horizon + horizon)])
                # record the sample index when splitting
                sample_idxs.append(idx)

        return np.array(Xs), np.array(Ys), np.array(sample_idxs)


class MIMICDataLoader:
    """
    Load data into desired formats for training/validation/testing, including preprocessing, especially for MIMIC data.
    """

    def __init__(self, horizon, back_horizon):
        self.horizon = horizon
        self.back_horizon = back_horizon
        self.scaler = list()

    def preprocessing(
        self,
        y,
        normalize=False,
        sequence_stride=1,
        ablation_horizon=None,
    ):
        self.sequence_stride = sequence_stride
        y = y.copy().astype("float")

        # training/testing split ==> timestep-wise
        # each split: one horizon ahead
        # input_steps+T +T +T = n_timesteps_total
        self.X_train, self.Y_train = (
            y.copy()[:, : self.back_horizon],
            y.copy()[:, self.back_horizon : (self.back_horizon + self.horizon)],
        )
        self.X_val, self.Y_val = (
            y.copy()[:, self.horizon : (self.back_horizon + self.horizon)],
            y.copy()[
                :,
                (self.back_horizon + self.horizon) : (
                    self.back_horizon + self.horizon + self.horizon
                ),
            ],
        )
        self.X_test, self.Y_test = (
            y.copy()[
                :,
                (self.horizon + self.horizon) : (
                    self.back_horizon + self.horizon + self.horizon
                ),
            ],
            y.copy()[
                :,
                (self.back_horizon + self.horizon + self.horizon) : (
                    self.back_horizon + self.horizon + self.horizon + self.horizon
                ),
            ],
        )
        self.test_idxs = np.arange(self.X_test.shape[0])

        # y.shape = n_samples x n_timesteps
        if normalize:
            for idx in range(y.shape[0]):
                scaler = MinMaxScaler(feature_range=(0, 1), clip=False)

                self.X_train[idx] = remove_extra_dim(
                    scaler.fit_transform(add_extra_dim(self.X_train[idx]))
                )
                self.Y_train[idx] = remove_extra_dim(
                    scaler.transform(add_extra_dim(self.Y_train[idx]))
                )

                self.X_val[idx] = remove_extra_dim(
                    scaler.transform(add_extra_dim(self.X_val[idx]))
                )
                self.Y_val[idx] = remove_extra_dim(
                    scaler.transform(add_extra_dim(self.Y_val[idx]))
                )

                self.X_test[idx] = remove_extra_dim(
                    scaler.transform(add_extra_dim(self.X_test[idx]))
                )
                self.Y_test[idx] = remove_extra_dim(
                    scaler.transform(add_extra_dim(self.Y_test[idx]))
                )

                self.scaler.append(scaler)

        self.X_train, self.Y_train = add_extra_dim(self.X_train), add_extra_dim(
            self.Y_train
        )
        self.X_val, self.Y_val = add_extra_dim(self.X_val), add_extra_dim(self.Y_val)
        self.X_test, self.Y_test = add_extra_dim(self.X_test), add_extra_dim(
            self.Y_test
        )

        # for ablation study of the forecasting horizon
        if ablation_horizon is not None:
            self.Y_train = self.Y_train[:, :ablation_horizon, :]
            self.Y_val = self.Y_val[:, :ablation_horizon, :]
            self.Y_test = self.Y_test[:, :ablation_horizon, :]

    @staticmethod
    def create_sequences(series_lst, horizon, back_horizon, sequence_stride):
        Xs, Ys, sample_idxs = list(), list(), list()

        for idx, series in enumerate(series_lst):
            len_series = series.shape[0]
            if len_series < (horizon + back_horizon):
                print(
                    f"Error - not enough timesteps to split for sample {idx}, len: {len_series}, horizon: {horizon}, back: {back_horizon}."
                )

            for i in range(0, len_series - back_horizon - horizon, sequence_stride):
                Xs.append(series[i : (i + back_horizon)])
                Ys.append(series[(i + back_horizon) : (i + back_horizon + horizon)])
                # record the sample index when splitting
                sample_idxs.append(idx)

        return np.array(Xs), np.array(Ys), np.array(sample_idxs)


def load_dataset(dataset_name, data_path):
    if dataset_name == "cif2016":
        df = pd.read_csv(
            data_path + "cif_2016_dataset.tsf",
            sep=":|,",
            encoding="cp1252",
            header=None,
            index_col=0,
        )
        # only use the series with 12 forecasting horizon
        df = df[df.iloc[:, 0] == 12]
        df = df.drop([1], axis=1)
    elif dataset_name == "sp500":
        df = pd.read_csv(data_path + "data_all_stocks_5yr.csv")
        df = df[["date", "open", "Name"]]

        # pivot the table: row -> company, column -> date
        df = pd.pivot_table(df, values="open", index=["Name"], columns=["date"])

        df = df.dropna(axis=0)
    elif dataset_name == "nn5":
        df = pd.read_csv(
            data_path + "data_nn5_daily_dataset_without_missing_values.tsf",
            sep=":|,",
            encoding="cp1252",
            header=None,
            index_col=0,
        )

        df = df.drop([1], axis=1)
    elif dataset_name == "tourism":
        df = pd.read_csv(
            data_path + "data_tourism_monthly_dataset.tsf",
            sep=":",
            encoding="cp1252",
            header=None,
            index_col=0,
        )
        df = df.loc[:, 2].str.split(",", expand=True)
        df = df.astype("float")
    elif dataset_name == "m4":
        df = pd.read_csv(data_path + "/m4-Monthly-train.csv", index_col=0)

        data_cat = pd.read_csv(data_path + "/m4_info.csv", index_col=0)
        data_cat = data_cat[data_cat.index.map(lambda x: x.startswith("M"))]
        # TODO: use .iloc() or .loc() to match it?
        df = df[data_cat["category"] == "Finance"]
    elif dataset_name == "mimic":
        import json

        data = pd.read_csv(data_path + "/data_map.csv")

        df = data["interpolated"].apply(lambda x: np.asarray(json.loads(x)))
        df = np.asarray(df)
        df = np.stack(df, axis=0)
    else:
        print("Not implemented.")
    return df


# remove an extra dimension
def remove_extra_dim(input_array):
    # 2d to 1d
    if len(input_array.shape) == 2:
        return np.reshape(input_array, (-1))
    # 3d to 2d (remove the last empty dim)
    elif len(input_array.shape) == 3:
        return np.squeeze(np.asarray(input_array), axis=-1)
    else:
        print("Not implemented.")


# add an extra dimension
def add_extra_dim(input_array):
    # 1d to 2d
    if len(input_array.shape) == 1:
        return np.reshape(input_array, (-1, 1))
    # 2d to 3d
    elif len(input_array.shape) == 2:
        return np.asarray(input_array)[:, :, np.newaxis]
    else:
        print("Not implemented.")


# conditional padding
def conditional_padding(array):
    num = array.shape[1]

    if num % 4 != 0:
        # find the next integer that can be divided by 4
        next_num = (int(num / 4) + 1) * 4
        padding_size = next_num - num
        # pad for the 3d array => pre-padding (at the beginning)
        array_padded = np.pad(array, pad_width=((0, 0), (padding_size, 0), (0, 0)))

        return array_padded, padding_size

    # else return the original array (padding size = 0)
    return array, 0


# remove padding and the last (empty) dimension
def remove_paddings(cf_samples, padding_size):
    if padding_size != 0:
        # use np.squeeze() to remove the last time-series dimension, for evaluation
        cf_samples = np.squeeze(cf_samples[:, padding_size:, :], axis=-1)
    else:
        cf_samples = np.squeeze(cf_samples, axis=-1)
    return cf_samples


###################### Evaluation metrics ########################
def forecast_metrics(dataset, Y_pred):
    X_test_original, Y_test_original, Y_pred_original = list(), list(), list()
    for i in range(dataset.Y_test.shape[0]):
        idx = dataset.test_idxs[i]
        X_test_original.append(dataset.scaler[idx].inverse_transform(dataset.X_test[i]))
        Y_test_original.append(dataset.scaler[idx].inverse_transform(dataset.Y_test[i]))
        Y_pred_original.append(
            dataset.scaler[idx].inverse_transform(Y_pred[i].reshape(-1, 1))
        )
    X_test_original, Y_test_original, Y_pred_original = (
        np.array(X_test_original),
        np.array(Y_test_original),
        np.array(Y_pred_original),
    )

    def smape(Y_test, Y_pred):
        # src: https://github.com/ServiceNow/N-BEATS/blob/c746a4f13ffc957487e0c3279b182c3030836053/common/metrics.py
        def smape_sample(actual, forecast):
            return 200 * np.mean(
                np.abs(forecast - actual) / (np.abs(actual) + np.abs(forecast))
            )

        return np.mean([smape_sample(Y_test[i], Y_pred[i]) for i in range(len(Y_pred))])

    def mase(Y_test, Y_pred, X_test):
        # src: https://github.com/ServiceNow/N-BEATS/blob/c746a4f13ffc957487e0c3279b182c3030836053/common/metrics.py
        def mase_sample(actual, forecast, insample, m=1):
            # num = np.mean(np.abs(actual - forecast))
            denum = np.mean(np.abs(insample[:-m] - insample[m:]))

            # divide by 1.0 instead of 0.0, in case when denom is zero the enumerator will be 0.0 anyway (TODO)
            if denum == 0.0:
                denum = 1.0
            return np.mean(np.abs(actual - forecast)) / denum

        return np.mean(
            [mase_sample(Y_test[i], Y_pred[i], X_test[i]) for i in range(len(Y_pred))]
        )

    mean_smape = smape(Y_test_original, Y_pred_original)
    mean_mase = mase(Y_test_original, Y_pred_original, X_test_original)
    return mean_smape, mean_mase


def cf_metrics(
    desired_max_lst,
    desired_min_lst,
    X_test,
    cf_samples,
    z_preds,
    input_indices,
    label_indices,
):
    validity = validity_ratio(
        pred_values=z_preds,
        desired_max_lst=desired_max_lst,
        desired_min_lst=desired_min_lst,
    )
    proximity = euclidean_distance(X=X_test, cf_samples=cf_samples)
    compactness = compactness_score(X=X_test, cf_samples=cf_samples)
    cumsum_auc, cumsum_valid_steps, cumsum_counts = cumulative_valid_steps(
        pred_values=z_preds, max_bounds=desired_max_lst, min_bounds=desired_min_lst
    )
    slope_diff = slope_difference(
        X_originals=X_test, X_CFs=cf_samples, input_indices=input_indices
    )
    slope_diff_preds = slope_difference_preds(
        cf_preds=z_preds, label_indices=label_indices, upper_bounds=desired_max_lst
    )

    return (
        validity,
        proximity,
        compactness,
        cumsum_valid_steps,
        cumsum_counts,
        cumsum_auc,
        slope_diff,
        slope_diff_preds,
    )


def euclidean_distance(X, cf_samples, average=True):
    paired_distances = np.linalg.norm(X - cf_samples, axis=1)
    return np.mean(paired_distances) if average else paired_distances


# originally from: https://github.com/isaksamsten/wildboar/blob/859758884677ba32a601c53a5e2b9203a644aa9c/src/wildboar/metrics/_counterfactual.py#L279
def compactness_score(X, cf_samples):
    # absolute tolerance atol=0.01, 0.001, OR 0.0001?
    c = np.isclose(X, cf_samples, atol=0.01)
    compact_lst = np.mean(c, axis=1)
    return compact_lst.mean()


# validity ratio
def validity_ratio(pred_values, desired_max_lst, desired_min_lst):
    # pred_values.shape = (n_batch, n_timesteps, n_features)
    validity_lst = np.logical_and(
        pred_values <= desired_max_lst, pred_values >= desired_min_lst
    ).mean(axis=1)
    return validity_lst.mean()


def cumulative_valid_steps(pred_values, max_bounds, min_bounds):
    input_array = np.logical_and(pred_values <= max_bounds, pred_values >= min_bounds)
    until_steps_valid = np.empty(input_array.shape[0])
    n_samples, n_steps_total, _ = pred_values.shape
    for i in range(input_array.shape[0]):
        step_counts = 0
        for step in range(input_array.shape[1]):
            if input_array[i, step] == True:
                step_counts += 1
                until_steps_valid[i] = step_counts
            elif input_array[i, step] == False:
                until_steps_valid[i] = step_counts
                break
            else:
                print("Wrong input: cumulative_valid_steps.")

    valid_steps, counts = np.unique(until_steps_valid, return_counts=True)
    cumsum_counts = np.flip(np.cumsum(np.flip(counts)))
    # remove the valid_step=0 (no valid cf preds) in the trapz calculation
    valid_steps, cumsum_counts = fillna_cumsum_counts(
        n_steps_total, valid_steps, cumsum_counts
    )

    cumsum_auc = np.trapz(
        cumsum_counts[1:] / n_samples, valid_steps[1:] / n_steps_total
    )

    return cumsum_auc, valid_steps, cumsum_counts


def fillna_cumsum_counts(n_steps_total, valid_steps, cumsum_counts):
    df = pd.DataFrame(
        [{key: val for key, val in zip(valid_steps, cumsum_counts)}],
        columns=list(range(0, n_steps_total + 1)),
    )
    df = df.sort_index(ascending=True, axis=1)
    # backfill the previous valid steps
    df = df.fillna(method="backfill", axis=1)
    # fill 0s for the right hand nas
    df = df.fillna(method=None, value=0)
    valid_steps, cumsum_counts = df.columns.to_numpy(), df.values[0]
    return valid_steps, cumsum_counts


def slope_difference(X_originals, X_CFs, input_indices):
    slope_lst = list()
    for i in range(len(X_originals)):
        slope, intercept = np.polyfit(
            input_indices, remove_extra_dim(X_originals[i]), 1
        )
        slope2, intercept2 = np.polyfit(input_indices, remove_extra_dim(X_CFs[i]), 1)
        slope_diff = slope2 - slope

        slope_lst.append(slope_diff)

    return slope_lst


def slope_difference_preds(cf_preds, label_indices, upper_bounds):
    """
    upper_bounds: could be either the upper bounds or lower bounds, either of which can define the slope of desired prediction
    """
    slope_lst = list()
    for i in range(len(cf_preds)):
        slope, intercept = np.polyfit(
            label_indices, remove_extra_dim(upper_bounds[i]), 1
        )
        slope2, intercept2 = np.polyfit(label_indices, remove_extra_dim(cf_preds[i]), 1)
        slope_diff = slope2 - slope

        slope_lst.append(slope_diff)

    return slope_lst
