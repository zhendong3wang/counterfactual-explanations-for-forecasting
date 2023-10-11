import tensorflow as tf
import numpy as np
from sklearn.neighbors import NearestNeighbors
from _helper import remove_extra_dim, add_extra_dim


class ForecastCF:
    """Explanations by generating a counterfacutal sample for a desired forecasting outcome.
    References
    ----------
    Counterfactual Explanations for Time Series Forecasting,
    Wang, Z., Miliou, I., Samsten, I., Papapetrou, P., 2023.
    in: International Conference on Data Mining (ICDM 2023)
    """

    def __init__(
        self,
        *,
        tolerance=1e-6,
        max_iter=100,
        optimizer=None,
        pred_margin_weight=1.0,  # weighted_steps_weight = 1 - pred_margin_weight
        step_weights="local",
        random_state=None,
    ):
        """
        Parameters
        ----------
        probability : float, optional
            The desired probability assigned by the model
        tolerance : float, optional
            The maximum difference between the desired and assigned probability
        optimizer :
            Optimizer with a defined learning rate
        max_iter : int, optional
            The maximum number of iterations
        """
        self.optimizer_ = (
            tf.keras.optimizers.legacy.Adam(learning_rate=0.0001)
            if optimizer is None
            else optimizer
        )
        # self.mse_loss_ = tf.keras.losses.MeanSquaredError()
        self.tolerance_ = tf.constant(tolerance)
        self.max_iter = max_iter

        # Weights of the different loss components
        self.pred_margin_weight = pred_margin_weight
        self.weighted_steps_weight = 1 - self.pred_margin_weight

        self.step_weights = step_weights
        self.random_state = random_state

        self.MISSING_MAX_BOUND = np.inf
        self.MISSING_MIN_BOUND = -np.inf

    def fit(self, model):
        """Fit a new counterfactual explainer to the model parameters
        ----------
        model : keras.Model
            The model
        """
        self.model_ = model
        return self

    def predict(self, x):
        """Compute the difference between the desired and actual forecasting predictions
        ---------
        x : Variable
            Variable of the sample
        """

        return self.model_(x)

    # The "forecast_margin_loss" is designed to measure the prediction probability to the desired decision boundary
    def margin_mse(self, prediction, max_bound, min_bound):
        masking_vector = tf.logical_not(
            tf.logical_and(prediction <= max_bound, prediction >= min_bound)
        )
        unmasked_preds = tf.boolean_mask(prediction, masking_vector)

        if unmasked_preds.shape == 0:
            return 0

        mse_loss_ = tf.keras.losses.MeanSquaredError(
            reduction=tf.keras.losses.Reduction.SUM
        )

        if tf.reduce_any(max_bound != self.MISSING_MAX_BOUND):
            dist_max = mse_loss_(max_bound, unmasked_preds)
        else:
            dist_max = 0

        if tf.reduce_any(min_bound != self.MISSING_MIN_BOUND):
            dist_min = mse_loss_(min_bound, unmasked_preds)
        else:
            dist_min = 0

        return dist_max + dist_min

    # An auxiliary MAE loss function to measure the proximity with step_weights
    def weighted_mae(self, original_sample, cf_sample, step_weights):
        return tf.math.reduce_mean(
            tf.math.multiply(tf.math.abs(original_sample - cf_sample), step_weights)
        )

    # additional input of step_weights
    def compute_loss(
        self, original_sample, z_search, step_weights, max_bound, min_bound
    ):
        loss = tf.zeros(shape=())
        decoded = z_search
        pred = self.model_(decoded)

        #         kl_loss = tf.keras.losses.KLDivergence(reduction=tf.keras.losses.Reduction.SUM)
        #         y_true = tf.concat((original_sample, tf.expand_dims(self.model_(original_sample), axis=0)), axis=1)
        #         y_pred = tf.concat((decoded, tf.expand_dims(pred, axis=0)), axis=1)
        #         forecast_margin_loss = kl_loss(y_true, y_pred)

        #         def kl_divergence_uniform(p_min, p_max, q_min, q_max): # p_min <= q_min < q_max <= p_max
        #             return tf.math.log((q_max - q_min)/(p_max - p_min))
        #         forecast_margin_loss = kl_divergence_uniform(p_min=min_bound, p_max=max_bound, q_min=0, q_max=pred)

        forecast_margin_loss = self.margin_mse(pred, max_bound, min_bound)
        loss += self.pred_margin_weight * forecast_margin_loss

        weighted_steps_loss = self.weighted_mae(
            original_sample=tf.cast(original_sample, dtype=tf.float32),
            cf_sample=tf.cast(decoded, dtype=tf.float32),
            step_weights=tf.cast(step_weights, tf.float32),
        )
        loss += self.weighted_steps_weight * weighted_steps_loss

        return loss, forecast_margin_loss, weighted_steps_loss

    # TODO: compatible with the counterfactuals of wildboar
    #       i.e., define the desired output target per label
    def transform(self, x, max_bound_lst=None, min_bound_lst=None):
        """Generate counterfactual explanations
        x : array-like of shape [n_samples, n_timestep, n_dims]
            The samples
        """
        # TODO: make the parameter check more properly
        try:
            print(
                f"Validating threshold input: {len(max_bound_lst)==x.shape[0] or len(min_bound_lst)==x.shape[0]}"
            )
        except:
            print("Wrong parameter inputs, at least one threshold should be provided.")

        result_samples = np.empty(x.shape)
        losses = np.empty(x.shape[0])
        # `weights_all` needed for debugging
        weights_all = np.empty((x.shape[0], 1, x.shape[1], x.shape[2]))

        for i in range(x.shape[0]):
            if i % 25 == 0:
                print(f"{i+1} samples been transformed.")

            step_weights = self.step_weights

            # Check the condition of desired CF: larger/smaller/bound
            max_bound = (
                max_bound_lst[i] if max_bound_lst != None else self.MISSING_MAX_BOUND
            )
            min_bound = (
                min_bound_lst[i] if min_bound_lst != None else self.MISSING_MIN_BOUND
            )
            # print(max_bound, min_bound)
            x_sample, loss = self._transform_sample(
                x[np.newaxis, i], step_weights, max_bound, min_bound
            )

            result_samples[i] = x_sample
            losses[i] = loss
            weights_all[i] = step_weights

        print(f"{i+1} samples been transformed, in total.")

        return result_samples, losses, weights_all

    def _transform_sample(self, x, step_weights, max_bound, min_bound):
        """Generate counterfactual explanations
        x : array-like of shape [n_samples, n_timestep, n_dims]
            The samples
        """
        # TODO: check_is_fitted(self)
        z = tf.Variable(x, dtype=tf.float32)
        it = 0

        with tf.GradientTape() as tape:
            loss, forecast_margin_loss, weighted_steps_loss = self.compute_loss(
                x, z, step_weights, max_bound, min_bound
            )

        pred = self.model_(z)

        # # uncomment for debug
        # print(
        #     f"current loss: {loss}, forecast_margin_loss: {forecast_margin_loss}, weighted_steps_loss: {weighted_steps_loss}, desired range: {min_bound,max_bound}, pred prob:{pred}, iter: {it}."
        # )

        while (tf.reduce_any(pred > max_bound) or tf.reduce_any(pred < min_bound)) and (
            it < self.max_iter if self.max_iter else True
        ):
            # Get gradients of loss wrt the sample
            grads = tape.gradient(loss, z)
            # Update the weights of the sample
            self.optimizer_.apply_gradients([(grads, z)])
            #             self.optimizer_.apply_gradients(grads)

            with tf.GradientTape() as tape:
                loss, forecast_margin_loss, weighted_steps_loss = self.compute_loss(
                    x, z, step_weights, max_bound, min_bound
                )
            it += 1

            pred = self.model_(z)

        # # uncomment for debug
        # print(
        #     f"current loss: {loss}, forecast_margin_loss: {forecast_margin_loss}, weighted_steps_loss: {weighted_steps_loss}, desired range: {min_bound,max_bound}, pred prob:{pred}, iter: {it}. \n"
        # )

        res = z.numpy()
        return res, float(loss)


class BaselineShiftCF:
    """Explanations by generating a counterfacutal sample."""

    def __init__(self, *, desired_percent_change):
        """
        Parameters
        ----------
        desired_percent_change : float, optional
            The desired percent change of the counterfactual
        """
        self.desired_change = desired_percent_change

    # TODO: compatible with the counterfactuals of wildboar
    #       i.e., define the desired output target per label
    def transform(self, x):
        """Generate counterfactual explanations
        x : array-like of shape [n_samples, n_timestep, n_dims]
            The samples
        """
        result_samples = x * (1 + self.desired_change)
        return result_samples


class BaselineNNCF:
    """Explanations by generating a counterfacutal sample."""

    def __init__(self):
        pass

    def fit(self, X_train, Y_train):
        """Fit a new counterfactual explainer to the model parameters
        ----------
        model : keras.Model
            The model
        """
        self.nn_model_ = NearestNeighbors(n_neighbors=1, metric="euclidean")
        # Y_train.shape: (n_queries, n_features)
        self.nn_model_.fit(Y_train)
        self.train_samples = X_train
        return self

    # TODO: compatible with the counterfactuals of wildboar
    #       i.e., define the desired output target per label
    def transform(self, desired_max_lst, desired_min_lst):
        """Generate counterfactual explanations
        x : array-like of shape [n_samples, n_timestep, n_dims]
            The samples
        """
        # desired_preds.shape: (n_queries, n_features)
        desired_preds = np.asarray(
            [
                (desired_max + desired_min) / 2
                for desired_max, desired_min in zip(desired_max_lst, desired_min_lst)
            ]
        )
        closest_idx = self.nn_model_.kneighbors(
            remove_extra_dim(desired_preds), return_distance=False
        )
        result_samples = self.train_samples[remove_extra_dim(closest_idx)]
        return add_extra_dim(result_samples)
