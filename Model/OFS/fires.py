import numpy as np
from warnings import warn
from scipy.stats import norm
from sklearn.preprocessing import MinMaxScaler
from Model.OFS.ofs_ac import OnlineFeatureSelectionAC





class Fires(OnlineFeatureSelectionAC):
    """
    implementation of Fires algorithm
    """
    DEFAULT_PARAMS = {
        'n_total_ftr': None,
        'target_values': None,
        'mu_init': 0,
        'sigma_init': 1,
        'penalty_s': 0.01,
        'penalty_r': 0.01,
        'epochs': 1,
        'lr_mu': 0.01,
        'lr_sigma': 0.01,
        'scale_weights': True,
        'num_selected_features': 5,
        'transform_binary': True,
        'model': 'probit'
    }

    def __init__(self):
        super().__init__(name='FIRES', parameters=Fires.DEFAULT_PARAMS)

    def run(self, X, Y):
        from Model.Simulation.experiment import Experiment
        Y = Experiment.transform_binary(Y) # transform target to binary - due to legacy implementation
        self.parameters['n_total_ftr'] = X.shape[1]
        self.parameters['target_values'] = [0,1]

        additional_parmas = ['num_selected_features','transform_binary']
        model = Fires.FiresLegacy(**{key: val for key, val in self.parameters.items() if key not in additional_parmas })
        features_weights = model.weigh_features(X, Y)  # Get feature weights with FIRES
        selected_features = np.argsort(features_weights)[::-1][:self.parameters['num_selected_features']]
        return selected_features

    class FiresLegacy:
        def __init__(self, n_total_ftr, target_values, mu_init=0, sigma_init=1, penalty_s=0.01, penalty_r=0.01,
                     epochs=1,
                     lr_mu=0.01, lr_sigma=0.01, scale_weights=True, model='multi'):
            """
            FIRES: Fast, Interpretable and Robust Evaluation and Selection of features

            cite:
            Haug et al. 2020. Leveraging Model Inherent Variable Importance for Stable Online Feature Selection.
            In Proceedings of the 26th ACM SIGKDD Conference on Knowledge Discovery and Data Mining (KDD ’20),
            August 23–27, 2020, Virtual Event, CA, USA.

            :param n_total_ftr: (int) Total no. of features
            :param target_values: (np.ndarray) Unique target values (class labels)
            :param mu_init: (int/np.ndarray) Initial importance parameter
            :param sigma_init: (int/np.ndarray) Initial uncertainty parameter
            :param penalty_s: (float) Penalty factor for the uncertainty (corresponds to gamma_s in the paper)
            :param penalty_r: (float) Penalty factor for the regularization (corresponds to gamma_r in the paper)
            :param epochs: (int) No. of epochs that we use each batch of observations to update the parameters
            :param lr_mu: (float) Learning rate for the gradient update of the importance
            :param lr_sigma: (float) Learning rate for the gradient update of the uncertainty
            :param scale_weights: (bool) If True, scale feature weights into the range [0,1]
            :param model: (str) Name of the base model to compute the likelihood (default is 'probit')
            """

            self.n_total_ftr = n_total_ftr
            self.target_values = target_values
            self.mu = np.ones(n_total_ftr) * mu_init
            self.sigma = np.ones(n_total_ftr) * sigma_init
            self.penalty_s = penalty_s
            self.penalty_r = penalty_r
            self.epochs = epochs
            self.lr_mu = lr_mu
            self.lr_sigma = lr_sigma
            self.scale_weights = scale_weights
            self.model = model

            # Additional model-specific parameters
            self.model_param = {}

            # Probit model
            if self.model == 'probit' and tuple(target_values) != (-1, 1):
                if len(np.unique(target_values)) == 2:
                    self.model_param[
                        'probit'] = True  # Indicates that we need to encode the target variable into {-1,1}
                    # warn('FIRES WARNING: The target variable will be encoded as: {} = -1, {} = 1'.format(
                    #     self.target_values[0], self.target_values[1]))
                else:
                    raise ValueError('The target variable y must be binary.')

            # ### ADD YOUR OWN MODEL PARAMETERS HERE #######################################

        def weigh_features(self, x, y):
            """
            Compute feature weights, given a batch of observations and corresponding labels

            :param x: (np.ndarray) Batch of observations
            :param y: (np.ndarray) Batch of labels
            :return: feature weights
            :rtype np.ndarray
            """

            # Update estimates of mu and sigma given the predictive model
            if self.model == 'probit':
                self.__probit(x, y)
            # ### ADD YOUR OWN MODEL HERE ##################################################
            elif self.model == 'multi':
                self.__yourModel(x, y)
            else:
                raise NotImplementedError('The given model name does not exist')

            # Limit sigma to range [0, inf]
            if sum(n < 0 for n in self.sigma) > 0:
                self.sigma[self.sigma < 0] = 0
                warn('Sigma has automatically been rescaled to [0, inf], because it contained negative values.')

            # Compute feature weights
            return self.__compute_weights()

        def __probit(self, x, y):
            """
            Update the distribution parameters mu and sigma by optimizing them in terms of the (log) likelihood.
            Here we assume a Bernoulli distributed target variable. We use a Probit model as our base model.
            This corresponds to the FIRES-GLM model in the paper.

            :param x: (np.ndarray) Batch of observations (numeric values only, consider normalizing data for better results)
            :param y: (np.ndarray) Batch of labels: type binary, i.e. {-1,1} (bool, int or str will be encoded accordingly)
            """

            for epoch in range(self.epochs):
                # Shuffle the observations
                random_idx = np.random.permutation(len(y))
                x = x[random_idx]
                y = y[random_idx]

                # Encode target as {-1,1}
                if 'probit' in self.model_param:
                    y[y == self.target_values[0]] = -1
                    y[y == self.target_values[1]] = 1

                # Iterative update of mu and sigma
                try:
                    # Helper functions
                    dot_mu_x = np.dot(x, self.mu)
                    rho = np.sqrt(1 + np.dot(x ** 2, self.sigma ** 2))

                    # Gradients
                    nabla_mu = norm.pdf(y / rho * dot_mu_x) * (y / rho * x.T)
                    nabla_sigma = norm.pdf(y / rho * dot_mu_x) * (
                                - y / (2 * rho ** 3) * 2 * (x ** 2 * self.sigma).T * dot_mu_x)

                    # Marginal Likelihood
                    marginal = norm.cdf(y / rho * dot_mu_x)

                    # Update parameters
                    self.mu += self.lr_mu * np.mean(nabla_mu / marginal, axis=1)
                    self.sigma += self.lr_sigma * np.mean(nabla_sigma / marginal, axis=1)
                except TypeError as e:
                    raise TypeError('All features must be a numeric data type.') from e

        # ### ADD YOUR OWN MODEL HERE ##################################################

        def __compute_weights(self):
            """
            Compute optimal weights according to the objective function proposed in the paper.
            We compute feature weights in a trade-off between feature importance and uncertainty.
            Thereby, we aim to maximize both the discriminative power and the stability/robustness of feature weights.

            :return: feature weights
            :rtype np.ndarray
            """

            # Compute optimal weights
            weights = (self.mu ** 2 - self.penalty_s * self.sigma ** 2) / (2 * self.penalty_r)

            if self.scale_weights:  # Scale weights to [0,1]
                weights = MinMaxScaler().fit_transform(weights.reshape(-1, 1)).flatten()

            return weights
