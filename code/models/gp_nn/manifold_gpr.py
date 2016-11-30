import numpy as np

import tensorflow as tf
import GPflow


def init_xavier(shape):
    return np.random.normal(scale=(3. / np.sum(shape)) * np.ones(shape))

class ManifoldGPR(GPflow.model.GPModel):
    """
    Gaussian Process Regression.
    This is a vanilla implementation of GP regression with a Gaussian
    likelihood.  Multiple columns of Y are treated independently.
    The log likelihood i this models is sometimes referred to as the 'marginal log likelihood', and is given by
    .. math::
       \\log p(\\mathbf y \\,|\\, \\mathbf f) = \\mathcal N\\left(\\mathbf y\,|\, 0, \\mathbf K + \\sigma_n \\mathbf I\\right)
    """
    def __init__(self, X, Y, kern, mean_function=GPflow.mean_functions.Zero(), graph_type='fc'):
        """
        X is a data matrix, size N x D
        Y is a data matrix, size N x R
        kern, mean_function are appropriate GPflow objects
        """
        likelihood = GPflow.likelihoods.Gaussian()
        X = GPflow.param.DataHolder(X, on_shape_change='pass')
        Y = GPflow.param.DataHolder(Y, on_shape_change='pass')
        GPflow.model.GPModel.__init__(self, X, Y, kern, likelihood, mean_function)
        self.num_latent = Y.shape[1]

        if graph_type == 'fc':
            self._graph_create_params = self._graph_create_params_fc
            self._graph_inference = self._graph_inference_fc
        else:
            raise NotImplementedError('Graph type {0} not valid'.format(graph_type))

        self._graph_create_params()

    #####################
    ### Graph methods ###
    #####################

    def _graph_get_params(self, name):
        """
        Tries to find 'name{0}'.format(i) for more i
        :param name:
        :return:
        """
        params = []
        i = 0
        while True:
            p = getattr(self, '{0}{1}'.format(name, i), None)
            if p is None:
                break

            params.append(p)
            i += 1

        return params

    def _graph_create_params_fc(self):
        n_hidden = 40
        self.weights0 = GPflow.param.Param(init_xavier((self.X.shape[1], n_hidden)))
        self.weights1 = GPflow.param.Param(init_xavier((n_hidden, n_hidden)))
        self.weights2 = GPflow.param.Param(init_xavier((n_hidden, self.kern.input_dim)))
        self.biases0 = GPflow.param.Param(np.random.normal(scale=0.1*np.ones(n_hidden,)))
        self.biases1 = GPflow.param.Param(np.random.normal(scale=0.1 * np.ones(n_hidden,)))
        self.biases2 = GPflow.param.Param(np.random.normal(scale=0.1*np.ones((self.kern.input_dim,))))

    def _graph_inference_fc(self, x):
        weights = self._graph_get_params('weights')
        biases = self._graph_get_params('biases')

        layer = x
        for i, (weight, bias) in enumerate(zip(weights, biases)):
            layer = tf.add(tf.matmul(layer, weight), bias)
            if i < len(weights) - 1:
                layer = tf.nn.relu(layer)

        return layer

    #####################
    ### Build methods ###
    #####################

    def build_likelihood(self):
        """
        Construct a tensorflow function to compute the likelihood.
            \log p(Y | theta).
        """
        X = self._graph_inference(self.X)
        K = self.kern.K(X) + GPflow.tf_wraps.eye(tf.shape(X)[0]) * self.likelihood.variance
        L = tf.cholesky(K)
        m = self.mean_function(X)

        return GPflow.densities.multivariate_normal(self.Y, m, L)

    def build_predict(self, Xnew, full_cov=False):
        """
        Xnew is a data matrix, point at which we want to predict
        This method computes
            p(F* | Y )
        where F* are points on the GP at Xnew, Y are noisy observations at X.
        """
        X = self._graph_inference(self.X)
        Xnew = self._graph_inference(Xnew)

        Kx = self.kern.K(X, Xnew)
        K = self.kern.K(X) + GPflow.tf_wraps.eye(tf.shape(X)[0]) * self.likelihood.variance
        L = tf.cholesky(K)
        A = tf.matrix_triangular_solve(L, Kx, lower=True)
        V = tf.matrix_triangular_solve(L, self.Y - self.mean_function(X))
        fmean = tf.matmul(tf.transpose(A), V) + self.mean_function(Xnew)
        if full_cov:
            fvar = self.kern.K(Xnew) - tf.matmul(tf.transpose(A), A)
            shape = tf.pack([1, 1, tf.shape(self.Y)[1]])
            fvar = tf.tile(tf.expand_dims(fvar, 2), shape)
        else:
            fvar = self.kern.Kdiag(Xnew) - tf.reduce_sum(tf.square(A), 0)
            fvar = tf.tile(tf.reshape(fvar, (-1, 1)), [1, tf.shape(self.Y)[1]])
        return fmean, fvar

if __name__ == '__main__':
    import matplotlib.pyplot as plt

    np.random.seed(0)
    tf.set_random_seed(0)

    N = 100
    D = 1
    # X = np.random.rand(N, D)
    # X = np.vstack([np.linspace(0.4, 0.6, N) for _ in xrange(D)]).T
    X = np.random.normal(size=(N, D))
    # Y = np.sin(12 * X) + 0.66 * np.cos(25 * X) + np.random.randn(N, D) * 0.1
    Y = (X > 0.).astype(float) + np.random.randn(N, D) * 0.01

    A = 1. * np.ones((X.shape[1], Y.shape[1]))
    b = np.zeros(Y.shape[1], dtype=float)
    mean_function_gpr = GPflow.mean_functions.Linear(A=A, b=b)
    kern_gpr = GPflow.kernels.Matern52(D, lengthscales=0.3, ARD=False)
    gpr = GPflow.gpr.GPR(X, Y, kern_gpr, mean_function_gpr)

    A = 1. * np.ones((2, Y.shape[1]))
    b = np.zeros(Y.shape[1], dtype=float)
    mean_function_mgpr = GPflow.mean_functions.Linear(A=A, b=b)
    kern_mgpr = GPflow.kernels.Matern52(2, lengthscales=0.3, ARD=False)
    mgpr = ManifoldGPR(X, Y, kern_mgpr, mean_function_mgpr)

    for model in (gpr, mgpr):
        model.likelihood.variance = 0.1

        param_values_before = [(p.name, p.value) for p in model.sorted_params if hasattr(p, 'value')]
        mean, std = model.predict_f(X)
        err_before = np.linalg.norm(Y - mean, axis=1).mean()

        model.optimize(method=tf.train.AdamOptimizer(learning_rate=0.001), maxiter=5000)
        # gpr.optimize()

        param_values_after = [(p.name, p.value) for p in model.sorted_params if hasattr(p, 'value')]
        mean, std = model.predict_f(X)
        err_after = np.linalg.norm(Y - mean, axis=1).mean()


        def plot(m):
            xx = np.linspace(-5., 5., 500)[:, None]
            xx = np.tile(xx, (1, D))
            mean, var = m.predict_y(xx)
            mean = mean[:, 0]
            var = var[:, 0]
            plt.figure(figsize=(12, 6))
            plt.plot(X, Y, 'kx', mew=2)
            plt.plot(xx, mean, 'b', lw=2)
            plt.fill_between(xx[:, 0], mean - 2 * np.sqrt(var), mean + 2 * np.sqrt(var),
                             color='blue', alpha=0.2)
            plt.xlim(-5., 5.)

        plot(model)
        plt.show(block=False)

        print('\n\n\n')
        print('Prediction error before: {0}'.format(err_before))
        print('Prediction error after: {0}'.format(err_after))

        print('\n')
        print('Parameter values before')
        for name, value in param_values_before:
            print('\t{0}: {1}'.format(name, value))
        print('\n')
        print('Parameter values after')
        for name, value in param_values_after:
            print('\t{0}: {1}'.format(name, value))

    import IPython; IPython.embed()