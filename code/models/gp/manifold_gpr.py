
import tensorflow as tf
import GPflow

class ManifoldGPR(GPflow.model.GPModel):
    """
    Gaussian Process Regression.
    This is a vanilla implementation of GP regression with a Gaussian
    likelihood.  Multiple columns of Y are treated independently.
    The log likelihood i this models is sometimes referred to as the 'marginal log likelihood', and is given by
    .. math::
       \\log p(\\mathbf y \\,|\\, \\mathbf f) = \\mathcal N\\left(\\mathbf y\,|\, 0, \\mathbf K + \\sigma_n \\mathbf I\\right)
    """
    def __init__(self, X, Y, kern, mean_function=GPflow.mean_functions.Zero()):
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

    def build_likelihood(self):
        """
        Construct a tensorflow function to compute the likelihood.
            \log p(Y | theta).
        """
        K = self.kern.K(self.X) + GPflow.tf_wraps.eye(tf.shape(self.X)[0]) * self.likelihood.variance
        L = tf.cholesky(K)
        m = self.mean_function(self.X)

        return GPflow.densities.multivariate_normal(self.Y, m, L)

    def build_predict(self, Xnew, full_cov=False):
        """
        Xnew is a data matrix, point at which we want to predict
        This method computes
            p(F* | Y )
        where F* are points on the GP at Xnew, Y are noisy observations at X.
        """
        Kx = self.kern.K(self.X, Xnew)
        K = self.kern.K(self.X) + GPflow.tf_wraps.eye(tf.shape(self.X)[0]) * self.likelihood.variance
        L = tf.cholesky(K)
        A = tf.matrix_triangular_solve(L, Kx, lower=True)
        V = tf.matrix_triangular_solve(L, self.Y - self.mean_function(self.X))
        fmean = tf.matmul(tf.transpose(A), V) + self.mean_function(Xnew)
        if full_cov:
            fvar = self.kern.K(Xnew) - tf.matmul(tf.transpose(A), A)
            shape = tf.pack([1, 1, tf.shape(self.Y)[1]])
            fvar = tf.tile(tf.expand_dims(fvar, 2), shape)
        else:
            fvar = self.kern.Kdiag(Xnew) - tf.reduce_sum(tf.square(A), 0)
            fvar = tf.tile(tf.reshape(fvar, (-1, 1)), [1, tf.shape(self.Y)[1]])
        return fmean, fvar