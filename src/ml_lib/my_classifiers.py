import numpy as np
import sys


class Perceptron(object):
    """
    IMPORTANT: to extend the perceptron to One-vs-All (OvA) classification problems, we can train n classifiers, where
    n is the number of classes by putting each time the class of reference as 1s and all the others as -1s. Then,
    we can assing the class to new data by associating the class with the greatest confidence, thus the highest value
    of the net input (one must return explicitly the net_input() in that case)
    """

    def __init__(self, eta=0.01, n_iter=50, ran_state=42, weights=None, errors=[]):
        """
        Parameters
        :param eta: [float] learning rate
        :param n_iter: [int] number of epochs
        :param ran_state: [int] random number generator seed for weights initialisation
        :param weights: {array-like} initialised to None and then reinitialised
        :param errors: {list-like} list of errors made in the prediction
        """
        self.eta = eta
        self.n_iter = n_iter
        self.ran_state = ran_state
        self.w_ = weights
        self.errors_ = errors

    def fit(self, X, y):
        """
        Parameters
        :param X: {array-like; shape = [n_samples, n_features]} Training vectors
        :param y: {array-like; shape = [n_samples, 1]} Target values (labels)
        :return: self: object
        """

        """
        IMPORTANT: the initialisation of the weights to small numbers is due to the fact that if we do initialise them
        to zero values, the learning rate has no effect on the classification outcome, but it would affect only
        the scale of the weight vector and not the direction
        """
        r_gen = np.random.RandomState(self.ran_state)
        self.w_ = r_gen.normal(loc=0.00, scale=0.01, size=1 + X.shape[1])

        for _ in range(self.n_iter):
            errors = 0
            for xi, lab in zip(X, y):
                delta_w = self.eta * (lab - self.predict(xi))
                self.w_[1:] += delta_w * xi
                self.w_[0] += delta_w
                errors += int(delta_w != 0.0)
            self.errors_.append(errors)

        return self

    def net_input(self, X):
        """
        Calculate the net input
        :param X: training data
        :return: [float] dot product between training data and weights
        """
        return np.dot(X, self.w_[1:]) + self.w_[0]

    def predict(self, X):
        """
        :param X: input data
        :return: [int] return 1 for correct prediction, -1 otherwise
        """
        return np.where(self.net_input(X) >= 0, 1, -1)


class Adaline(object):
    """ADAptive LInear NEuron classifier. Put SGD = True for stochastic gradient descent

    Parameters
    ------------
    eta : float
      Learning rate (between 0.0 and 1.0)
    n_iter : int
      Passes over the training dataset.
    sgd: bool (default: False)
      Apply stochastic gradient descent.
    shuffle : bool (default: True)
      Shuffles training data every epoch if True to prevent cycles.
    cost_list: list (default: None)
    weights: list (default: None)
      List of weights to be initialized.
    random_state : int
      Random number generator seed for random weight
      initialization.

    Attributes
    -----------
    w_ : 1d-array
      Weights after fitting.
    cost_ : list
      Sum-of-squares cost function value averaged over all
      training samples in each epoch.

    -----------
    -- Adaline adds an activation function to the idea of the perceptron (the identity function), which
    allows to introduce a differentiable error (cost) function to be minimised. The error function is the sum
    of squared errors, which being convex allows to use a gradient descent optimisation algorithm to find the
    minimum of the cost function
    """

    def __init__(self, eta=0.01, n_iter=10, sgd=False, shuffle=True, cost_list=None, weights=None, random_state=None):

        self.eta = eta
        self.n_iter = n_iter
        self.w_initialized = False
        self.shuffle = shuffle
        self.random_state = random_state
        if cost_list is None:
            cost_list = []
            self.cost_ = cost_list
        self.w_ = weights
        self.sgd = sgd

    def fit(self, X, y):
        """ Fit training data.

        Parameters
        ----------
        X : {array-like}, shape = [n_samples, n_features]
          Training vectors, where n_samples is the number of samples and
          n_features is the number of features.
        y : array-like, shape = [n_samples]
          Target values.

        Returns
        -------
        self : object

        """

        self._initialize_weights(X.shape[1])

        if self.sgd:
            for _ in range(self.n_iter):
                if self.shuffle:
                    X, y = self._shuffle(X, y)
                cost = []
                for xi, target in zip(X, y):
                    cost.append(self._update_weights(xi, target))
                avg_cost = sum(cost) / len(y)
                self.cost_.append(avg_cost)
        else:
            for _ in range(self.n_iter):
                net_input = self.net_input(X)
                output = self.activation(net_input)
                errors = (y - output)
                self.w_[1:] += self.eta * X.T.dot(errors)
                self.w_[0] += self.eta * errors.sum()
                cost = (errors ** 2).sum() / 2.0
                self.cost_.append(cost)

        return self

    def partial_fit(self, X, y):
        """Fit training data without reinitializing the weights"""
        if not self.w_initialized:
            self._initialize_weights(X.shape[1])
        if y.ravel().shape[0] > 1:
            for xi, target in zip(X, y):
                self._update_weights(xi, target)
        else:
            self._update_weights(X, y)
        return self

    def _shuffle(self, X, y):
        """Shuffle training data"""
        r = self.rgen.permutation(len(y))
        return X[r], y[r]

    def _initialize_weights(self, m):
        """Initialize weights to small random numbers"""
        self.rgen = np.random.RandomState(self.random_state)
        self.w_ = self.rgen.normal(loc=0.0, scale=0.01, size=1 + m)
        self.w_initialized = True

    def _update_weights(self, xi, target):
        """Apply Adaline learning rule to update the weights"""
        output = self.activation(self.net_input(xi))
        error = (target - output)
        self.w_[1:] += self.eta * xi.dot(error)
        self.w_[0] += self.eta * error
        cost = 0.5 * error ** 2
        return cost

    def net_input(self, X):
        """Calculate net input"""
        return np.dot(X, self.w_[1:]) + self.w_[0]

    @staticmethod
    def activation(X):
        """Compute linear activation"""
        return X

    def predict(self, X):
        """Return class label after unit step"""
        return np.where(self.activation(self.net_input(X)) >= 0.0, 1, -1)


class LogisticRegressionGD(object):
    """Logistic Regression Classifier using gradient descent.

    Parameters
    ------------
    eta : float
      Learning rate (between 0.0 and 1.0)
    n_iter : int
      Passes over the training dataset.
    ran_state : int
      Random number generator seed for random weight
      initialization.


    Attributes
    -----------
    w_ : 1d-array
      Weights after fitting.
    cost_ : list
      Sum-of-squares cost function value in each epoch.

    Note: -- Logistic Regression is similar to Adaline, but working with the logarithm of the odds ratio (the logarithm
    takes values in [0, 1] and returns values in the full real-number range, which we can use to express linear
    relationship between feature values and log-odds). We want to predict the probability that a sample belongs to a
    particular class, which is the inverse form of the logit function. It is also called logistic sigmoid function.
    This function does the opposite of the logit, returning from a net input a value in [0,1] which can be interpreted
    as a probability (and we can then convert into binary outcomes in case). Instead of minimizing the sum-squared-err
    we now want to maximize the (log-)likelihood (which allows to transform the products into sums when we have
    independent conditional probabilities in the equation, and also limits potential underflows
    """

    def __init__(self, eta=0.05, n_iter=100, ran_state=1, weights=None, cost_list=[]):
        self.eta = eta
        self.n_iter = n_iter
        self.ran_state = ran_state
        self.w_ = weights
        self.cost_ = cost_list

    def fit(self, X, y):
        """
        Parameters
        :param X: {array-like; shape = [n_samples, n_features]} Training vectors
        :param y: {array-like; shape = [n_samples, 1]} Target values (labels)
        :return: self: object
        """

        """
        IMPORTANT: the initialisation of the weights to small numbers is due to the fact that if we do initialise them
        to zero values, the learning rate has no effect on the classification outcome, but it would affect only
        the scale of the weight vector and not the direction
        """
        self._initialize_weights(X.shape[1])

        for i in range(self.n_iter):
            net_input = self.net_input(X)
            output = self.activation(net_input)
            errors = (y - output)
            self.w_[1:] += self.eta * X.T.dot(errors)
            self.w_[0] += self.eta * errors.sum()
            cost = -y.dot(np.log(output)) - ((1 - y).dot(np.log(1 - output)))
            self.cost_.append(cost)

        return self

    def _initialize_weights(self, m):
        """
        Initialise weights to small random numbers and change the status of w_initialised variable
        :param m: [int] length of the weight vector (1 to add for the bias unit)
        :return: void
        """
        self.r_gen = np.random.RandomState(self.ran_state)
        self.w_ = self.r_gen.normal(loc=0.0, scale=0.01, size=1 + m)
        self.w_initialised = True

    def net_input(self, X):
        """
        Calculate the net input
        :param X: training data
        :return: [float] dot product between training data and weights
        """
        return np.dot(X, self.w_[1:]) + self.w_[0]

    @staticmethod
    def activation(z):
        """
        Compute sigmoidal activation function
        :param z: training data
        :return: [float] logistic output
        """
        return 1. / (1. + np.exp(-np.clip(z, -250, 250)))

    def predict(self, X):
        """
        :param X: input data
        :return: [int] return 1 for correct prediction, 0 otherwise
        """
        return np.where(self.activation(self.net_input(X)) >= 0.5, 1, 0)


class NeuralNetMLP(object):
    """ Feedforward neural network / Multi-layer perceptron classifier.

    Parameters
    ------------
    n_hidden : int (default: 30)
        Number of hidden units.
    l2 : float (default: 0.)
        Lambda value for L2-regularization.
        No regularization if l2=0. (default)
    epochs : int (default: 100)
        Number of passes over the training set.
    eta : float (default: 0.001)
        Learning rate.
    shuffle : bool (default: True)
        Shuffles training data every epoch if True to prevent circles.
    minibatch_size : int (default: 1)
        Number of training samples per minibatch.
    seed : int (default: None)
        Random seed for initalizing weights and shuffling.

    Attributes
    -----------
    eval_ : dict
      Dictionary collecting the cost, training accuracy,
      and validation accuracy for each epoch during training.

    """
    def __init__(self, n_hidden=30,
                 l2=0., epochs=100, eta=0.001,
                 shuffle=True, minibatch_size=1, seed=None):

        self.random = np.random.RandomState(seed)
        self.n_hidden = n_hidden
        self.l2 = l2
        self.epochs = epochs
        self.eta = eta
        self.shuffle = shuffle
        self.minibatch_size = minibatch_size

    def _onehot(self, y, n_classes):
        """Encode labels into one-hot representation

        Parameters
        ------------
        y : array, shape = [n_samples]
            Target values.

        Returns
        -----------
        onehot : array, shape = (n_samples, n_labels)

        """
        onehot = np.zeros((n_classes, y.shape[0]))
        for idx, val in enumerate(y.astype(int)):
            onehot[val, idx] = 1.
        return onehot.T

    def _sigmoid(self, z):
        """Compute logistic function (sigmoid)"""
        return 1. / (1. + np.exp(-np.clip(z, -250, 250)))

    def _forward(self, X):
        """Compute forward propagation step"""

        # step 1: net input of hidden layer
        # [n_samples, n_features] dot [n_features, n_hidden]
        # -> [n_samples, n_hidden]
        z_h = np.dot(X, self.w_h) + self.b_h

        # step 2: activation of hidden layer
        a_h = self._sigmoid(z_h)

        # step 3: net input of output layer
        # [n_samples, n_hidden] dot [n_hidden, n_classlabels]
        # -> [n_samples, n_classlabels]

        z_out = np.dot(a_h, self.w_out) + self.b_out

        # step 4: activation output layer
        a_out = self._sigmoid(z_out)

        return z_h, a_h, z_out, a_out

    def _compute_cost(self, y_enc, output):
        """Compute cost function.

        Parameters
        ----------
        y_enc : array, shape = (n_samples, n_labels)
            one-hot encoded class labels.
        output : array, shape = [n_samples, n_output_units]
            Activation of the output layer (forward propagation)

        Returns
        ---------
        cost : float
            Regularized cost

        """
        L2_term = (self.l2 *
                   (np.sum(self.w_h ** 2.) +
                    np.sum(self.w_out ** 2.)))

        term1 = -y_enc * (np.log(output))
        term2 = (1. - y_enc) * np.log(1. - output)
        cost = np.sum(term1 - term2) + L2_term
        
        # If you are applying this cost function to other
        # datasets where activation
        # values maybe become more extreme (closer to zero or 1)
        # you may encounter "ZeroDivisionError"s due to numerical
        # instabilities in Python & NumPy for the current implementation.
        # I.e., the code tries to evaluate log(0), which is undefined.
        # To address this issue, you could add a small constant to the
        # activation values that are passed to the log function.
        #
        # For example:
        #
        # term1 = -y_enc * (np.log(output + 1e-5))
        # term2 = (1. - y_enc) * np.log(1. - output + 1e-5)
        
        return cost

    def predict(self, X):
        """Predict class labels

        Parameters
        -----------
        X : array, shape = [n_samples, n_features]
            Input layer with original features.

        Returns:
        ----------
        y_pred : array, shape = [n_samples]
            Predicted class labels.

        """
        z_h, a_h, z_out, a_out = self._forward(X)
        y_pred = np.argmax(z_out, axis=1)
        return y_pred

    def fit(self, X_train, y_train, X_valid, y_valid):
        """ Learn weights from training data.

        Parameters
        -----------
        X_train : array, shape = [n_samples, n_features]
            Input layer with original features.
        y_train : array, shape = [n_samples]
            Target class labels.
        X_valid : array, shape = [n_samples, n_features]
            Sample features for validation during training
        y_valid : array, shape = [n_samples]
            Sample labels for validation during training

        Returns:
        ----------
        self

        """
        n_output = np.unique(y_train).shape[0]  # number of class labels
        n_features = X_train.shape[1]

        ########################
        # Weight initialization
        ########################

        # weights for input -> hidden
        self.b_h = np.zeros(self.n_hidden)
        self.w_h = self.random.normal(loc=0.0, scale=0.1,
                                      size=(n_features, self.n_hidden))

        # weights for hidden -> output
        self.b_out = np.zeros(n_output)
        self.w_out = self.random.normal(loc=0.0, scale=0.1,
                                        size=(self.n_hidden, n_output))

        epoch_strlen = len(str(self.epochs))  # for progress formatting
        self.eval_ = {'cost': [], 'train_acc': [], 'valid_acc': []}

        y_train_enc = self._onehot(y_train, n_output)

        # iterate over training epochs
        for i in range(self.epochs):

            # iterate over minibatches
            indices = np.arange(X_train.shape[0])

            if self.shuffle:
                self.random.shuffle(indices)

            for start_idx in range(0, indices.shape[0] - self.minibatch_size +
                                   1, self.minibatch_size):
                batch_idx = indices[start_idx:start_idx + self.minibatch_size]

                # forward propagation
                z_h, a_h, z_out, a_out = self._forward(X_train[batch_idx])

                ##################
                # Backpropagation
                ##################

                # [n_samples, n_classlabels]
                sigma_out = a_out - y_train_enc[batch_idx]

                # [n_samples, n_hidden]
                sigmoid_derivative_h = a_h * (1. - a_h)

                # [n_samples, n_classlabels] dot [n_classlabels, n_hidden]
                # -> [n_samples, n_hidden]
                sigma_h = (np.dot(sigma_out, self.w_out.T) *
                           sigmoid_derivative_h)

                # [n_features, n_samples] dot [n_samples, n_hidden]
                # -> [n_features, n_hidden]
                grad_w_h = np.dot(X_train[batch_idx].T, sigma_h)
                grad_b_h = np.sum(sigma_h, axis=0)

                # [n_hidden, n_samples] dot [n_samples, n_classlabels]
                # -> [n_hidden, n_classlabels]
                grad_w_out = np.dot(a_h.T, sigma_out)
                grad_b_out = np.sum(sigma_out, axis=0)

                # Regularization and weight updates
                delta_w_h = (grad_w_h + self.l2*self.w_h)
                delta_b_h = grad_b_h # bias is not regularized
                self.w_h -= self.eta * delta_w_h
                self.b_h -= self.eta * delta_b_h

                delta_w_out = (grad_w_out + self.l2*self.w_out)
                delta_b_out = grad_b_out  # bias is not regularized
                self.w_out -= self.eta * delta_w_out
                self.b_out -= self.eta * delta_b_out

            #############
            # Evaluation
            #############

            # Evaluation after each epoch during training
            z_h, a_h, z_out, a_out = self._forward(X_train)
            
            cost = self._compute_cost(y_enc=y_train_enc,
                                      output=a_out)

            y_train_pred = self.predict(X_train)
            y_valid_pred = self.predict(X_valid)

            train_acc = ((np.sum(y_train == y_train_pred)).astype(np.float) /
                         X_train.shape[0])
            valid_acc = ((np.sum(y_valid == y_valid_pred)).astype(np.float) /
                         X_valid.shape[0])

            sys.stderr.write('\r%0*d/%d | Cost: %.2f '
                             '| Train/Valid Acc.: %.2f%%/%.2f%% ' %
                             (epoch_strlen, i+1, self.epochs, cost,
                              train_acc*100, valid_acc*100))
            sys.stderr.flush()

            self.eval_['cost'].append(cost)
            self.eval_['train_acc'].append(train_acc)
            self.eval_['valid_acc'].append(valid_acc)

        return self
