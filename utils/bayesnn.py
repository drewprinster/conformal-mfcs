import logging

import autograd
import autograd.numpy as np


ln2pi = np.log(2.0 * np.pi)


class MLP:
    def __init__(self, n_inputs, n_hidden):
        self.n_inputs = n_inputs
        self.n_hidden = n_hidden
        self.n_outputs = 1
        self.layer_sizes = [self.n_inputs, self.n_hidden, self.n_outputs]

    def init_params(self, rng):
        W1, b1 = init_layer(self.n_inputs, self.n_hidden, rng)
        W2, b2 = init_layer(self.n_hidden, self.n_outputs, rng)
        params = serialize_layers([(W1, b1), (W2, b2)])
        return params

    def layers(self, params):
        return deserialize_layers(self.layer_sizes, params)

    def predict(self, params, inputs):
        layers = self.layers(params)
        W1, b1 = layers[0]
        W2, b2 = layers[1]

        z1 = np.dot(inputs, W1) + b1
        a1 = np.tanh(z1)
        z2 = np.dot(a1, W2) + b2

        m = z2[:, 0]

        return m


def serialize_layers(layers):
    params = []
    for W, b in layers:
        params.append(W.ravel())
        params.append(b)

    return np.concatenate(params)


def deserialize_layers(layer_sizes, params):
    i_sizes = layer_sizes[:-1]
    o_sizes = layer_sizes[1:]

    layers = []
    stream = params
    for n_i, n_o in zip(i_sizes, o_sizes):
        W, stream = deserialize_array(stream, (n_i, n_o))
        b, stream = deserialize_array(stream, (n_o,))
        layers.append((W, b))

    return layers


def deserialize_array(stream, shape):
    n = np.prod(shape)
    a = stream[:n].reshape(shape)
    return a, stream[n:]


def init_layer(n_in, n_out, rng):
    W = np.sqrt(1.0 / n_in) * rng.normal(size=(n_in, n_out))
    b = np.zeros(n_out)
    return W, b


def make_objective(model, alphas, beta, n_train, weights):
    ## Note that alphas control L2 regularization / Gaussian prior: lambda = alpha^2
    s_w1 = 1.0 / alphas[0]
    s_w2 = 1.0 / alphas[1]
    s_y = 1.0 / beta

    def objective(params, inputs, targets, weights):
        batch_size = len(inputs)
        weight = n_train / batch_size
        lnl = likelihood(params, inputs, targets, weights)
        lnp = prior(params)
#         print("-(weight * np.sum(lnl) + lnp)", -(weight * np.sum(lnl) + lnp))
        return -(weight * np.sum(lnl) + lnp)

    def likelihood(params, inputs, targets, weights):
        m = model.predict(params, inputs)
#         print("len(targets)", len(targets))
#         print("len(m)", len(m))
#         print("len(norm_logpdf(targets, m, s_y)) :", len(norm_logpdf(targets, m, s_y)))
#         print("len(weights)", len(weights))
#         print("norm_logpdf(targets, m, s_y)*weights : ", norm_logpdf(targets, m, s_y)*weights)
#         print("norm_logpdf(targets, m, s_y)", norm_logpdf(targets, m, s_y))
        return norm_logpdf(targets, m, s_y)*weights ## Drew added weights here

    def prior(params):
        layers = model.layers(params)

        W1 = layers[0][0]
        m1 = W1.size
        lnp1 = norm_logpdf(W1, 0.0, s_w1)

        W2 = layers[1][0]
        m2 = W2.size
        lnp2 = norm_logpdf(W2, 0.0, s_w2)

        return np.sum(lnp1) + np.sum(lnp2)
    
    def likelihood_all(params, inputs, targets, weights):
        return np.sum(likelihood(params, inputs, targets, weights)) ## + 0.5*np.linalg.norm(params) ## Adding weight decay

    return objective, likelihood, prior, likelihood_all


def train(obj, params, inputs, targets, config, weights, epoch_callback=None):
    rng = np.random.RandomState(config['seed'])

    params = np.array(params)
    n_train = len(inputs)
    grad = autograd.grad(obj)
    
    for i in range(0, config['n_epochs']):
        
        p = rng.permutation(n_train)
        X = inputs[p]
        y = targets[p]
        weights_p = weights[p]

        for j in range(0, n_train, config['batch_size']):
#             d = config['decay'] ** (i // config['decay_period']) ## Does this line do anything?
            k = j + config['batch_size']
            g = grad(params, X[j:k], y[j:k], weights_p[j:k])
            params -= config['learning_rate'] * g

        msg = 'epoch={:04d}, objective={:.04f}, grad_norm={:.02f}'
        f = obj(params, X, y, weights_p)
        n = np.sqrt(np.sum(grad(params, X, y, weights_p)**2))
        msg = msg.format(i + 1, f, n)
#         logging.info(msg) ## Removed this to not have verbose training
        
#         if (n < 0.5):
#             break

        if epoch_callback:
            epoch_callback(params)

    return params


def init_sgd_config():
    return {
        'n_epochs': 100,
        'batch_size': 10,
        'learning_rate': 1e-4,
        'decay': 1.0,
        'decay_period': 10,
        'seed': 0
    }


def norm_logpdf(y, m, s):
    return -0.5 * (ln2pi + 2.0*np.log(s) + ((y - m) / s)**2)
