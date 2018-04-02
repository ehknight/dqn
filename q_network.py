import keras
import tensorflow as tf
import numpy as np

Dense = tf.layers.Dense

class QNetwork(object):
    def __init__(self, inps, outs, hidden_sizes, str_optim,
            discount_factor=1.0, activation=tf.nn.tanh, lr=0.001, damping=0.01,
            cov_decay=0.95):
        self.sess = tf.Session() # create new tf session
        self.discount_factor = discount_factor
        self.lr = lr
        self.damping = damping
        self.cov_decay = cov_decay

        self.state_var = tf.placeholder('float32', [None, inps])
        previous_layer_out = self.state_var
        layer_sizes = hidden_sizes + [outs]

        for i, layer_size in enumerate(layer_sizes):
            if i+1 == len(layer_sizes): # last layer
                layer_activation = None
            else:
                layer_activation = activation

            layer = Dense(units=layer_size, activation=None, use_bias=True,
                    kernel_initializer=tf.random_normal_initializer())
            preactivated = layer(previous_layer_out)
            if layer_activation is None:
                print 'NO ACTIVATION'
                activated = preactivated
            else:
                print 'YES ACTIVATION'
                activated = layer_activation(preactivated)
            to_add = {'ws': layer.kernel, 'bs':layer.bias,
                    'prev_outs': previous_layer_out, 'preacs': preactivated}
            print to_add
            network_params.append(to_add)
            previous_layer_out = activated

        self.q_pred_var = activated
        self.q_true_var = tf.placeholder('float32', [None, outs])
        self.loss = tf.losses.mean_squared_error(self.q_pred_var, self.q_true_var)

        if str_optim == 'adam':
            optimizer = tf.train.AdamOptimizer(self.lr)

        else:
            raise NotImplementedError("Unknown optimizer")

        self.train_fn = optimizer.minimize(self.loss)
        self.sess.run(tf.initialize_all_variables())

    def get_action_values(self, state):
        if len(state.shape) == 1:
            state = state.reshape(1, -1)
        return self.sess.run(self.q_pred_var, feed_dict={self.state_var: state})

    def fit_transitions(self, transitions):
        minibatch = [] # will have state-value tuples
        for transition in transitions:
            if transition.done:
                target_q_val = transition.reward
            else:
                est_future_reward = self.get_action_values(transition.state_next).reshape(-1)
                max_future_reward = max(est_future_reward)
                target_q_val = transition.reward + self.discount_factor * max_future_reward

            target_action_values = self.get_action_values(transition.state)
            target_action_values[0,transition.action] = target_q_val

            minibatch.append((transition.state, target_action_values))

        states, true_action_values = zip(*minibatch) # transpose minibatch
        states, true_action_values = map(np.asarray, [states, true_action_values])
        true_action_values = true_action_values.reshape(states.shape[0], -1)
        _, loss = self.sess.run([self.train_fn, self.loss],
                feed_dict={self.state_var: states,
                self.q_true_var: true_action_values})
        return loss
