import tensorflow as tf
from utils import *
flags = tf.app.flags
FLAGS = flags.FLAGS


class LayerBase(object):
    """
    # Properties
        name: String, defines the variable scope of the layer.
        logging: Boolean, switches Tensorflow histogram logging on/off

    # Methods
        _call(inputs): Defines computation graph of layer
            (i.e. takes input, returns output)
        __call__(inputs): Wrapper for _call()
        _log_vars(): Log all variables
    """

    def __init__(self, **kwargs):
        allowed_kwargs = {'name', 'logging'}
        for kwarg in kwargs.keys():
            assert kwarg in allowed_kwargs, 'Invalid keyword argument: ' + kwarg
        name = kwargs.get('name')
        if not name:
            layer = self.__class__.__name__.lower()
            name = layer + '_' + str(get_layer_uid(layer))
        self.name = name
        self.vars = {}
        logging = kwargs.get('logging', False)
        self.logging = logging
        self.input_is_sparse = False

    def forward_sub(self, inputs):
        return inputs

    def forward(self, inputs):
        with tf.name_scope(self.name):
            if self.logging and not self.input_is_sparse:
                tf.summary.histogram(self.name + '/inputs', inputs)
            outputs = self.forward_sub(inputs)
            if self.logging:
                tf.summary.histogram(self.name + '/outputs', outputs)
            return outputs

    def _log_vars(self):
        for var in self.vars:
            tf.summary.histogram(self.name + '/vars/' + var, self.vars[var])


class Dense(LayerBase):
    """Dense layer."""
    def __init__(self, input_dim, output_dim, placeholders, dropout=0., input_is_sparse=False,
                 act=tf.nn.relu, bias=False, featureless=False, **kwargs):
        super(Dense, self).__init__(**kwargs)

        if dropout:
            self.dropout = placeholders['dropout']
        else:
            self.dropout = 0.

        self.act = act
        self.input_is_sparse = input_is_sparse
        self.featureless = featureless
        self.bias = bias

        # helper variable for sparse dropout
        self.num_features_nonzero = placeholders['num_features_nonzero']

        with tf.variable_scope(self.name + '_vars'):
            self.vars['weights'] = glorot([input_dim, output_dim],
                                          name='weights')
            if self.bias:
                self.vars['bias'] = zeros([output_dim], name='bias')

        if self.logging:
            self._log_vars()

    def forward_sub(self, inputs):
        x = inputs #debug看是49216个(node_id,feature_id)=value,实际是[2708,1433]

        # dropout
        if self.input_is_sparse:
            x = sparse_dropout(x, 1-self.dropout, self.num_features_nonzero)
        else:
            x = tf.nn.dropout(x, 1-self.dropout)

        # transform
        # layer1 [2708,1433]*[1433*16]->[2708,16]
        # layer2 [2708,16]*[16,7]->[2708,7]
        output = dot(x, self.vars['weights'], sparse=self.input_is_sparse)

        # bias
        if self.bias:
            output += self.vars['bias']

        return self.act(output)



class MultiDense(LayerBase):
    """All dense layer in one."""
    def __init__(self, input_dim, output_dim, placeholders, dropout=0., input_is_sparse=False,
                 act=tf.nn.relu, bias=False, featureless=False, **kwargs):
        super(MultiDense, self).__init__(**kwargs)

        if dropout:
            self.dropout = placeholders['dropout']
        else:
            self.dropout = 0.

        self.act = act
        self.hidden_dim = 64
        self.input_is_sparse = input_is_sparse
        self.featureless = featureless
        self.bias = bias

        # helper variable for sparse dropout
        self.num_features_nonzero = placeholders['num_features_nonzero']

        with tf.variable_scope(self.name + '_vars'):
            self.vars['weights'] = glorot([input_dim, self.hidden_dim],
                                          name='weights')
            self.vars['weights2'] = glorot([self.hidden_dim, output_dim],
                                          name='weights')
            if self.bias:
                self.vars['bias'] = zeros([self.hidden_dim], name='bias')
                self.vars['bias2'] = zeros([output_dim], name='bias')

        if self.logging:
            self._log_vars()

    def forward_sub(self, inputs):
        x = inputs #debug看是49216个(node_id,feature_id)=value,实际是[2708,1433]

        # if self.input_is_sparse:
        #     x = sparse_dropout(x, keep_prob=self.dropout, noise_shape=self.num_features_nonzero)
        # else:
        #     x = tf.nn.dropout(x, keep_prob=self.dropout)

        # transform
        # layer1 [2708,1433]*[1433*16]->[2708,16]
        # layer2 [2708,16]*[16,7]->[2708,7]
        output = dot(x, self.vars['weights'], sparse=self.input_is_sparse)

        # bias
        if self.bias:
            output += self.vars['bias']

        output = tf.nn.dropout(output, keep_prob=self.dropout)
        input2 = self.act(output)
        output2 = tf.matmul(input2, self.vars['weights2'])
        if self.bias:
            output2 += self.vars['bias2']
        return self.act(output2)