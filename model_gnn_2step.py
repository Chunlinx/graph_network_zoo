from model_base import ModelBase
import time
from util.common_function import *
flags = tf.app.flags
FLAGS = flags.FLAGS
class GNN(ModelBase):
    def __init__(self, placeholders, batch_size=10, state_dim=4, node_num=8, edge_type_num=2, **kwargs):
        super(GNN, self).__init__(**kwargs)
        self.edge_type_num = edge_type_num
        self.node_embeddings1 = []
        self.node_embeddings2 = []
        for i in range(self.edge_type_num):
            self.node_embeddings1.append(tf.keras.layers.Dense(state_dim, activation=None, use_bias=False))
            self.node_embeddings2.append(tf.keras.layers.Dense(state_dim, activation=None, use_bias=False))

        self.placeholders = placeholders
        self.node_num = node_num
        self.state_dim = state_dim
        self.merge_mlp =tf.keras.layers.Dense(state_dim, activation="tanh")

        self.out_mlp =  tf.keras.Sequential([
            tf.keras.layers.Dense(state_dim, activation="tanh"),
            tf.keras.layers.Dense(1, activation=None),
        ])
        self.optimizer = tf.train.AdamOptimizer(learning_rate=FLAGS.learning_rate)

    def forward(self):
        A_in = self.placeholders["adj_matrix"][:, :, :self.node_num * self.edge_type_num]
        A_out = self.placeholders["adj_matrix"][:, :, self.node_num * self.edge_type_num:]

        node_states1=[]
        node_states2=[]
        for i in range(self.edge_type_num):
            node_states1.append(self.node_embeddings1[i](self.placeholders["init_input"]))
            node_states2.append(self.node_embeddings2[i](self.placeholders["init_input"]))
        node_states1=tf.reshape(tf.transpose(tf.stack(node_states1),[1,0,2,3]),[tf.shape(A_in)[0],-1,self.state_dim])
        node_states2=tf.reshape(tf.transpose(tf.stack(node_states2),[1,0,2,3]),[tf.shape(A_in)[0],-1,self.state_dim])
        output1 = tf.matmul(A_in, node_states1)
        output2 = tf.matmul(A_out, node_states2)
        output_state = self.merge_mlp(tf.concat([self.placeholders["init_input"],output1,output2],axis=-1))

        #second step:
        node_states1 = []
        node_states2 = []
        for i in range(self.edge_type_num):
            node_states1.append(self.node_embeddings1[i](output_state))
            node_states2.append(self.node_embeddings2[i](output_state))
        node_states1 = tf.reshape(tf.transpose(tf.stack(node_states1), [1, 0, 2, 3]),
                                  [tf.shape(A_in)[0], -1, self.state_dim])
        node_states2 = tf.reshape(tf.transpose(tf.stack(node_states2), [1, 0, 2, 3]),
                                  [tf.shape(A_in)[0], -1, self.state_dim])
        output1 = tf.matmul(A_in, node_states1)
        output2 = tf.matmul(A_out, node_states2)
        output_state = self.merge_mlp(tf.concat([output_state, output1, output2], axis=-1))

        output = self.out_mlp(output_state)
        self.output = tf.reshape(output,[tf.shape(A_in)[0],self.node_num])

        self._loss()
        self._accuracy()
        self.opt_op = self.optimizer.minimize(self.loss)

    def _accuracy(self):
        correct_prediction = tf.equal(tf.cast(tf.argmax(self.output, 1),tf.int32), self.placeholders['target'])
        accuracy_all = tf.cast(correct_prediction, tf.float32)
        self.accuracy = tf.reduce_mean(accuracy_all)

    def _loss(self):
        target = tf.one_hot(self.placeholders['target'],self.node_num)
        self.loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=self.output, labels=target))

    def evaluate(self, sess, cur_batch, placeholders, construct_feed_dict):
        t_test = time.time()
        feed_dict = construct_feed_dict(cur_batch.init_input_batch, cur_batch.adjacency_matrix_batch,
                                        cur_batch.annotation_batch, cur_batch.target_batch, placeholders)
        # feed_dict.update({placeholders['dropout']: 1})
        outs_val = sess.run([self.loss, self.accuracy], feed_dict=feed_dict)
        return outs_val[0], outs_val[1], (time.time() - t_test)