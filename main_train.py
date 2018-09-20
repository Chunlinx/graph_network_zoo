import tensorflow as tf
import numpy as np
import time
from model_mlp import MLP
from model_gnn_1step import GNN as GNN1step
from model_gnn_2step import GNN as GNN2step
from util.babi_dataset import *
from util.common_function import *

seed = 123
np.random.seed(seed)
tf.set_random_seed(seed)
flags = tf.app.flags
FLAGS = flags.FLAGS

# cora, citeseer, pubmed --> dense
# you can see that gnn1step can not solve babi-15, and gnn2step can solve babi-15
# baba-4 --> gnn1step, gnn2step
# babi-15 --> gnn2step

# cora: 2708node，1433feature

# cora, citeseer, pubmed, babi-4 ,babi-15
flags.DEFINE_string('dataset', 'babi-4', 'Dataset string.')

flags.DEFINE_string('model', 'gnn-1step', 'Model string.')  # gcn, gcn_cheby, dense，gnn-1step, gnn-2step
flags.DEFINE_float('learning_rate', 0.01, 'Initial learning rate.')
flags.DEFINE_integer('epochs', 200, 'Number of epochs to train.')
flags.DEFINE_float('dropout', 0.5, 'Dropout rate (1 - keep probability).')
flags.DEFINE_float('weight_decay', 5e-4, 'Weight for L2 loss on embedding matrix.')
flags.DEFINE_integer('early_stopping', 10, 'Tolerance for early stopping (# of epochs).')
flags.DEFINE_integer('max_degree', 3, 'Maximum Chebyshev polynomial degree.')
flags.DEFINE_integer('batch_size', 10, 'Number of epochs to train.')

if str.split(FLAGS.dataset,"-")[0]=="babi":
    dataroot = 'data/babi_data/processed_1/train/%d_graphs.txt' % int(str.split(FLAGS.dataset,"-")[-1])
    dataroot_test = 'data/babi_data/processed_1/test/%d_graphs.txt' % int(str.split(FLAGS.dataset,"-")[-1])
    question_type = 0 # 只能回答一类这类question_id的问题
    train_dataset = bAbIDataset(dataroot, task_id=question_type, is_train=True, batch_size=FLAGS.batch_size)
    test_dataset = bAbIDataset(dataroot_test, task_id=question_type, is_train=False, batch_size=FLAGS.batch_size)

    model_type = str.split(FLAGS.model,"-")[0]
    model_sub_type = str.split(FLAGS.model,"-")[1]
    if model_type == "gnn":
        if model_sub_type=="1step":
            model_func = GNN1step
        if model_sub_type=="2step":
            model_func = GNN2step
        state_dim = 4
        annotation_dim = 1
        edge_type_num = train_dataset.n_edge_types
        placeholders = {
            'target': tf.placeholder(tf.int32, shape=(None)),
            'init_input': tf.placeholder(tf.float32, shape=(None, train_dataset.node_num, state_dim)),
            'adj_matrix': tf.placeholder(tf.float32,
                                         shape=(None, train_dataset.node_num, train_dataset.node_num * edge_type_num * 2)),
            'annotation': tf.placeholder(tf.float32, shape=(None, train_dataset.node_num, annotation_dim)),
        }

        def construct_feed_dict(init_input, adj_matrix, annotation, target, placeholders):
            feed_dict = dict()
            feed_dict.update({placeholders['init_input']: init_input})
            feed_dict.update({placeholders['adj_matrix']: adj_matrix})
            feed_dict.update({placeholders['annotation']: annotation})
            feed_dict.update({placeholders['target']: target})
            return feed_dict

    model = model_func(placeholders, batch_size=FLAGS.batch_size, state_dim=state_dim, node_num=train_dataset.node_num, edge_type_num=edge_type_num)

else:

    adj, features, y_train, y_val, y_test, train_mask, val_mask, test_mask = load_data(FLAGS.dataset)

    features = preprocess_features(features) #debug看是49216个(node_id,feature_id)=value,实际是[2708,1433]

    if FLAGS.model == 'dense':
        support = [preprocess_adj(adj)]  # Not used in dense model
        num_supports = 1
        model_func = MLP
        placeholders = {
            'support': [tf.sparse_placeholder(tf.float32) for _ in range(num_supports)],
            'features': tf.sparse_placeholder(tf.float32, shape=tf.constant(features[2], dtype=tf.int64)),
            'labels': tf.placeholder(tf.float32, shape=(None, y_train.shape[1])),
            'labels_mask': tf.placeholder(tf.int32),
            'dropout': tf.placeholder_with_default(0., shape=()),
            'num_features_nonzero': tf.placeholder(tf.int32)  # helper variable for sparse dropout
        }
        def construct_feed_dict(features, support, labels, labels_mask, placeholders):
            feed_dict = dict()
            feed_dict.update({placeholders['labels']: labels})
            feed_dict.update({placeholders['labels_mask']: labels_mask})
            feed_dict.update({placeholders['features']: features})
            feed_dict.update({placeholders['support'][i]: support[i] for i in range(len(support))})
            feed_dict.update({placeholders['num_features_nonzero']: features[1].shape})
            return feed_dict


    model = model_func(placeholders, input_dim=features[2][1], logging=True)

model.forward()
sess = tf.Session()

sess.run(tf.global_variables_initializer())

cost_val = []


# Train model
for epoch in range(FLAGS.epochs):

    t = time.time()
    model_type = str.split(FLAGS.model, "-")[0]
    if FLAGS.model == 'dense':
        feed_dict = construct_feed_dict(features, support, y_train, train_mask, placeholders)
        feed_dict.update({placeholders['dropout']: 0.7})
        outs = sess.run([model.opt_op, model.loss, model.accuracy], feed_dict=feed_dict)
        cost, acc, duration = model.evaluate(sess, features, support, y_val, val_mask, placeholders,
                                             construct_feed_dict)
        cost_val.append(cost)
    elif model_type == 'gnn':
        for i in range(train_dataset.num_batch):
            cur_batch = train_dataset.next_batch()
            feed_dict = construct_feed_dict(cur_batch.init_input_batch, cur_batch.adjacency_matrix_batch, cur_batch.annotation_batch, cur_batch.target_batch, placeholders)

            outs = sess.run([model.opt_op, model.loss, model.accuracy], feed_dict=feed_dict)

        cost, acc, duration = model.evaluate(sess, cur_batch, placeholders,
                                             construct_feed_dict)
        cost_val.append(cost)


    # Print results
    print("Epoch:", '%04d' % (epoch + 1), "train_loss=", "{:.5f}".format(outs[1]),
          "train_acc=", "{:.5f}".format(outs[2]), "val_loss=", "{:.5f}".format(cost),
          "val_acc=", "{:.5f}".format(acc), "time=", "{:.5f}".format(time.time() - t))

    if epoch > FLAGS.early_stopping and cost_val[-1] > np.mean(cost_val[-(FLAGS.early_stopping+1):-1]):
        print("Early stopping...")
        break


print("Optimization Finished!")
model_type = str.split(FLAGS.model, "-")[0]
if FLAGS.model == 'dense':
  # Testing
    test_cost, test_acc, test_duration = model.evaluate(sess, features, support, y_test, test_mask, placeholders,
                                                        construct_feed_dict)
    print("Test set results:", "cost=", "{:.5f}".format(test_cost),
      "accuracy=", "{:.5f}".format(test_acc), "time=", "{:.5f}".format(test_duration))
elif model_type == 'gnn':
    for i in range(test_dataset.num_batch):
        cur_batch = test_dataset.next_batch()
        test_cost, test_acc, test_duration = model.evaluate(sess, cur_batch, placeholders,
                                         construct_feed_dict)
        print("Test set results:", "cost=", "{:.5f}".format(test_cost),
          "accuracy=", "{:.5f}".format(test_acc), "time=", "{:.5f}".format(test_duration))
