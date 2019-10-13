import random
import numpy as np
from util import is_x_better_than_y, BatchGenerator
from .model import Model
import tensorflow as tf
import pandas as pd
import pickle
import os

class SimpleModel(Model):

    def __init__(self,
                 num_params_per_design,
                 spec_kwrd_list,
                 logger,
                 compare_nn_hidden_dim_list,
                 feat_ext_hidden_dim_list,
                 learning_rate=None,
                 **kwargs,
                 ):
        self.num_params_per_design = num_params_per_design
        self.spec_kwrd_list = spec_kwrd_list
        self.feat_ext_dim_list = [num_params_per_design] + feat_ext_hidden_dim_list
        self.compare_nn_dim_list = \
            [2*feat_ext_hidden_dim_list[-1]] + compare_nn_hidden_dim_list + [2]
        self.lr = learning_rate
        self.logger = logger

        self.evaluate_flag = True if 'eval_save_to_path' in kwargs.keys() else False
        if self.evaluate_flag:
            self.file_base_name = 'acc'
            self._initialize_evaluation(**kwargs)

    def _initialize_evaluation(self, **kwargs):
        self.eval_save_to = kwargs['eval_save_to_path']
        if self.eval_save_to == 'log_path':
            self.eval_save_to = self.logger.log_path
        os.makedirs(self.eval_save_to, exist_ok=True)

        self.df_accuracy = pd.DataFrame()
        oracle_db_loc = kwargs['oracle_db_loc']
        with open(oracle_db_loc, 'rb') as f:
            oracle_data = pickle.load(f)

        self.df = pd.DataFrame.from_dict(oracle_data)
        keys = oracle_data['inputs1'][0].specs.keys()

        # creates a vector for all designs indicating whether input1 is better with respect to
        # critical designs
        self.oracle_is_1_better = []
        for index, row in self.df.iterrows():
            is_1_better = all(row[row['critical_specs']])
            self.oracle_is_1_better.append(is_1_better)

        self.oracle_input1 = np.array(self.df["inputs1"].tolist())
        self.oracle_input2 = np.array(self.df["inputs2"].tolist())

        # for true labels we should provide one hot encoded versions, so:
        # 1. get the colomn df.as_matrix(columns=[kwrd] as matrix and flatten it.
        # 2. multiply it by 1 to get all 1s and 0s.
        # 3. use it as the indices and create one hot encoded vector
        self.labels = dict()
        for kwrd in keys:
            labels = np.zeros((len(self.df), 2))
            col_num = (self.df.as_matrix(columns=[kwrd]).flatten())*1
            labels[np.arange(len(self.df)), col_num] = 1
            self.labels[kwrd] = labels

        self.acc_txt_file = os.path.join(self.eval_save_to, self.file_base_name + ".txt")
        if os.path.exists(self.acc_txt_file):
            os.remove(self.acc_txt_file)

    def _init_tf_sess(self):
        self.saver = tf.train.Saver()
        tf_config = tf.ConfigProto(inter_op_parallelism_threads=1, intra_op_parallelism_threads=1)
        tf_config.gpu_options.allow_growth = True  # may need if using GPU
        self.sess = tf.Session(config=tf_config)
        self.sess.__enter__()  # equivalent to `with self.sess:`
        tf.global_variables_initializer().run()  # pylint: disable=E1101

    def _define_placeholders(self):
        self.input1 = tf.placeholder(tf.float32, shape=[None, self.num_params_per_design],
                                     name='in1')
        self.input2 = tf.placeholder(tf.float32, shape=[None, self.num_params_per_design],
                                     name='in2')
        self.true_labels = {}

        for kwrd in self.spec_kwrd_list:
            self.true_labels[kwrd] = tf.placeholder(tf.float32, shape=[None, 2],
                                                    name='labels_' + kwrd)

    def _normalize(self):
        with tf.variable_scope('normalizer'):
            self.mu = tf.Variable(tf.zeros([self.num_params_per_design], dtype=tf.float32),
                                  name='mu', trainable=False)
            self.std = tf.Variable(tf.zeros([self.num_params_per_design], dtype=tf.float32),
                                   name='std', trainable=False)
            input1_norm = (self.input1 - self.mu) / (self.std + tf.constant(1e-6))
            input2_norm = (self.input2 - self.mu) / (self.std + tf.constant(1e-6))

        return input1_norm, input2_norm

    def _feature_extraction_model(self, input_data, name='feature_model', reuse=False):
        layer = input_data
        with tf.variable_scope(name):
            for i, layer_dim in enumerate(self.feat_ext_dim_list[1:]):
                layer = tf.layers.dense(layer, layer_dim, activation=tf.nn.relu,
                                        reuse=reuse, name='feat_fc'+str(i))

        return layer

    def _sym_fc_layer(self, input_data, layer_dim, activation_fn=None, reuse=False, scope='sym_fc'):
        assert input_data.shape[1]%2==0

        with tf.variable_scope(scope):
            weight_elements = tf.get_variable(name='W', shape=[input_data.shape[1]//2, layer_dim],
                                              initializer=tf.random_normal_initializer)
            bias_elements = tf.get_variable(name='b', shape=[layer_dim//2],
                                            initializer=tf.zeros_initializer)

            Weight = tf.concat([weight_elements, weight_elements[::-1, ::-1]],
                               axis=0, name='Weights')
            Bias = tf.concat([bias_elements, bias_elements[::-1]], axis=0, name='Bias')

            out = tf.add(tf.matmul(input_data, Weight), Bias)
            if activation_fn == None:
                pass
            elif activation_fn == 'Relu':
                out = tf.nn.relu(out)
            elif activation_fn == 'tanh':
                out = tf.nn.tanh(out)
            else:
                print('activation does not exist')

            return out, Weight, Bias

    def _comparison_model(self, input_data, name='compare_model', reuse=False):
        layer = input_data
        w_list, b_list = [], []
        with tf.variable_scope(name):
            for i, layer_dim in enumerate(self.compare_nn_dim_list[1:-1]):
                layer, w, b = self._sym_fc_layer(layer, layer_dim, activation_fn='Relu',
                                                 reuse=reuse, scope=name+str(i))
                w_list.append(w)
                b_list.append(b)

            logits, w, b = self._sym_fc_layer(layer, 2, reuse=reuse, scope='fc_out')
            w_list.append(w)
            b_list.append(b)

        return logits, w_list, b_list

    def _build_policy(self):

        input1_norm, input2_norm = self._normalize()
        features1 = self._feature_extraction_model(input1_norm, name='feat_model', reuse=False)
        features2 = self._feature_extraction_model(input2_norm, name='feat_model', reuse=True)
        input_features = tf.concat([features1, features2[:, ::-1]], axis=1)

        self.out_logits, w_list, b_list = {}, {}, {}
        for kwrd in self.spec_kwrd_list:
            self.out_logits[kwrd], w_list[kwrd], b_list[kwrd] = \
                self._comparison_model(input_features, name='cmp_model_' + kwrd, reuse=False)

    def _build_loss(self):
        self.total_loss = 0
        neg_likelihoods, self.loss = {}, {}
        self.out_predictions = {}
        for kwrd in self.spec_kwrd_list:
            with tf.variable_scope("loss_"+kwrd):
                self.out_predictions[kwrd] = tf.nn.softmax(self.out_logits[kwrd])
                neg_likelihoods[kwrd] = \
                    tf.nn.softmax_cross_entropy_with_logits_v2(labels=self.true_labels[kwrd],
                                                               logits=self.out_logits[kwrd])
                self.loss[kwrd] = tf.reduce_mean(neg_likelihoods[kwrd])
                self.total_loss += self.loss[kwrd]

    def _build_accuracy(self):
        self.accuracy = {}
        for kwrd in self.spec_kwrd_list:
            correct_predictions = tf.equal(tf.argmax(self.out_predictions[kwrd], axis=1),
                                           tf.argmax(self.true_labels[kwrd], axis=1))
            self.accuracy[kwrd] = tf.reduce_mean(tf.cast(correct_predictions, tf.float32))

    def _build_computation_graph(self):

        self._define_placeholders()
        self._build_policy()
        self._build_loss()
        self._build_accuracy()

        if self.lr:
            self.update_op = tf.train.AdamOptimizer(self.lr).minimize(self.total_loss)
        else:
            self.update_op = tf.train.AdamOptimizer().minimize(self.total_loss)

    def init(self):
        self._build_computation_graph()
        self._init_tf_sess()

    def get_train_valid_ds(self, db, k_top, eval_core, validation_frac):

        assert k_top <= len(db), "ktop={} should be smaller than " \
                                 "train_set_len={}".format(k_top, len(db))

        db_cost_sorted = sorted(db, key=lambda x: x.cost)[:k_top]

        category = {}
        nn_input1, nn_input2 = [], []
        nn_labels = {}
        for kwrd in self.spec_kwrd_list:
            category[kwrd], nn_labels[kwrd] = [], []

        n = len(db_cost_sorted)
        for i in range(n-1):
            for j in range(i+1, n):
                rnd = random.random()
                if rnd < 0.5:
                    nn_input1.append(db_cost_sorted[i])
                    nn_input2.append(db_cost_sorted[j])
                    for kwrd in self.spec_kwrd_list:
                        label = 1 if is_x_better_than_y(eval_core=eval_core,
                                                        x=db_cost_sorted[i].specs[kwrd],
                                                        y=db_cost_sorted[j].specs[kwrd],
                                                        kwrd=kwrd) else 0
                        category[kwrd].append(label)
                else:
                    nn_input1.append(db_cost_sorted[j])
                    nn_input2.append(db_cost_sorted[i])
                    for kwrd in self.spec_kwrd_list:
                        label = 0 if is_x_better_than_y(eval_core=eval_core,
                                                        x=db_cost_sorted[i].specs[kwrd],
                                                        y=db_cost_sorted[j].specs[kwrd],
                                                        kwrd=kwrd) else 1
                        category[kwrd].append(label)

        for kwrd in self.spec_kwrd_list:
            nn_labels[kwrd] = np.zeros((len(category[kwrd]), 2))
            nn_labels[kwrd][np.arange(len(category[kwrd])), category[kwrd]] = 1

        nn_input1 = np.array(nn_input1)
        nn_input2 = np.array(nn_input2)

        self.logger.log_text("[INFO] dataset size:%d" % len(db))
        self.logger.log_text("[INFO] ktop: %d" % k_top)
        self.logger.log_text("[INFO] combine size:%d" % len(nn_input1))

        permutation = np.random.permutation(len(nn_input1))
        nn_input1 = nn_input1[permutation]
        nn_input2 = nn_input2[permutation]
        for kwrd in self.spec_kwrd_list:
            nn_labels[kwrd] = nn_labels[kwrd][permutation]

        boundry_index = len(nn_input1) - int(len(nn_input1)*validation_frac)

        train_input1 = nn_input1[:boundry_index]
        train_input2 = nn_input2[:boundry_index]
        valid_input1 = nn_input1[boundry_index:]
        valid_input2 = nn_input2[boundry_index:]

        train_labels, valid_labels = {}, {}
        for kwrd in self.spec_kwrd_list:
            train_labels[kwrd] = nn_labels[kwrd][:boundry_index]
            valid_labels[kwrd] = nn_labels[kwrd][boundry_index:]

        ds = {
            'training_ds': dict(
                train_input1=train_input1,
                train_input2=train_input2,
                train_labels=train_labels,
            ),
            'validation_ds': dict(
                valid_input1=valid_input1,
                valid_input2=valid_input2,
                valid_labels=valid_labels,
            )
        }
        return ds

    def train(self, data_set, batch_size, num_epochs, ckpt_step, log_step):

        train_input1 = data_set['training_ds']['train_input1']
        train_input2 = data_set['training_ds']['train_input2']
        train_labels = data_set['training_ds']['train_labels']
        valid_input1 = data_set['validation_ds']['valid_input1']
        valid_input2 = data_set['validation_ds']['valid_input2']
        valid_labels = data_set['validation_ds']['valid_labels']

        train_mean = np.mean(np.concatenate([train_input1, train_input2], axis=0), axis=0)
        train_std = np.std(np.concatenate([train_input1, train_input2], axis=0), axis=0)

        batch_generator = BatchGenerator(len(train_input1), batch_size)
        total_n_batches = int(len(train_input1) // batch_size)

        self.logger.log_text("[info] training the model with dataset ....")
        self.logger.log_text("[info] number of total batches: %d" % total_n_batches)
        self.logger.log_text(30*"-")

        self.mu.assign(train_mean).op.run()
        self.std.assign(train_std).op.run()

        for epoch in range(num_epochs+1):
            avg_loss, avg_train_acc, avg_valid_acc = {}, {}, {}
            avg_total_loss = 0.

            for kwrd in self.spec_kwrd_list:
                avg_loss[kwrd] = 0.
                avg_train_acc[kwrd] = 0.
                avg_valid_acc[kwrd] = 0.

            for iter in range(total_n_batches):
                index = batch_generator.next()
                batch_input1, batch_input2 = train_input1[index], train_input2[index]
                feed_dict = {self.input1         :batch_input1,
                             self.input2         :batch_input2}
                for kwrd in self.spec_kwrd_list:
                    feed_dict[self.true_labels[kwrd]] = train_labels[kwrd][index]

                _, l, t_l, train_acc = self.sess.run([self.update_op, self.loss,
                                                      self.total_loss, self.accuracy],
                                                     feed_dict=feed_dict)

                feed_dict = {self.input1         :valid_input1,
                             self.input2         :valid_input2}

                for kwrd in self.spec_kwrd_list:
                    feed_dict[self.true_labels[kwrd]] = valid_labels[kwrd]

                valid_acc, = self.sess.run([self.accuracy], feed_dict=feed_dict)

                for kwrd in self.spec_kwrd_list:
                    avg_total_loss += t_l / total_n_batches
                    avg_loss[kwrd] += l[kwrd] / total_n_batches
                    avg_train_acc[kwrd] += train_acc[kwrd] / total_n_batches
                    avg_valid_acc[kwrd] += valid_acc[kwrd] / total_n_batches

            if epoch % ckpt_step == 0:
                self.logger.store_model(self.saver, self.sess)
            if epoch % log_step == 0:
                self.logger.log_text(10*"-")
                self.logger.log_text("[epoch %d] total_loss: %f " %(epoch, avg_total_loss))
                for kwrd in self.spec_kwrd_list:
                    self.logger.log_text("".format(kwrd))
                    self.logger.log_text("[%s] loss: %f" %(kwrd, avg_loss[kwrd]))
                    self.logger.log_text("[%s] train_acc = %.2f%%, "
                                         "valid_acc = %.2f%%" %(kwrd,
                                                                avg_train_acc[kwrd]*100,
                                                                avg_valid_acc[kwrd]*100))
        if self.evaluate_flag:
            self.evaluate()


    def query(self, input1, input2):

        nn_input1 = np.array(input1)
        nn_input2 = np.array(input2)

        if nn_input1.ndim == 1:
            nn_input1 = nn_input1[None, :]
        if nn_input2.ndim == 1:
            nn_input2 = nn_input2[None, :]

        feed_dict = {
            self.input1: nn_input1,
            self.input2: nn_input2,
        }

        prediction = self.sess.run(self.out_predictions, feed_dict=feed_dict)
        return prediction

    def evaluate(self):
        "A function that evaluates the nn with oracle data to see how they compare"
        assert self.evaluate_flag, 'To evaluate the evalute flage must be set to True'

        oracle_feed_dict = {
            self.input1: self.oracle_input1,
            self.input2: self.oracle_input2,
        }

        for kwrd, tensor in self.true_labels.items():
            oracle_feed_dict[tensor] = self.labels[kwrd]

        accuracy, predictions = self.sess.run([self.accuracy, self.out_predictions],
                                              feed_dict=oracle_feed_dict)

        # see if nn says input1 is better than input2 for all rows according to the critical specs
        nn_is_1_better = []
        for i in range(len(self.df)):
            is_1_better = all([random.random() > predictions[kwrd][i][0]
                               for kwrd in self.df['critical_specs'][i]])
            nn_is_1_better.append(is_1_better)


        # compute all accuracy numbers (oracle_nn): false_false, true_true, false_true, true_false
        ff, tt = 0, 0
        ft, tf = 0, 0
        for nn_vote, oracle_vote in zip(nn_is_1_better, self.oracle_is_1_better):
            if not nn_vote and not oracle_vote:
                ff+=1
            elif nn_vote and oracle_vote:
                tt+=1
            elif not nn_vote and oracle_vote:
                tf+=1
            elif nn_vote and not oracle_vote:
                ft+=1
        total_accuracy = (tt+ff)/(tt+ff+tf+ft)
        # how many of those that oracle says are good nn says are good: very important, should be 1
        a1 = tt/(tf+tt)
        # how many of those that nn says good are actually good: very important, should be 1,
        a2 = tt/(ft+tt)
        # indicates that nn doesn't add useless data
        # how many of those that oracle says are bad nn says are bad: should be 1, indicates that
        a3 = ff/(ff+ft)
        #  nn can prune out the space efficiently
        # how many of those that nn says bad are actually bad: should be 1
        a4 = ff/(tf+ff)

        accuracy["total_acc"] = total_accuracy
        accuracy["a1"] = a1
        accuracy["a2"] = a2
        accuracy["a3"] = a3
        accuracy["a4"] = a4
        accuracy["tt"] = tt
        accuracy["ff"] = ff
        accuracy["tf"] = tf
        accuracy["ft"] = ft
        self.df_accuracy = self.df_accuracy.append(accuracy, ignore_index=True)

        self.logger.store_db(self.df_accuracy, fpath=os.path.join(self.eval_save_to,
                                                                  self.file_base_name + '.pkl'))
        self.logger.log_text(accuracy, stream_to_stdout=False, fpath=self.acc_txt_file)
