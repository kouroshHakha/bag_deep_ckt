from .vannilla import SimpleModel
import tensorflow as tf
import numpy as np
import os.path as osp
import random


class DropOutModel(SimpleModel):

    def __init__(self, *args, **kwargs):
        SimpleModel.__init__(self, *args, **kwargs)
        self.keep_prob = kwargs.get('keep_prob')
        self.n_bayes_samples = kwargs.get('n_bayes_samples', 1)

    def _feature_extraction_model(self, input_data, drop_out_prob=None, name='feature_model',
                                  reuse=False):
        if not drop_out_prob:
            drop_out_prob = tf.constant(self.keep_prob, dtype=tf.float32)
        layer = input_data
        with tf.variable_scope(name):
            for i, layer_dim in enumerate(self.feat_ext_dim_list[1:]):
                layer = tf.layers.dense(layer, layer_dim, activation=tf.nn.relu,
                                        reuse=reuse, name='feat_fc'+str(i))
                layer = tf.nn.dropout(layer, drop_out_prob)
        return layer

    def _comparison_model(self, input_data, drop_out_prob=None, name='compare_model', reuse=False):
        if not drop_out_prob:
            drop_out_prob = tf.constant(self.keep_prob, dtype=tf.float32)
        layer = input_data
        w_list, b_list = [], []
        with tf.variable_scope(name):
            for i, layer_dim in enumerate(self.compare_nn_dim_list[1:-1]):
                layer, w, b = self._sym_fc_layer(layer, layer_dim, activation_fn='Relu',
                                                 reuse=reuse, scope=name+str(i))
                layer = tf.nn.dropout(layer, drop_out_prob)
                w_list.append(w)
                b_list.append(b)

            logits, w, b = self._sym_fc_layer(layer, 2, reuse=reuse, scope='fc_out')
            w_list.append(w)
            b_list.append(b)

        return logits, w_list, b_list

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

        avg_prediction = dict()
        for kwrd in self.spec_kwrd_list:
            avg_prediction[kwrd] = 0

        for _ in range(self.n_bayes_samples):
            prediction_dict = self.sess.run(self.out_predictions, feed_dict=feed_dict)
            for kwrd in self.spec_kwrd_list:
                avg_prediction[kwrd] += prediction_dict[kwrd]/self.n_bayes_samples
        self.logger.log_text('{}'.format(avg_prediction), stream_to_stdout=False,
                             fpath=osp.join(self.logger.log_path, 'avg_prediction.txt'))
        return avg_prediction

    def evaluate(self):
        """
        A function that evaluates the nn with oracle data to see how they compare
        """
        assert self.evaluate_flag, 'To evaluate the evalute flage must be set to True'

        oracle_feed_dict = {
            self.input1: self.oracle_input1,
            self.input2: self.oracle_input2,
        }

        for kwrd, tensor in self.true_labels.items():
            oracle_feed_dict[tensor] = self.labels[kwrd]

        avg_accuracy, avg_predictions = {}, {}
        for kwrd in self.spec_kwrd_list:
            avg_accuracy[kwrd] = 0
            avg_predictions[kwrd] = 0

        for _ in range(self.n_bayes_samples):
            accuracy, predictions = self.sess.run([self.accuracy, self.out_predictions],
                                                  feed_dict=oracle_feed_dict)
            for kwrd in self.spec_kwrd_list:
                avg_predictions[kwrd] += predictions[kwrd]/self.n_bayes_samples
                avg_accuracy[kwrd] += accuracy[kwrd]/self.n_bayes_samples

        # see if nn says input1 is better than input2 for all rows according to the critical specs
        nn_is_1_better = []
        for i in range(len(self.df)):
            is_1_better = all([random.random() > avg_predictions[kwrd][i][0]
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

        avg_accuracy["total_acc"] = total_accuracy
        avg_accuracy["a1"] = a1
        avg_accuracy["a2"] = a2
        avg_accuracy["a3"] = a3
        avg_accuracy["a4"] = a4
        avg_accuracy["tt"] = tt
        avg_accuracy["ff"] = ff
        avg_accuracy["tf"] = tf
        avg_accuracy["ft"] = ft
        self.df_accuracy = self.df_accuracy.append(avg_accuracy, ignore_index=True)
        self.logger.store_db(self.df_accuracy, fpath=osp.join(self.eval_save_to,
                                                                  self.file_base_name + '.pkl'))
        self.logger.log_text(avg_accuracy, stream_to_stdout=False, fpath=self.acc_txt_file)