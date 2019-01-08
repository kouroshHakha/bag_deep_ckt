"""
This one shares parameters across each spec's network so the information is shared across them

"""
import copy
import importlib
import os
import pickle
import time

import bag_deep_ckt.es as es
from bag.io import read_yaml
from bag_deep_ckt.util import *

np.random.seed(10)
random.seed(10)


class DeepCktAgent(object):
    def __init__(self,
                 eval_core,
                 spec_range,
                 n_init_samples=200,
                 n_new_samples=5,
                 num_designs=2,
                 num_params_per_design=7,
                 num_classes=2,
                 valid_frac=0.2,
                 max_n_retraining=80,
                 k_top=199,
                 ref_dsn_idx=20,
                 max_data_set_size=600,
                 num_epochs=100,
                 batch_size=64,
                 display_step=25,
                 ckpt_step=25,
                 size=20,
                 learning_rate=0.03,
                 ):

        self.eval_core = eval_core
        self.spec_range = spec_range
        self.spec_list = list(spec_range.keys())

        # data generation and preprocessing. for new environments these numbers should be readjusted
        self.n_init_samples = n_init_samples
        self.n_new_samples = n_new_samples
        self.num_designs = num_designs
        self.num_params_per_design = num_params_per_design
        self.num_classes = num_classes
        self.num_nn_features = num_designs * num_params_per_design
        self.valid_frac = valid_frac
        self.max_n_retraining = max_n_retraining

        self.k_top = k_top  #during training only consider comparison between k_top ones and the others
        self.ref_dsn_idx = ref_dsn_idx #during inference compare new randomly generated samples with this design in the sorted dataset
        self.max_data_set_size = max_data_set_size

        # training settings
        self.num_epochs = num_epochs
        self.batch_size = batch_size
        self.display_step = display_step
        self.ckpt_step = ckpt_step

        # nn hyper parameters
        self.feat_ext_dim_list = [num_params_per_design, size, size]
        self.compare_nn_dim_list = [2*size, size, num_classes]

        self.learning_rate = learning_rate

        self.critical_specs = []

    def init_tf_sess(self):
        tf_config = tf.ConfigProto(inter_op_parallelism_threads=1, intra_op_parallelism_threads=1)
        tf_config.gpu_options.allow_growth = True  # may need if using GPU
        self.sess = tf.Session(config=tf_config)
        self.sess.__enter__()  # equivalent to `with self.sess:`
        tf.global_variables_initializer().run()  # pylint: disable=E1101

    def _define_placeholders(self):
        self.input1 = tf.placeholder(tf.float32, shape=[None, self.num_params_per_design], name='in1')
        self.input2 = tf.placeholder(tf.float32, shape=[None, self.num_params_per_design], name='in2')
        self.true_labels = {}

        for kwrd in self.spec_list:
            self.true_labels[kwrd] = tf.placeholder(tf.float32, shape=[None, self.num_classes], name='labels_'+kwrd)

    def _normalize(self):
        with tf.variable_scope('normalizer'):
            self.mu = tf.Variable(tf.zeros([self.num_params_per_design], dtype=tf.float32), name='mu', trainable=False)
            self.std = tf.Variable(tf.zeros([self.num_params_per_design], dtype=tf.float32), name='std', trainable=False)
            input1_norm = (self.input1 - self.mu) / (self.std + 1e-6)
            input2_norm = (self.input2 - self.mu) / (self.std + 1e-6)

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

            Weight = tf.concat([weight_elements, weight_elements[::-1, ::-1]], axis=0, name='Weights')
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
                layer, w, b = self._sym_fc_layer(layer, layer_dim, activation_fn='Relu', reuse=reuse, scope=name+str(i))
                w_list.append(w)
                b_list.append(b)

            logits, w, b = self._sym_fc_layer(layer, self.num_classes, reuse=reuse, scope='fc_out')
            w_list.append(w)
            b_list.append(b)

        return logits, w_list, b_list

    def _build_policy(self):

        input1_norm, input2_norm = self._normalize()
        features1 = self._feature_extraction_model(input1_norm, name='feat_model', reuse=False)
        features2 = self._feature_extraction_model(input2_norm, name='feat_model', reuse=True)
        input_features = tf.concat([features1, features2[:, ::-1]], axis=1)

        self.out_logits, w_list, b_list = {}, {}, {}
        for kwrd in self.spec_list:
            self.out_logits[kwrd], w_list[kwrd], b_list[kwrd] = self._comparison_model(input_features,
                                                                                       'cmp_model_' + kwrd,
                                                                                       reuse=False)

    def _build_loss(self):
        self.total_loss = 0
        neg_likelihoods, self.loss = {}, {}
        self.out_predictions = {}
        for kwrd in self.spec_list:
            with tf.variable_scope("loss_"+kwrd):
                self.out_predictions[kwrd] = tf.nn.softmax(self.out_logits[kwrd])
                neg_likelihoods[kwrd] = tf.nn.softmax_cross_entropy_with_logits_v2(labels=self.true_labels[kwrd],
                                                                                   logits=self.out_logits[kwrd])
                self.loss[kwrd] = tf.reduce_mean(neg_likelihoods[kwrd])
                self.total_loss += self.loss[kwrd]

    def _build_accuracy(self):
        self.accuracy = {}
        for kwrd in self.spec_list:
            correct_predictions = tf.equal(tf.argmax(self.out_predictions[kwrd], axis=1), tf.argmax(self.true_labels[kwrd], axis=1))
            self.accuracy[kwrd] = tf.reduce_mean(tf.cast(correct_predictions, tf.float32))

    def build_computation_graph(self):

        self._define_placeholders()
        self._build_policy()
        self._build_loss()
        self._build_accuracy()

        self.update_op = tf.train.AdamOptimizer().minimize(self.total_loss)


    def combine(self, train_set):

        # this combine function is used when generating nn data for training
        # label [0 1] means design1 is "sufficiently" worse than than design2 (e.g gain1 is at
        # least %10 higher than gain2) and label [1 0] means it is not the case.
        # it is really important to combine those pairs that are going to be useful during inference
        # since during inference one of the designs is always good we should make sure that during training also
        # this bias on some level exists. For this reason we will sort the data set and produce pairs that always have at least
        # one design from top k_top designs.

        assert self.k_top < len(train_set), "ktop={}, train_set_len={}".format(self.k_top, len(train_set))

        db_cost_sorted = sorted(train_set, key=lambda x: x.cost)[:self.k_top+1]

        category = {}
        nn_input1, nn_input2 = [], []
        nn_labels = {}
        for kwrd in self.spec_list:
            category[kwrd], nn_labels[kwrd] = [], []

        n = len(db_cost_sorted)

        for i in range(n-1):
            for j in range(i+1, n):
                rnd = random.random()
                if rnd < 0.5:
                    nn_input1.append(db_cost_sorted[i])
                    nn_input2.append(db_cost_sorted[j])
                    for kwrd in self.spec_list:
                        label = 1 if is_x_better_than_y(eval_core=self.eval_core,
                                                        x=db_cost_sorted[i].specs[kwrd],
                                                        y=db_cost_sorted[j].specs[kwrd],
                                                        kwrd=kwrd) else 0
                        category[kwrd].append(label)
                else:
                    nn_input1.append(db_cost_sorted[j])
                    nn_input2.append(db_cost_sorted[i])
                    for kwrd in self.spec_list:
                        label = 0 if is_x_better_than_y(eval_core=self.eval_core,
                                                        x=db_cost_sorted[i].specs[kwrd],
                                                        y=db_cost_sorted[j].specs[kwrd],
                                                        kwrd=kwrd) else 1
                        category[kwrd].append(label)

        for kwrd in self.spec_list:
            nn_labels[kwrd] = np.zeros((len(category[kwrd]), self.num_classes))
            nn_labels[kwrd][np.arange(len(category[kwrd])), category[kwrd]] = 1

        nn_input1 = np.array(nn_input1)
        nn_input2 = np.array(nn_input2)

        return nn_input1, nn_input2, nn_labels

    def train(self, db):
        t_minus = time.time()
        all_vars = tf.global_variables()
        saver = tf.train.Saver(all_vars)

        nn_input1, nn_input2, nn_labels = self.combine(db)

        permutation = np.random.permutation(len(nn_input1))
        nn_input1 = nn_input1[permutation]
        nn_input2 = nn_input2[permutation]
        for kwrd in self.spec_list:
            nn_labels[kwrd] = nn_labels[kwrd][permutation]

        boundry_index = len(nn_input1) - int(len(nn_input1)*self.valid_frac)

        train_input1 = nn_input1[:boundry_index]
        train_input2 = nn_input2[:boundry_index]
        valid_input1 = nn_input1[boundry_index:]
        valid_input2 = nn_input2[boundry_index:]
        train_labels, valid_labels = {}, {}
        for kwrd in self.spec_list:
            train_labels[kwrd] = nn_labels[kwrd][:boundry_index]
            valid_labels[kwrd] = nn_labels[kwrd][boundry_index:]

        # find the mean and std of dataset for normalizing
        train_mean = np.mean(np.concatenate([train_input1, train_input2], axis=0), axis=0)
        train_std = np.std(np.concatenate([train_input1, train_input2], axis=0), axis=0)
        # print(train_mean)
        # print(train_std)

        print("[info] dataset size:%d" % len(db))
        print("[info] combine size:%d" %(len(train_input1)+len(valid_input1)))
        for kwrd in self.spec_list:
            print("[info][%s] train_dataset: positive_samples/total ratio : %d/%d" %(kwrd, np.sum(train_labels[kwrd], axis=0)[0], train_labels[kwrd].shape[0]))
            print("[info][%s] valid_dataset: positive_samples/total ratio : %d/%d" %(kwrd, np.sum(valid_labels[kwrd], axis=0)[0], valid_labels[kwrd].shape[0]))

        # although we have shuffled the training dataset once, there is going to be another shuffle inside the batch generator
        batch_generator = BatchGenerator(len(train_input1), self.batch_size)
        print("[info] training the model with dataset ....")

        total_n_batches = int(len(train_input1) // self.batch_size)
        print("[info] number of total batches: %d" %total_n_batches)
        print(30*"-")

        # tf.global_variables_initializer().run()
        self.mu.assign(train_mean).op.run()
        self.std.assign(train_std).op.run()

        for epoch in range(self.num_epochs+1):

            avg_loss, avg_train_acc, avg_valid_acc = {}, {}, {}
            avg_total_loss = 0.

            for kwrd in self.spec_list:
                avg_loss[kwrd] = 0.
                avg_train_acc[kwrd] = 0.
                avg_valid_acc[kwrd] = 0.

            feed_dict = {}
            for iter in range(total_n_batches):
                index = batch_generator.next()
                batch_input1, batch_input2 = train_input1[index], train_input2[index]
                batch_labels = {}
                feed_dict = {self.input1         :batch_input1,
                             self.input2         :batch_input2}
                for kwrd in self.spec_list:
                    feed_dict[self.true_labels[kwrd]] = train_labels[kwrd][index]

                # print(session.run(loss, feed_dict=feed_dict))
                _, l, t_l, train_acc = self.sess.run([self.update_op, self.loss, self.total_loss, self.accuracy],
                                                     feed_dict=feed_dict)
                # print(np.sum(list(l.values())), t_l)
                # print(session.run(loss, feed_dict=feed_dict))
                feed_dict = {self.input1         :valid_input1,
                             self.input2         :valid_input2}
                for kwrd in self.spec_list:
                    feed_dict[self.true_labels[kwrd]] = valid_labels[kwrd]

                valid_acc, = self.sess.run([self.accuracy], feed_dict=feed_dict)

                for kwrd in self.spec_list:
                    avg_total_loss += t_l / total_n_batches
                    avg_loss[kwrd] += l[kwrd] / total_n_batches
                    avg_train_acc[kwrd] += train_acc[kwrd] / total_n_batches
                    avg_valid_acc[kwrd] += valid_acc[kwrd] / total_n_batches

            if epoch % self.ckpt_step == 0:
                saver.save(self.sess, 'bag_deep_ckt/checkpoint/saver/checkpoint.ckpt')
                with open('bag_deep_ckt/checkpoint/db/data.pkl', 'wb') as f:
                    pickle.dump(db, f)
            if epoch % self.display_step == 0:
                print(10*"-")
                print("[epoch %d] total_loss: %f " %(epoch, avg_total_loss))
                for kwrd in self.spec_list:
                    print("".format(kwrd))
                    print("[%s] loss: %f" %(kwrd, avg_loss[kwrd]))
                    print("[%s] train_acc = %.2f%%, valid_acc = %.2f%%" %(kwrd, avg_train_acc[kwrd]*100,
                                                                          avg_valid_acc[kwrd]*100))

        t_plus = time.time()
        print("[info] training done %.2fSec" % (t_plus - t_minus))

    def run_model(self, db, pop_dict, max_iter=1000):
        pop_size = len(pop_dict['cost'])
        # extract the most influential spec on cost function
        critical_spec_kwrd = find_critic_spec(self.eval_core, db, self.spec_range, self.critical_specs,
                                              ref_idx=self.ref_dsn_idx, k=self.k_top)

        # if critical spec is nothing it means that everything in the top population on average meets the spec
        if critical_spec_kwrd == '':
            return [], True

        if critical_spec_kwrd not in self.critical_specs:
            self.critical_specs.append(critical_spec_kwrd)

        pop_dict['critical_specs'] = find_ciritical_pop(self.eval_core, db, k=pop_size, specs=self.critical_specs)

        ref_design = pop_dict['critical_specs'][self.ref_dsn_idx]

        print(30*"-")
        print("[debug] ref design {} -> {}".format(ref_design, ref_design.cost))
        debug_str = ''
        for i, key in enumerate(ref_design.specs.keys()):
            debug_str += '{} -> '.format(ref_design.specs[key])
        print("[debug] "+debug_str)

        all_crit_specs_except_last = self.critical_specs.copy()
        all_crit_specs_except_last.remove(critical_spec_kwrd)

        parent1 = find_ciritical_pop(self.eval_core, db, k=pop_size, specs=all_crit_specs_except_last)
        parent2 = pop_dict[critical_spec_kwrd]

        print("///////------------> parent1")
        penalties = compute_critical_penalties(self.eval_core, parent1, all_crit_specs_except_last)
        for i, penalty in enumerate(penalties[:self.ref_dsn_idx]):
            print("{} -> {}".format(parent1[i], penalty))
        print("///////------------> parent2")
        for ind in parent2[:self.ref_dsn_idx]:
            print("{} -> {}".format(ind, ind.specs[critical_spec_kwrd]))

        print(30*"-")
        print("[info] running model ... ")

        offsprings = []
        n_iter = 0

        while len(offsprings) < self.n_new_samples and n_iter < max_iter:
            new_designs = es.gen_children_from_two_pops(copy.deepcopy(parent1),
                                                        copy.deepcopy(parent2),
                                                        self.eval_core)

            for new_design in new_designs:
                if any([(new_design == row) for row in db]) or any([(new_design == row) for row in offsprings]):
                    # if design is already in the design pool skip ...
                    print("[debug] design {} already exists".format(new_design))
                    continue

                n_iter += 1
                nn_input1 = np.array(new_design)
                nn_input2 = np.array(ref_design)
                feed_dict = {
                    self.input1: nn_input1[None, :],
                    self.input2: nn_input2[None, :],
                }
                prediction = self.sess.run(self.out_predictions, feed_dict=feed_dict)

                # sample from output distribution and see if new design in better than ref design in almost all critical
                # design metrics
                is_new_design_better = [random.random() > prediction[kwrd][0][0] for kwrd in self.critical_specs]

                if all(is_new_design_better):
                    offsprings.append(new_design)

        if (len(offsprings) < self.n_new_samples):
            return [], True

        print(30*"-")
        design_results = self.eval_core.evaluate(offsprings)
        list_to_be_removed = []
        for i, design in enumerate(offsprings):
            design_result = design_results[i]
            if design_result['valid']:
                design.cost = design_result['cost']
                for key in design.specs.keys():
                    design.specs[key] = design_result[key]

                print("[debug] design {} with cost {} was added".format(design, design.cost))
                debug_str = ''
                for key in design.specs.keys():
                    debug_str += '{} -> '.format(design.specs[key])
                print("[debug] "+debug_str)
                print(10*'-')
            else:
                print("[debug] design {} did not produce valid results".format(design))
                list_to_be_removed.append(design)

        for design in list_to_be_removed:
            offsprings.remove(design)

        print("[info] new designs tried: %d" %n_iter)
        print("[info] new candidates size: %d " %len(offsprings))

        return offsprings, False

def main():

    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('design_specs_fname', type=str,
                        help='The top level yaml file that contains everything')
    parser.add_argument('env_name', type=str,
                        help='The name used for logdir creation as well as the time stamp')
    parser.add_argument('--max_n_retraining', '-mnr', type=int, default=100,
                        help='the maximum number of retrainings (the algorithm can finish sooner than this)')
    parser.add_argument('--n_init_samples', '-n', type=int, default=80,
                        help='if init_data is given n random ones are picked from it ,'
                             ' if not, eval_core will generate and evaluate n samples, at the beginning')
    parser.add_argument('--max_iter', type=int, default=200000,
                        help='max number of iterations that we call nn prediction on offsprings')
    parser.add_argument('--evict_old_data', '-eod', type=bool, default=False,
                        help='A variant of algorithm that will evict old data from training dataset and keeps the'
                             'training dataset size equal to n_init_samples, hence the training time constant,'
                             'making this true does not usually help')
    parser.add_argument('--ref_dsn_idx', '-ref', type=int, default=10,
                        help='The rank number of the design in the population that is used for comparison')

    args = parser.parse_args()

    content = read_yaml(args.design_specs_fname)

    # setup database directory
    db_dir = content['database_dir']
    os.makedirs(db_dir, exist_ok=True)

    # create evaluation core instance
    eval_module = importlib.import_module(content['eval_core_package'])
    eval_cls = getattr(eval_module, content['eval_core_class'])
    eval_core = eval_cls(design_specs_fname=args.design_specs_fname)

    # load/create db
    if os.path.isfile(db_dir+'/init_data.pickle'):
        with open(db_dir+'/init_data.pickle', 'rb') as f:
            db = pickle.load(f)
    else:
        db = eval_core.generate_data_set(args.n_init_samples, evaluate=True)
        with open(db_dir+'/init_data.pickle', 'wb') as f:
            pickle.dump(db, f)


    # check len(db)
    db = clean(db)
    # check len(db): it should be the same, since no invalid design is in db anymore
    # check d.cost for some ds
    db = relable(db, eval_core)
    # check d.cost fo the same ds and check the relableling function
    db = sorted(db, key=lambda x: x.cost)
    db = random.choices(db, k=args.n_init_samples)
    n_init_samples = len(db)

    agent = DeepCktAgent(
        eval_core=eval_core,
        spec_range=eval_core.spec_range,
        n_init_samples=n_init_samples,
        n_new_samples=5,
        num_designs=2,
        num_params_per_design=eval_core.num_params,
        num_classes=2,
        valid_frac=0.2,
        max_n_retraining=args.max_n_retraining,
        k_top=len(db)-1,
        ref_dsn_idx=args.ref_dsn_idx,
        max_data_set_size=600,
        num_epochs=100,
        batch_size=64,
        display_step=25,
        ckpt_step=25,
        size=20,
        learning_rate=0.001,
    )

    agent.build_computation_graph()
    agent.init_tf_sess()
    set_random_seed(10)


    data_set_list = []
    data_set_list.append(db)

    agent.train(db)
    for i in range(args.max_n_retraining):
        pop_dict = find_pop(db, k=n_init_samples, spec_range=eval_core.spec_range)

        offsprings, isConverged = agent.run_model(db, pop_dict, max_iter=args.max_iter)

        if isConverged:
            break
        elif len(offsprings) == 0:
            continue

        if args.evict_old_data:
            db = db[:-len(offsprings)] + offsprings
        else:
            db = db + offsprings
        data_set_list.append(offsprings)

        agent.train(db)


    log_dir = 'bag_deep_ckt/log_files/'+ args.env_name +'_'+'MDNN_'+ \
              time.strftime("%d-%m-%Y_%H-%M-%S")+'.pkl'

    with open(log_dir, 'wb') as f:
        pickle.dump(data_set_list, f)

    sorted_db = sorted(db, key=lambda x: x.cost)
    print("[finished] total_n_evals = {}".format(len(db)))
    print("[finished] best_solution = {}".format(sorted_db[0]))
    print("[finished] id = {}".format(sorted_db[0].id))
    print("[finished] cost = {}".format(sorted_db[0].cost))
    print("[finished] performance \n{} ".format(sorted_db[0].specs))
    for ind in sorted_db[:args.ref_dsn_idx]:
        print("{} -> {} -> {}".format(ind, ind.cost, ind.specs))

    # re-evaluate the best design:
    # eval_core.evaluate([sorted_db[0]])

if __name__ == '__main__':
    main()