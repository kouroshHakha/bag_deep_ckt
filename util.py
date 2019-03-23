"""
This one shares parameters across each spec's network so the information is shared across them

"""
import numpy as np
import random
import tensorflow as tf
import sys
import time
from copy import deepcopy

from typing import TYPE_CHECKING, List, Optional, Dict, Any, Tuple, Sequence

## function and classes related to this specific problem and dealing with the evaluation core
#########################
##   Algorithm Helpers   ##
#########################
def find_pop(db, k, spec_range):
    """
    :param db: list of designs
    :param k: integer indicating the cut off rank
    :param spec_range: dictionary indicating the desired specifications
    :return:
    pop: dictionary: keywords are the desired specs as well as 'cost'.values are db entries
    sorted by the keyword criteria
    """
    assert k <= len(db), "k={} is not less that len(db)={}".format(k, len(db))
    pop = {}
    pop['cost'] = sorted(db, key=lambda x: x.cost)[:k]
    for kwrd in spec_range.keys():
        spec_min, spec_max, _ = spec_range[kwrd]
        reverse = True if spec_min is not None else False
        pop[kwrd] = sorted(db, key=lambda x: x.specs[kwrd], reverse=reverse)[:k]

    return pop

def find_ciritical_pop(eval_core, db, k, specs):
    """
    specs is a list of strings indicating the spec_kwrds that have been important.
    This function looks at db and returns a list of designs sorted by the cumulative
    penalties induced by specs.
    :param db:
    :param eval_core:
    :param k:
    :param specs:
    :return:
    """
    if len(specs)==0:
        return random.sample(db, k)

    from operator import add
    sum_penalties = [0 for ind in db]
    for kwrd in specs:
        penalties = eval_core.compute_penalty([ind.specs[kwrd] for ind in db], kwrd)
        sum_penalties = list(map(add, penalties, sum_penalties))

    indices = sorted(range(len(db)), key=lambda x: sum_penalties[x])
    critical_pop = [db[i] for i in indices[:k]]
    return critical_pop

def find_critic_spec(eval_core, db, spec_range, critical_specs, ref_idx=20, k=200):
    penalty = {}
    worst_specs = {}
    if len(critical_specs) == 0:
        pop = sorted(db, key=lambda x: x.cost)[:k]
    else:
        pop = find_ciritical_pop(eval_core, db, k=k, specs=critical_specs)

    """
    for each spec_kwrd we,see what the worst number is among top ref_idx designs in pop
    and compute the penalties for each of them. this allows us to select the spec_kwrd
    that contributes the most to the overall cost function. This further means if we learn
    how to imporve that spec the results are going to get better faster.
    """
    for spec_kwrd in spec_range:
        spec_min, spec_max, _ = spec_range[spec_kwrd]
        if spec_min is not None:
            worst_specs[spec_kwrd] = np.min([ind.specs[spec_kwrd] for ind in pop[:ref_idx]])
        elif spec_max is not None:
            worst_specs[spec_kwrd] = np.max([ind.specs[spec_kwrd] for ind in pop[:ref_idx]])

        penalty[spec_kwrd] = eval_core.compute_penalty(worst_specs[spec_kwrd], spec_kwrd)[0]

    critical_list_sorted = sorted(penalty.keys(), key=lambda x: penalty[x], reverse=True)

    critical_spec_kwrd = ''

    # This part makes sure that performance metrics like ibias are not picked as critical specs
    for spec in critical_list_sorted:
        if penalty[spec] != 0: #and spec not in performance_specs:
            critical_spec_kwrd = spec
            break

    print('worst_specs:', worst_specs)
    print('penalties of worst_specs:', penalty)
    print('critical_spec_kwrd', critical_spec_kwrd)
    return critical_spec_kwrd

def is_x_better_than_y(eval_core, x,y, kwrd):
    if eval_core.compute_penalty(x, kwrd) <= eval_core.compute_penalty(y, kwrd):
        return True
    return False

def compute_critical_penalties(eval_core, designs, specs):
    if type(designs) is not list:
        designs = [designs]

    from operator import add
    sum_penalties = [0 for ind in designs]
    for kwrd in specs:
        penalties = eval_core.compute_penalty([ind.specs[kwrd] for ind in designs], kwrd)
        sum_penalties = list(map(add, penalties, sum_penalties))

    return sum_penalties


##########################
##   Data structures    ##
##########################
class IDEncoder(object):
    # example: input = [1,2,3], bases = [10, 3, 8]
    # [10, 3, 8] -> [3x8, 8, 0]
    # [1, 2, 3] x [3x8, 8, 0] = [24, 16 , 0] -> 24+16+0 = 40
    # 40 -> [k] in base 62 (0,...,9,a,...,z,A,....,Z) and then we pad it to [0,k] and then return '0k'

    def __init__(self, params_vec):
        self._bases = np.array([len(vec) for vec in params_vec.values()])
        self._mulipliers = self._compute_multipliers()
        self._lookup_dict = self._create_lookup()
        self.length_letters = self._compute_length_letter()

    def _compute_length_letter(self):
        cur_multiplier = 1
        for base in self._bases:
            cur_multiplier = cur_multiplier * base

        max_number = cur_multiplier
        ret_vec = self._convert_2_base_letters(max_number)
        return len(ret_vec)

    def _create_lookup(self):
        lookup = {}
        n_letters = ord('z') - ord('a') + 1
        for i in range(10):
            lookup[i] = str(i)
        for i in range(n_letters):
            lookup[i+10] = chr(ord('a')+i)
        for i in range(n_letters):
            lookup[i+10+n_letters] = chr(ord('A')+i)
        return lookup

    def _compute_multipliers(self):
        """
        computes [3x8, 8, 0] in our example
        :return:
        """
        cur_multiplier = 1
        ret_list = []
        for base in self._bases[::-1]:
            cur_multiplier = cur_multiplier * base
            ret_list.insert(0, cur_multiplier)
            assert cur_multiplier < sys.float_info.max, 'search space too large, cannot be represented by this machine'

        print(cur_multiplier)
        ret_list[:-1] = ret_list[1:]
        ret_list[-1] = 0
        return np.array(ret_list)

    def _convert_2_base_10(self, input_vec):
        assert len(input_vec) == len(self._bases)
        return np.sum(np.array(input_vec) * self._mulipliers)

    def _convert_2_base_letters(self, input_base_10):
        x = input_base_10
        ret_list = []
        while x:
            key = int(x % len(self._lookup_dict))
            ret_list.insert(0, self._lookup_dict[key])
            x = int(x / len(self._lookup_dict))

        return ret_list

    def _pad(self, input_base_letter):
        while len(input_base_letter) < self.length_letters:
            input_base_letter.insert(0, '0')
        return input_base_letter

    def convert_list_2_id(self, input_list):
        base10 = self._convert_2_base_10(input_list)
        base_letter = self._convert_2_base_letters(base10)
        padded = self._pad(base_letter)
        return ''.join(padded)

class Design(list):
    def __init__(self, spec_range, id_encoder, seq=()):
        """
        :param spec_range: Dict[Str : [a,b]] -> kwrds are used for spec property creation of Design objects
        :param params_vec: Dict[Str : List] -> len(List) is used for id generation of Design object
        :param seq: input parameter vector as a List
        """
        list.__init__(self, seq)
        self.cost =     None
        self.fitness =  None
        self.specs = {}
        self.id_encoder = id_encoder
        # # virtuoso cannot handle lvs for long named files, so id should be the shortest while unique at the same time
        # self.id = int((time.time() - G.ref_id) * 1e6)
        # print(G.ref_id)
        # # assure unique ids, delay for 1000 us
        # time.sleep(0.001)
        for spec_kwrd in spec_range.keys():
            self.specs[spec_kwrd] = None


        self.parent1 = None
        self.parent2 = None
        self.sibling = None

    def set_parents_and_sibling(self, parent1, parent2, sibling):
        self.parent1 = parent1
        self.parent2 = parent2
        self.sibling = sibling

    def is_init_population(self):
        if self.parent1 is None:
            return True
        else:
            return False
    def is_mutated(self):
        if self.parent1 is not None:
            if self.parent2 is None:
                return True
        else:
            return False

    @property
    def id(self):
        return self.id_encoder.convert_list_2_id(list(self))

    @property
    def cost(self):
        return self.__cost

    @property
    def fitness(self):
        return self.__fitness

    @cost.setter
    def cost(self, x):
        self.__cost = x
        self.__fitness = -x if x is not None else None

    @fitness.setter
    def fitness(self, x):
        self.__fitness = x
        self.__cost = -x if x is not None else None

    @staticmethod
    def recreate_design(spec_range, old_design, eval_core):
        dsn = Design(spec_range, eval_core.id_encoder, old_design)
        dsn.specs.update(**old_design.specs)
        for attr in dsn.__dict__.keys():
            if (attr in old_design.__dict__.keys()) and (attr not in ['specs']):
                dsn.__dict__[attr] = deepcopy(old_design.__dict__[attr])
        return dsn

    @staticmethod
    def genocide(*args):
        for dsn in args:
            dsn.parent1 = None
            dsn.parent2 = None
            dsn.sibling = None

def clean(db, eval_core):
    new_spec_range = eval_core.spec_range
    list_to_be_removed = []
    for data in db:
        if data.cost is None:
            list_to_be_removed.append(data)
    for data in list_to_be_removed:
        db.remove(data)

    for i in range(len(db)):
        d_dummy = Design.recreate_design(new_spec_range, db[i], eval_core)
        db[i] = d_dummy
    return db


def relable(db, eval_core):
    for design in db:
        design.cost = 0
        for spec_kwrd in eval_core.spec_range.keys():
            design.cost += eval_core.compute_penalty(design.specs[spec_kwrd], spec_kwrd)[0]

    return db

###########################
##   Training helpers    ##
###########################

class BatchGenerator(object):
    def __init__(self, data_set_size, batch_size):
        self._data_size = data_set_size
        self._batch_size = batch_size
        self._segment = self._data_size // batch_size
        self.last_index = 0
        self._permutations = list(range(data_set_size))
        random.shuffle(self._permutations)

    def next(self):

            if ((self.last_index+1)*self._batch_size > self._data_size):
                indices1 = self._permutations[self.last_index * self._batch_size:]
                indices2 = self._permutations[:((self.last_index+1)*self._batch_size)%self._data_size]
                indices = indices1 + indices2
            else:
                indices = self._permutations[self.last_index * self._batch_size:(self.last_index + 1) * self._batch_size]

            self.last_index = (self.last_index+1) % (self._segment+1)
            return indices


def set_random_seed(seed):
    tf.set_random_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
