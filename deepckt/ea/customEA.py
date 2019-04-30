from util import Design
import random
import math
from deap import tools
from .ea import EA

class CustomEA(EA):

    def __init__(self, cxpb, mutpb, eval_core):
        self.eval_core = eval_core
        self.cxpb = cxpb
        self.mutpb = mutpb

        self.ups, self.lows = [], []
        for value in self.eval_core.params.values():
            self.lows.append(0)
            len_param_vec = math.floor((value[1]-value[0])/value[2])
            self.ups.append(len_param_vec-1)

    def get_next_generation_candidates(self, parents1, parents2):
        # parent 1: good in everything but the critical spec
        # parent 2: good only in critical spec
        if len(parents1) == 0:
            parents1 = parents2
        assert (self.cxpb + self.mutpb) <= 1.0, (
            "The sum of the crossover and mutation probabilities must be smaller "
            "or equal to 1.0.")
        op_choice = random.random()
        offsprings = []
        if op_choice <= self.cxpb:            # Apply crossover
            parent1, parent2 = self._select_parents_from_two_pops(parents1, parents2)
            ind1 = parent1.copy()
            ind2 = parent2.copy()
            ind1, ind2 = self._mate(ind1, ind2, low=self.lows, up=self.ups)
            Design.genocide(ind1, ind2)
            ind1.set_parents_and_sibling(parent1, parent2, ind2)
            ind2.set_parents_and_sibling(parent1, parent2, ind1)
            offsprings += [ind1, ind2]
        elif op_choice < self.cxpb + self.mutpb:      # Apply mutation
            parent1 = self._select_for_mut_based_on_order(parents1)
            new_ind = parent1.copy()
            new_ind, = self._mutate(new_ind, low=self.lows, up=self.ups)
            Design.genocide(new_ind)
            new_ind.set_parents_and_sibling(parent1, None, None)
            offsprings.append(new_ind)

        return offsprings

    @staticmethod
    def _select_for_mut_based_on_order(ordered_pop):
        weights = range(len(ordered_pop), 0, -1)
        return random.choices(ordered_pop, weights=weights)[0]

    @staticmethod
    def _select_parents_from_two_pops(parents1, parents2):
        weights = range(len(parents1), 0, -1)
        ind1 = random.choices(parents1, weights=weights)[0]
        ind2 = random.choices(parents2, weights=weights)[0]
        return ind1, ind2

    @staticmethod
    def _mate(ind1, ind2, low, up, blend_prob=0.5):
        # a mixture of blend and 2 point crossover
        if random.random() < blend_prob:
            ind1, ind2 = tools.cxBlend(ind1, ind2, alpha=0.5)
            size = min(len(ind1), len(ind2))
            for i, u, l in zip(range(size), up, low):
                ind1[i] = math.floor(ind1[i])
                ind2[i] = math.ceil(ind2[i])
                if ind1[i] > u:
                    ind1[i] = u
                elif ind1[i] < l:
                    ind1[i] = l
                if ind2[i] > u:
                    ind2[i] = u
                elif ind2[i] < l:
                    ind2[i] = l
            return ind1, ind2
        else:
            return tools.cxUniform(ind1, ind2, indpb=0.5)

    @staticmethod
    def _mutate(ind, low, up):
        return tools.mutUniformInt(ind, low=low, up=up, indpb=0.5)

    def prepare_for_generation(self, db, n):
        # does nothing for preparation
        pass