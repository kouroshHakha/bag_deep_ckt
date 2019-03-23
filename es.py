import random
import numpy as np
from deap import tools
import math
from copy import deepcopy
from util import Design
import pdb

#### modified version, compatible with multi objective optimization

def select_for_mut_based_on_order(ordered_pop):
    weights = range(len(ordered_pop), 0, -1)
    return random.choices(ordered_pop, weights=weights)[0]


def select_parents_from_two_pops(parents1, parents2):
    weights = range(len(parents1), 0, -1)
    ind1 = random.choices(parents1, weights=weights)[0]
    ind2 = random.choices(parents2, weights=weights)[0]
    return ind1, ind2

def gen_children_from_two_pops(parents1, parents2, eval_core):
    # parent 1: good in everything but the critical spec
    # parent 2: good only in critical spec
    if len(parents1) == 0:
        parents1 = parents2
    assert (G.cxpb + G.mutpb) <= 1.0, (
        "The sum of the crossover and mutation probabilities must be smaller "
        "or equal to 1.0.")
    op_choice = random.random()
    offsprings = []
    lows, ups = [], []
    for value in eval_core.params.values():
        lows.append(0)
        len_param_vec = math.floor((value[1]-value[0])/value[2])
        ups.append(len_param_vec-1)
    if op_choice <= G.cxpb:            # Apply crossover
        ind1, ind2 = select_parents_from_two_pops(parents1, parents2)
        parent1 = deepcopy(ind1)
        parent2 = deepcopy(ind2)
        ind1, ind2 = mate(ind1, ind2, low=lows, up=ups)
        Design.genocide(ind1, ind2)
        ind1.set_parents_and_sibling(parent1, parent2, ind2)
        ind2.set_parents_and_sibling(parent1, parent2, ind1)
        offsprings += [ind1, ind2]
    elif op_choice < G.cxpb + G.mutpb:      # Apply mutation
        ind = select_for_mut_based_on_order(parents1)
        parent1 = deepcopy(ind)
        ind, = mutate(ind, low=lows, up=ups)
        Design.genocide(ind)
        ind.set_parents_and_sibling(parent1, None, None)
        offsprings.append(ind)
    return offsprings

##############################################################

def generate_offspring(population, eval_core, cxpb, mutpb):

    assert (cxpb + mutpb) <= 1.0, (
        "The sum of the crossover and mutation probabilities must be smaller "
        "or equal to 1.0.")
    op_choice = random.random()
    offsprings = []
    lows, ups = [], []
    for value in eval_core.params.values():
        lows.append(0)
        len_param_vec = math.floor((value[1]-value[0])/value[2])
        ups.append(len_param_vec-1)

    if op_choice <= cxpb:            # Apply crossover
        ind1, ind2 = selectParents(population)
        ind1, ind2 = mate(ind1, ind2, low=lows, up=ups)
        offsprings += [ind1, ind2]
    elif op_choice < cxpb + mutpb:      # Apply mutation
        ind = select_for_mut(population)
        ind, = mutate(ind, low=lows, up=ups)
        offsprings.append(ind)
    return offsprings

def selParentRandom(population):
    return random.sample(population, 2)

def selectParents(population):
    return selParentRandom(population)

def mate(ind1, ind2,low, up, blend_prob=0.5):
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
        # ind1 = [math.floor(ind1[i]) for i in range(len(ind1))]
        # ind2 = [math.ceil(ind2[i]) for i in range(len(ind2))]
        return ind1, ind2

    else:
        return tools.cxUniform(ind1, ind2, indpb=0.5)
    # return tools.cxOnePoint(ind1, ind2)
    # return tools.cxTwoPoint(ind1, ind2)

def mutate(ind, low, up):
    return tools.mutUniformInt(ind, low=low, up=up, indpb=0.5)

def select(pop, mu):
    # The best sample in pop is going to have prob=1 but the other samples
    # are going to have linearly diminishing chance in being in the next generation
    # do this until mu designs are chosen
    # sorted_pop = sorted(pop, key=lambda x: x.cost)
    # prob = [(1-i/(len(pop)-1)) for i in range(len(pop))]
    # selected_individuals = []
    # index = 0
    # while len(selected_individuals) < mu:
    #     if not (any([(sorted_pop[index] == row) for row in selected_individuals])):
    #         if random.random() < prob[index]:
    #             selected_individuals.append(sorted_pop[index])
    #
    #     index += 1
    #     index = index % len(sorted_pop)
    # return selected_individuals
    return tools.selBest(pop, mu, fit_attr='fitness')
def select_new(pop, offspring):
    if len(offspring) > 0:
        return pop[:-len(offspring)]+offspring

    return pop

def select_for_mut(population):
    reverse_sorted_pop = sorted(population, key=lambda x: x.cost, reverse=True)
    ranks = np.arange(1, len(reverse_sorted_pop)+1, 1)
    prob = ranks / np.sum(ranks)
    ind = random.choices(reverse_sorted_pop, weights=prob)[0]
    return ind

class G:
    cxpb = 0.6
    mutpb = 0.4