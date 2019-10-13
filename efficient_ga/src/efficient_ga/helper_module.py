import random


class DecisionBox:

    def __init__(self, ref_index, eval_core, logger):
        # during inference compare new randomly generated samples with this design in the sorted
        # dataset
        self.ref_index = ref_index
        self.eval_core = eval_core
        self.spec_range = eval_core.spec_range
        self.logger = logger
        self.critical_specs = []
        self.critical_spec_kwrd = None
        self.pop_dict = dict()

    def _find_pop(self, db, k_top):
        """
        :param db: list of designs
        :return:
        pop: dictionary: keywords are the desired specs as well as 'cost'.values are db entries
        sorted by the keyword criteria
        """
        assert k_top <= len(db), "ktop={} should be smaller than " \
                                 "train_set_len={}".format(k_top, len(db))
        pop = dict()
        pop['cost'] = sorted(db, key=lambda x: x.cost)[:k_top]
        for kwrd in self.spec_range.keys():
            spec_min, spec_max, _ = self.spec_range[kwrd]
            reverse = True if spec_min is not None else False
            pop[kwrd] = sorted(db, key=lambda x: x.specs[kwrd], reverse=reverse)[:k_top]

        return pop

    def _find_ciritical_pop(self, db, specs, k_top):
        """
        specs is a list of strings indicating the spec_kwrds that have been important.
        This function looks at db and returns a list of designs sorted by the cumulative
        penalties induced by specs.
        :param db:
        :param specs:
        """

        if len(specs) == 0:
            return random.sample(db, k_top)

        from operator import add
        sum_penalties = [0]*len(db)
        for kwrd in specs:
            penalties = self.eval_core.compute_penalty([ind.specs[kwrd] for ind in db], kwrd)
            sum_penalties = list(map(add, penalties, sum_penalties))

        indices = sorted(range(len(db)), key=lambda x: sum_penalties[x])
        critical_pop = [db[i] for i in indices[:k_top]]
        return critical_pop

    def _find_critic_spec(self, db, k_top):
        penalty = {}
        worst_specs = {}
        if len(self.critical_specs) == 0:
            pop = sorted(db, key=lambda x: x.cost)[:k_top]
        else:
            pop = self._find_ciritical_pop(db, self.critical_specs, k_top)

        """
        for each spec_kwrd we see what the worst number is among top ref_idx designs in pop
        and compute the penalties for each of them. this allows us to select the spec_kwrd
        that contributes the most to the overall cost function. This further means if we learn
        how to imporve that spec the results are going to get better faster.
        """
        for spec_kwrd in self.spec_range:

            spec_nums = [ind.specs[spec_kwrd] for ind in pop[:self.ref_index]]
            worst_specs[spec_kwrd], penalty[spec_kwrd] = self.eval_core.find_worst(spec_nums,
                                                                                   spec_kwrd,
                                                                                   ret_penalty=True)
            # spec_min, spec_max, _ = self.spec_range[spec_kwrd]
            # if spec_min is not None:
            #     worst_specs[spec_kwrd] = np.min([ind.specs[spec_kwrd] for ind in pop[
            #                                                                      :self.ref_index]])
            # elif spec_max is not None:
            #     worst_specs[spec_kwrd] = np.max([ind.specs[spec_kwrd] for ind in pop[
            #                                                                      :self.ref_index]])
            #
            # penalty[spec_kwrd] = self.eval_core.compute_penalty(worst_specs[spec_kwrd],
            #                                                     spec_kwrd)[0]

        critical_list_sorted = sorted(penalty.keys(), key=lambda x: penalty[x], reverse=True)

        critical_spec_kwrd = ''

        # This part makes sure that performance metrics like ibias are not picked as critical specs
        for spec in critical_list_sorted:
            if penalty[spec] != 0:  # and spec not in performance_specs:
                critical_spec_kwrd = spec
                break

        self.logger.log_text('worst_specs: {}'.format(worst_specs))
        self.logger.log_text('penalties of worst_specs: {}'.format(penalty))
        self.logger.log_text('critical_spec_kwrd: {}'.format(critical_spec_kwrd))
        return critical_spec_kwrd

    def _compute_critical_penalties(self, designs, specs):
        if type(designs) is not list:
            designs = [designs]

        from operator import add
        sum_penalties = [0]*len(designs)
        for kwrd in specs:
            penalties = self.eval_core.compute_penalty([ind.specs[kwrd] for ind in designs], kwrd)
            sum_penalties = list(map(add, penalties, sum_penalties))

        return sum_penalties

    def update_heuristics(self, db, k_top):

        self.pop_dict = self._find_pop(db, k_top)
        # extract the most influential spec on cost function
        self.critical_spec_kwrd = self._find_critic_spec(db, k_top)

        # if critical spec is nothing it means that everything in the top population on average
        # meets the spec
        if self.critical_spec_kwrd == '':
            return

        if self.critical_spec_kwrd not in self.critical_specs:
            self.critical_specs.append(self.critical_spec_kwrd)

        self.logger.log_text('[debug] critical_specs: {}'.format(self.critical_specs))
        self.pop_dict['critical_specs'] = self._find_ciritical_pop(db, self.critical_specs, k_top)

    def get_parents_and_ref_design(self, db, k_top):
        ref_design = self.pop_dict['critical_specs'][self.ref_index]

        self.logger.log_text(30*"-")
        self.logger.log_text("[debug] ref design {} -> {}".format(ref_design, ref_design.cost))
        self.logger.log_text("[debug] {}".format(ref_design.specs))

        all_crit_specs_except_last = self.critical_specs.copy()
        all_crit_specs_except_last.remove(self.critical_spec_kwrd)

        parent1 = self._find_ciritical_pop(db, all_crit_specs_except_last, k_top)
        parent2 = self.pop_dict[self.critical_spec_kwrd]

        self.logger.log_text("///////------------> parent1", stream_to_file=False)
        penalties = self._compute_critical_penalties(parent1, all_crit_specs_except_last)
        for i, penalty in enumerate(penalties[:self.ref_index]):
            self.logger.log_text("{} -> {}".format(parent1[i], penalty), stream_to_file=False)
        self.logger.log_text("///////------------> parent2", stream_to_file=False)
        for ind in parent2[:self.ref_index]:
            self.logger.log_text("{} -> {}".format(ind, ind.specs[self.critical_spec_kwrd]),
                                 stream_to_file=False)

        return parent1, parent2, ref_design

    def has_converged(self):

        # if critical spec is nothing it means that everything in the top population on average
        # meets the spec
        if self.critical_spec_kwrd == '':
            return True

        # if found even one solution exit the process
        # if self.pop_dict['cost'][0].cost == 0:
        #     return [], True

    def accept_new_design(self, prediction):
        # sample from output distribution and see if new design in better than ref design in almost
        # all critical design metrics
        is_new_design_better = all([random.random() > prediction[kwrd][0][0] for kwrd in
                                    self.critical_specs])
        # if we want to account uncertainty we only reject designs that we are sure are worst than
        # the reference design with 20% confidence percentage
        # is_new_design_worst = any([(random.random() < prediction[kwrd][0][0] and
        #                             prediction[kwrd][0][0] < 0.2) for kwrd in self.critical_specs])
        # is_new_design_better = not is_new_design_worst
        return is_new_design_better
