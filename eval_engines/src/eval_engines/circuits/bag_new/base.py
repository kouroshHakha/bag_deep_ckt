from typing import Optional, Dict, Any, Sequence

from bag_mp.manager import EvaluationManager

from ...util.design import Design
from ...util.importlib import import_cls
from ..base import CircuitsEngineBase, SpecSeqType

import pdb


class BagNetEngineBase(CircuitsEngineBase):

    def __init__(self, yaml_fname: Optional[str] = None,
                 specs: Optional[Dict[str, Any]] = None, **kwargs) -> None:
        CircuitsEngineBase.__init__(self, yaml_fname, specs, **kwargs)
        _eval_cls_str = self.specs['eval_module_cls']
        _eval_temp = self.specs['eval_module_template']
        _eval_cls = import_cls(_eval_cls_str)
        self.eval_module: EvaluationManager = _eval_cls(_eval_temp, **kwargs)

    def generate_rand_designs(self, n: int = 1, evaluate: bool = False, seed: Optional[int] = None,
                              **kwargs) -> Sequence[Design]:
        if seed:
            self.set_seed(seed)
        tried_designs, valid_designs = [], []
        remaining = n
        while remaining != 0:
            trying_designs = []
            useless_iter_count = 0
            while len(trying_designs) < remaining:
                rand_design = self.get_rand_sample()
                if rand_design in tried_designs or rand_design in trying_designs:
                    useless_iter_count += 1
                    if useless_iter_count > n * 10:
                        raise ValueError(f'large amount randomly selected samples failed {n}')
                    continue
                trying_designs.append(rand_design)

            if evaluate:
                self.evaluate(trying_designs)
                n_valid = 0
                for design in trying_designs:
                    tried_designs.append(design)
                    if design['valid']:
                        n_valid += 1
                        valid_designs.append(design)
                remaining = remaining - n_valid
            else:
                remaining = remaining - len(trying_designs)

            pdb.set_trace()

        generator_efficiency = len(valid_designs) / len(tried_designs)
        print(f'Genrator Efficiency: {generator_efficiency}')
        return valid_designs

    def evaluate(self, designs: Sequence[Design], *args, **kwargs) -> Any:
        designs_interpreted = [self.interpret(dsn) for dsn in designs]
        results = self.eval_module.batch_evaluate(designs_interpreted, sync=True)
        self.update_designs_with_results(designs, results)
        return designs

    def compute_penalty(self, spec_nums: SpecSeqType, spec_kwrd: str) -> SpecSeqType:
        """
        Parameters
        ----------
        spec_nums
        spec_kwrd

        Returns
        -------

        """
        if not hasattr(spec_nums, '__iter__'):
            list_spec_nums = [spec_nums]
        else:
            list_spec_nums = spec_nums

        penalties = []
        for spec_num in list_spec_nums:
            penalty = 0
            ret = self.spec_range[spec_kwrd]
            if len(ret) == 3:
                spec_min, spec_max, w = ret
            else:
                spec_min, spec_max = ret
                w = 1
            if spec_max is not None:
                if spec_num > spec_max:
                    # if (spec_num + spec_max) != 0:
                    #     penalty += w*abs((spec_num - spec_max) / (spec_num + spec_max))
                    # else:
                    #     penalty += 1000
                    penalty += w * abs(spec_num - spec_max) / abs(spec_num)
                    # penalty += w * abs(spec_num - spec_max) / self.avg_specs[spec_kwrd]
            if spec_min is not None:
                if spec_num < spec_min:
                    # if (spec_num + spec_min) != 0:
                    #     penalty += w*abs((spec_num - spec_min) / (spec_num + spec_min))
                    # else:
                    #     penalty += 1000
                    penalty += w * abs(spec_num - spec_min) / abs(spec_min)
                    # penalty += w * abs(spec_num - spec_min) / self.avg_specs[spec_kwrd]
            penalties.append(penalty)
        return penalties

    def update_designs_with_results(self, designs: Sequence[Design],
                                    results: Sequence[Dict[str, Any]]) -> None:
        """
        Override this method to change the behavior of appending the results to the Design objects.
        This method updates the designs in-place.
        Parameters
        ----------
        designs: Sequence[Design]
            the sequence of designs
        results: Sequence[Dict[str, Any]]
            the sequence of dictionaries each representing the result of simulating designs in
            the order that was given

        Returns
        -------
            None
        """
        if len(designs) != len(results):
            raise ValueError('lengths do not match between the designs and the results')
        for design, result in zip(designs, results):
            try:
                for k, v in result.items():
                    design[k] = v
                design['valid'] = True
                design['id'] = design.id(self.id_encoder)
                self.post_process_design(design)
            except AttributeError:
                design['valid'] = False
                design['id'] = design.id(self.id_encoder)

    # noinspection PyMethodMayBeStatic
    def post_process_design(self, design: Design) -> None:
        """
        override this method to do post-processing of the design object. Use this function to
        compute cost function.
        Parameters
        ----------
        design: Design
            the Design object under consideration.

        Returns
        -------
        None
            This function should manipulate design object in-place.
        """
        cost = 0
        for spec_kwrd in self.spec_range:
            cost += self.compute_penalty(design[spec_kwrd], spec_kwrd)[0]
        design['cost'] = cost
