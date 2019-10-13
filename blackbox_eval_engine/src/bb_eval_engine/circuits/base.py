from typing import Optional, Dict, Any, Union, List, Sequence

import abc
import numpy as np
import random

from ..base import EvaluationEngineBase
from ..util.design import Design
from .util.id import IDEncoder


SpecType = Union[float, int]
SpecSeqType = Union[Sequence[SpecType], SpecType]


class CircuitsEngineBase(EvaluationEngineBase, abc.ABC):

    def __init__(self, yaml_fname: Optional[str] = None,
                 specs: Optional[Dict[str, Any]] = None, **kwargs) -> None:
        EvaluationEngineBase.__init__(self, yaml_fname, specs, **kwargs)

        self.spec_range = specs['spec_range']
        self.params_vec = {}
        self.search_space_size = 1
        for key, value in self.specs['params'].items():
            listed_value = np.arange(value[0], value[1], value[2]).tolist()
            self.params_vec[key] = listed_value
            self.search_space_size = self.search_space_size * len(list(listed_value))

        self.id_encoder = IDEncoder(self.params_vec)

    @staticmethod
    def set_seed(seed):
        random.seed(seed)
        np.random.seed(seed)

    def get_rand_sample(self):
        """
        override this method to change the meaning of each design value
        """
        design_list = []
        for key, vec in self.params_vec.items():
            rand_idx = random.randrange(len(list(vec)))
            design_list.append(rand_idx)
        attrs = self.spec_range.keys()
        return Design(design_list, attrs)

    def interpret(self, design: Design) -> Dict[str, Any]:
        """
        override this method to change the interpretation of each design parameter
        Parameters
        ----------
        design: Design
            design object under consideration
        Returns
        -------
        values: Dict[str, Any]
            a dictionary representing the values of design parameters
        """
        param_values = {}
        for param_idx, key in zip(design['value'], self.params_vec.keys()):
            param_values[key] = self.params_vec[key][param_idx]
        return param_values

    @abc.abstractmethod
    def compute_penalty(self, spec_nums: SpecSeqType, spec_kwrd: str) -> SpecSeqType:
        """
        implement this method to compute the penalty(s) of a given spec key word based on the
        what the provided numbers for that specification.
        Parameters
        ----------
        spec_nums: SpecSeqType
            Either a single number or a sequence of numbers for a given specification.
        spec_kwrd: str
            The keyword of the specification of interest.

        Returns
        -------
            Either a single number or a sequence of numbers representing the penalty number for
            that specification
        """
        raise NotImplementedError
