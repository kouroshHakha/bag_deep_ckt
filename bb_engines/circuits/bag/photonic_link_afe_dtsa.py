from typing import Optional, Dict, Any

from bb_eval_engine.circuits.bag_new.base import BagNetEngineBase
from bb_eval_engine.util.design import Design
import re


class PhotonicLinkEvalEngine(BagNetEngineBase):

    def __init__(self, yaml_fname: Optional[str] = None,
                 specs: Optional[Dict[str, Any]] = None, **kwargs) -> None:
        BagNetEngineBase.__init__(self, yaml_fname, specs, **kwargs)

    def interpret(self, design: Design) -> Dict[str, Any]:
        """
        This implementation overrides the inherited behavior by first running its parents'
        implementation and then converting design params to integer.
        """
        design_specs = BagNetEngineBase.interpret(self, design)
        for k, v in design_specs.items():
            match_seg_w = re.match(r'\w*_seg_\w*', k) or re.match(r'\w*_w_\w*', k)
            match_nser_npar = re.match(r'\w*_npar', k) or re.match(r'\w*_nser', k)
            if match_seg_w or match_nser_npar:
                design_specs[k] = int(v)
        dsn_id = design.id(self.id_encoder)
        base_name = self.specs['base_name']
        design_specs['impl_lib'] = f'{base_name}_{dsn_id}'
        design_specs['impl_cell'] = f'{base_name}'

        # impose some constraints for the layout
        # limit the aspect ratio of caps in ctle to 3
        aratio = self.specs['cap_aratio']
        cap_w = design_specs['ctle_cap_w']
        cap_h = design_specs['ctle_cap_h']

        if (cap_w / cap_h) > aratio:
            cap_w = aratio * cap_h
        elif (cap_h / cap_w) > aratio:
            cap_h = aratio * cap_w

        design_specs['ctle_cap_w'] = cap_w
        design_specs['ctle_cap_h'] = cap_h

        return design_specs
