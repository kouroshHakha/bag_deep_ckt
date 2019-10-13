from typing import Optional, Dict, Any

from bb_eval_engine.circuits.bag_new.base import BagNetEngineBase
from bb_eval_engine.util.design import Design


class DTSAEvalEngine(BagNetEngineBase):

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
            design_specs[k] = int(v)
        dsn_id = design.id(self.id_encoder)
        base_name = self.specs['base_name']
        design_specs['impl_lib'] = f'{base_name}_{dsn_id}'
        design_specs['impl_cell'] = f'{base_name}'
        return design_specs
