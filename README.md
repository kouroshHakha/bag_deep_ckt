# bag_deep_ckt
genetic and neural net optimization for circuit design

Because of lack of time and avoiding messing up on-going collaborations at the moment, the master branch does not have the most up-to-date software structure. The refactored (and more up-to-date version) is on the branch **kourosh**.

NGspice:

NGspice 27 should be installed on the system, this code was developed on a linux machine but also worked on macos.

Models need to be added in eval_engines/NGspice/ngspice_inputs/spice_models and then run:
python correct_inputs.py

This will make all the netlist files point to the correct 45nm model file which is located in eval_engines/NGspice/ngspice_inputs/spice_models/45nm_bulk.txt

make sure in the yaml file, the .cir files are pointing to the correct files.
