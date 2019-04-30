import pprint
import os
import time
from shutil import copyfile
import pickle

class Logger:

    def __init__(self, log_path, time_stamped=True):
        if os.path.isfile(log_path):
            raise ValueError('{} is not a file path, please provide a directory path')

        self.log_path = os.path.abspath(log_path)
        if time_stamped:
            self.log_path = self.log_path + '_' + time.strftime("%d-%m-%Y_%H-%M-%S")

        os.makedirs(self.log_path, exist_ok=True)

        self.log_txt_fname = os.path.join(self.log_path, 'progress_log.txt')
        if os.path.exists(self.log_txt_fname):
            os.remove(self.log_txt_fname)

        self.log_db_fname = os.path.join(self.log_path, 'db.pkl')

        log_model_path = os.path.join(self.log_path, 'checkpoint')
        self.log_model_fname = os.path.join(log_model_path, 'checkpoint.ckpt')


    def log_text(self, str, stream_to_file=True, stream_to_stdout=True, pretty=False, fpath=None):
        if fpath:
            stream = open(fpath, 'a')
        else:
            stream = open(self.log_txt_fname, 'a')

        if pretty:
            printfn = pprint.pprint
        else:
            printfn = print

        if stream_to_file:
            printfn(str, file=stream)
        if stream_to_stdout:
            printfn(str)

        stream.close()


    def store_db(self, db, fpath=None):
        if fpath is None:
            fpath = self.log_db_fname

        with open(fpath, 'wb') as f:
            pickle.dump(db, f)


    def store_model(self, tf_saver, tf_session):
        tf_saver.save(tf_session, self.log_model_fname)

    def store_settings(self, agent_yaml, circuit_yaml):
        agent_fname = os.path.abspath(agent_yaml)
        circuit_fname = os.path.abspath(circuit_yaml)
        copyfile(agent_fname, os.path.join(self.log_path, 'agent.yaml'))
        copyfile(circuit_fname, os.path.join(self.log_path, 'circuit.yaml'))
