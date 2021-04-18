from Backend.OFS.alpha_investing import run_AI
from Backend.Streaming.pystreaming.algorithms.ofs import run_ofs
from Backend.OFS.Saola import run_saola
from Backend.OFS.OSFS import run_SOSFS


Algorithms = ['Alpha Investing','Saola','OSFS','FOSFS','OFS']

class OFSAlgo():
    def __init__(self):
        pass

    @staticmethod
    def get_algo(name):
        if name == 'Alpha Investing':
            return run_AI, 'Alpha Investing'
        elif name == 'Saola':
            return run_saola, 'Saola'
        elif name == 'OSFS':
            return run_SOSFS, 'OSFS'
        elif name == 'FOSFS':
            return run_SOSFS, 'FOSFS'
        else:
            return run_ofs,'Online feature selection (OFS)'
