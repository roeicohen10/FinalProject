from Backend.OFS.alpha_investing import run_AI
from Backend.Streaming.pystreaming.algorithms.ofs import run_ofs

Algorithms = ['Alpha Investing']

class OFSAlgo():
    def __init__(self):
        pass

    @staticmethod
    def get_algo(name):
        if (name == 'Alpha Investing'):
            return run_AI, 'Online feature selection (OFS)'
        else:
            return run_ofs,'Online feature selection (OFS)'
