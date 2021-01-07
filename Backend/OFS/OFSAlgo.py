from Backend.OFS.alpha_investing import run_AI
from pystreamfs.algorithms import ofs

Algorithms = ['Alpha Investing']

class OFSAlgo():
    def __init__(self):
        pass

    @staticmethod
    def get_algo(name):
        if (name == 'Alpha Investing'):
            return run_AI, 'Online feature selection (OFS)'
        else:
            return ofs.run_ofs,'Online feature selection (OFS)'
