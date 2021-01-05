from pystreamfs.algorithms import ofs,efs,cancelout,mcnn,fsds
from OFS.alpha_investing import run_AI

class OFSAlgo():
    def __init__(self):
        pass

    @staticmethod
    def get_algo():
        # return ofs.run_ofs,'Online feature selection (OFS)'
        return run_AI, 'Online feature selection (OFS)'