from Backend.OFS.OSFSn import run_osfs, OSFS_PARAMS, FOSFS_PARAMS
from Backend.OFS.Saola import run_saola, SAOLA_PARAMS
from Backend.OFS.alpha_investing import run_AI, AI_PARAMS
from Backend.OFS.fires import init_fires,FIRES_PARAMS

OFS_ALGO = [
    {
        "name": "-",
        "func": None,
        "params": None

    },
    {
        "name": "SAOLA",
        "func": run_saola,
        "params": SAOLA_PARAMS,

    },
    {
        "name": "Alpha Investing",
        "func": run_AI,
        "params": AI_PARAMS
    },
    {
        "name": "OSFS",
        "func": run_osfs,
        "params": OSFS_PARAMS
    }, {
        "name": "Fast OSFS",
        "func": run_osfs,
        "params": FOSFS_PARAMS
    }
    ,
    # {
    #     "name": "Fires",
    #     "func": init_fires,
    #     "params": FIRES_PARAMS
    # }
]