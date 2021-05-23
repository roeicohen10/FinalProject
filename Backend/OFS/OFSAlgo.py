from Backend.OFS.OSFSn import run_osfs, OSFS_PARAMS, FOSFS_PARAMS
from Backend.OFS.Saola import run_saola, SAOLA_PARAMS
from Backend.OFS.alpha_investing import run_AI, AI_PARAMS
from Backend.OFS.fires import init_fires,FIRES_PARAMS,run_fires

OFS_ALGO = [
    {
        "name": "-",
        "func": None,
        "params": None,
        "params_str":''
    },
    {
        "name": "SAOLA",
        "func": run_saola,
        "params": SAOLA_PARAMS,
        "params_str":'alpha=0.05'

    },
    {
        "name": "Alpha Investing",
        "func": run_AI,
        "params": AI_PARAMS,
        "params_str":'alpha=0.05, dw=0.05'
    },
    {
        "name": "OSFS",
        "func": run_osfs,
        "params": OSFS_PARAMS,
        "params_str":'alpha=0.05'
    },
    {
        "name": "Fast OSFS",
        "func": run_osfs,
        "params": FOSFS_PARAMS,
        "params_str":'alpha=0.05'
    }
    ,
    {
        "name": "FIRES",
        "func": run_fires,
        "params": {},
        "init_func": init_fires,
        "init_params": FIRES_PARAMS,
        "params_str":'lr_mu=0.01, lr_sigma=0.01'
    }
]