import os
import sys

import streamlit.cli as stcli

from energypredictor.main.utils.common.Constants import Environment as envCons
from energypredictor.main.utils.common.Constants import args


def main():
    
    sys.argv = [
        "streamlit", "run", "{}/main/Dashboard.py".format(envCons.app), "--",
        "--env={}".format(args.env)
    ]
    sys.exit(stcli.main())


if __name__ == "__main__":
    main()
