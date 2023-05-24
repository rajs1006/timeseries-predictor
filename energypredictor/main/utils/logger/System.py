import datetime
import logging
import os
import sys

from jsonformatter import JsonFormatter

from energypredictor.main.utils.common.Constants import Environment as envCons
from energypredictor.main.utils.decorator.Singleton import Singleton

"""
Converts the timestamp in UTC format in the logger file (time).
"""
CUSTOM_FORMAT = {"asctime": lambda: datetime.datetime.now()}
"""
Format of JSON logger used for logging purpose.
"""
STRING_FORMAT = """{
    "time":           "%(asctime)s",
    "name":            "name",
    "levelname":       "levelname",
    "message":         "message"
}"""


def logger(fileName: str) -> (logging.log):
    """
    Creates and returns Singleton instance of python.

    Arguments:
        name {str} -- name of the file from where the method hs been called.

    Returns:
        system logger : used to log the system logs, like time of methods,
                        some print statement inside a method etc.
    """
    class Logger:
        def __init__(self):
            ### Using streamhandler, it generates logs as standard error.
            ### for local python file there is no file handler.
            try:
                self.handler = logging.handlers.TimedRotatingFileHandler(
                    envCons.logFile, when='D', interval=1, backupCount=1)
            except:
                self.handler = logging.StreamHandler(stream=sys.stdout)

            self.handler.setFormatter(
                JsonFormatter(STRING_FORMAT,
                              record_custom_attrs=CUSTOM_FORMAT))

        def __call__(self, name):
            ## extracting name from file
            name = os.path.basename(name)

            log = logging.getLogger(name)
            log.setLevel(envCons.logLevel)
            log.addHandler(self.handler)

            return log

    ### Singleton instance of DashboardLogger, don't want to generate handlers again.
    logger = Singleton(Logger)()(fileName)

    return logger
