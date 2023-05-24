import bz2
import math
import os
import pickle
from copy import deepcopy

import numpy as np
import pandas as pd
import statsmodels.api as sm
from pmdarima import auto_arima
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.stattools import adfuller, kpss

from energypredictor.main.utils.common.Constants import Resource as resCons
from energypredictor.main.utils.decorator.Log import time_log_this
from energypredictor.main.utils.logger.System import logger

log = logger(__file__)


class Sarima:
    def __init__(self):
        self.trainArgs = dict(
            start_p=0,
            start_q=0,
            max_p=2,
            max_q=2,
            m=12,
            seasonal=True,
            d=0,
            trace=False,
            D=1,
            error_action="ignore",
            suppress_warnings=True,
            stepwise=True,
        )

        self.testArgs = dict(return_conf_int=True)

    @time_log_this
    def build(self, data, exog=None):
        self.trainArgs["exogenous"] = None
        if exog is not None:
            self.trainArgs["exogenous"] = exog

        self.model = auto_arima(data, **self.trainArgs)

        return self

    def update(self, data, exog=None):
        self.model.update(data, exogenous=exog)

    def predict(self, exog=None, steps=1, alpha=0.05, return_conf_int=False):
        self.trainArgs["exogenous"] = None
        if exog is not None:
            self.testArgs["exogenous"] = exog

        self.testArgs["return_conf_int"] = True
        self.testArgs["alpha"] = alpha

        if return_conf_int:
            forecast, confidenceInterval = self.model.predict(steps, **self.testArgs)
            return forecast[0], confidenceInterval[0][0], confidenceInterval[0][1]
        else:
            forecast = self.model.predict(steps, **self.testArgs)
            return forecast[0]

        log.debug("AutoArima one step ahead prediction = {}".format(forecast))

    def predictInSample(self, exog=None, start=None, end=None):
        return self.model.predict_in_sample(exogenous=exog, start=start, end=end)

    def getResidual(self):
        return self.model.resid()

    def save(self, folder, file):
        try:
            with open(
                os.path.join(
                    "" if folder is None else folder,
                    "{}{}".format(file, resCons.extensions.modelExt),
                ),
                "wb",
            ) as modelFile:
                pickle.dump(self, bz2.BZ2File(modelFile, "wb"))
        except:
            raise Exception("Could not save Sarima model, please try again")

    def load(self, folder, file):
        try:
            with open(
                os.path.join(
                    "" if folder is None else folder,
                    "{}{}".format(file, resCons.extensions.modelExt),
                ),
                "rb",
            ) as modelFile:
                model = pickle.load(bz2.BZ2File(modelFile, "rb"))
        except:
            raise Exception("Could not load Sarima model, please try again")
        return model

    def copy(self):
        return deepcopy(self)
