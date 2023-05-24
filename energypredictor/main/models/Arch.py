import bz2
import os
import pickle
import random
import sys
import warnings
from copy import deepcopy
from heapq import nsmallest
from types import SimpleNamespace

import numpy as np
import pandas as pd
from arch import arch_model

from energypredictor.main.utils.common.Constants import Resource as resCons
from energypredictor.main.utils.common.Utility import progressBar, stProgressBar
from energypredictor.main.utils.decorator.Log import time_log_this
from energypredictor.main.utils.logger.System import logger

warnings.filterwarnings("ignore")

log = logger(__file__)


class Arch:
    """
        Arch model is used to model the data and forecast the volatility of data in future.

        This model we are using to model the residual of SARIMA model prediction to boost 
        the predicted values.
    """

    def __init__(self):
        d = dict(
            distances=["skewt"],
            # distances=['Normal', 'skewt', 'studentst'],
            means=["AR", "HAR"],
            P=range(1, 5),
            Q=range(1, 5),
            minimal={"p1": 999, "p2": 999, "p3": 999},
        )
        self.args = SimpleNamespace(**d)
        self.residualMultiplier = 10

    @time_log_this
    def validate(self, trueVal, forecast, residual, exog, end_loc, steps):
        """
            Runs the Grid search validation for Arch model
        """

        parameters = {}
        errors = []
        count = 0

        for d in self.args.distances:
            for mn in self.args.means:
                lag = np.sort(random.sample(range(1, 30), 10))
                for r in range(5, 30, 5):
                    for l in range(1, 15):
                        lag = np.arange(1, r, l)
                        for p in self.args.P:
                            for q in self.args.Q:

                                forecasts = {}

                                m = arch_model(
                                    self.residualMultiplier * residual,
                                    x=exog,
                                    mean=mn,
                                    vol="Garch",
                                    p=p,
                                    o=0,
                                    q=q,
                                    dist=d,
                                    lags=lag,
                                )

                                for i in range(steps + 1):

                                    progressBar("iteration {}".format(count), i, steps)

                                    try:
                                        res = m.fit(
                                            first_obs=i,
                                            last_obs=i + end_loc,
                                            disp="off",
                                            show_warning=False,
                                        )
                                        temp = res.forecast(horizon=1).mean

                                        fcast = temp.iloc[i + end_loc - 1]
                                        forecasts[fcast.name] = fcast

                                    except:
                                        log.warn(
                                            "Arch model has executed with error for params = {}".format(
                                                {
                                                    "distance": d,
                                                    "mean": mn,
                                                    "P": p,
                                                    "Q": q,
                                                }
                                            )
                                        )

                                count += 1

                                fCast = pd.DataFrame(forecasts).T

                                error = round(
                                    np.sqrt(
                                        (
                                            (
                                                (
                                                    forecast
                                                    - np.divide(
                                                        fCast["h.1"][-steps:],
                                                        self.residualMultiplier,
                                                    )
                                                )
                                                - trueVal
                                            )
                                            ** 2
                                        ).mean()
                                    ),
                                    5,
                                )

                                errors.append(error)

                                if self.args.minimal["p3"] > error:
                                    if self.args.minimal["p2"] > error:
                                        if self.args.minimal["p1"] > error:
                                            self.args.minimal["p1"] = error
                                            parameters["p1"] = {
                                                "m": mn,
                                                "distance": d,
                                                "p": p,
                                                "q": q,
                                                "lag": lag,
                                                "error": error,
                                            }
                                            break
                                        self.args.minimal["p2"] = error
                                        parameters["p2"] = {
                                            "m": mn,
                                            "distance": d,
                                            "p": p,
                                            "q": q,
                                            "lag": lag,
                                            "error": error,
                                        }
                                        break
                                    self.args.minimal["p3"] = error
                                    parameters["p3"] = {
                                        "m": mn,
                                        "distance": d,
                                        "p": p,
                                        "q": q,
                                        "lag": lag,
                                        "error": error,
                                    }
                                    break

        log.debug(
            "Sarima error = {} : Arch errors = {}".format(
                np.sqrt((((forecast - trueVal) ** 2).mean())), self.args.minimal
            )
        )

        return parameters

    @time_log_this
    def build(self, residual, exog=None, parameters=None):
        """
            Builds the model with given parameters. These parameters are 
            decided by the validation method.
        """
        if parameters is not None:
            parameters = SimpleNamespace(**parameters["p1"])
        else:
            raise Exception(
                "Parameters file is not found, please pass the parameters in the arguments"
            )

        log.debug("Building the models with parameters = {}".format(parameters))

        self.model = arch_model(
            self.residualMultiplier * residual,
            # x=exog,
            mean=parameters.m,
            vol="Garch",
            p=parameters.p,
            o=0,
            q=parameters.q,
            dist=parameters.distance,
            lags=parameters.lag,
        )

        return self

    def fit(self, start_loc, end_loc=None):
        log.debug("Arch model fiting on dates {} = {}".format(start_loc, end_loc))
        res = self.model.fit(
            first_obs=start_loc, last_obs=end_loc, disp="off", show_warning=False
        )
        return res

    def predict(self, result, start_loc, alpha, horizon=1):
        """
            Forecasts the future mean of residual. By using confidence interval 
            VaR(values at risk) can be determined. 
        """
        forecast = result.forecast(horizon=horizon, start=start_loc)
        ## Get the last value of forecast of only 1st horizon (h.1)
        lastMeanForecast = forecast.mean.iloc[start_loc]
        lastCondVarForecast = forecast.variance.values[start_loc]

        log.debug(
            "Result param forecast for date {} is {}".format(
                start_loc, result.params[-2:]
            )
        )

        q = self.model.distribution.ppf(
            [(1 - (np.array(alpha) / 100))], result.params[-2:]
        )

        valueAtRiskLow = forecast.mean.values[-1, :] - np.squeeze(
            np.sqrt(lastCondVarForecast) * q[None, :]
        )
        valueAtRiskHigh = forecast.mean.values[-1, :] + np.squeeze(
            np.sqrt(lastCondVarForecast) * q[None, :]
        )

        log.debug(
            "Arch mean forecast for date {} is {}".format(start_loc, lastMeanForecast)
        )

        meanVal = np.divide(lastMeanForecast.to_numpy(), self.residualMultiplier)
        if type(meanVal) == np.ndarray:
            meanVal = meanVal.item()
            valueAtRiskLow = valueAtRiskLow.item()
            valueAtRiskHigh = valueAtRiskHigh.item()

        return meanVal, valueAtRiskLow, valueAtRiskHigh

    def save(self, parameters, folder, file):
        try:
            with open(
                os.path.join(
                    "" if folder is None else folder,
                    "{}{}".format(file, resCons.extensions.paramExt),
                ),
                "wb",
            ) as modelFile:
                pickle.dump(parameters, bz2.BZ2File(modelFile, "wb"))
        except:
            raise Exception("Could not save arch model, please try again")

    def load(self, folder, file):
        try:
            with open(
                os.path.join(
                    "" if folder is None else folder,
                    "{}{}".format(file, resCons.extensions.paramExt),
                ),
                "rb",
            ) as modelFile:
                parameters = pickle.load(bz2.BZ2File(modelFile, "rb"))
        except:
            raise Exception("Could not load arch model, please try again")
        return parameters

    def copy(self):
        return deepcopy(self)
