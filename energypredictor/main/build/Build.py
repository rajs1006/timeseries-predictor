import bz2
import os
import pickle
from datetime import datetime

import altair as alt
import numpy as np
import pandas as pd
from dateutil.relativedelta import *

from energypredictor.main.models import Arch, Sarima, Scalar
from energypredictor.main.utils.common.Constants import Environment as envCons
from energypredictor.main.utils.common.Constants import Resource as resCons
from energypredictor.main.utils.decorator.Log import time_log_this
from energypredictor.main.utils.logger.System import logger
from energypredictor.main.utils.Transformation import alignDates
from joblib import Parallel, delayed

log = logger(__file__)


class Model:
    """
    [summary]
    """

    def __init__(self):

        self.scalar = Scalar.Scalar()

        ## Models.
        self.sarima = Sarima.Sarima()
        self.arch = Arch.Arch()

    @time_log_this
    def trainAndValidate(
        self, energyData, weatherData, st, sliceTime, validationSlice: int = 3
    ):
        """
        
        """
        ######################## Prepare frames ################################
        ## Slice time represents the last time stamp of training data set
        # and start time for prediction, getting the minimum of valid index
        # of both weather and energy data will make sure that index is in sync.
        trainSliceTime = (
            sliceTime
            if sliceTime
            else min(energyData.last_valid_index(), weatherData.last_valid_index())
        )

        trainSliceTime = trainSliceTime + relativedelta(months=+1)

        energyDataTrain, weatherDataTrain = energyData, weatherData = self.__prepare(
            energyData, weatherData, sliceTime=trainSliceTime
        )

        trainFromValidate = weatherDataTrain.index[0]
        trainToValidate = weatherDataTrain.index[-1]

        weatherDataValidate = weatherData.iloc[-validationSlice:]
        energyDataValidate = energyData.iloc[-validationSlice:]

        ######################## Build models ################################
        Parallel(n_jobs=envCons.coreProcess)(
            delayed(Model.__weatherTrainValidate)(
                i, columnName, columnData, self.sarima.copy()
            )
            for i, (columnName, columnData) in enumerate(weatherDataTrain.iteritems())
        )

        validationStart = len(energyDataTrain) - validationSlice

        Parallel(n_jobs=envCons.coreProcess)(
            delayed(Model.__energyTrainValidate)(
                i,
                columnName,
                columnData,
                self.sarima.copy(),
                self.arch.copy(),
                weatherDataTrain,
                weatherDataValidate,
                energyDataValidate,
                validationStart,
                validationSlice,
            )
            for i, (columnName, columnData) in enumerate(energyDataTrain.iteritems())
        )

        ## Save column names to load them while predicting to ensure only trained models are tested
        self.save(weatherDataTrain.columns, envCons.paramFolder, envCons.weatherFile)
        self.save(energyDataTrain.columns, envCons.paramFolder, envCons.energyFile)

        ## Save the last trained time in file
        trainingTime = datetime.now().strftime("%d/%m/%Y, %H:%M:%S")
        self.save(
            [trainingTime, trainSliceTime], envCons.paramFolder, envCons.trainFile
        )

        return trainingTime, trainFromValidate, trainToValidate

    @time_log_this
    def predictAndModel(
        self, sliceTime, steps, alpha, model_residual=True, model_VaR=False
    ):
        """
            To predict the values using Sarima and then model the residual using Arch

            Args:
                sliceTime (datetime): day, from which the prediction needs to be run
                steps (int): No. of future steps for prediction
                alpha (list): list of confidence interval
                model_residual (bool, optional): whether to model the residual or not (whether to run Arch). 
                                                    Defaults to True.
                model_VaR (bool, optional): whether to model the confindece interval by Arch. Defaults to False.

            Returns:
                tuple: final predictio , upper bound, lower bound
        """
        ## Load parameters
        (
            energyColumns,
            weatherColumns,
            lastSliceTime,
            lastTrainedTime,
            msg,
        ) = self.__loadParameters(sliceTime)

        ## Sarima results
        predicted, predictedLower, predictedUpper = self.__predict(
            energyColumns, weatherColumns, lastSliceTime, steps, alpha
        )

        ## Arch results
        residual, VaRLow, VaRHigh = self.__model(
            energyColumns, lastSliceTime, alpha, steps
        )

        if model_residual:
            prediction = self.__inverseScale(
                predicted - residual, envCons.energyFile, "prediction"
            )
            prediction = self.__truncate(prediction)
        else:
            prediction = self.__inverseScale(
                predicted, envCons.energyFile, "prediction"
            )
            prediction = self.__truncate(prediction)

        ## Only run arch model if residual modeling is needed
        if model_VaR:
            lowerPrediction = self.__inverseScale(
                predicted - VaRLow, envCons.energyFile, "lowerPrediction"
            )
            lowerPrediction = self.__truncate(lowerPrediction)

            upperPrediction = self.__inverseScale(
                predicted - VaRHigh, envCons.energyFile, "upperPrediction"
            )
            upperPrediction = self.__truncate(upperPrediction)
        else:
            lowerPrediction = self.__inverseScale(
                predictedLower, envCons.energyFile, "lowerPrediction"
            )
            lowerPrediction = self.__truncate(lowerPrediction)

            upperPrediction = self.__inverseScale(
                predictedUpper, envCons.energyFile, "upperPrediction"
            )
            upperPrediction = self.__truncate(upperPrediction)

        return (
            lastTrainedTime,
            lastSliceTime,
            msg,
            (prediction, lowerPrediction, upperPrediction),
        )

    def plotAndSave(self, dataDict, alphaList, st):
        """
            Plot the results and save in a csv file. 

            Args:
                dataDict: dictionnary containg boostedPrediction, prediction, lowerPrediction,
                            upperPrediction for each alpha in alphaList
                alphaList: a list of user-inputed alphas
                st: running streamlit object
        """
        if envCons.resultScaling == "1":
            yTitle = "Balancing energy (MWh)"
        elif envCons.resultScaling == "1000":
            yTitle = "Balancing energy (GWh)"
        elif envCons.resultScaling == "1000000":
            yTitle = "Balancing energy (TWh)"
        else:
            raise Exception("Unkown results scaling")

        prediction, lowerPrediction, upperPrediction = dataDict[alphaList[0]]

        for column in prediction:
            ## Create source ##
            dataSource = pd.DataFrame(index=prediction.index)
            dataSource["Forecast"] = prediction[column].div(
                float(envCons.resultScaling)
            )

            for alpha in alphaList:
                prediction, lowerPrediction, upperPrediction = dataDict[alpha]
                colNameLower = "lowerLimit_{}".format((1 - alpha) * 100)
                colNameUpper = "upperLimit_{}".format((1 - alpha) * 100)
                dataSource[colNameLower] = lowerPrediction[column].div(
                    float(envCons.resultScaling)
                )
                dataSource[colNameUpper] = upperPrediction[column].div(
                    float(envCons.resultScaling)
                )

            st.markdown("## {}".format(column))
            data = dataSource.reset_index().melt("index")
            line_chart = (
                alt.Chart(data)
                .mark_line()
                .encode(
                    x=alt.X("index", axis=alt.Axis(title="")),
                    y=alt.Y("value", axis=alt.Axis(title=yTitle)),
                    color="variable",
                )
                .properties(width=700, height=400)
            )

            st.altair_chart(line_chart)
            st.markdown("Total predicted sum")
            st.table(dataSource.sum(axis=0, skipna=True))
            st.markdown("Monthly prediction")
            st.table(dataSource)
            self.__saveResults(dataSource, column)

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
            raise Exception("Could not save model, please try again")

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
            raise Exception("Could not load model, please try again")
        return parameters

    ########################### Private methods #########################################

    def __loadParameters(self, sliceTime):
        ## Load columns for energy trained data from parameter file
        energyColumns = self.load(envCons.paramFolder, envCons.energyFile)
        weatherColumns = self.load(envCons.paramFolder, envCons.weatherFile)

        lastTrainedTime, trainSlicedTime = self.load(
            envCons.paramFolder, envCons.trainFile
        )

        msg = "Values predicted from {}".format(trainSlicedTime)

        return energyColumns, weatherColumns, trainSlicedTime, lastTrainedTime, msg

    def __predict(self, energyColumns, weatherColumns, sliceTime, steps, alpha):

        ######################## Walk forward prediction ################################
        forecastIndex = pd.date_range(start=sliceTime, periods=steps, freq="MS")

        ######################## Predict step ################################
        weatherPredictions = Parallel(n_jobs=envCons.coreProcess)(
            delayed(Model.__weatherPredict)(columnName, self.sarima, steps, sliceTime)
            for columnName in weatherColumns
        )

        weatherForecast = self.__weatherForecast(weatherPredictions, forecastIndex)

        energyPredictions = Parallel(n_jobs=envCons.coreProcess)(
            delayed(Model.__energyPredict)(
                columnName, weatherForecast, self.sarima, alpha, steps, sliceTime
            )
            for columnName in energyColumns
        )

        (
            energyForecast,
            energyForecastLower,
            energyForecastUpper,
        ) = self.__energyForecast(energyPredictions, forecastIndex)

        return energyForecast, energyForecastLower, energyForecastUpper

    def __model(self, energyColumns, sliceTime, alpha, steps):
        """
            Models the residual after the prediction of values using ARCH Model
        """
        residualIndex = pd.date_range(start=sliceTime, periods=steps, freq="MS")

        modeledResidual = Parallel(n_jobs=envCons.coreProcess)(
            delayed(Model.__energyModel)(
                columnName, self.sarima, self.arch, alpha, steps
            )
            for columnName in energyColumns
        )

        residualForecast, residualVaRLow, residualVaRHigh = self.__residualForecast(
            modeledResidual, residualIndex
        )
        return residualForecast, residualVaRLow, residualVaRHigh

    def __prepare(self, energyData, weatherData, sliceTime):
        """
            Call alignDates to make sure weather and energy span the same period from start to sliceTime
            Scale using a standard scaler - results will be inverse scaled before being displayed
        """
        energyData = energyData[:sliceTime]
        weatherData = weatherData[:sliceTime]

        energyDataAligned, weatherDataAligned = alignDates(energyData, weatherData)
        energyDataScaled = self.__scale(energyDataAligned, envCons.energyFile)
        weatherDataScaled = self.__scale(weatherDataAligned, envCons.weatherFile)

        return energyDataScaled, weatherDataScaled

    def __scale(self, data, file):
        scalar = self.scalar.copy().fit(data.values)
        scalar.save(envCons.scalarFolder, file)

        scaled = scalar.scale(data.values)
        scaled_df = pd.DataFrame(data=scaled, index=data.index, columns=data.columns)
        return scaled_df

    def __inverseScale(self, data, file, dfName):
        scalar = self.scalar.load(envCons.scalarFolder, file)

        unscaled = scalar.inverseScale(data.values)
        unscaled_df = pd.DataFrame(
            data=unscaled, index=data.index, columns=data.columns
        )
        unscaled_df.name = dfName

        return unscaled_df

    def __truncate(self, df):
        """
            Clip the results at 0 since negative balancing does not make physical sense
        """
        df[df < 0] = 0
        return df

    ### Static Methods ###

    ####################### Training method with parallelization #####################
    @staticmethod
    def __weatherTrainValidate(i, columnName, columnData, sarima):

        sarimaModel = sarima.build(columnData)
        sarimaModel.save(envCons.weatherModelFolder, columnName)

    @staticmethod
    def __energyTrainValidate(
        i,
        columnName,
        columnData,
        sarima,
        arch,
        weatherData,
        weatherDataValidate,
        energyDataValidate,
        validationStart,
        validationSlice,
    ):

        st = datetime.now()
        sarimaModel = sarima.build(columnData, exog=weatherData)
        sarimaModel.save(envCons.energySarimaModelFolder, columnName)

        residual = sarimaModel.getResidual()
        log.debug("Residual = {}".format(len(residual)))

        predicted = sarimaModel.predictInSample(
            exog=weatherDataValidate, start=validationStart
        )

        parameters = arch.validate(
            energyDataValidate[columnName],
            predicted,
            residual,
            weatherData,
            validationStart,
            validationSlice,
        )

        ## Save arch paramneters
        arch.save(parameters, envCons.energyArchModelFolder, columnName)

    ####################### Prediction method with parallelization #####################
    @staticmethod
    def __weatherPredict(columnName, sarima, steps, sliceTime):
        ## Load weather model from file system
        weatherModel = sarima.load(envCons.weatherModelFolder, columnName)

        ## Start predict at 1 month ahead of training end
        predictAt = sliceTime
        predictions = []

        for s in range(0, steps):
            ## First fit and then predict the model in step wise
            predicted = weatherModel.predict()
            weatherModel.update(predicted)

            predictions.append(predicted)
            ## save it to forcasetd value and in the last of step loop, append it to training data

            if s == steps - 1:
                ## Save the final models, as it is needed for residual modeling
                weatherModel.save(
                    envCons.weatherModelFolder, "{}_Online".format(columnName)
                )

            predictAt = sliceTime + relativedelta(months=+(s + 1))

        return {columnName: predictions}

    @staticmethod
    def __energyPredict(columnName, weatherForecast, sarima, alpha, steps, sliceTime):
        ## Load sarima model from file
        energyModel = sarima.load(envCons.energySarimaModelFolder, columnName)

        ## Start predict at 1 month ahead of training end
        predictAt = sliceTime

        predictions = []
        predictionLower = []
        predictionUpper = []

        for s in range(0, steps):
            ## First fit and then predict the model in step wise
            predicted, predictedLower, predictedUpper = energyModel.predict(
                exog=pd.DataFrame(weatherForecast.loc[predictAt]).T,
                return_conf_int=True,
                alpha=alpha,
            )

            energyModel.update(
                predicted, exog=pd.DataFrame(weatherForecast.loc[predictAt]).T
            )

            predictions.append(predicted)
            predictionLower.append(predictedLower)
            predictionUpper.append(predictedUpper)

            if s == steps - 1:
                ## Save the final models, as it is needed for residual modeling
                energyModel.save(
                    envCons.energySarimaModelFolder, "{}_Online".format(columnName)
                )

            predictAt = sliceTime + relativedelta(months=+(1 + s))

        return (
            pd.Series(predictions, name=columnName),
            pd.Series(predictionLower, name=columnName),
            pd.Series(predictionUpper, name=columnName),
        )

    @staticmethod
    def __energyModel(columnName, sarima, arch, alpha, steps):
        energyModel = sarima.load(
            envCons.energySarimaModelFolder, "{}_Online".format(columnName)
        )

        residual = energyModel.getResidual()

        ## Arch, model the residual
        archParams = arch.load(envCons.energyArchModelFolder, columnName)
        log.debug("Residual = {} : Parameters = {}".format(len(residual), archParams))

        end_loc = len(residual) - steps
        arch = arch.build(residual, None, parameters=archParams)

        residuals = []
        varLow = []
        varHigh = []

        for s in range(0, steps):
            ## First fit and then predict the model in step wise, as it is already
            # trained on the last date of training dataset
            archRes = arch.fit(start_loc=s, end_loc=s + end_loc)
            modeledResidual, modeledVaRLow, modeledVaRHigh = arch.predict(
                archRes, start_loc=s + end_loc - 1, alpha=alpha
            )
            residuals.append(modeledResidual)
            varLow.append(modeledVaRLow)
            varHigh.append(modeledVaRHigh)

        return (
            pd.Series(residuals, name=columnName),
            pd.Series(varLow, name=columnName),
            pd.Series(varHigh, name=columnName),
        )

    @staticmethod
    def __weatherForecast(weatherPredictions, forecastIndex):

        weatherForecast = pd.DataFrame(
            {k: v for d in weatherPredictions for k, v in d.items()},
            index=forecastIndex,
        )

        return weatherForecast

    @staticmethod
    def __energyForecast(energyPredictions, forecastIndex):

        prediction, predictionLower, predictionUpper = [], [], []
        for e in energyPredictions:
            prediction.append(e[0])
            predictionLower.append(e[1])
            predictionUpper.append(e[2])

        energyForecast = pd.DataFrame(prediction).T.set_index(forecastIndex)
        energyForecastLower = pd.DataFrame(predictionLower).T.set_index(forecastIndex)
        energyForecastUpper = pd.DataFrame(predictionUpper).T.set_index(forecastIndex)

        return energyForecast, energyForecastLower, energyForecastUpper

    @staticmethod
    def __residualForecast(modeledResidual, residualIndex):

        residual, varLow, varHigh = [], [], []
        for e in modeledResidual:
            residual.append(e[0])
            varLow.append(e[1])
            varHigh.append(e[2])

        residualForecast = pd.DataFrame(residual).T.set_index(residualIndex)
        residualVaRLow = pd.DataFrame(varLow).T.set_index(residualIndex)
        residualVaRHigh = pd.DataFrame(varHigh).T.set_index(residualIndex)

        return residualForecast, residualVaRLow, residualVaRHigh

    @staticmethod
    def __saveResults(df, name):
        df.to_csv(os.path.join(envCons.resultFolder, name + resCons.extensions.csvExt))

