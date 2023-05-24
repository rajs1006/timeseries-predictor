import os
from datetime import datetime, timedelta

import altair as alt
import pandas as pd
import streamlit as st

from energypredictor.main.build.Build import Model
from energypredictor.main.data.Energy import Energy
from energypredictor.main.data.Weather import Weather
from energypredictor.main.utils.common.Constants import Environment as envCons
from energypredictor.main.utils.decorator.Singleton import Singleton


class Streamlit:
    def __init__(self):
        self.model = Singleton(Model)()
        self.weather = Singleton(Weather)()
        self.energy = Singleton(Energy)()

    def input(self):
        """  
        Read and validate  the user input arguments and save them as instance variables;
        The input arguments are:
            - the train-to date (has to be the latest last month)
            - the number of steps in months to be predicted - warning if it is more than 18
            - a list of confidence intervals in %. alpha is then converted to the format accepted by the model
            - weather data fetching method - if self.weatherFetching is false, it will read the saved data
            - energy data fetching bool - if self.balancingFetching is false, it will read the saved data

        Returns:
            trainButton (bool): is true when user clicks on the train button
            predictButton (bool): is true when user clicks on the predict button
        """
        st.sidebar.title("Input")
        lastMonth = datetime.now().replace(day=1) - timedelta(days=1)
        userInput = st.sidebar.text_input(
            "Enter: mm/yyy,steps,space separated confidence in %, for example: 08/2020,3,95 90 85.",
            value="{},12,95 85".format(lastMonth.strftime("%m/%Y")),
            key="input1",
            type="default",
        )

        dateText, stepsText, alphaText = [x.strip() for x in userInput.split(",")]
        self.alpha = [(100 - float(i)) / 100 for i in alphaText.split(" ")]
        st.sidebar.text("---------------------------------------")
        if any((a > 1.0) | (a <= 0.0) for a in self.alpha):
            st.write("Error: confidency can only vary between 0 to 100")

        self.totalSteps = int(stepsText)
        if self.totalSteps > 18:
            st.write("Error: too many steps ahead")

        self.trainTo = pd.to_datetime(dateText, format="%m/%Y").replace(day=1)
        if self.trainTo > datetime.now():
            st.write("Error: cannot start the prediction in the future")

        manualMsg = "Use manually downloaded weather files"
        autoMsg = "Fetch weather data automatically from DWD"
        savedMsgWeather = "Use saved weather data without fetching again"
        savedMsgEnergy = "Use saved balancing data without fetching again"

        weatherSource = st.sidebar.radio(
            "", (manualMsg, autoMsg, savedMsgWeather), index=0
        )

        if weatherSource == manualMsg:
            self.manualWeatherDataFlag = True
            self.weatherFetching = True
        elif weatherSource == autoMsg:
            self.manualWeatherDataFlag = False
            self.weatherFetching = True
        else:
            self.manualWeatherDataFlag = False
            self.weatherFetching = False

        self.balancingFetching = not st.sidebar.checkbox(savedMsgEnergy, value=False)

        trainButton = st.sidebar.button("Train")
        st.sidebar.text("---------------------------------------")
        predictButton = st.sidebar.button("Predict")

        return trainButton, predictButton

    def load(self):
        """  
        Call the get methods of the weather and energy instances and save the results as dataframes to instance variables
        """
        self.energyData = self.energy.get(
            os.path.join(envCons.energyDataFolder, envCons.energyFile),
            envCons.granularity,
            envCons.externalBalancingFile,
            envCons.comercialConversionFile,
            save=self.balancingFetching,
        )

        self.weatherData = self.weather.get(
            os.path.join(envCons.weatherDataFolder, envCons.weatherFile),
            granularity=envCons.granularity,
            save=self.weatherFetching,
            useManual=self.manualWeatherDataFlag,
        )

    def train(self):
        """  
        Call the train method which trains until the user-defined trainTo month
        Plot the training data in the meantime
        """
        infoText = st.sidebar.text("Training in progress... it can take up a while(up to 2 hours)")
        st.markdown("### Model is getting trained using the following data ")

        self.energy.plot(st)
        self.weather.plot(st)

        trainedTime, trainFromValidate, trainToValidate = self.__train(self.trainTo)

        infoText.text(
            "Training done @{} on data from {} until {}".format(
                trainedTime, trainFromValidate, trainToValidate
            )
        )

    @st.cache
    def __train(self, trainTo):
        """  
        Call the model train method.

        Return:
            lastTrainedTime (datetime): time of training, used to display a message
        """
        lastTrainedTime = self.model.trainAndValidate(
            self.energyData, self.weatherData, st, sliceTime=trainTo
        )
        return lastTrainedTime

    def test(self):
        """  
        Call the test method which predicts totalSteps ahead for each alpha in the user-defined list
        dataDict contains the prediction results per each alpha
            key: alpha
            Item: boostedPrediction, prediction, lowerPrediction, upperPrediction
        Call the plotAndSave to plot on Streamlit and save the restults to csv
        """
        infoText = st.sidebar.text("Prediction in progress...")
        st.markdown("# Results")

        dataDict = {i: "" for i in self.alpha}
        for alpha in self.alpha:
            lastTrainedTime, lastTestTime, msg, dataDict[alpha] = self.__test(
                self.trainTo, self.totalSteps, alpha
            )

        self.model.plotAndSave(dataDict, self.alpha, st)

        infoText.text(
            " {} \n Prediction done @{} based on training done until date @{}".format(
                msg, lastTestTime, lastTrainedTime
            )
        )

    @st.cache
    def __test(self, trainTo, totalSteps, alpha):
        """  
        Call the model test method.

        Return:
            lastTrainedTime (datetime)   : time of training, used to display a message
            lastTestTime (datetime)      : time of prediction, used to display a message
            msg (str)                    : message to be displayed to the user in Streamlit
            data (list)                  : boostedPrediction, prediction, lowerPrediction, upperPrediction - 
                                            each of these dataframes contains month datetime index and endogenous 
                                            variables as columns

        """
        lastTrainedTime, lastSliceTime, msg, data = self.model.predictAndModel(
            sliceTime=trainTo, steps=totalSteps, alpha=alpha
        )
        lastTestTime = datetime.now().strftime("%d/%m/%Y, %H:%M:%S")

        return lastSliceTime, lastTestTime, msg, data
