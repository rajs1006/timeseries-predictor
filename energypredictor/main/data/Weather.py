import os
import urllib.request
from io import BytesIO
from zipfile import ZipFile

import dask.dataframe as dd
import numpy as np
import pandas as pd
import requests
from bs4 import BeautifulSoup

from energypredictor.main.utils.common.Constants import Environment as envCons
from energypredictor.main.utils.common.Constants import Resource as resCons


class Weather:
    """  
    A class to encapsulate all methods for weather data fetching and preparation.
    """

    def get(
        self,
        weatherFile: str,
        granularity: str,
        save: bool = True,
        useManual: bool = True,
    ):
        """
        Fetch and transform weather data according to the user input. The fetching options are:
            - Automatically fetch and transform all available on DWD open data server
            - Load and transform the manually downloaded csv files from DWD which have been placed in a designated folder (see config.env) in the correct format
            - Use the data peviously fetched and saved in a pickle

        Args:
            weatherFile (str): path to where the weather data is saved once collected and prepared
            granularity (str): model granularity. For now, only 'm' (month) is supported
            save (bool): if true collect and save data, if false use the data saved from a previous run
            useManual (bool): if true, use the manually downloaded weather csv's which are saved in a designated folder, 
                if false collect data from DWD automatically
        
        Returns:
            weatherData (dataframe): Dataframe with monthly datetime index and weather variables as columns
        """
        weatherFile = weatherFile + resCons.extensions.dataExt

        ## if data is to be collected
        if save:
            ## if data is already downloaded and saved in the correct format in the designed folder
            if useManual:
                dataFrames = None
                for root, directories, files in os.walk(
                    envCons.manualWeatherFolder, topdown=False
                ):
                    dataFrames = [
                        self.__readManualWeatherFiles(
                            os.path.join(envCons.manualWeatherFolder, file)
                        )
                        for file in files
                        if file.endswith(".csv")
                    ]
                    break
                if dataFrames:
                    weatherData = pd.concat(dataFrames, axis=1)
                else:
                    raise Exception(
                        "Something went wrong with the files, please check 'yaml' file for configuration"
                    )
                self.weatherData = self.__transformManual(weatherData)

            ## if data needs to be fetched automatically from DWD's server.
            # Note that data is fetched from the 'recent' url and 'historical' url and then appended together.
            # There is no sharp date separating recent from historical, that is why fetching only 'recent' data will lead to errors and inconsistent data.
            else:
                if granularity == resCons.common.DAILY_RESAMPLE:
                    raise Exception("Daily granularity is not yet supported, WIP !!")

                elif granularity == resCons.common.MONTHLY_RESAMPLE:
                    weatherDataRecent = self.__getMonthly(envCons.urlMonthlyRecent)
                    weatherDataHistorical = self.__getMonthly(
                        envCons.urlMonthlyHistorical
                    )
                    weatherData = weatherDataRecent.append(
                        weatherDataHistorical, ignore_index=False, sort=True
                    )
                    suffix = resCons.commonWeather.WEATHER_MONTHLY_SUFFIX

                else:
                    raise Exception("'granularity' is not in ['m', 'd']")

                self.weatherData = self.__transformAuto(weatherData)

        ## if train using weather data previously saved and do not collect again
        else:
            if granularity == resCons.common.MONTHLY_RESAMPLE:
                suffix = resCons.commonWeather.WEATHER_MONTHLY_SUFFIX
            elif granularity == resCons.common.DAILY_RESAMPLE:
                raise Exception("Daily granularity is not yet supported, WIP !!")
            else:
                raise Exception("'granularity' is not in ['m', 'd']")

            self.weatherData = self.__getData(weatherFile)

        ## save compressed data files
        self.weatherData.to_pickle(weatherFile, compression="bz2")

        return self.weatherData

    def plot(self, st):
        """
        Plot the weather data in Streamlit while the model in training

        Args:
            st (streamlit object): the streamlit object
        """
        for column, columnData in self.weatherData.iteritems():
            st.line_chart(columnData)

    def __transformManual(self, weatherData):
        """
        Transform the data manually downloaded from DWD's website

        Args:
            weatherData (dataframe): weather data which has been read from the csv files and saved into a dataframe
        
        Returns:
            weatherData (dataframe): Dataframe with datetime index and monthly groupby mean weather variables as columns
        """
        weatherData.reset_index(
            level=[resCons.weatherManual.stationIdChanged], drop=True, inplace=True
        )
        weatherData.rename(
            columns=resCons.weatherManual.variablesReq,  ## add more if necessary
            inplace=True,
        )

        weatherDataAvg = weatherData.groupby(
            level=resCons.weatherManual.monthIdxChanged
        ).mean()
        weatherDataAvg = weatherDataAvg[
            list(resCons.weatherManual.variablesReq.values())
        ]

        return weatherDataAvg

    def __transformAuto(self, weatherData):
        """
        Transform the data automatically fetched from DWD's open data server

        Args:
            weatherData (dataframe): data read from DWD's server saved in a dataframe
        
        Returns:
            weatherData (dataframe): Dataframe with datetime index and monthly groupby mean weather variables as columns
        """
        weatherData[resCons.weatherAuto.indexColumn] = pd.to_datetime(
            weatherData[resCons.weatherAuto.indexColumn], format="%Y%m%d"
        )
        weatherData.set_index(resCons.weatherAuto.indexColumn, inplace=True, drop=True)
        weatherData = weatherData.groupby(level="Date").mean()
        weatherData = weatherData.loc[
            weatherData.index >= pd.to_datetime("2011-01-01")
        ]  # very old data is causing issues

        return weatherData

    def __getData(self, file):
        """  
        Load previously prepared and saved weather data if save is false

        Return:
            dataframe (dataframe): loaded weather data 
        """
        dataframe = pd.read_pickle(file, compression="bz2")
        return dataframe

    def __getMonthly(self, url):
        """  
        Call the method __readDwdFiles that fetches data from DWD's open data server. 
        The DWD-specific title of the date column is different for monthly and daily data - hence the separation into __getMonthly and __getDaily
        Return:
            Dataframe of data fetched from and "monthly" recent or historical DWD url. 
        """
        return self.__readDwdFiles(
            url=url,
            dateColumn=resCons.weatherAuto.dateColumnMonthly,
            variables=resCons.weatherAuto.variablesMonthly,
        )

    def __getDaily(self, url):
        """  
        Call the method __readDwdFiles that fetches data from DWD's open data server. 
        The DWD-specific title of the date column is different for monthly and daily data - hence the separation into __getMonthly and __getDaily
        Return:
            Dataframe of data fetched from and "daily" recent or historical DWD url
        """
        return self.__readDwdFiles(
            url=url,
            dateColumn=resCons.weatherAuto.dateColumnDaily,
            variables=resCons.weatherAuto.variablesDaily,
        )

    def __readDwdFiles(self, url, dateColumn, variables):
        """  
        Fetch all the data from the files found in the given DWD url and transform it to a dataframe
        Args:
            url (str): from which DWD url to fetch data - there is a different url for monthly, daily, recent and historical
            dateColumn (str): the title of the column containing the date information of each entry - 
            this is DWD-specific and is different for monthly and daily - see config.eng
            variables (dict): dictionnary of variables to be read and a mapping of the DWD title to a more intuitive naming
        Return:
            df_full (dataframe): data fetched from DWD, cleaned and saved into a dataframe
        """
        df_full = pd.DataFrame(columns=variables.values())
        for file in Weather.__listFD(url):
            url = urllib.request.urlopen(file)
            with ZipFile(BytesIO(url.read())) as my_zip_file:
                productFile = [
                    i for i in my_zip_file.namelist() if i.startswith("produkt")
                ][0]
                df = pd.read_csv(my_zip_file.open(productFile), sep=";")
                df.columns = df.columns.str.strip()
                df[resCons.weatherAuto.indexColumn] = pd.to_datetime(
                    df[dateColumn], format="%Y%m%d"
                )
                df.rename(columns=variables, inplace=True)
                df = df.loc[:, df.columns.isin(variables.values())]
                df.loc[:, df.columns != resCons.weatherAuto.indexColumn] = df.loc[
                    :, df.columns != resCons.weatherAuto.indexColumn
                ].replace(-999, np.nan)
                df_full = df_full.append(df, ignore_index=True, sort=False)

        return df_full

    def __readManualWeatherFiles(self, filename):
        """  
        Read a csv file that has been manually downloaded from the DWD website. DWD gives one file per variable
        Args:
            filename (str): which file to read
        Return:
            df (dataframe): csv content in dataframe
        """
        df = pd.read_csv(filename)
        df = df[
            [
                resCons.weatherManual.productCode,
                resCons.weatherManual.monthIdx,
                resCons.weatherManual.stationId,
                resCons.weatherManual.valueColumn,
            ]
        ]

        df[resCons.weatherManual.monthIdx] = pd.to_datetime(
            df[resCons.weatherManual.monthIdx], format="%Y%m"
        )

        df.rename(
            columns={
                resCons.weatherManual.valueColumn: df[
                    resCons.weatherManual.productCode
                ][0]
            },
            inplace=True,
        )
        df.index = pd.MultiIndex.from_arrays(
            df[
                [resCons.weatherManual.monthIdx, resCons.weatherManual.stationId]
            ].values.T,
            names=[
                resCons.weatherManual.monthIdxChanged,
                resCons.weatherManual.stationIdChanged,
            ],
        )

        df.drop(
            [
                resCons.weatherManual.productCode,
                resCons.weatherManual.monthIdx,
                resCons.weatherManual.stationId,
            ],
            axis=1,
            inplace=True,
        )

        return df

    @staticmethod
    def __listFD(url, ext="zip"):
        """  
        List all files on an ftp server that have a extension ext.
        Args:
            url (str): where to look for files
            ext (str): read all files that have this extension
        Return:
            list of file names
        """
        page = requests.get(url).text
        soup = BeautifulSoup(page, "html.parser")
        return [
            url + "/" + node.get("href")
            for node in soup.find_all("a")
            if node.get("href").endswith(ext)
        ]
