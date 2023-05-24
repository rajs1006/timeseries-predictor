import os
from functools import lru_cache
from multiprocessing import Process

import altair as alt
import numpy as np
import pandas as pd
import xmltodict
from energypredictor.main.utils.common.Constants import Environment as envCons
from energypredictor.main.utils.common.Constants import Resource as resCons


class Energy:
    """
    Class for all energy fetching and preparation methods.
    """

    def get(
        self,
        energyFile: str,
        granularity: str,
        externalBalancingFile: str = None,
        comercialConversionFile: str = None,
        save: bool = True,
    ) -> "DataFrame : Fetched data":
        """
        Loads the data given file path. File extension should be .xml with DTD validation appended in the same file.

        Args:
            externalBalancingFile (str): pass the path of eneregy data file : os.path.join(dataFolder, 'NCG_external_balancing_gas.xml')
            comercialConversionFile (str): pass the path of data file : os.path.join(dataFolder, 'NCG_commercial_conversion_final_data.xml')

        """
        ## Adding extension to file
        energyFile = energyFile + resCons.extensions.dataExt
        if save:
            if (externalBalancingFile is None) | (comercialConversionFile is None):
                raise Exception(
                    "if 'save' is True, pass the externalBalancingFile and comercialConversionFile file paths"
                )

            if not os.path.exists(externalBalancingFile):
                raise Exception(
                    "Path of External Balancing file is not correct : {}".format(
                        externalBalancingFile
                    )
                )

            if not os.path.exists(comercialConversionFile):
                raise Exception(
                    "Path of Comercial conversion file is not correct : {}".format(
                        comercialConversionFile
                    )
                )

            externalBalancingData = self.__getData(
                externalBalancingFile, resCons.commonGas.key
            )
            comercialConversionData = self.__getData(
                comercialConversionFile, resCons.commonConv.key
            )

            (
                buyHBalancingDataFinal,
                buyLBalancingDataFinal,
                sellHBalancingDataFinal,
                sellLBalancingDataFinal,
            ) = self.__transform(externalBalancingData, comercialConversionData)

            if granularity == resCons.common.DAILY_RESAMPLE:
                buyLBalancingDataFinal = buyLBalancingDataFinal.resample(
                    resCons.common.DAILY_RESAMPLE
                ).sum()
                sellLBalancingDataFinal = sellLBalancingDataFinal.resample(
                    resCons.common.DAILY_RESAMPLE
                ).sum()
                buyHBalancingDataFinal = buyHBalancingDataFinal.resample(
                    resCons.common.DAILY_RESAMPLE
                ).sum()
                sellHBalancingDataFinal = sellHBalancingDataFinal.resample(
                    resCons.common.DAILY_RESAMPLE
                ).sum()

                raise Exception("Daily granularity is not yet supported, WIP !!")

            elif granularity == resCons.common.MONTHLY_RESAMPLE:
                buyLBalancingDataFinal = buyLBalancingDataFinal.resample(
                    resCons.common.MONTHLY_RESAMPLE
                ).sum()
                sellLBalancingDataFinal = sellLBalancingDataFinal.resample(
                    resCons.common.MONTHLY_RESAMPLE
                ).sum()
                buyHBalancingDataFinal = buyHBalancingDataFinal.resample(
                    resCons.common.MONTHLY_RESAMPLE
                ).sum()
                sellHBalancingDataFinal = sellHBalancingDataFinal.resample(
                    resCons.common.MONTHLY_RESAMPLE
                ).sum()

            else:
                raise Exception("'granularity' is not in ['m', 'd']")

            concatenatedEnergy = pd.concat(
                [
                    buyHBalancingDataFinal,
                    buyLBalancingDataFinal,
                    sellHBalancingDataFinal,
                    sellLBalancingDataFinal,
                ],
                axis=1,
                keys=[
                    resCons.gas.buyHBalancingDataName,
                    resCons.gas.buyLBalancingDataName,
                    resCons.gas.sellHBalancingDataName,
                    resCons.gas.sellLBalancingDataName,
                ],
            )
            concatenatedEnergy.to_pickle(energyFile, compression="bz2")

            self.energyData = self.__combine(
                buyHBalancingDataFinal,
                buyLBalancingDataFinal,
                sellHBalancingDataFinal,
                sellLBalancingDataFinal,
            )
        else:
            if not os.path.exists(energyFile):
                raise Exception(
                    "The energy data has not been collected yet, please unselect the checkbox and run the model again: {}".format(
                        energyFile
                    )
                )

            energyFile = pd.read_pickle(energyFile, compression="bz2")

            buyHBalancingDataFinal = energyFile[resCons.gas.buyHBalancingDataName]
            buyLBalancingDataFinal = energyFile[resCons.gas.buyLBalancingDataName]
            sellHBalancingDataFinal = energyFile[resCons.gas.sellHBalancingDataName]
            sellLBalancingDataFinal = energyFile[resCons.gas.sellLBalancingDataName]

            self.energyData = self.__combine(
                buyHBalancingDataFinal,
                buyLBalancingDataFinal,
                sellHBalancingDataFinal,
                sellLBalancingDataFinal,
            )

        return self.energyData

    def plot(self, st):

        dataSource = self.energyData.reset_index().melt("Day")

        line_chart = (
            alt.Chart(dataSource)
            .mark_line()
            .encode(
                x=alt.X("Day", axis=alt.Axis(title="")),
                y=alt.Y("value", axis=alt.Axis(title="Balancing energy")),
                color="variable",
            )
            .properties(width=700, height=400)
        )

        st.altair_chart(line_chart)

    def __combine(
        self,
        buyHBalancingDataFinal,
        buyLBalancingDataFinal,
        sellHBalancingDataFinal,
        sellLBalancingDataFinal,
    ):
        ## Combine dataframes ##
        buyHBalancingDataFinal = buyHBalancingDataFinal.rename(
            columns={resCons.changeGas.quantityColumn: "HBuy"}
        )
        buyLBalancingDataFinal = buyLBalancingDataFinal.rename(
            columns={resCons.changeGas.quantityColumn: "LBuy"}
        )
        sellHBalancingDataFinal = sellHBalancingDataFinal.rename(
            columns={resCons.changeGas.quantityColumn: "HSell"}
        )
        sellLBalancingDataFinal = sellLBalancingDataFinal.rename(
            columns={resCons.changeGas.quantityColumn: "LSell"}
        )

        concatenatedFile = pd.concat(
            [
                buyHBalancingDataFinal,
                buyLBalancingDataFinal,
                sellHBalancingDataFinal,
                sellLBalancingDataFinal,
            ],
            axis=1,
        )

        return concatenatedFile

    def __transform(self, externalBalancingData, comercialConversionData):

        self.__validateEnergyData(externalBalancingData)

        externalBalancingData = self.__transformBalancingData(externalBalancingData)
        comercialConversionData = self.__transformConversionData(
            comercialConversionData
        )

        (
            buyHBalancingData,
            buyLBalancingData,
            sellHBalancingData,
            sellLBalancingData,
        ) = self.__filterData(externalBalancingData)
        (
            buyHBalancingDataFinal,
            buyLBalancingDataFinal,
            sellHBalancingDataFinal,
            sellLBalancingDataFinal,
        ) = self.__buildAndSubstractConversionData(
            comercialConversionData,
            buyHBalancingData,
            buyLBalancingData,
            sellHBalancingData,
            sellLBalancingData,
        )

        return (
            buyHBalancingDataFinal,
            buyLBalancingDataFinal,
            sellHBalancingDataFinal,
            sellLBalancingDataFinal,
        )

    def __validateEnergyData(self, externalBalancingData):

        areColumnsPresent = np.all(
            [
                c in externalBalancingData.columns
                for c in [
                    resCons.gas.indexColumn,
                    resCons.gas.quantityColumn,
                    resCons.gas.productColumn,
                    resCons.gas.directionColumn,
                    resCons.gas.gasColumn,
                ]
            ]
        )

        if not areColumnsPresent:
            raise Exception(
                "Required columns do not exists in the External Balancing Dataframe, kindly check the resource file 'energy.yml' "
            )

    def __getData(self, file, key):

        with open(file) as fd:
            doc = xmltodict.parse(fd.read())

        dataframe = pd.DataFrame(doc[key][key])
        return dataframe

    def __transformBalancingData(self, externalBalancingData):
        """
        Transforms energy data.

        Args:
            externalBalancingFile ([type], optional): [pass the path of data file : os.path.join(dataFolder, 'NCG_external_balancing_gas.xml')].

        Returns:
            DataFrame: Tranmsformed dataframe for energy data
        """

        externalBalancingData = externalBalancingData.set_index(
            pd.to_datetime(externalBalancingData.DayOfUse)
        )
        externalBalancingData.index.name = resCons.changeGas.indexColumn
        externalBalancingData[resCons.gas.quantityColumn] = externalBalancingData[
            resCons.gas.quantityColumn
        ].astype("float64")

        ### Removing corrupted data with 0 DailyQuantity value
        externalBalancingData = externalBalancingData[
            externalBalancingData.DailyQuantity != resCons.common.ZERO_FLOAT
        ]

        return externalBalancingData

    def __transformConversionData(self, comercialConversionData):
        """
        Transforms the Conversion data

        Args:
            comercialConversionFile ([type], optional): [description]. Defaults to os.path.join(dataFolder, 'NCG_commercial_conversion_final_data.xml').

        Returns:
            DataFrame: Tranmsformed dataframe for energy data
        """

        comercialConversionData = comercialConversionData.set_index(
            pd.to_datetime(comercialConversionData.DayOfUse)
        )
        comercialConversionData[resCons.conv.HLConversionCol] = comercialConversionData[
            resCons.conv.HLConversionCol
        ].astype("float64")
        comercialConversionData[resCons.conv.LHConversionCol] = comercialConversionData[
            resCons.conv.LHConversionCol
        ].astype("float64")
        comercialConversionData.index.name = "Day"

        if (resCons.gas.energyUnit == resCons.commonGas.MEGA_WATT) & (
            resCons.conv.energyUnit == resCons.commonGas.KILO_WATT
        ):

            ### Converting the data to conform with balancing unit (MWh)
            comercialConversionData[resCons.conv.HLConversionCol] = (
                comercialConversionData[resCons.conv.HLConversionCol] / 1000
            )
            comercialConversionData[
                resCons.conv.HLUnitColumn
            ] = resCons.commonGas.MEGA_WATT

            comercialConversionData[resCons.conv.LHConversionCol] = (
                comercialConversionData[resCons.conv.LHConversionCol] / 1000
            )
            comercialConversionData[
                resCons.conv.LHUnitColumn
            ] = resCons.commonGas.KILO_WATT

        return comercialConversionData

    def __filterData(self, externalBalancingData):
        ### Filtering out Hourly data
        externalBalancingDataWOHour = externalBalancingData[
            externalBalancingData.ProductType != resCons.gas.excludedProduct
        ]

        ### Separate buy and sell data
        externalBalancingBuyData = externalBalancingDataWOHour[
            externalBalancingDataWOHour.DirectionOfFlow.isin(
                resCons.gas.buyDirectionVal
            )
        ]
        externalBalancingSellData = externalBalancingDataWOHour[
            externalBalancingDataWOHour.DirectionOfFlow.isin(
                resCons.gas.sellDirectionVal
            )
        ]

        ### Buy H and L gas
        buyHBalancingData = externalBalancingBuyData[
            externalBalancingBuyData.Gas.isin([resCons.gas.hGas])
        ]
        buyHBalancingData = (
            buyHBalancingData.loc[:, [resCons.gas.quantityColumn]]
            .resample(resCons.common.DAILY_RESAMPLE)
            .sum()
        )
        buyHBalancingData = buyHBalancingData.rename(
            columns={resCons.gas.quantityColumn: resCons.changeGas.quantityColumn}
        )

        buyLBalancingData = externalBalancingBuyData[
            externalBalancingBuyData.Gas.isin([resCons.gas.lGas])
        ]
        buyLBalancingData = (
            buyLBalancingData.loc[:, [resCons.gas.quantityColumn]]
            .resample(resCons.common.DAILY_RESAMPLE)
            .sum()
        )
        buyLBalancingData = buyLBalancingData.rename(
            columns={resCons.gas.quantityColumn: resCons.changeGas.quantityColumn}
        )

        ### Sell H and L gas
        sellHBalancingData = externalBalancingSellData[
            externalBalancingSellData.Gas.isin([resCons.gas.hGas])
        ]
        sellHBalancingData = (
            sellHBalancingData.loc[:, [resCons.gas.quantityColumn]]
            .resample(resCons.common.DAILY_RESAMPLE)
            .sum()
        )
        sellHBalancingData = sellHBalancingData.rename(
            columns={resCons.gas.quantityColumn: resCons.changeGas.quantityColumn}
        )

        sellLBalancingData = externalBalancingSellData[
            externalBalancingSellData.Gas.isin([resCons.gas.lGas])
        ]
        sellLBalancingData = (
            sellLBalancingData.loc[:, [resCons.gas.quantityColumn]]
            .resample(resCons.common.DAILY_RESAMPLE)
            .sum()
        )
        sellLBalancingData = sellLBalancingData.rename(
            columns={resCons.gas.quantityColumn: resCons.changeGas.quantityColumn}
        )

        return (
            buyHBalancingData,
            buyLBalancingData,
            sellHBalancingData,
            sellLBalancingData,
        )

    def __buildAndSubstractConversionData(
        self,
        comercialConversionData,
        buyHBalancingData,
        buyLBalancingData,
        sellHBalancingData,
        sellLBalancingData,
    ):
        ### Substracting Conversion data

        comercialConversionHLData = (
            comercialConversionData.loc[:, [resCons.conv.HLConversionCol]]
            .resample(resCons.common.DAILY_RESAMPLE)
            .sum()
        )
        comercialConversionHLData = comercialConversionHLData.rename(
            columns={resCons.conv.HLConversionCol: resCons.changeGas.quantityColumn}
        )

        comercialConversionLHData = (
            comercialConversionData.loc[:, [resCons.conv.LHConversionCol]]
            .resample(resCons.common.DAILY_RESAMPLE)
            .sum()
        )
        comercialConversionLHData = comercialConversionLHData.rename(
            columns={resCons.conv.LHConversionCol: resCons.changeGas.quantityColumn}
        )

        ### Substract conversion-(HL) from sell-H and buy-L
        sellHBalancingDataFinal = (
            (sellHBalancingData.copy() - comercialConversionHLData)
            .combine_first(comercialConversionHLData)
            .reindex_like(comercialConversionHLData)
        )
        buyLBalancingDataFinal = (
            (buyLBalancingData.copy() - comercialConversionHLData)
            .combine_first(comercialConversionHLData)
            .reindex_like(comercialConversionHLData)
        )

        ### Substract conversion-(LH) from sell-L and buy-H
        sellLBalancingDataFinal = (
            (sellLBalancingData.copy() - comercialConversionLHData)
            .combine_first(comercialConversionLHData)
            .reindex_like(comercialConversionLHData)
        )
        buyHBalancingDataFinal = (
            (buyHBalancingData.copy() - comercialConversionLHData)
            .combine_first(comercialConversionLHData)
            .reindex_like(comercialConversionLHData)
        )

        return (
            buyHBalancingDataFinal,
            buyLBalancingDataFinal,
            sellHBalancingDataFinal,
            sellLBalancingDataFinal,
        )

