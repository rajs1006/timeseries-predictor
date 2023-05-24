import argparse
import multiprocessing
import os
import sys
from types import SimpleNamespace

import importlib_resources
import yaml
from dotenv import load_dotenv


def loadArgumets():
    parser = argparse.ArgumentParser(description="Temprature based energy predictor")

    parser.add_argument("--env", help="pass the full path of environment file", required=True)
    args = parser.parse_args()

    return args


################ Arguments Parser ##############

## parse arguments, use this variable as constant to get
# command line varibale values
args = loadArgumets()

################################################


def readEnvVar(envVar: str, defaultValue: any = None) -> str:
    """
    Gets the value of an environment variable.

    Returns:
        the value of the environment variable.

    Throws:
        EnvironmentError if the environment variable isn't set.
    """

    val = os.getenv(envVar, defaultValue)

    if val is None:
        raise EnvironmentError("Environment variable '{}' must be set".format(envVar))

    return val


def checkFolderExist(folder):
    os.makedirs(folder, exist_ok=True)
    return folder


def checkPathExists(path, isHome=True):
    try:
        isPath = os.path.exists(path)
        if isPath:
            if isHome:
                os.chdir(path)
            else:
                pass
        else:
            raise Exception
    except:
        if isHome:
            raise Exception(
                "Location of HOME is not correct '{}', pass the path of the folder where your 'data' folder resides.".format(
                    path
                )
            )
        else:
            raise Exception(
                "Location of file is not correct '{}', make sure the file exists".format(
                    path
                )
            )
    return path


def loadResource(resourceFile: str):
    """
    This method loads the yaml file.

    Args:
        resourceFile : path of resource file

    Returns:
        Dict: simplenamespace dict of resource variables
    """
    with open(resourceFile) as f:
        return yaml.load(f, Loader=yaml.FullLoader)


### COntains the constants of environment file at one place
class Environment:

    ## Load env file when executed with stremlit.
    load_dotenv(args.env, verbose=True)

    home = checkPathExists(readEnvVar("homeFolderPath"), isHome=True)
    app = importlib_resources.files("energypredictor")

    logLevel = readEnvVar("logLevel")
    logFolder = checkFolderExist(readEnvVar("logFolder"))
    logFile = readEnvVar("logFile")

    trainFile = readEnvVar("trainFile")

    granularity = readEnvVar("granularity").upper()

    coreProcess = int(readEnvVar("coreProcess", 3 * multiprocessing.cpu_count()))

    resultFolder = checkFolderExist(readEnvVar("result"))

    scalarFolder = checkFolderExist(readEnvVar("scalarFolder"))
    paramFolder = checkFolderExist(readEnvVar("paramFolder"))

    ## Energy specific var
    energyDataFolder = checkFolderExist(readEnvVar("energyDataFolder"))
    energySarimaModelFolder = checkFolderExist(readEnvVar("energySarimaModelFolder"))
    energyArchModelFolder = checkFolderExist(readEnvVar("energyArchModelFolder"))

    energyFile = readEnvVar("energyFile")

    ## Energy files downloaded from NCG website
    externalBalancingFile = readEnvVar("externalBalancingFile")
    comercialConversionFile = readEnvVar("comercialConversionFile")

    ## Weather specific var
    weatherDataFolder = checkFolderExist(readEnvVar("weatherDataFolder"))
    weatherModelFolder = checkFolderExist(readEnvVar("weatherModelFolder"))

    weatherFile = readEnvVar("weatherFile")

    ## Automatic weather related details
    urlMonthlyRecent = readEnvVar("urlMonthlyRecent")
    urlMonthlyHistorical = readEnvVar("urlMonthlyHistorical")

    ## Manual weather related files
    manualWeatherFolder = readEnvVar("manualWeatherFolder")

    ## Results
    resultScaling = readEnvVar("resultScaling")


class Resource:
    ## This file path is written here as this file is packaged with the build
    # and need not be accessed from outside the build (Build specific configurations)
    resourceFilePath = "resource/resource.yml"
    resource = checkPathExists(Environment.app / resourceFilePath, isHome=False)

    ## load data
    energyResource = loadResource(resource)

    ### Properies for GAS and CONVERSION data
    energy = energyResource["GAS"]
    conversion = energyResource["CONVERSION"]

    commonGas = SimpleNamespace(**energy["common"])
    gas = SimpleNamespace(**energy["validate"])
    changeGas = SimpleNamespace(**energy["change"])

    commonConv = SimpleNamespace(**conversion["common"])
    conv = SimpleNamespace(**conversion["validate"])

    ### Properies for WEATHER data
    weather = energyResource["WEATHER"]
    weatherAuto = SimpleNamespace(**weather["automated"])
    weatherManual = SimpleNamespace(**weather["manual"])
    commonWeather = SimpleNamespace(**weather["common"])

    ### Common properties
    common = SimpleNamespace(**energyResource["COMMON"])

    ### File extensions
    extensions = SimpleNamespace(**energyResource["EXTENSIONS"])
