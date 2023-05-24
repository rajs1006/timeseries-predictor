import pandas as pd
import numpy as np



def alignDates(data1, data2):
    ### set all days to one ## 
    data1.index = data1.index.map(lambda t: t.replace(day=1))
    data2.index = data2.index.map(lambda t: t.replace(day=1))

    ### align dates ##
    start = max(data1.index[0], data2.index[0])
    end = min(data1.index[-1], data2.index[-1])
    data11 = data1.loc[start:end, :]
    data22 = data2.loc[start:end, :]
    return [data11, data22]
