import bz2
import os
import pickle
from copy import deepcopy

from sklearn.preprocessing import StandardScaler

from energypredictor.main.utils.common.Constants import Resource as resCons


class Scalar:
    def __init__(self):
        self.scalar = StandardScaler()

    def fit(self, data):
        self.scalar.fit(data)
        return self

    def scale(self, data):
        return self.scalar.transform(data)

    def inverseScale(self, data):
        return self.scalar.inverse_transform(data)

    def save(self, folder, file):
        try:
            with open(
                os.path.join(
                    "" if folder is None else folder,
                    "{}{}".format(file, resCons.extensions.scalarExt),
                ),
                "wb",
            ) as f:
                pickle.dump(self, bz2.BZ2File(f, "wb"))
        except:
            raise Exception("Could not save scalar model, please try again")

    def load(self, folder, file):
        try:
            with open(
                os.path.join(
                    "" if folder is None else folder,
                    "{}{}".format(file, resCons.extensions.scalarExt),
                ),
                "rb",
            ) as f:
                return pickle.load(bz2.BZ2File(f, "rb"))
        except:
            raise Exception("Could not load scalar model, please try again")

    def copy(self):
        return deepcopy(self)
