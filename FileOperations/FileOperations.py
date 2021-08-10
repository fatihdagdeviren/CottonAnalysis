import os
# conda install -c anaconda joblib
import joblib
from Utilities import Cons

def CreatePickleFromFile(fileName, object):
    try:
        joblib.dump(object, fileName)
        return Cons.SucessVal, Cons.SucessMessage
    except BaseException as e:
        return Cons.ErrorVal, "{0}-{1}".format(Cons.ErrorMessage, str(e))

def LoadPickleFromFile(fileName):
    try:
        data =joblib.load(fileName)
        return Cons.SucessVal, data
    except BaseException as e:
        return Cons.ErrorVal, "{0}-{1}".format(Cons.ErrorMessage, str(e))
