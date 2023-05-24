import datetime

from energypredictor.main.utils.logger.System import logger


def time_log_this(original_function):
    """
    Decorator to call original_function and log its name and the execution
    time of the call.

    Usage: Put {@time_log_this} before the method.
    Example:
            @time_log_this
            def PRC_Data:
                --- method body ---

    Arguments:
        original_function {Object} -- The function to be called for logging
    """

    ### Instantiating logger with qualified name of method (<clsName>.<methodName>)
    log = logger(original_function.__qualname__)

    def new_function(*args, **kwargs):
        log.debug("START -> {0}".format(original_function.__name__))
        before = datetime.datetime.now()
        x = original_function(*args, **kwargs)
        after = datetime.datetime.now()
        log.debug("END -> {0}: Elapsed Time = {1} seconds".format(
            original_function.__name__, (after - before).total_seconds()))
        return x

    return new_function


def time_log_all(Cls):
    """
    Decorator to be executed with all methods of the class and log the name
    and the execution time of the methods.

    Usage: Put {@time_log_all} before the class

    Example:
            @time_log_all
            class DBTimeseries:
                --- class body ---

    Arguments:
        Cls {Object} -- Instance of class
    """
    class NewCls(object):
        def __init__(self, *args, **kwargs):
            self.oInstance = Cls(*args, **kwargs)

        def __getattribute__(self, s: str):
            """
            This is called whenever any attribute of a NewCls object is accessed.
            This first tries to get the attribute of NewCls. If it fails then
            it tries to fetch the attribute from self.oInstance (an instance
            of the decorated class). If it manages to fetch the attribute from
            self.oInstance, and the attribute is an instance method then
            `time_this` is applied.

            Arguments:
                s {str} -- the name of attribute

            """

            try:
                return super(NewCls, self).__getattribute__(s)
            except AttributeError:
                pass

            x = self.oInstance.__getattribute__(s)
            if type(x) == type(self.__init__):  ### it is an instance method
                ### This is equivalent of just decorating the method with time_this
                return time_log_this(x)
            else:
                return x

    return NewCls
