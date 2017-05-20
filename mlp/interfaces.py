"""
Interfaces
"""


class Model(object):

    def fit(self, X, y):
        raise NotImplemented('%s is not implemented' % self.__name__)

    def transform(self, X):
        raise NotImplemented('%s is not implemented' % self.__name__)

    def coefficients(self):
        raise NotImplemented('%s is not implemented' % self.__name__)
