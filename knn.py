from random import random


data = [[(random() + i) for _ in range(4)] for i in range(100)]


class knn(object):

    def __init__(self, k=3):
        self.k = k

    def fit(self, x, y):
        self.x_train = x
        self.y_train = y

    def predict(self, x, y):
        predict_lable
