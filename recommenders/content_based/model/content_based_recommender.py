import time
from abc import ABC
from urllib.request import urlopen

from recommenders.i_recommend_model import IRecommendModel
import pickle
from datasketch import MinHash
import pandas as pd
import numpy as np


class ContentBasedRecommender(IRecommendModel, ABC):
    def __init__(self, model_path: str, perms):
        self.__load_model(model_path)
        self.perms = perms

    def __load_model(self, model_path: str):
        self.model = pickle.load(urlopen(model_path))

    def predict(self, encode_id, dataset: pd.DataFrame, num_results):
        m = MinHash(num_perm=self.perms)

        features = dataset.loc[dataset.index == encode_id]['features_col']
        for feature in features:
            m.update(feature.encode('utf-8'))
        idx_array = np.array(self.model.query(m, num_results))
        if len(idx_array) == 0:
            return None  # if your query is empty, return none
        result = dataset.iloc[idx_array].index.tolist()
        return result
