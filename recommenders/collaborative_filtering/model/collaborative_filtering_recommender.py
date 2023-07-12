import pickle
from urllib.request import urlopen

from recommenders.i_recommend_model import IRecommendModel
from abc import ABC
from fuzzywuzzy import fuzz


class CollaborativeFilteringRecommender(IRecommendModel, ABC):
    def __init__(self, model_path: str):
        self.__load_model(model_path)

    def __load_model(self, model_path: str):
        self.model = pickle.load(urlopen(model_path))

    def predict(self, encode_id, dataset, num_results):
        """

        :param encode_id: id of the song
        :param dataset: song_play_matrix (dataframe)
        :param num_results: num of result
        :return: Song recommendations
        """
        query_index = None
        ratio_tuples = []

        for i in dataset.index:
            ratio = fuzz.ratio(i, encode_id)
            if ratio == 100:
                current_index_query = dataset.index.tolist().index(i)
                ratio_tuples.append((i, ratio, current_index_query))

        try:
            query_index = max(ratio_tuples, key=lambda x: x[1])[2]
        except:
            return None

        distances, indices = self.model.kneighbors(dataset.iloc[query_index, :].values.reshape(1, -1),
                                                   n_neighbors=num_results + 1)
        recommended_songs = []
        for i in range(0, len(distances.flatten())):
            if i == 0:
                continue
            else:
                recommended_songs.append(dataset.index[indices.flatten()[i]])
        return recommended_songs
