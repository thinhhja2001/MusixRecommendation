from abc import ABC, abstractmethod


class IRecommendModel(ABC):
    @abstractmethod
    def predict(self, encode_id, dataset, num_results):
        pass
