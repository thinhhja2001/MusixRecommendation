# This is a sample Python script.
from io import BytesIO
from typing import Annotated

import eyed3
# Press Ctrl+F5 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.
import pandas as pd
import uvicorn
from fastapi import FastAPI, HTTPException, File, UploadFile, Form
from tinytag import TinyTag
import tempfile

from copyright_checker.model.copyright_checker import CopyrightChecker
from recommenders.collaborative_filtering.model.collaborative_filtering_recommender import \
    CollaborativeFilteringRecommender
from recommenders.content_based.model.content_based_recommender import ContentBasedRecommender
from recommenders.models.ListSongId import ListSongId
from recommenders.models.recommender_response import RecommenderResponse

content_based_recommender = ContentBasedRecommender(
    model_path="https://drive.google.com/uc?export=download&id=1zOc44sCk4NSsCq5bzk7ZP5LOXrXRh1V0&confirm=t&uuid=cf31f9ab-ac3c-4c67-9a20-a7b9c3604215&at=ALt4Tm0UGG52LRwEKGKQgHP3yyPn:1689270819563",
    perms=128
)
collab_recommender = CollaborativeFilteringRecommender(
    model_path="https://drive.google.com/uc?export=download&id=1EHRzZIU0TcPfYYQYmdTVi0mnL21BJK1M")
print('Hell')
app = FastAPI()


@app.get("/next_song/")
async def recommend_next_song(song_id: str, count: int):
    try:
        song_data = load_dataset("recommenders/content_based/dataset/song_data.csv", index_col="encode_id")
        # content_based_recommender = ContentBasedRecommender(
        #     model_path="https://firebasestorage.googleapis.com/v0/b/musix-cfd8e.appspot.com/o/content_based_model.sav?alt=media&token=https://firebasestorage.googleapis.com/v0/b/musix-cfd8e.appspot.com/o/content_based_model.sav?alt=media&token=c8779635-b363-4306-93e1-0d49075754f9",
        #     perms=128
        # )
        recommend_song_ids = content_based_recommender.predict(encode_id=song_id, dataset=song_data, num_results=count)
        result = RecommenderResponse(status_code=200, message="Success", data=recommend_song_ids)
        return result
    except Exception as e:
        print(e)
        raise HTTPException(status_code=500, detail="Server error")


@app.get("/generated_recommend_playlist")
async def generate_recommend_playlist(song_ids: ListSongId):
    try:
        wide_song_data = pd.read_csv("recommenders/collaborative_filtering/dataset/song_play_record.csv",
                                     index_col="song_id")
        recommended_song_ids = []
        for song_id in song_ids.song_ids:
            recommended_id = collab_recommender.predict(encode_id=song_id, dataset=wide_song_data, num_results=5)
            if recommended_id is not None:
                recommended_song_ids.extend(recommended_id)
        recommended_song_ids = set(recommended_song_ids)
        result = RecommenderResponse(status_code=200, message="Success", data=recommended_song_ids)
        return result
    except Exception as e:
        print(e)
        raise HTTPException(status_code=500, detail="Server error")


@app.get("/check_copyright")
async def check_copyright(song_file: UploadFile):
    file_bytes = await song_file.read()
    bytes_io = BytesIO(file_bytes)
    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as tmp:
        print(tmp.name)
        tmp.write(bytes_io.getvalue())
        tmp.seek(0)
        tag = TinyTag.get(tmp.name)
        if tag.title is not None or tag.artist is not None:
            return {"status_code": 200, "result": False, "message": "This song violated the copyright"}
        return {"status_code": 200, "result": True, "message": "This song is not violated the copyright"}


def load_dataset(file_path, index_col):
    song_data = pd.read_csv(file_path, index_col=index_col)
    song_data['album'].fillna(" ")

    song_data['features_col'] = song_data['artists_name'] + song_data['genres_id']
    return song_data

    # # Press the green button in the gutter to run the script.


if __name__ == '__main__':
    uvicorn.run(app, port=5000)

# # See PyCharm help at https://www.jetbrains.com/help/pycharm/
