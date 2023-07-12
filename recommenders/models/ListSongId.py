from pydantic import BaseModel


class ListSongId(BaseModel):
    song_ids: list
