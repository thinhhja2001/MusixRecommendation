from pydantic import BaseModel


class CopyrightChecker(BaseModel):
    data: bytes
