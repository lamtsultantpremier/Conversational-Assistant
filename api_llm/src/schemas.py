from pydantic import BaseModel


class LinkBase(BaseModel):
    url: str


class LinkCreate(LinkBase):
    pass


class LinkRead(LinkBase):
    id_link: int

    class Config:
        from_attributes = True


class QuestionInput(BaseModel):
    question: str
    chat_history: list[dict[str, str]] = []
