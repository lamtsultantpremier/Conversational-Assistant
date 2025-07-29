from sqlalchemy import String
from sqlalchemy.orm import Mapped, mapped_column

from src.database import Base


class Link(Base):
    __tablename__ = "links"

    id_link: Mapped[int] = mapped_column(primary_key=True, init=False)
    url: Mapped[str] = mapped_column(String, unique=True, nullable=False)
