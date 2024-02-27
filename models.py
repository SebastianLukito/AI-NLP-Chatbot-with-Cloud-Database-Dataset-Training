from sqlalchemy import Column, Integer, String, Float, ForeignKey
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import relationship
from sqlalchemy.orm import Mapped
from database import engine
from typing import List

Base = declarative_base()

class Tag(Base):
    __tablename__ = 'tag'

    id = Column(Integer, primary_key=True)
    name = Column(String(1000))
    pattern: Mapped[List["Pattern"]] = relationship('Pattern', backref='tag')
    response: Mapped[List["Response"]] = relationship('Response', backref='tag')
    
class Pattern(Base):
    __tablename__ = 'pattern'

    id = Column(Integer, primary_key=True)
    value = Column(String(1000))
    tag_id = Column(Integer, ForeignKey('tag.id'))

class Response(Base):
    __tablename__ = 'response'

    id = Column(Integer, primary_key=True)
    tag_id = Column(Integer, ForeignKey('tag.id'))
    # pattern_id = Column(Integer, ForeignKey('pattern.id'))
    value = Column(String(1000))


Base.metadata.create_all(engine)
