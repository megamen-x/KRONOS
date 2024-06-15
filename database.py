import secrets
import json
import copy
from datetime import datetime, timezone

from sqlalchemy import create_engine, Column, Integer, String, Text, MetaData
from sqlalchemy.orm import declarative_base, sessionmaker


Base = declarative_base()
metadata = MetaData()


class Messages(Base):
    __tablename__ = 'messages'
    id = Column(Integer, primary_key=True)
    role = Column(String, nullable=False)
    user_id = Column(Integer, nullable=True)
    user_name = Column(String, nullable=True)
    content = Column(Text, nullable=False)
    conv_id = Column(String, nullable=False, index=True)
    timestamp = Column(Integer, nullable=False)
    message_id = Column(Integer)
    system_prompt = Column(Text)


class Conversations(Base):
    __tablename__ = 'current_conversations'
    id = Column(Integer, primary_key=True)
    user_id = Column(Integer, nullable=False, index=True)
    conv_id = Column(String, nullable=False, unique=True)
    timestamp = Column(Integer, nullable=False)


class SystemPrompts(Base):
    __tablename__ = 'system_prompts'
    id = Column(Integer, primary_key=True)
    user_id = Column(Integer, nullable=False, index=True)
    prompt = Column(Text, nullable=False)


class Likes(Base):
    __tablename__ = 'likes'
    id = Column(Integer, primary_key=True)
    user_id = Column(Integer, nullable=False, index=True)
    message_id = Column(Integer, nullable=False, index=True)
    feedback = Column(String, nullable=False)
    is_correct = Column(Integer, nullable=False)


class Database:
    def __init__(self, db_path: str):
        self.engine = create_engine(f'sqlite:///{db_path}')
        Base.metadata.create_all(self.engine)
        self.Session = sessionmaker(bind=self.engine)

    @staticmethod
    def get_current_ts():
        return int(datetime.now().replace(tzinfo=timezone.utc).timestamp())

    def create_conv_id(self, user_id):
        conv_id = secrets.token_hex(nbytes=16)
        with self.Session() as session:
            new_conv = Conversations(user_id=user_id, conv_id=conv_id, timestamp=self.get_current_ts())
            session.add(new_conv)
            session.commit()
        return conv_id

    def get_current_conv_id(self, user_id):
        with self.Session() as session:
            conv = (
                session.query(Conversations)
                .filter(Conversations.user_id == user_id)
                .order_by(Conversations.timestamp.desc())
                .first()
            )
            if conv is None:
                return self.create_conv_id(user_id)
            return conv.conv_id

    def get_current_feedback(self, message_id):
        with self.Session() as session:
            likes = (
                session.query(Likes)
                .filter(Likes.message_id == message_id)
                .first()
            )
            if likes:
                return likes.feedback
            return 'Error'

    def fetch_conversation(self, conv_id, include_meta: bool = False):
        with self.Session() as session:
            messages = (
                session.query(Messages)
                .filter(Messages.conv_id == conv_id)
                .order_by(Messages.timestamp)
                .all()
            )
            if not messages:
                return []
            clean_messages = []
            for m in messages:
                message = {
                    "role": m.role,
                    "text": self._parse_content(m.content),
                }
                if include_meta:
                    message['message_id'] = m.message_id
                    message["system_prompt"] = m.system_prompt
                    message["timestamp"] = m.timestamp
                clean_messages.append(message)
            return clean_messages

    def get_system_prompt(self, user_id, default_prompt):
        with self.Session() as session:
            prompt = (
                session.query(SystemPrompts)
                .filter(SystemPrompts.user_id == user_id)
                .first()
            )
            if prompt:
                return prompt.prompt
            return default_prompt

    def set_system_prompt(self, user_id: int, text: str):
        with self.Session() as session:
            prompt = (
                session.query(SystemPrompts)
                .filter(SystemPrompts.user_id == user_id)
                .first()
            )
            if prompt:
                prompt.prompt = text
            else:
                new_prompt = SystemPrompts(
                    user_id=user_id,
                    prompt=text
                )
                session.add(new_prompt)
            session.commit()

    def save_user_message(self, content: str, conv_id: str, user_id: int, user_name: str = None):
        with self.Session() as session:
            new_message = Messages(
                role="user",
                content=self._serialize_content(content),
                conv_id=conv_id,
                user_id=user_id,
                user_name=user_name,
                timestamp=self.get_current_ts()
            )
            session.add(new_message)
            session.commit()

    def save_assistant_message(self, content: str, conv_id: str, message_id: int, system_prompt: str):
        with self.Session() as session:
            new_message = Messages(
                role="assistant",
                content=content,
                conv_id=conv_id,
                timestamp=self.get_current_ts(),
                message_id=message_id,
                system_prompt=system_prompt
            )
            session.add(new_message)
            session.commit()

    def save_feedback(self, feedback: str, user_id: int, message_id: int):
        with self.Session() as session:
            new_feedback = Likes(
                user_id=user_id,
                message_id=message_id,
                feedback=feedback,
                is_correct=True
            )
            session.add(new_feedback)
            session.commit()

    def get_all_conv_ids(self):
        with self.Session() as session:
            conversations = session.query(Conversations).all()
            return [conv.conv_id for conv in conversations]

    def _serialize_content(self, content):
        if isinstance(content, str):
            return content
        return json.dumps(content)

    def _parse_content(self, content):
        try:
            parsed_content = json.loads(content)
            if not isinstance(parsed_content, list):
                return content
            for m in parsed_content:
                if not isinstance(m, dict):
                    return content
            return parsed_content
        except json.JSONDecodeError:
            return content