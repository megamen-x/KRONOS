import asyncio
import csv
import os
import traceback
import uuid

import fire
from aiogram import Bot, Dispatcher, F
from aiogram.client.default import DefaultBotProperties
from aiogram.enums import ParseMode
from aiogram.filters import Command
from aiogram.types import Message, InlineKeyboardButton, CallbackQuery, FSInputFile
from aiogram.utils.keyboard import InlineKeyboardBuilder
from rag import *
from database import Database
import prettytable as pt

Settings.embed_model = HuggingFaceEmbedding(
        model_name="./embedder"
    )
tokenizer, llm, retriever, reranker = None, None, None, None


class LlmBot:
    def __init__(
        self,
        bot_token: str,
        db_path: str,
        history_max_tokens: int,
        chunk_size: int,
    ):
        self.default_prompt = "Ты — Сайга, русскоязычный автоматический ассистент. Ты разговариваешь с людьми и помогаешь им."
        assert self.default_prompt

        # Параметры
        self.history_max_tokens = history_max_tokens
        self.chunk_size = chunk_size

        # База
        self.db = Database(db_path)

        # Клавиатуры
        self.start_kb = InlineKeyboardBuilder()
        self.start_kb.add(InlineKeyboardButton(text='Узнать команды', callback_data='commands'))
        self.likes_kb = InlineKeyboardBuilder()
        self.likes_kb.add(InlineKeyboardButton(
            text="👍",
            callback_data="feedback:like"
        ))
        self.likes_kb.add(InlineKeyboardButton(
            text="👎",
            callback_data="feedback:dislike"
        ))

        # Бот
        self.bot = Bot(token=bot_token, default=DefaultBotProperties(parse_mode=ParseMode.MARKDOWN))
        self.dp = Dispatcher()

        self.dp.message.register(self.start, Command("start"))
        self.dp.message.register(self.reset, Command("reset_history"))
        self.dp.message.register(self.history, Command("history"))
        self.dp.message.register(self.about_team, Command("team"))
        self.dp.message.register(self.set_system, Command("set_system"))
        self.dp.message.register(self.get_system, Command("get_system"))
        self.dp.message.register(self.reset_system, Command("reset_system"))
        self.dp.message.register(self.add_knowledge, Command("add_knowledge"))
        self.dp.message.register(self.generate)

        self.dp.callback_query.register(self.save_feedback, F.data.startswith("feedback:"))
        self.dp.callback_query.register(self.show_commands, F.data.startswith('commands'))


    async def start_polling(self):
        await self.dp.start_polling(self.bot)

    async def start(self, message: Message):
        chat_id = message.chat.id
        self.db.create_conv_id(chat_id)
        markup = self.start_kb.as_markup()
        await message.reply("Привет! Я ассистент, отвечающий на вопросы на темы по платформе 1С: Предприятие. Доступно 2 возможности узнать список команд\n   - Кнопка 'Узнать команды'\n   - Кнопка 'Menu', расположенная слева от строки запроса", reply_markup=markup)

    async def set_system(self, message: Message):
        chat_id = message.chat.id
        text = message.text.replace("/setsystem", "").strip()
        self.db.set_system_prompt(chat_id, text)
        self.db.create_conv_id(chat_id)
        await message.reply(f"Новый системный промпт задан:\n\n{text}")

    async def get_system(self, message: Message):
        chat_id = message.chat.id
        prompt = self.db.get_system_prompt(chat_id, self.default_prompt)
        if prompt.strip():
            await message.reply(prompt)
        else:
            await message.reply("Системный промпт пуст")

    async def reset_system(self, message: Message):
        chat_id = message.chat.id
        self.db.set_system_prompt(chat_id, self.default_prompt)
        self.db.create_conv_id(chat_id)
        await message.reply("Системный промпт сброшен!")

    async def reset(self, message: Message):
        chat_id = message.chat.id
        self.db.create_conv_id(chat_id)
        await message.reply("История сообщений сброшена!")

    async def history(self, message: Message):
        chat_id = message.chat.id
        conv_id = self.db.get_current_conv_id(chat_id)
        history = self.db.fetch_conversation(conv_id, include_meta=True)
        feedback = {'like': "Ответ понравился", 'dislike': "Ответ не понравился"}
        user_question = ''
        bot_answer = ''
        for i, m in enumerate(history[-10:]):
            if not isinstance(m["text"], str):
                m["text"] = "Не текст"
            if i % 2 == 0:
                user_question = m['text']
            else:
                bot_answer = m['text']
                text_feedback = self.db.get_current_feedback(m['message_id'])
                await self.bot.send_message(text=f'Запрос пользователя: {user_question}\n------------------------------------------------------------------------------------\nОтвет бота: {bot_answer}\n------------------------------------------------------------------------------------\nОценка: {feedback[text_feedback]}', chat_id=chat_id)

    async def about_team(self, message: Message):
        photo_team = FSInputFile("photo_team.png")
        await self.bot.send_photo(photo=photo_team, chat_id=message.chat.id)
    
    # Дописать!
    async def add_knowledge(self, message: Message):
        with open('rag_app\dataset_eda.csv', mode='a') as file:
            writer = csv.writer(file)
            writer.writerow(['question', 'answer'])

    def get_user_name(self, message: Message):
        return message.from_user.full_name if message.from_user.full_name else message.from_user.username

    async def generate(self, message: Message):
        user_id = message.from_user.id
        user_name = self.get_user_name(message)
        chat_id = user_id
        conv_id = self.db.get_current_conv_id(chat_id)
        system_prompt = self.db.get_system_prompt(chat_id, self.default_prompt)

        content = await self._build_content(message)
        if not isinstance(content, str):
            await message.answer("Выбранная модель не может обработать ваше сообщение")
            return
        if content is None:
            await message.answer("Такой тип сообщений (ещё) не поддерживается")
            return

        self.db.save_user_message(content, conv_id=conv_id, user_id=user_id, user_name=user_name)
        placeholder = await message.answer("💬")

        try:
            answer = await self.query_api(
                user_content=content,
                system_prompt=system_prompt
            )
            chunk_size = self.chunk_size
            answer_parts = [answer[i:i + chunk_size] for i in range(0, len(answer), chunk_size)]
            markup = self.likes_kb.as_markup()
            new_message = await placeholder.edit_text(answer_parts[0], parse_mode=None, reply_markup=markup)

            self.db.save_assistant_message(
                content=answer,
                conv_id=conv_id,
                message_id=new_message.message_id,
                system_prompt=system_prompt
            )

        except Exception:
            traceback.print_exc()
            await placeholder.edit_text("Что-то пошло не так")


    async def save_feedback(self, callback: CallbackQuery):
        user_id = callback.from_user.id
        message_id = callback.message.message_id
        feedback = callback.data.split(":")[1]
        self.db.save_feedback(feedback, user_id=user_id, message_id=message_id)
        await self.bot.edit_message_reply_markup(
            chat_id=callback.message.chat.id,
            message_id=message_id,
            reply_markup=None
        )

    async def show_commands(self, callback: CallbackQuery):
        table = pt.PrettyTable(['Команда', 'Описание'])
        data = [
            ('start', 'Приветственное сообщение'),
            ('history', 'Просмотреть историю'),
            ('reset_history', 'Очистить историю'),
            ('set_system', 'Установить новый системный промпт'),
            ('get_system', 'Просмотреть текущий системный промпт'),
            ('reset_system', 'Сбросить системный промпт до значения по умолчанию'),
            ('team', 'Узнать информацию о команде разработчиков'),
        ]
        for command, description in data:
            table.add_row([command, description])
        await callback.message.edit_text(
            f'В боте доступны следующие команды:\n```python\n{table}```',
            parse_mode='MarkdownV2'
        )
    

    @staticmethod
    def _merge_messages(messages):
        new_messages = []
        prev_role = None
        for m in messages:
            content = m["text"]
            role = m["role"]
            if content is None:
                continue
            if role == prev_role:
                is_current_str = isinstance(content, str)
                is_prev_str = isinstance(new_messages[-1]["text"], str)
                if is_current_str and is_prev_str:
                    new_messages[-1]["text"] += "\n\n" + content
                    continue
            prev_role = role
            new_messages.append(m)
        return new_messages

    def _crop_content(self, content):
        if isinstance(content, str):
            return content.replace("\n", " ")[:40]
        return "Not text"

    async def query_api(self, user_content, system_prompt: str) -> tuple:
        names, pages, chunks, relevant_score = top_k_rerank(user_content, retriever, reranker)
        if relevant_score >= 0.52:
            answer = vllm_infer(tokenizer, llm, chunks, user_content, system_prompt)
            if answer[0] == 'Я не могу ответить на ваш вопрос.':
                return answer[0]
            else:
                generated_text = '''{llm_gen}\n===================================\nИсточники дополнительной информации:\nДокумент {doc_name}, {page_number}'''
                formatted_answer = generated_text.format(
                    llm_gen=answer[0],
                    doc_name=names[0], page_number=pages[0]
                )
                return formatted_answer
        return 'Данный вопрос выходит за рамки компетенций бота. Пожалуйста, переформулируйте вопрос или попросите вызвать сотрудника.'

    
    async def _get_text_from_audio(audio):
        pass

    async def _build_content(self, message: Message):
        content_type = message.content_type
        if content_type == "text":
            text = message.text
            return text
        elif content_type == 'audio':
            voice = message.voice.get_file()
            file_name = uuid.uuid4()
            os.makedirs('voices', exist_ok=True)
            await self.bot.download_file(file_path=voice.file_path, destination=os.path.join('voices', file_name))
            text = self._get_text_from_audio(os.path.join('voices', file_name))
            os.remove(os.path.join('voices', file_name))
            return text
        return None


def main(
    bot_token: str,
    db_path: str,
    history_max_tokens: int = 4500,
    chunk_size: int = 2000,
) -> None:
    global tokenizer, llm, retriever, reranker
    tokenizer, llm, retriever, reranker = start_rag()
    bot = LlmBot(
        bot_token=bot_token,
        db_path=db_path,
        history_max_tokens=history_max_tokens,
        chunk_size=chunk_size,
    )
    asyncio.run(bot.start_polling())


if __name__ == "__main__":
    fire.Fire(main)
