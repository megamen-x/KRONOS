<a name="readme-top"></a>  
<img width="100%" src="https://github.com/megamen-x/KRONOS/blob/main/github-assets/pref_github.png" alt="megamen banner">
<div align="center">
  <p align="center">
    <!--<h1 align="center">KRONOS</h1>-->
  </p>
  <p align="center">
    <p></p>
    <!-- <p><strong>Умный ассистент РОСАТОМ для 1С: Предприятие.</strong></p> -->
    Создано <strong>megamen</strong>, совместно с <br /> <strong> Госкорпорацией «РОСАТОМ»</strong>
    <br /><br />
    <a href="https://github.com/megamen-x/KRONOS/issues" style="color: black;">Сообщить об ошибке</a>
    ·
    <a href="https://github.com/megamen-x/KRONOS/discussions/1" style="color: black;">Предложить улучшение</a>
  </p>
</div>

**Содержание:**
- [Проблематика задачи](#title1)
- [Описание решения](#title2)
- [Тестирование решения](#title3)
- [Обновления](#title4)

## <h3 align="start"><a id="title1">Проблематика задачи</a></h3> 
Необходимо создать, с применением технологий искусственного интеллекта, сервис, куда можно задавать вопросы текстом (полноценно работающий ботпомощник) и получать генеративный ответ с указанием ссылки на источники, откуда был
взят ответ.

Ключевые функции решения:
* Использование векторной базы данных;
* Переранжирование ответов при помощи модели реранкера;
* Квантизация эмбеддингов для ускорения работы решения;
* Возможность задать кастомный system prompt генерирующей модели.

<p align="right">(<a href="#readme-top"><i>Вернуться наверх</i></a>)</p>

## <h3 align="start"><a id="title2">Описание решения</a></h3>

**Machine Learning:**

[![Python](https://img.shields.io/badge/python-3670A0?style=for-the-badge&logo=python&logoColor=ffdd54)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-%23EE4C2C.svg?style=for-the-badge&logo=PyTorch&logoColor=white)](https://pytorch.org/)

 - **Общая схема решения:**

  <img width="100%" src="https://github.com/megamen-x/KRONOS/blob/main/github-assets/sheme_github.png" alt="megamen banner">

 - **Использованные модели:**
    - **```Embedder```**:
      - intfloat/multilingual-e5-large;
    - **```LLM```**:
      - IlyaGusev/saiga_llama3_8b;
    - **```Reranker```**:
      - PitKoro/cross-encoder-ru-msmarco-passage.

Ссылка на телеграмм-бота для тестирования решения:
<!-- <button onclick="location.href='https://t.me/Kronos_rag_bot'" type="button">KRONOS-bot</button> -->
[<img src="https://github.com/megamen-x/KRONOS/blob/main/github-assets/tg-btn.png" height="60"/>](https://t.me/Kronos_rag_bot)

**Клиентская часть**

[![Telegram](https://img.shields.io/badge/Telegram-2CA5E0?style=for-the-badge&logo=telegram&logoColor=white)](https://core.telegram.org/)

**Серверная часть**

[![FastAPI](https://img.shields.io/badge/FastAPI-005571?style=for-the-badge&logo=fastapi)](https://fastapi.tiangolo.com/)
[![Docker](https://img.shields.io/badge/docker-%230db7ed.svg?style=for-the-badge&logo=docker&logoColor=white)](https://www.docker.com/)


<p align="right">(<a href="#readme-top"><i>Вернуться наверх</i></a>)</p>


## <h3 align="start"><a id="title3">Тестирование решения</a></h3> 

Данный репозиторий предполагает следующую конфигурацию тестирования решения:
  
  **```Telegram-bot + FastAPI + ML-models;```**

<details>
  <summary> <strong><i> Инструкция по запуску TelegramAPI-бота:</i></strong> </summary>
  
  - В Visual Studio Code (**Windows-PowerShell activation recommended**) через терминал последовательно выполнить следующие команды:
  
    - Клонирование репозитория:
    ```
    git clone https://github.com/megamen-x/KRONOS.git
    ```
    - Создание и активация виртуального окружения (Протестировано на **Python 3.10.10**):
    ```
    cd ./KRONOS
    python -m venv .venv
    .venv\Scripts\activate
    ```
    - Уставновка зависимостей (при использовании **CUDA 12.1**):
    ```
    pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
    pip3 install -r requirements.txt
    ```
    - После установки зависимостей (5-7 минут):
    ```
    python bot.py --bot_token='Token_bot' --db_path='Path_to_sqlite_DB'
    ```

</details> 

</br> 

<details>
  <summary> <strong><i> Инструкция по локальному запуску FastAPI-сервера:</i></strong> </summary>
  
  - В Visual Studio Code (**Windows-PowerShell activation recommended**) через терминал последовательно выполнить следующие команды:
  
    - Клонирование репозитория:
    ```
    git clone https://github.com/megamen-x/KRONOS.git
    ```
    - Создание и активация виртуального окружения (Протестировано на **Python 3.10.10**):
    ```
    cd ./KRONOS
    python -m venv .venv
    .venv\Scripts\activate
    ```
    - Уставновка зависимостей (при использовании **CUDA 12.1**):
    ```
    pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
    pip3 install -r requirements.txt
    ```
    - После установки зависимостей (5-7 минут):
    ```
    python main.py
    ```

</details> 

</br> 


<p align="right">(<a href="#readme-top"><i>Вернуться наверх</i></a>)</p>


## <h3 align="start"><a id="title4">Обновления</a></h3> 

***Все обновления и нововведения будут размещаться здесь!***

<p align="right">(<a href="#readme-top"><i>Вернуться наверх</i></a>)</p>
