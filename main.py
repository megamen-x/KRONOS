from fastapi import FastAPI, HTTPException
from contextlib import asynccontextmanager
from pydantic import BaseModel, Field, ValidationError
from typing import List, Union
import uvicorn
from rag import *


tokenizer, llm, retriever, reranker = None, None, None, None
system_prompt = 'Ты — Сайга, русскоязычный автоматический ассистент. Ты разговариваешь с людьми и помогаешь им.'

@asynccontextmanager
async def lifespan(app: FastAPI):
    global tokenizer, llm, retriever, reranker
    Settings.embed_model = HuggingFaceEmbedding(
        model_name="./embedder"
    )
    tokenizer, llm, retriever, reranker = start_rag()
    yield
    del tokenizer
    del llm
    del retriever
    del reranker


app = FastAPI(
    title="Assistant API",
    version="0.1.0",
    lifespan=lifespan
)

class ValidationErrorDetail(BaseModel):
    loc: List[Union[str, int]]
    msg: str
    type: str

class HTTPValidationError(BaseModel):
    detail: List[ValidationErrorDetail]

class Request(BaseModel):
    query: str = Field(..., title="Query")

class Response(BaseModel):
    text: str = Field(..., title="Text")

@app.post("/assist", response_model=Response, responses={422: {"model": HTTPValidationError}})
async def assist(request: Request):
    global retriever, reranker
    try:
        # Process the query
        names, pages, chunks, relevant_score = top_k_rerank(request.query, retriever, reranker)
        if relevant_score >= 0.52:
            answer = vllm_infer(tokenizer, llm, chunks, request.query, system_prompt)
            if answer[0] == 'Я не могу ответить на ваш вопрос.':
                return Response(text=answer[0])
            else:
                formatted_answer = generated_text.format(
                    llm_gen=answer[0],
                    doc_name=names[0], page_number=pages[0]
                )
                return Response(text=formatted_answer)
        return Response(text='Данный вопрос выходит за рамки компетенций бота. Пожалуйста, переформулируйте вопрос или попросите вызвать сотрудника.')
        # Create the response
        
    except ValidationError as e:
        raise HTTPException(
            status_code=422,
            detail=[{"loc": ["body", error['loc'][0]], "msg": error['msg'], "type": error['type']} for error in e.errors()]
        )

if __name__ == "__main__":
    uvicorn.run('main:app', port=9875, host='0.0.0.0', reload=False)