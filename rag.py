import numpy as np
import os
import torch
import requests

from llama_index.core import Document
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.core import VectorStoreIndex
from llama_index.core.postprocessor import SimilarityPostprocessor
from llama_index.core import Settings
from llama_index.llms.vllm import Vllm
from llama_index.core import PromptTemplate
from transformers import BitsAndBytesConfig, AutoModel, AutoTokenizer
from sentence_transformers import CrossEncoder
from vllm import LLM, SamplingParams
from langchain_community.document_loaders import TextLoader, Docx2txtLoader, PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from warnings import filterwarnings
filterwarnings('ignore')


def file_to_chunks(file_name, sep, chunk_size, chunk_overlap):
    file_ext = file_name.split('.')[-1]
    file_path = file_name
    
    overall_chunks = []
    overall_pages = []
    
    # Загружаем содержимое файла 
    if file_ext == 'txt':
        loader = TextLoader(file_path, encoding='utf-8')
    elif file_ext == 'docx':
        loader = Docx2txtLoader(file_path)
    elif file_ext == 'pdf':
        loader = PyPDFLoader(file_path)
    else:
        return
    file = loader.load()
    
    text_splitter = RecursiveCharacterTextSplitter(
        separators = sep,
        chunk_size = chunk_size,
        chunk_overlap = chunk_overlap,
        length_function = len,
        is_separator_regex = False,
        add_start_index = False
    )
    
    for docs in file:
        content = docs.page_content
        chunks = text_splitter.split_text(content)
        
        page_number = chunks[0][chunks[0].find('Страница'):].strip().split('\n')[0].strip()
        overall_chunks.append(chunks[0][chunks[0].find('Версия') + 11:].strip())
        overall_pages.append(page_number)

    return overall_chunks, overall_pages


def create_docs():
    documents = []
    sep = '\n'
    chunk_size = 2048
    chunk_overlap = 128
    
    for file in os.listdir('./docs_for_rag_v2'):
        file_name = os.path.join('./docs_for_rag_v2', file)
        try:
            chunks, pages = file_to_chunks(file_name, sep, chunk_size, chunk_overlap)
        except:
            print(file_name)
        
        for chunk, page in zip(chunks, pages):
            metadata = {
                "название документа": file,
                "страница в документе": page,
                "описание": chunk
            }

            documents.append(Document(text=chunk, metadata=metadata,
                             excluded_embed_metadata_keys=["название документа", "страница в документе"]))

    return documents


def top_k_rerank(query: str, retriever, reranker, top_k: int = 2):
    documents = retriever.retrieve(query)
    # relevant_score = max(doc.score for doc in documents)
    relevant_score = documents[0].score
    print(f'Наивысшее знаение релевантности документов: {relevant_score}')
    
    candidate_texts = [x.text for x in documents]
    candidate_names = [x.metadata['название документа'] for x in documents]
    candidate_pages = [x.metadata['страница в документе'] for x in documents]
    
    rerank_scores = reranker.predict(list(zip([query] * len(candidate_texts), candidate_texts)))
    ranked_indices = np.argsort(rerank_scores)[::-1]
    
    names = [candidate_names[i] for i in ranked_indices][:top_k]
    pages = [candidate_pages[i] for i in ranked_indices][:top_k]
    texts = [candidate_texts[i] for i in ranked_indices][:top_k]
    
    return names, pages, texts, relevant_score


generated_text = '''
    {llm_gen}
    ===================================
    Источники дополнительной информации:
    Документ {doc_name}, {page_number}
    '''

def vllm_infer(
    tokenizer,
    wrapped_llm: LLM,
    texts, 
    query,
    system_prompt,
    temperature: float = 0.2,
    top_p: float = 0.9,
    top_k: int = 30,
    max_tokens: int = 512,
    repetition_penalty: float = 1.1
):
    
    user_prompt = '''Используй только следующий контекст, чтобы кратко ответить на вопрос в конце.
        Не пытайся выдумывать ответ. Если контекст не соотносится с вопросом, скажи, что ты не можешь ответить на данный вопрос. 
        Если вопрос не соотносится с банковской тематикой, выведи фразу "Я не могу ответить на ваш вопрос." и не выводи ничего больше.
        Контекст:
        ===========
        {texts}
        ===========
        Вопрос:
        ===========
        {query}'''.format(texts=texts, query=query)
    
    sampling_params = SamplingParams(
        temperature=temperature,
        top_p=top_p,
        top_k=top_k,
        max_tokens=max_tokens,
        repetition_penalty=repetition_penalty
    )
    
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt}
    ]
    
    answers = []
    
    prompt = wrapped_llm.llm_engine.tokenizer.tokenizer.apply_chat_template(conversation=messages, add_generation_prompt=True, tokenize=False)
    prompts = [prompt]
    
    outputs = wrapped_llm.generate(prompts, sampling_params)
    
    
    for output in outputs:
        generated_text = output.outputs[0].text
        answers.append(generated_text)
        # display(Markdown(generated_text))
    
    torch.cuda.empty_cache()
    return answers
    


def start_rag():
    docs = create_docs()
    tokenizer = AutoTokenizer.from_pretrained('./llm')
    llm = LLM(model='./llm',  dtype=torch.float16, max_seq_len_to_capture=8192) # 8192 for llama
    index = VectorStoreIndex.from_documents(documents=docs, show_progress=True)
    retriever = index.as_retriever(similarity_top_k=7, node_postprocessors=[
                               SimilarityPostprocessor(similarity_cutoff=0.85)])
    reranker = CrossEncoder('./reranker')
    return tokenizer, llm, retriever, reranker

