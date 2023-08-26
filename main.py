import configparser
import os

import numpy as np
import pandas as pd
import torch
import uvicorn
from fastapi import FastAPI, UploadFile
from pydantic import BaseModel
from scipy.spatial.distance import cdist
from sentence_transformers import SentenceTransformer
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, GenerationConfig, AutoModelForSequenceClassification

import model_like_service as ms

if __name__ == '__main__':
    uvicorn.run('main:app', workers=1, host="0.0.0.0", port=9000)


def generate(prompt):
    data = tokenizer(f"{prompt}", return_tensors="pt").to(model.device)
    output_ids = fred_t5_large.generate(
        **data,
        generation_config=generation_config
    )[0]
    out = tokenizer.decode(output_ids.tolist(), skip_special_tokens=True)
    return out.replace('<extra_id_0>', '').strip()


def simplify_text(text):
    text = text.lower()

    # remove_extra_spaces
    text = " ".join(text.split())
    # line break to spaces
    text = text.replace("\n", " ")

    text = text.replace(";", ",")
    text = text.replace("- ", "-")

    # punctuation simplifier
    # translation_table = str.maketrans("", "", string.punctuation.replace("-", ""))
    return text


def get_best(query, K=3):
    query_embedding = model.encode([query])
    distances = cdist(query_embedding, faq_embeddings, "cosine")[0]
    ind = np.argsort(distances, axis=0)
    print("\n" + query)
    text = ""
    regression = ms.order_with_like(list(zip(distances[ind], ind)))
    index = None

    queryes = []
    questions = []
    answers = []
    for c, i in regression[:K]:
        question = simplify_text(question_dataset[i])
        answer = simplify_text(answer_dataset.loc[i])
        print(c, question, answer)
        if c > 0.035:
            continue
        questions.append(question)
        queryes.append(query)
        answers.append(answer)

    if len(answers) == 0:
        return [-1, "Не знаю ответа"]

    features = cross_encoder_tokenizer(queryes, answers, padding=True, truncation=True, return_tensors="pt")
    cross_encoder_model.eval()
    with torch.no_grad():
        index = torch.argmax(cross_encoder_model(**features).logits).item()
        text = f'''Вопрос: {questions[index]}\nОтвет: {answers[index]}\n'''
        print("Выбрали вариант:", text)

    prompt = '''<SC6>Текст: {}\nВопрос: {}\nОтвет: <extra_id_0>'''.format(text.strip(), query)
    result = generate(prompt)
    return [index, result]


# device = torch.cuda.current_device() if torch.cuda.is_available() and torch.cuda.mem_get_info()[0] >= 2*1024**3 else -1
#device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device = torch.device("cpu")

skipAiTraining = False
# Проверка существования файла
if os.path.exists('config.ini'):
    # Создание объекта ConfigParser
    config = configparser.ConfigParser()
    # Чтение файла конфигурации
    config.read('config.ini')

    # Чтение значения свойства property1
    if config.get('SETTINGS', 'env'):
        device = torch.device("cpu")
    if config.get('SETTINGS', 'aiModelDisabled'):
        skipAiTraining = True

generation_config = None
tokenizer = None
fred_t5_large = None
question_dataset = None
answer_dataset = None
model = None
faq_embeddings = None
cross_encoder_model = None
cross_encoder_tokenizer = None

if not skipAiTraining:
    generation_config = GenerationConfig.from_pretrained("Den4ikAI/FRED-T5-LARGE_text_qa")
    tokenizer = AutoTokenizer.from_pretrained("Den4ikAI/FRED-T5-LARGE_text_qa")
    fred_t5_large = AutoModelForSeq2SeqLM.from_pretrained("Den4ikAI/FRED-T5-LARGE_text_qa").to(device)
    fred_t5_large.eval()
    question_dataset = pd.read_excel('train_dataset_Датасет.xlsx', 'Лист1')['QUESTION'].astype(str).tolist()
    answer_dataset = pd.read_excel('train_dataset_Датасет.xlsx', 'Лист1')['ANSWER'].astype(str)
    # model = SentenceTransformer('intfloat/multilingual-e5-large', device=device)
    # model = SentenceTransformer('Den4ikAI/sbert_large_mt_ru_retriever', device=device)
    model = SentenceTransformer('sentence-transformers/quora-distilbert-multilingual', device=device)
    faq_embeddings = model.encode(question_dataset, normalize_embeddings=True)

    cross_encoder_model = AutoModelForSequenceClassification.from_pretrained('cross-encoder/mmarco-mMiniLMv2-L12-H384-v1')
    cross_encoder_tokenizer = AutoTokenizer.from_pretrained('cross-encoder/mmarco-mMiniLMv2-L12-H384-v1')


def process_query(query):
    return get_best(query, 3)


class QADataModel(BaseModel):
    question: str


app = FastAPI()


@app.post("/upload_dataset")
async def upload_dataset(file: UploadFile):
    global question_dataset
    global answer_dataset
    global faq_embeddings
    contents = await file.read()

    with open(f"train_dataset_Датасет.xlsx", "wb") as f:
        f.write(contents)

    question_dataset = pd.read_excel('train_dataset_Датасет.xlsx', 'Лист1')['QUESTION'].astype(str).tolist()
    answer_dataset = pd.read_excel('train_dataset_Датасет.xlsx', 'Лист1')['ANSWER'].astype(str)
    faq_embeddings = model.encode(question_dataset, normalize_embeddings=True)


@app.post("/answering")
async def qa(input_data: QADataModel):
    result = process_query((input_data.question + "?").replace("??", "?"))
    return {"index": str(result[0]), "answer": result[1]}


class AnswerScoreDto(BaseModel):
    index: str
    like: bool


@app.post("/rate")
async def rate(input_data: AnswerScoreDto):
    ms.store_rating(int(input_data.index), input_data.like)
    return "OK"
