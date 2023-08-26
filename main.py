import string
import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
from scipy.spatial.distance import cdist
import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, GenerationConfig
import uvicorn
from fastapi import FastAPI
from pydantic import BaseModel


def generate(prompt):
    data = tokenizer(f"{prompt}", return_tensors="pt").to(model.device)
    output_ids = fred_t5_large.generate(
        **data,
        generation_config=generation_config
    )[0]
    out = tokenizer.decode(output_ids.tolist(), skip_special_tokens=True)
    return out.replace('<extra_id_0>', '').strip()


def lowercase_text(text):
    return text.lower()


def remove_extra_spaces(text):
    return " ".join(text.split())


def remove_punctuation(text):
    translation_table = str.maketrans("", "", string.punctuation)
    return text.translate(translation_table)


def get_best(query, K=3):
    query_embedding = model.encode([query])
    distances = cdist(query_embedding, faq_embeddings, "cosine")[0]
    ind = np.argsort(distances, axis=0)
    print("\n" + query)
    for c, i in list(zip(distances[ind], ind))[:K]:
        # print(c, question_dataset[i], answer_dataset.loc[i], sep="\t")
        question = remove_punctuation(
            lowercase_text(remove_extra_spaces(remove_punctuation(question_dataset[i])))).replace("\n", "")
        answer = lowercase_text(remove_extra_spaces(remove_punctuation(answer_dataset.loc[i]))).replace("\n", "")
        text = f'''Вопрос: {question}?  Ответ: {answer}'''
        question = query
        prompt = '''<SC6>Текст: {}\nВопрос: {}\nОтвет: <extra_id_0>'''.format(text, question)
        result = generate(prompt)
        return result


# device = torch.cuda.current_device() if torch.cuda.is_available() and torch.cuda.mem_get_info()[0] >= 2*1024**3 else -1
device = torch.device("cpu")
generation_config = GenerationConfig.from_pretrained("Den4ikAI/FRED-T5-LARGE_text_qa")
tokenizer = AutoTokenizer.from_pretrained("Den4ikAI/FRED-T5-LARGE_text_qa")
fred_t5_large = AutoModelForSeq2SeqLM.from_pretrained("Den4ikAI/FRED-T5-LARGE_text_qa").to(device)
fred_t5_large.eval()
question_dataset = pd.read_excel('train_dataset_Датасет.xlsx', 'Лист1')['QUESTION'].astype(str).tolist()  # [:-50]
answer_dataset = pd.read_excel('train_dataset_Датасет.xlsx', 'Лист1')['ANSWER'].astype(str)
# model = SentenceTransformer('intfloat/multilingual-e5-large', device=device)
model = SentenceTransformer('sentence-transformers/quora-distilbert-multilingual', device=device)
faq_embeddings = model.encode(question_dataset, normalize_embeddings=True)


def process_query(query):
    return get_best(query, 1)


class QADataModel(BaseModel):
    question: str


app = FastAPI()


@app.post("/answering")
async def qa(input_data: QADataModel):
    result = process_query(input_data.question)
    return {"answer": result}


if __name__ == '__main__':
    uvicorn.run('main:app', workers=5)