import string

import pandas as pd

import numpy as np
from sentence_transformers import SentenceTransformer
from scipy.spatial.distance import cdist
import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, GenerationConfig


def generate(prompt):
    data = tokenizer(f"{prompt}", return_tensors="pt").to(model.device)
    output_ids = fredT5Large.generate(
        **data,
        generation_config=generation_config
    )[0]
    out = tokenizer.decode(output_ids.tolist(), skip_special_tokens=True)
    return out.replace('<extra_id_0>', '').strip()


def simplify_text(text):
    text.lower()

    #remove_extra_spaces
    " ".join(text.split())
    #line break to spaces
    text.replace("\n", " ")

    #punctuation simplifier
    translation_table = str.maketrans("", "", string.punctuation.replace("-", ""))
    return text.translate(translation_table)


def get_best(query, K=3):
    query_embedding = model.encode([query])
    distances = cdist(query_embedding, faq_embeddings, "cosine")[0]
    ind = np.argsort(distances, axis=0)
    print("\n" + query)
    for c, i in list(zip(distances[ind], ind))[:K]:
        # print(c, question_dataset[i], answer_dataset.loc[i], sep="\t")
        question = simplify_text(question_dataset[i])
        answer = simplify_text(answer_dataset.loc[i])
        text = f'''Вопрос: {question}?  Ответ: {answer}'''
        question = query
        prompt = '''<SC6>Текст: {}\nВопрос: {}\nОтвет: <extra_id_0>'''.format(text, question)
        print(prompt)
        tmpAns=generate(prompt)
        print(tmpAns)


if __name__ == '__main__':
    use_cuda = torch.cuda.is_available()
    device = torch.device("cpu")
    generation_config = GenerationConfig.from_pretrained("Den4ikAI/FRED-T5-LARGE_text_qa")
    tokenizer = AutoTokenizer.from_pretrained("Den4ikAI/FRED-T5-LARGE_text_qa")
    fredT5Large = AutoModelForSeq2SeqLM.from_pretrained("Den4ikAI/FRED-T5-LARGE_text_qa").to(device)
    fredT5Large.eval()
    question_dataset = pd.read_excel('train_dataset_Датасет.xlsx', 'Лист1')['QUESTION'].astype(str).tolist()  # [:-50]
    answer_dataset = pd.read_excel('train_dataset_Датасет.xlsx', 'Лист1')['ANSWER'].astype(str)
    # model = SentenceTransformer('intfloat/multilingual-e5-large', device=device)
    model = SentenceTransformer('sentence-transformers/quora-distilbert-multilingual', device=device)
    faq_embeddings = model.encode(question_dataset, normalize_embeddings=True)
    get_best("""На какую сумму может рассчитывать заявитель при оформлении услуги "Предоставлять меру социальной поддержки детям-сиротам, детям, оставшимся без попечения родителей, лицам из числа детей-сирот и детей, оставшихся без попечения родителей, в виде денежной компенсации стоимости путевки в организации отдыха детей и молодежи и их оздоровления в случае самостоятельного приобретения путевок в организации отдыха детей и молодежи и их оздоровления опекунами (попечителями), приемными родителями детей-сирот и детей, оставшихся без попечения родителей, или лицами из числа детей-сирот и детей, оставшихся без попечения родителей"?""", 1)
