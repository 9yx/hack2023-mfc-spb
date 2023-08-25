import pandas as pd

import numpy as np
from sentence_transformers import SentenceTransformer
from scipy.spatial.distance import cdist
import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, GenerationConfig


def generate(prompt):
    data = tokenizer(f"{prompt}", return_tensors="pt").to(model.device)
    output_ids = model.generate(
        **data,
        generation_config=generation_config
    )[0]
    out = tokenizer.decode(output_ids.tolist(), skip_special_tokens=True)
    return out.replace('<extra_id_0>', '').strip()


if __name__ == '__main__':
    use_cuda = torch.cuda.is_available()
    device = torch.device("cpu")
    generation_config = GenerationConfig.from_pretrained("Den4ikAI/FRED-T5-LARGE_text_qa")
    tokenizer = AutoTokenizer.from_pretrained("Den4ikAI/FRED-T5-LARGE_text_qa")
    fredT5Large = AutoModelForSeq2SeqLM.from_pretrained("Den4ikAI/FRED-T5-LARGE_text_qa").to(device)
    fredT5Large.eval()

    question_dataset = pd.read_excel('train_dataset_Датасет.xlsx', 'Лист1')['QUESTION'].astype(str).tolist()  # [:-50]
    answer_dataset = pd.read_excel('train_dataset_Датасет.xlsx', 'Лист1')['ANSWER'].astype(str)
    model = SentenceTransformer('intfloat/multilingual-e5-large', device=device)

    faq_embeddings = model.encode(question_dataset, normalize_embeddings=True)


    def get_best(query, K=3):
        query_embedding = model.encode([query])
        distances = cdist(query_embedding, faq_embeddings, "cosine")[0]
        ind = np.argsort(distances, axis=0)
        print("\n" + query)
        for c, i in list(zip(distances[ind], ind))[:K]:
            print(c, question_dataset[i], answer_dataset.loc[i], sep="\t")
            text = f'''Вопрос: {question_dataset[i]}?  Ответ: {answer_dataset.loc[i]}'''
            question = query
            prompt = '''<SC6>Текст: {}\nВопрос: {}\nОтвет: <extra_id_0>'''.format(text, question)
            print(generate(prompt))


    get_best("Какие условия в услуге 33?", 3)
