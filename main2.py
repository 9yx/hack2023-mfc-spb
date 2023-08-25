import pandas as pd
import sklearn
import numpy as np
from sentence_transformers import SentenceTransformer

if __name__ == '__main__':

    dataset = pd.read_excel('train_dataset_Датасет.xlsx', 'Лист1')['QUESTION'].astype(str).tolist()  #[:-50]
    answer_dataset = pd.read_excel('train_dataset_Датасет.xlsx', 'Лист1')['ANSWER'].astype(str)
    #test_dataset = pd.read_excel('train_dataset_Датасет.xlsx', 'Лист1')['QUESTION'].astype(str).tail(50).tolist()

    #model = SentenceTransformer("quora-distilbert-multilingual")
    model = SentenceTransformer('intfloat/multilingual-e5-large')

    faq_embeddings = model.encode(dataset, normalize_embeddings=True)
    # test_q_emb = model.encode(test_dataset)
    from scipy.spatial.distance import cdist

    # for q, qe in zip(test_dataset, test_q_emb):
    #     distances = cdist([qe], faq_embeddings, "cosine")[0]
    #     ind = np.argsort(distances, axis=0)[:3]
    #     print("\n Test Question: \n " + q)
    #     for i, (dis, text) in enumerate(zip(distances[ind], [dataset[i] for i in ind])):
    #         print(dis, ind[i], text, sep="\t")


    def get_best(query, K=3):
        query_embedding = model.encode([query])
        distances = cdist(query_embedding, faq_embeddings, "cosine")[0]
        ind = np.argsort(distances, axis=0)
        print("\n" + query)
        for c, i in list(zip(distances[ind], ind))[:K]:
            print(c, dataset[i], answer_dataset.loc[i], sep="\t")


    get_best("Какие условия в услуге 33?", 3)
