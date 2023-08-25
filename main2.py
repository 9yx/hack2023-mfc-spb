import pandas as pd
import sklearn
import numpy as np

if __name__ == '__main__':
    wwf_faq=["I haven’t received my adoption pack. What should I do?",
             "How quickly will I receive my adoption pack?",
             "How can I renew my adoption?",
             "How do I change my address or other contact details?",
             "Can I adopt an animal if I don’t live in the UK?",
             "If I adopt an animal, will I be the only person who adopts that animal?",
    "My pack doesn't contain a certicate",
    "My adoption is a gift but won’t arrive on time. What can I do?",
    "Can I pay for an adoption with a one-off payment?",
    "Can I change the delivery address for my adoption pack after I’ve placed my order?",
    "How long will my adoption last for?",
    "How often will I receive updates about my adopted animal?",
    "What animals do you have for adoption?",
    "How can I nd out more information about my adopted animal?",
    "How is my adoption money spent?",
    "What is your refund policy?",
    "An error has been made with my Direct Debit payment, can I receive a refund?",
    "How do I change how you contact me?"]

    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer("quora-distilbert-multilingual")

    faq_embeddings = model.encode(wwf_faq)
    test_questions = ["What should be done, if the adoption pack did not reach to me?",
                      " How fast is my adoption pack delivered to me?",
                      "What should I do to renew my adoption?",
                      "What should be done to change adress and contact details ?",
                      "I live outside of the UK, Can I still adopt an animal?"]
    test_q_emb = model.encode(test_questions)
    from scipy.spatial.distance import cdist

    for q, qe in zip(test_questions, test_q_emb):
        distances = cdist([qe], faq_embeddings, "cosine")[0]
        ind = np.argsort(distances, axis=0)[:3]
        print("\n Test Question: \n " + q)
        for i, (dis, text) in enumerate(zip(distances[ind], [wwf_faq[i] for i in ind])):
            print(dis, ind[i], text, sep="\t")


    def get_best(query, K=3):
        query_embedding = model.encode([query])
        distances = cdist(query_embedding, faq_embeddings, "cosine")[0]
        ind = np.argsort(distances, axis=0)
        print("\n" + query)
        for c, i in list(zip(distances[ind], ind))[:K]:
            print(c, wwf_faq[i], sep="\t")


    get_best("How do I change my contact info?", 3)
    get_best("How do I get my plane ticket if I bought it online?")