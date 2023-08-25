from transformers import AutoModelForQuestionAnswering, AutoTokenizer, pipeline
import torch
import pandas as pd
import openpyxl

if __name__ == '__main__':
    device = torch.cuda.current_device() if torch.cuda.is_available() and torch.cuda.mem_get_info()[
        0] >= 2 * 1024 ** 3 else -1

    dataset = pd.read_excel('train_dataset_Датасет.xlsx', 'Лист1')['ANSWER'].astype(str)
    #print(dataset[:5])
    context = ' '.join(dataset[:10])
    #print(context)
    #
    #
    # model_name = "Mathnub/ruRoberta-sberquad"
    #
    # nlp = pipeline('question-answering', model=model_name, tokenizer=model_name, device=device)
    # QA_input = {
    #     'question': 'Какие основания для детской карты?',
    #     'context': context
    # }
    #
    # res = nlp(QA_input)
    # print(res)

    # Initialize tokenizer and model
    tokenizer = AutoTokenizer.from_pretrained('Mathnub/ruRoberta-sberquad')
    model = AutoModelForQuestionAnswering.from_pretrained('Mathnub/ruRoberta-sberquad')

    # Example large context

    question = "Кто является заявителем?"

    # Splitting context into chunks
    max_length = tokenizer.model_max_length - tokenizer.num_special_tokens_to_add(pair=True) - len(context) - 8
    chunked_contexts = [context[i:i + max_length] for i in range(0, len(context), max_length)]



    # Initializing variables
    all_answers = []

    # Loop over chunked contexts
    for chunk in chunked_contexts:
        # Tokenize question and chunked context
        encoding = tokenizer.encode_plus(question, chunk, add_special_tokens=True, return_tensors="pt")

        # Predict answer
        with torch.no_grad():
            outputs = model(**encoding)
            start_logits = outputs.start_logits
            end_logits = outputs.end_logits

        # Get the start and end index of the answer in the current chunk
        start_index = torch.argmax(start_logits)
        end_index = torch.argmax(end_logits) + 1

        # Convert token indices to tokens and join them
        answer = tokenizer.convert_tokens_to_string(
            tokenizer.convert_ids_to_tokens(encoding["input_ids"][0][start_index:end_index])
        )

        # Append answer to the list
        all_answers.append(answer)

    # Convert list of answers to a single string
    final_answer = " ".join(all_answers)

    print(final_answer)

