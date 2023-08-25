import torch
from transformers import GPT2Tokenizer, T5ForConditionalGeneration

if __name__ == '__main__':
    tokenizer = GPT2Tokenizer.from_pretrained('ai-forever/FRED-T5-1.7B',eos_token='</s>')
    model = T5ForConditionalGeneration.from_pretrained('ai-forever/FRED-T5-1.7B')
    device='cuda'
    model.to(device)

    #Prefix <LM>
    lm_text='<LM>Принялся Кутузов рассказывать свою историю как он сюда попал. Началось'
    input_ids=torch.tensor([tokenizer.encode(lm_text)]).to(device)
    outputs=model.generate(input_ids,eos_token_id=tokenizer.eos_token_id,early_stopping=True)
    print(tokenizer.decode(outputs[0][1:]))
