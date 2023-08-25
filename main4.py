import torch
import transformers
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, GPT2Tokenizer, T5ForConditionalGeneration, \
    GenerationConfig

if __name__ == '__main__':
    use_cuda = torch.cuda.is_available()
    device = torch.device("cpu")
    generation_config = GenerationConfig.from_pretrained("Den4ikAI/FRED-T5-LARGE_text_qa")
    tokenizer = AutoTokenizer.from_pretrained("Den4ikAI/FRED-T5-LARGE_text_qa")
    model = AutoModelForSeq2SeqLM.from_pretrained("Den4ikAI/FRED-T5-LARGE_text_qa").to(device)
    model.eval()


    def generate(prompt):
        data = tokenizer(f"{prompt}", return_tensors="pt").to(model.device)
        output_ids = model.generate(
            **data,
            generation_config=generation_config
        )[0]
        print(tokenizer.decode(data["input_ids"][0].tolist()))
        out = tokenizer.decode(output_ids.tolist())
        return out

    text = '''Название услуги: Предоставлять дополнительную меру социальной поддержки в виде единовременной компенсационной выплаты при рождении ребенка (усыновлении в возрасте до шести месяцев) для приобретения предметов детского ассортимента и продуктов детского питания. Вопрос: Какие условия являются обязательными для получения 33 услуги?  Ответ: ребенок - гражданин РФ, имеющий место жительства (постоянную регистрацию) в СПб И один из родителей (усыновителей) или оба родителя (усыновителя) - граждане РФ, имеющие место жительства (постоянную регистрацию) в СПб; Мать - встала на медицинский учет по поводу беременности в учреждении здравоохранения в срок до 20 недель (включительно) ИЛИ ребенка усыновили в возрасте до шести месяцев; Обращение последовало не позднее полутора лет со дня рождения ребенка.'''
    question = "Название услуги 33?"

    prompt = '''<SC6>Текст: {}\nВопрос: {}\nОтвет: <extra_id_0>'''.format(text, question)
    print(generate(prompt))

    # use_cuda = torch.cuda.is_available()
    # device = torch.device("cpu")
    #
    # tokenizer = GPT2Tokenizer.from_pretrained('ai-forever/FRED-T5-1.7B', eos_token='</s>')
    # model = T5ForConditionalGeneration.from_pretrained('ai-forever/FRED-T5-1.7B')
    # model.to(device)
    #
    # prompt = '<SC6>Текст: Какие условия являются обязательными для получения 33 услуги? Ребенок - гражданин РФ, ' \
    #          'имеющий место жительства (постоянную регистрацию) в СПб И один из родителей (усыновителей) или оба родителя (усыновителя) - ' \
    #          'граждане РФ, имеющие место жительства (постоянную регистрацию) в СПб, мать - встала на медицинский учет по поводу беременности в' \
    #          ' учреждении здравоохранения в срок до 20 недель (включительно) ИЛИ ребенка усыновили в возрасте до шести месяцев; ' \
    #          'обращение последовало не позднее полутора лет со дня рождения ребенка.\nВопрос: Какие условия в услуге 33?\nОтвет: <extra_id_0>'
    #
    # input_ids = torch.tensor([tokenizer.encode(prompt)]).to(device)
    # outputs=model.generate(input_ids, eos_token_id=tokenizer.eos_token_id, early_stopping=True,  max_length=512)
    # print(tokenizer.decode(outputs[0], skip_special_tokens=True))
