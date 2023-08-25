import torch
import transformers

if __name__ == '__main__':
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")

    t5_tokenizer = transformers.GPT2Tokenizer.from_pretrained("SiberiaSoft/SiberianPersonaFred")
    t5_model = transformers.T5ForConditionalGeneration.from_pretrained("SiberiaSoft/SiberianPersonaFred")

    prompt = '<SC6>Вопрос: Какие условия являются обязательными для получения 33 услуги?  Ответ: ребенок - ' \
             'гражданин РФ, имеющий место жительства (постоянную регистрацию) в СПб И один из родителей (усыновителей) ' \
             'или оба родителя (усыновителя) - граждане РФ, имеющие место жительства (постоянную регистрацию) в СПб; ' \
             'Мать - встала на медицинский учет по поводу беременности в учреждении здравоохранения в срок до 20 недель ' \
             '(включительно) ИЛИ ребенка усыновили в возрасте до шести месяцев; Обращение последовало не позднее полутора ' \
             'лет со дня рождения ребенка. Вопрос: Какие условия в услуге 33? Ответ: <extra_id_0>'

    input_ids = t5_tokenizer(prompt, return_tensors='pt').input_ids
    out_ids = t5_model.generate(input_ids=input_ids.to(device), do_sample=True, temperature=0.9,
                                max_new_tokens=512, top_p=0.85,
                                top_k=2, repetition_penalty=1.2)
    t5_output = t5_tokenizer.decode(out_ids[0][1:])
    if '</s>' in t5_output:
        t5_output = t5_output[:t5_output.find('</s>')].strip()
    t5_output = t5_output.replace('<extra_id_0>', '').strip()
    print('B:> {}'.format(t5_output))
