import shutil

import pandas as pd


def copyDataset():
    original_file = './train_dataset_Датасет.xlsx'
    copy_file = './train_dataset_Датасетcopy.xlsx'
    # Копирование файла
    shutil.copyfile(original_file, copy_file)
    print('Файл успешно скопирован.')



def storeRating(query, answer, rate):

    data = {'QUESTION': [query], 'ANSWER': [answer], 'Rating': [rate]}
    df = pd.DataFrame(data)

    try:
        existing_data = pd.read_excel('./train_dataset_Датасетcopy.xlsx')
        df = pd.concat([existing_data, df], ignore_index=True)
    except FileNotFoundError:
        print("file not found")
        pass


    df.to_excel('train_dataset_Датасетcopy.xlsx', index=False)

    print("Данные успешно добавлены в файл Excel.")