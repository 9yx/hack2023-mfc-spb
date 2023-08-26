FROM python:3.7

RUN pip install torch
RUN pip install fastapi numpy pandas pydantic uvicorn transformers sentence_transformers scipy
RUN pip install openpyxl

EXPOSE 80

COPY . /app
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "9000"]