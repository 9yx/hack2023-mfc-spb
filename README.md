# hack2023-mfc-spb
Хакатон 25.08.2023 ЦП СЗФО - Санкт-Петербургское государственное казенное учреждение «Многофункциональный центр предоставления государственных и муниципальных услуг» (сокращенно - СПб ГКУ «МФЦ»)


config.ini may be:
[SETTINGS]
env=dev_cpu
aiModelDisabled=true

## Запрос с вопросом от оператора возвращает ответ
```bash
curl --request POST \
  --url http://127.0.0.1:9000/answering \
  --header 'Content-Type: application/json' \
  --data '{
	"question": "Какие заявления брать у заявителя в услуге 33?"
}'
```