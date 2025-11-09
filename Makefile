# Makefile для ML проекта
.PHONY: build up down logs clean restart

# Сборка и запуск
build:
	sudo docker compose up -d --build

# Запуск
up:
	sudo docker compose up -d

# Остановка
down:
	sudo docker compose down

# Просмотр логов
logs:
	sudo docker compose logs -f ml-api

# Полная очистка
clean:
	sudo docker compose down -v --rmi all
	sudo docker system prune -a -f

# Перезапуск
restart:
	sudo docker compose restart

# Проверка статуса
status:
	sudo docker compose ps
	curl -s http://localhost:8000/health | jq .