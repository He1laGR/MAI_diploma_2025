#!/bin/bash

# Активация виртуального окружения
source venv/bin/activate

# Остановка предыдущего процесса, если он существует
if [ -f model_manager.pid ]; then
    kill $(cat model_manager.pid) 2>/dev/null || true
    rm model_manager.pid
fi

# Запуск model_manager.py в фоновом режиме
python model_manager.py > model_manager.log 2>&1 &
PID=$!

# Сохранение PID процесса
echo $PID > model_manager.pid

echo "Model manager started with PID $PID"
echo "Waiting for server to start..."

# Ждем, пока сервер запустится
for i in {1..30}; do
    if curl -s http://localhost:5000/health > /dev/null; then
        echo "Server is up and running!"
        exit 0
    fi
    echo "Waiting... ($i/30)"
    sleep 2
done

echo "Server failed to start. Check model_manager.log for details"
exit 1 