#!/usr/bin/env python3
"""
Запуск приложения Attentify

Этот скрипт запускает веб-приложение Attentify с настройками по умолчанию.
"""

import uvicorn
import os
import sys
from pathlib import Path

# Добавляем корневую директорию в Python path
sys.path.insert(0, str(Path(__file__).parent))

def main():
    """Основная функция запуска."""
    print("🧠 Запуск Attentify - Платформы прикладного внимания")
    print("=" * 60)
    
    # Проверяем наличие необходимых директорий
    check_directories()
    
    # Настройки запуска
    host = os.getenv("HOST", "0.0.0.0")
    port = int(os.getenv("PORT", "8000"))
    reload = os.getenv("RELOAD", "true").lower() == "true"
    
    print(f"📍 Хост: {host}")
    print(f"🚪 Порт: {port}")
    print(f"🔄 Автоперезагрузка: {'Включена' if reload else 'Отключена'}")
    print("=" * 60)
    
    # Запускаем приложение
    try:
        uvicorn.run(
            "app.main:app",
            host=host,
            port=port,
            reload=reload,
            log_level="info",
            access_log=True
        )
    except KeyboardInterrupt:
        print("\n🛑 Приложение остановлено пользователем")
    except Exception as e:
        print(f"❌ Ошибка при запуске: {e}")
        sys.exit(1)

def check_directories():
    """Проверяем и создаем необходимые директории."""
    directories = [
        "checkpoints",
        "uploads",
        "logs"
    ]
    
    for directory in directories:
        Path(directory).mkdir(exist_ok=True)
        print(f"✅ Директория {directory} готова")

if __name__ == "__main__":
    main()
