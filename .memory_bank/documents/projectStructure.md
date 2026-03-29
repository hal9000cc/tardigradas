---
core: true
lastModified: '2026-03-29T05:17:00Z'
tags:
- structure
- global
---

# Структура проекта

## Директории
- `src/tardigradas/` — исходный код пакета с текущим публичным API.
- `.memory_bank/` — служебные документы контекста.

## Модули
- `src/tardigradas/__init__.py` — основной модуль библиотеки с `Resolve`, `Tardigradas` и связанными перечислениями/исключениями.
- `src/tardigradas/gen_types.py` — перечисление `GenType`.

## Ключевые файлы
- `pyproject.toml` — метаданные пакета, build-system и настройка `src`-layout.
- `README.md` — описание библиотеки, структуры и локальной установки.
- `.gitignore` — исключения для Python-артефактов сборки и виртуальных окружений.

## Зависимости и поток данных
- Пакет `tardigradas` зависит от `numpy`.
- Внешний пользователь импортирует API из `tardigradas`, а типы генов — из `tardigradas.gen_types`.
- Сборка пакета выполняется через `setuptools` с поиском пакетов в каталоге `src`.