---
core: true
lastModified: '2026-03-29T06:16:23Z'
tags:
- structure
- global
---

# Структура проекта

## Директории
- `src/tardigradas/` — исходный код пакета с новым модульным API.
- `src/tardigradas/operators/` — функции генетических операторов.
- `.memory_bank/` — служебные документы контекста.

## Модули
- `src/tardigradas/__init__.py` — публичный API и реэкспорт основных сущностей.
- `src/tardigradas/engine.py` — движок `Tardigradas`, оркестратор эволюционного цикла.
- `src/tardigradas/problem.py` — абстрактный контракт `Problem` для описания задачи.
- `src/tardigradas/individual.py` — класс `Individual`, представляющий одну особь.
- `src/tardigradas/schema.py` — dataclass `ChromosomeSchema` с описанием структуры хромосомы.
- `src/tardigradas/gen_types.py` — перечисления `GenType`, `CrossoverBitType`, `CrossoverFloatType`.
- `src/tardigradas/exceptions.py` — исключение `TardigradasException` и legacy alias.
- `src/tardigradas/serialization.py` — сохранение и восстановление состояния поиска.
- `src/tardigradas/operators/crossover.py` — функции кроссовера.
- `src/tardigradas/operators/mutation.py` — функции мутации.
- `src/tardigradas/operators/selection.py` — ранжирование и выбор родителей.

## Ключевые файлы
- `pyproject.toml` — метаданные пакета, build-system и `src`-layout.
- `README.md` — документация по новому API, структуре и примеру использования.
- `.gitignore` — исключения для Python-артефактов и окружений.

## Зависимости и поток данных
- Пакет зависит от `numpy`.
- Пользователь определяет класс `Problem` и возвращает `ChromosomeSchema`.
- `Tardigradas` инициализирует схему, создаёт `Individual` и управляет популяцией.
- Модули из `operators/` используются движком для селекции, кроссовера и мутаций.
- `serialization.py` отвечает за сериализацию состояния эволюционного процесса.