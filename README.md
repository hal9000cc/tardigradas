# Tardigradas

`Tardigradas` — компактная Python-библиотека для задач оптимизации на основе генетического алгоритма.

После рефакторинга проект получил более явную архитектуру:

- `Problem` описывает задачу;
- `ChromosomeSchema` формализует структуру хромосомы;
- `Individual` представляет отдельную особь;
- `Tardigradas` выступает оркестратором эволюционного цикла;
- генетические операторы и сериализация вынесены в отдельные модули.

## Что умеет библиотека

- поддерживает три типа генов: `bit`, `int`, `float`;
- инициализирует популяцию случайными особями;
- выполняет ранговый отбор родителей;
- поддерживает кроссовер и мутацию;
- добавляет «свежую кровь» в популяцию;
- удаляет дубликаты;
- сохраняет и восстанавливает состояние поиска через `pickle`.

## Установка

Для локальной разработки:

```bash
pip install -e .
```

## Структура проекта

- `src/tardigradas/__init__.py` — публичный API пакета;
- `src/tardigradas/engine.py` — основной движок генетического алгоритма;
- `src/tardigradas/problem.py` — базовый абстрактный класс задачи;
- `src/tardigradas/individual.py` — объект отдельной особи;
- `src/tardigradas/schema.py` — dataclass `ChromosomeSchema`;
- `src/tardigradas/operators/` — селекция, кроссовер и мутация;
- `src/tardigradas/serialization.py` — сериализация состояния;
- `src/tardigradas/gen_types.py` — перечисления типов генов и стратегий кроссовера.

## Базовая идея использования

Пользователь описывает задачу через класс-наследник `Problem`, затем передаёт этот класс в `Tardigradas`.

### Что нужно реализовать в своём классе задачи

Минимально:

- `init_environment(tardigradas)` — инициализация окружения задачи;
- `gen_info(tardigradas) -> ChromosomeSchema` — описание структуры хромосомы;
- `fitness(individual)` — расчёт функции приспособленности;
- `chromo_valid(individual)` — опциональная дополнительная валидация.

## Формат `ChromosomeSchema`

`ChromosomeSchema` описывает:

- `gen_types` — список типов генов;
- `bounds` — нижние и верхние границы;
- `comments` — текстовые описания генов;
- `groups` — группы генов для совместного переключения;
- `defaults` — значения по умолчанию;
- `defaults_probability` — вероятности применения значений по умолчанию.

Пример:

```python
from tardigradas import ChromosomeSchema, GenType


schema = ChromosomeSchema(
    gen_types=[GenType.float, GenType.float, GenType.int],
    bounds=([0.0, 0.0, 0], [10.0, 10.0, 5]),
    comments=["x", "y", "mode"],
    groups=[0, 0, 1],
    defaults=[1.0, float("nan"), 2],
    defaults_probability=[0.3, 0.0, 0.5],
)
```

## Важная деталь про `fitness`

Метод `fitness()` должен возвращать либо одно число, либо последовательность чисел:

- первый элемент — основная метрика оптимизации;
- остальные элементы — дополнительные метрики, если они нужны.

Примеры:

```python
return -42.0
```

или

```python
return [-42.0, 0.1, 7.5]
```

## Пример использования

```python
from tardigradas import ChromosomeSchema, GenType, Individual, Problem, Tardigradas


class SphereProblem(Problem):
    @staticmethod
    def init_environment(tardigradas):
        pass

    @staticmethod
    def gen_info(tardigradas):
        return ChromosomeSchema(
            gen_types=[GenType.float, GenType.float],
            bounds=([-5.0, -5.0], [5.0, 5.0]),
            comments=["x", "y"],
            groups=[0, 0],
        )

    @staticmethod
    def fitness(individual: Individual):
        x = individual[0]
        y = individual[1]
        return [-(x ** 2 + y ** 2)]

    @staticmethod
    def chromo_valid(individual: Individual):
        return True


ga = Tardigradas(
    problem=SphereProblem,
    population_size=50,
    crossover_fraction=0.5,
    fresh_blood_fraction=0.1,
    gen_mutation_fraction=0.1,
    fitness_environment=None,
    n_elits=2,
)

ga.population_init()
ga.loop(max_iterations=100, loop_fun=lambda engine: False)

print("Лучшая оценка:", ga.best_score)
print("Лучшая особь:", ga.best_individual.chromo)
```

## Основные параметры `Tardigradas`

- `problem` — класс задачи;
- `population_size` — размер популяции;
- `crossover_fraction` — доля новых особей, получаемых кроссовером;
- `fresh_blood_fraction` — доля случайно добавляемых особей;
- `gen_mutation_fraction` — интенсивность мутации;
- `fitness_environment` — произвольное внешнее окружение;
- `n_elits` — число элитных особей.

## Основные методы

### Инициализация и цикл

- `population_init()` — создать начальную популяцию;
- `step()` — выполнить одну итерацию;
- `loop(...)` — запустить цикл оптимизации.

### Работа с результатами

- `best_score` — лучшая найденная оценка;
- `best_iteration` — номер итерации лучшего результата;
- `best_individual` — лучшая найденная особь;
- `step_best_individual` — лучшая особь текущего шага.

### Сохранение состояния

```python
ga.save_to_file("state.pkl")
ga.restore_from_file("state.pkl")
```

## Что улучшено в архитектуре

- монолитный `__init__.py` заменён на модульную структуру;
- описание задачи отделено от представления особи;
- хрупкий кортеж из `gen_info()` заменён на `ChromosomeSchema`;
- логика операторов выделена в отдельный пакет;
- сериализация вынесена из движка;
- удаление дубликатов работает через хеширование, а не через двойной цикл `O(n^2)`;
- движок хранит популяцию как `list[Individual]`, а не как `numpy`-массив объектов.

## Следующие логичные шаги

- добавить автоматические тесты;
- добавить отдельные примеры использования;
- при необходимости сделать стратегии кроссовера и мутации конфигурируемыми на уровне API.

## Опциональный MNIST benchmark с PyTorch

В тестовом контуре добавлен отдельный benchmark для сценария, где `Tardigradas` оптимизирует веса компактной сверточной сети на полном train split MNIST и затем валидирует лучшую особь на test split.

Особенности сценария:

- реализация находится в `tests/mnist_helpers.py` и `tests/test_engine_mnist_benchmark.py`;
- используется компактная CNN с двумя сверточными слоями и глобальным pooling;
- все веса сети кодируются в плоскую хромосому `float`-генов;
- primary fitness — отрицательная cross-entropy на полном train split;
- benchmark помечен как `slow` и `gpu`;
- benchmark не запускается по умолчанию и требует явного opt-in через переменную окружения.

### Требования

- в интерпретаторе тестов должны быть установлены `torch` и `torchvision`;
- по умолчанию ожидается CUDA-совместимое устройство;
- набор данных MNIST должен быть доступен локально, либо нужно разрешить его скачивание.

### Полезные переменные окружения

- `TARDIGRADAS_RUN_MNIST_BENCHMARK=1` — разрешить запуск benchmark;
- `TARDIGRADAS_MNIST_ROOT=/path/to/mnist` — указать каталог с MNIST;
- `TARDIGRADAS_MNIST_DOWNLOAD=1` — разрешить `torchvision` скачать датасет, если его нет локально.

### Ручной запуск

```bash
TARDIGRADAS_RUN_MNIST_BENCHMARK=1 python -m pytest tests/test_engine_mnist_benchmark.py -m "slow and gpu" -q
```

В рамках обычного `pytest`-прогона этот benchmark будет пропущен, пока не установлен `TARDIGRADAS_RUN_MNIST_BENCHMARK=1`.