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
- позволяет явно выбирать разные операторы кроссовера для `bit` и `float`/`int` генов;
- поддерживает adaptive-выбор операторов кроссовера по статистике успеха;
- добавляет «свежую кровь» в популяцию;
- удаляет дубликаты;
- сохраняет и восстанавливает состояние поиска через `pickle`.

## Установка

Для локальной разработки:

```bash
pip install -e .
```

Для использования встроенной панели графиков с `matplotlib`:

```bash
pip install -e .[plot]
```

## Структура проекта

- `src/tardigradas/__init__.py` — публичный API пакета;
- `src/tardigradas/engine.py` — основной движок генетического алгоритма;
- `src/tardigradas/problem.py` — базовый абстрактный класс задачи;
- `src/tardigradas/individual.py` — объект отдельной особи;
- `src/tardigradas/schema.py` — dataclass `ChromosomeSchema`;
- `src/tardigradas/crossover_policy.py` — конфигурация explicit/adaptive политик кроссовера;
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
from tardigradas import (
    ChromosomeSchema,
    CrossoverBitType,
    CrossoverFloatType,
    CrossoverPolicy,
    GenType,
    Individual,
    Problem,
    Tardigradas,
)


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
    crossover_policy=CrossoverPolicy.explicit(
        bit=CrossoverBitType.uniform,
        float=CrossoverFloatType.BLX,
    ),
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
- `n_elits` — число элитных особей;
- `crossover_policy` — explicit/adaptive политика выбора операторов кроссовера.

## Настройка кроссоверов

По умолчанию, если `crossover_policy` не передан, библиотека использует:

```python
CrossoverPolicy.explicit(
    bit=CrossoverBitType.uniform,
    float=CrossoverFloatType.uniform,
)
```

`int`-гены обрабатываются через ту же float-ветку кроссоверов с последующим округлением результата.

### Явный выбор операторов

```python
from tardigradas import CrossoverBitType, CrossoverFloatType, CrossoverPolicy


policy = CrossoverPolicy.explicit(
    bit=CrossoverBitType.two_point,
    float=CrossoverFloatType.BLX,
)
```

Доступные bit-операторы:
- `CrossoverBitType.uniform`
- `CrossoverBitType.one_point`
- `CrossoverBitType.two_point`

Доступные float-операторы:
- `CrossoverFloatType.uniform`
- `CrossoverFloatType.arithmetic`
- `CrossoverFloatType.BLX`

### Adaptive-выбор операторов

```python
from tardigradas import CrossoverBitType, CrossoverFloatType, CrossoverPolicy


policy = CrossoverPolicy.adaptive(
    bit_candidates=[
        CrossoverBitType.uniform,
        CrossoverBitType.one_point,
        CrossoverBitType.two_point,
    ],
    float_candidates=[
        CrossoverFloatType.uniform,
        CrossoverFloatType.arithmetic,
        CrossoverFloatType.BLX,
    ],
    reward="elite_survival",
)
```

В текущей версии adaptive-режим поддерживает reward-стратегию `"elite_survival"`: оператор получает успех, если ребёнок, созданный им, попадает в элитную часть популяции на следующем шаге.

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

## Встроенная панель графиков прогресса

Если установлен optional extra `plot`, библиотека может показывать live-панель прогресса через `matplotlib`.

Панель отображает:

- `best_score`, `step_score`, `population_mean_score`, `population_max_score`;
- `score_improvement`, `killed_doubles`, `elapsed_time_sec`;
- bar chart текущей популяции, отсортированной по убыванию score, с цветами по происхождению особей;
- adaptive probabilities операторов кроссовера, если включена adaptive policy.

Пример использования:

```python
from tardigradas import Tardigradas, create_progress_panel


engine = Tardigradas(problem=SphereProblem, population_size=50)
engine.population_init()
engine.estimate_population()

panel = create_progress_panel(title="Sphere progress")
panel.capture_initial_state(engine)

engine.loop(
    max_iterations=100,
    loop_fun=panel.loop_callback(),
)

panel.show(block=True)
```

Если `matplotlib` недоступен, `create_progress_panel()` всё равно вернёт helper-объект, но он будет просто собирать history без отрисовки окна.

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
- политики кроссовера выделены в отдельный публичный API;
- сериализация вынесена из движка;
- удаление дубликатов работает через хеширование, а не через двойной цикл `O(n^2)`;
- движок хранит популяцию как `list[Individual]`, а не как `numpy`-массив объектов.

## Следующие логичные шаги

- добавить автоматические тесты;
- добавить отдельные примеры использования;
- при необходимости расширить набор adaptive reward-стратегий и метрик успеха операторов.

## Benchmark-скрипты

Benchmark-сценарии вынесены в отдельную директорию `benchmarks/` и не входят в обычный pytest-прогон.

Доступные скрипты:

- `benchmarks/run_onemax.py`
- `benchmarks/run_sphere.py`
- `benchmarks/run_rastrigin.py`
- `benchmarks/run_royal_road.py`
- `benchmarks/run_rosenbrock.py`
- `benchmarks/run_ackley.py`
- `benchmarks/run_mnist.py`

Каждый скрипт:

- хранит свои параметры запуска прямо в файле для удобного экспериментирования;
- хранит и выводит `crossover_policy` как часть benchmark-конфигурации;
- печатает в консоль все основные параметры алгоритма;
- выводит стартовый лучший score и итоговый результат;
- может свободно кастомизироваться независимо от остальных benchmark-сценариев.

Примеры запуска:

```bash
python benchmarks/run_onemax.py
python benchmarks/run_sphere.py
python benchmarks/run_ackley.py
```

## Pytest benchmark-тесты

Обычные benchmark-задачи сохранены в `tests/test_engine_benchmarks.py` как regression-тесты качества.

Их можно запускать отдельно:

```bash
python -m pytest tests/test_engine_benchmarks.py -q
```

## Отдельный MNIST benchmark с PyTorch

MNIST-сценарий теперь оформлен как отдельный скрипт `benchmarks/run_mnist.py`, где `Tardigradas` оптимизирует веса компактной сверточной сети на полном train split MNIST и затем валидирует лучшую особь на test split.

Особенности сценария:

- реализация находится в `benchmarks/mnist_helpers.py` и `benchmarks/run_mnist.py`;
- используется компактная CNN с двумя сверточными слоями и глобальным pooling;
- все веса сети кодируются в плоскую хромосому `float`-генов;
- primary fitness — отрицательная cross-entropy на полном train split;
- benchmark запускается вручную отдельным скриптом.

### Требования

- в интерпретаторе тестов должны быть установлены `torch` и `torchvision`;
- по умолчанию ожидается CUDA-совместимое устройство;
- набор данных MNIST должен быть доступен локально, либо нужно разрешить его скачивание.

### Полезные переменные окружения

- `TARDIGRADAS_MNIST_ROOT=/path/to/mnist` — указать каталог с MNIST;
- `TARDIGRADAS_MNIST_DOWNLOAD=1` — разрешить `torchvision` скачать датасет, если его нет локально.

### Ручной запуск

```bash
python benchmarks/run_mnist.py
```

При необходимости можно предварительно задать переменные окружения, например:

```bash
TARDIGRADAS_MNIST_DOWNLOAD=1 python benchmarks/run_mnist.py
```