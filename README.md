# findex — лабораторні роботи 1–2

Навчальний проєкт пошукової системи `findex`, який поступово розширюється протягом курсу Python.

[![Quality Gate Status](https://sonarcloud.io/api/project_badges/measure?project=dfsgotl-lenya_findex&metric=alert_status)](https://sonarcloud.io/project/overview?id=dfsgotl-lenya_findex)

## Лабораторна робота №1

Тема: **ітератори, генератори та корпус текстів**.

Реалізовано:

- лінивий `iter_documents()`;
- потоковий `tokenize()` з NFC, `casefold()` та `re.finditer()`;
- правила обробки апострофів, дефісів і цифр;
- однопрохідну статистику корпусу через `Counter`;
- `--limit` через `itertools.islice`;
- порівняння eager/lazy підходів під `tracemalloc`;
- тести через `pytest`.

## Лабораторна робота №2

Тема: **інвертований індекс: словники, хешування та пам'ять**.

Мета — побудувати інвертований індекс поверх лінивого пайплайна з лабораторної №1, реалізувати Boolean-пошук, серіалізацію та виміряти вартість різних структур зберігання.

### Реалізовано

- `Posting` і `DocMeta` як `@dataclass(frozen=True, slots=True)`;
- побудова індексу через `defaultdict` за один прохід корпусу;
- відсортовані posting lists: `term → [(doc_id, tf), ...]`;
- метадані документів і довжини документів;
- Boolean-пошук `AND`, `OR`, `NOT`;
- двовказівниковий merge та альтернативний `set`-двигун;
- серіалізація у `pickle` та `JSON`;
- вимірювання часу та пікової пам'яті через `tracemalloc`;
- порівняння `plain dataclass` / `slots=True` / `array('I')`;
- GitHub Actions + SonarQube Cloud.

## Структура

```text
findex/
├── .github/workflows/sonar.yml
├── benchmarks/
│   ├── make_demo_corpus.py
│   ├── benchmark_search.py
│   └── measure_lab2.py
├── data/                         # локальний корпус; не комітиться
├── src/findex/
│   ├── corpus.py
│   ├── tokenize.py
│   ├── stats.py
│   ├── benchmark.py
│   ├── index.py
│   ├── search.py
│   └── store.py
├── tests/
├── pyproject.toml
├── sonar-project.properties
├── uv.lock
└── README.md
```

## Встановлення

У корені проєкту:

```powershell
uv sync
```

Перевірка:

```powershell
uv run pytest
uv run ruff check .
```

## Робота з індексом

Побудова та збереження у `pickle`:

```powershell
uv run python -m findex.index data/ --out index.bin
```

Побудова та збереження у JSON:

```powershell
uv run python -m findex.index data/ --out index.json --format json
```

Для швидкої перевірки на частині корпусу:

```powershell
uv run python -m findex.index data/ --out index.bin --limit 100
```

Пошук із збереженого `pickle`-індексу:

```powershell
uv run python -m findex.search index.bin "python search"
```

Пошук через `set`:

```powershell
uv run python -m findex.search index.bin "python OR search" --engine set
```

JSON визначається автоматично за розширенням `.json`:

```powershell
uv run python -m findex.search index.json "python NOT engine" --engine merge
```

Обидві CLI-команди показують час виконання та пікову пам'ять.

## Boolean-пошук

За замовчуванням суміжні терми означають `AND`.

Приклади:

```text
python search
python OR engine
python NOT engine
```

Для `AND` і `OR` використовується двовказівниковий merge по відсортованих posting lists. Для порівняння є `--engine set`.

## Бенчмарк merge vs set

Запуск:

```powershell
uv run python benchmarks/benchmark_search.py data/ --limit 300 --repeats 200
```

| Терм | Група | Двигун | Результатів | Загальний час (с) | Середній час (мс) |
|---|---|---|---:|---:|---:|
| токен | common | merge | 300 | 0.000959 | 0.0048 |
| токен | common | set | 300 | 0.000969 | 0.0048 |
| пошук | common | merge | 300 | 0.000998 | 0.0050 |
| пошук | common | set | 300 | 0.000950 | 0.0047 |
| raretoken0001 | rare | merge | 1 | 0.000193 | 0.0010 |
| raretoken0001 | rare | set | 1 | 0.000205 | 0.0010 |
| raretoken0002 | rare | merge | 1 | 0.000200 | 0.0010 |
| raretoken0002 | rare | set | 1 | 0.000201 | 0.0010 |

## Серіалізація

Підтримуються два формати:

| Формат | Переваги | Недоліки |
|---|---|---|
| `pickle` | простий і швидкий для Python | небезпечний для чужих файлів |
| `JSON` | безпечний і читабельний | більший та повільніший |

### Чому `pickle.load()` небезпечний

`pickle` може відновлювати довільний граф Python-об'єктів. Під час завантаження можуть виконуватися закодовані виклики/конструктори, тому файл `pickle` не можна бездумно відкривати, якщо він отриманий від стороннього джерела.

Вимірювання:

```powershell
uv run python -m findex.index data/ --out index.bin
uv run python -m findex.index data/ --out index.json --format json
```

| Формат | Розмір (MiB) | Збереження (с) | Завантаження (с) |
|---|---:|---:|---:|
| pickle | 0.26 | 0.020832 | 0.018150 |
| json | 0.41 | 0.033599 | 0.013174 |

## Дослідження пам'яті

Для однакового корпусу:

```powershell
uv run python benchmarks/measure_lab2.py data/ --limit 300
```

Порівнюються:

1. звичайний `@dataclass`;
2. `@dataclass(slots=True)`;
3. пари `array('I')`, де зберігаються `doc_id` та `tf` без мільйонів окремих Python-об'єктів.

| Варіант | Документи | Пікова пам'ять (MiB) | Розмір pickle (MiB) | Збереження (с) | Завантаження (с) |
|---|---:|---:|---:|---:|---:|
| plain | 300 | 2.39 | 0.31 | 0.010769 | 0.006774 |
| slots | 300 | 1.72 | 0.24 | 0.018702 | 0.015783 |
| array | 300 | 0.90 | 0.15 | 0.001073 | 0.000246 |

Очікувана причина різниці: у plain-варіанта кожен запис має звичайний `__dict__`; `slots=True` прибирає цей словник і зменшує накладні витрати; `array('I')` зберігає цілі числа компактно по 4 байти кожне замість окремих Python `int` та посилань зі списків. У звіті потрібно пояснити результат саме за своїми вимірюваннями.

## Лабораторний корпус

Корпус зберігається локально в `data/`. `data/` ігнорується Git.

Для навчального benchmark можна створити детермінований тестовий корпус:

```powershell
uv run python benchmarks/make_demo_corpus.py data/benchmark --documents 300 --copies-per-document 80
```

Реальний корпус семестру можна складати з `.txt` документів Project Gutenberg або іншого дозволеного джерела.

## Тести

```powershell
uv run pytest
```

Для лабораторної №2 додані тести на:

- хешованість `Posting` / `DocMeta`;
- побудову та сортування postings;
- частоту терма `tf`;
- merge `AND/OR/NOT`;
- однакові результати `merge` та `set`;
- round-trip `pickle`;
- round-trip `JSON`;
- пошук невідомого терма;
- `--limit` на етапі побудови.

## Git workflow для лабораторних

Лабораторні логічно розділені через Git-гілки та pull request:

```text
main
  └── lab-02
        └── Pull Request → main
```

Лабораторна №1 залишається доступною в історії та під тегом `lab-01`. Для лабораторної №2 використовується окрема гілка `lab-02`; після перевірки через тести, Ruff та SonarQube PR зливається в `main`, а вже після merge створюється тег `lab-02`.

## Reflection — питання для захисту

1. Як працює hash table і що відбувається при колізії?
Hash table використовує хеш ключа для швидкого пошуку; колізії вирішуються внутрішнім механізмом пошуку іншої позиції та перевіркою рівності ключів.
2. Який контракт між `__hash__()` та `__eq__()`?
Якщо a == b, то hash(a) == hash(b).
3. Чому `list` не може бути ключем `dict`, а `tuple` може?
list змінюваний і тому не hashable, а tuple незмінюваний і може бути ключем, якщо його елементи hashable.
4. Що дає `@dataclass(frozen=True, slots=True)`?
frozen=True робить об'єкт незмінним, slots=True зменшує пам'ять, прибираючи звичайний __dict__.
5. Куди йдуть байти у plain-об'єктах і що прибирає `__slots__`?
Plain-об'єкти зберігають атрибути через __dict__; __slots__ прибирає цю структуру.
6. Чому `array('I')` компактніший за `list[Posting]`?
array('I') зберігає числа компактно як типізовані елементи, тоді як list[Posting] має багато Python-об'єктів і посилань.
7. Яка складність двовказівникового merge та `set` і що показав benchmark?
merge — O(n+m), set — у середньому O(1) для пошуку; benchmark показує перевагу кожного підходу залежно від типу задачі.
8. Чому `pickle.load()` чужого файлу небезпечний?
pickle.load() може виконати шкідливий код із підготовленого файлу, тому чужі pickle-файли завантажувати небезпечно.
