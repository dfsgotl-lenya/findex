# Lab 8 · M1 — профіль до будь-яких змін

Корпус: 744 PEP (`data/peps.jsonl`, 13.1 МБ), індекс з позиціями `data/peps.bin` (14.6 МБ, 31 753 терми, 481 435 постингів). Машина: 2 vCPU, Python 3.13.

Команди (результати нижче — з коміту `lab-07`, до правок Lab 8):

```bash
uv run python -m cProfile -o search.prof -m findex search data/peps.bin "python type annotations" -k 10
uv run python -m cProfile -o index.prof -m findex index data/peps.jsonl --out ix.bin --positions --executor serial
uv run py-spy record --nonblocking --rate 500 -o docs/lab08_flame_search.svg -- .venv/bin/python -m findex search data/peps.bin "python type annotations" -k 10
uv run py-spy record --nonblocking --rate 200 -o docs/lab08_flame_index.svg -- .venv/bin/python -m findex index data/peps.jsonl --out ix.bin --positions --executor serial
uv run scalene run --cpu-only --program-path . benchmarks/profile_search_workload.py data/peps.bin
```

Flame graphs: [`lab08_flame_search.svg`](lab08_flame_search.svg), [`lab08_flame_index.svg`](lab08_flame_index.svg). (`--nonblocking` потрібен, бо в контейнері без нього py-spy зависав, зупиняючи процес на кожному семплі.)

## `findex search` — 3-термовий запит на повному корпусі

cProfile, увесь процес CLI — 2.85 с:

| Функція | cumtime | Що це |
|---|---:|---|
| `store.load` → `_pickle.load` | 2.55 с | розпаковування індексу |
| ↳ `dataclasses._dataclass_setstate` (482 179 викликів) | 2.16 с | відновлення кожного `Posting` |
| ↳ `dataclasses.fields` (482 197 викликів) | 1.61 с | `setstate` щоразу будує кортеж полів |
| сам `search` | ~0.02 с | парсинг, скоринг, сніпети |

Окремо — 50 пошуків в одному процесі (індекс уже в пам'яті, кеш запитів очищається), 0.223 с:

| Місце | tottime / cumtime | Частка |
|---|---:|---:|
| `snippets.make_snippet` (cumtime) | 0.160 с | 72 % |
| ↳ `json.decoder.raw_decode` | 0.049 с | |
| ↳ `str.casefold` | 0.031 с | |
| ↳ `str.replace`, `readline`, `open` | 0.032 с | |
| `ranking.score_terms` (cumtime) | 0.045 с | 20 % |
| ↳ `BM25.__call__` (10 200 викликів) | 0.036 с | |

Scalene (`benchmarks/profile_search_workload.py`: завантаження + 600 пошуків, 4.55 с), Python vs native:

| Рядок | Python | native | system |
|---|---:|---:|---:|
| `store.py:93  pickle.load(handle)` | 0 % | 48.4 % | 3.7 % |
| `snippets.py:20  json.loads(...)` | 0 % | 13.3 % | 0.5 % |
| `snippets.py:44  text.casefold()` | 0 % | 7.5 % | 0.1 % |
| `snippets.py:18–21, 40` (open/seek/readline/replace) | 0 % | 15.8 % | 0.4 % |
| `ranking.py:39  BM25.__call__` | 0.4 % | 0.8 % | 0.1 % |
| `index.py:97  num_docs` (викликається з `BM25.__call__`) | 0.9 % | 0 % | 0 % |

Scalene показує `pickle.load` як native-час, бо рядок — виклик C-функції. cProfile уточнює, що всередині цей C-код на кожен об'єкт викликає назад Python-функцію `_dataclass_setstate`: тобто це **Python-час, захований за native-викликом**.

## `findex index` (serial)

cProfile (з накладними витратами профайлера — 12.8 с замість ~6 с):

| Функція | tottime | cumtime |
|---|---:|---:|
| `parallel._index_one_file` (цикл підрахунку tf і позицій) | 2.45 с | 5.80 с |
| `tokenize.tokenize` (генератор, 2 026 412 токенів) | 1.41 с | 1.90 с |
| `dataclasses._dataclass_getstate` + `fields` (збереження pickle) | 0.54 с | 2.11 с |
| `dict.setdefault` / `list.append` у циклі | 1.21 с | |

## Три найзатратніші місця і гіпотези (записано до змін)

1. **Завантаження індексу: `pickle.load` 482 тис. об'єктів `Posting` — 2.55 с, ~90 % часу `findex search` і ~50 % робочого навантаження в Scalene.**
   Python-час за native-викликом: для `frozen=True, slots=True` dataclass Python 3.11+ генерує `__getstate__/__setstate__`, і кожен об'єкт викликає `dataclasses.fields()`.
   *Гіпотеза:* (а) власний `Posting.__reduce__` → pickle відновлює об'єкт одним викликом конструктора, без `fields()` — очікую 2–3× швидше; (б) постинги як кілька великих масивів NumPy — завантаження стає копіюванням буферів, мілісекунди замість секунд.

2. **Сніпети: для кожного результату заново відкривається JSONL, читається рядок, `json.loads` усього документа, `casefold` усього тексту — 72 % часу пошуку в процесі.**
   Переважно native-час (C-реалізації `json`, `str.casefold`, I/O), тож NumPy тут не допоможе.
   *Гіпотеза:* LRU-кеш тексту документів прибере повторне читання/декодування для популярних документів; у вебі одні й ті самі PEP повертаються постійно.

3. **Скоринг: `BM25.__call__` викликається на кожен постинг кожного терма (Python-цикл, виклик методу, `index.df()`, `index.num_docs`, `doc_length()` на кожній ітерації).**
   Python-час. На 3-термовому запиті це лише ~20 %, але лінійно росте з довжиною списку постингів: для `python` (729 документів) це вже основна робота скорера.
   *Гіпотеза:* переписати BM25/TF-IDF як вирази над масивами `int32` (`doc_ids`, `tfs`, `doc_len`) — один прохід у C на терм; очікую 10–50× на частих термах і майже нічого на рідкісних (короткий список, накладні витрати виклику NumPy).

## Результати після змін

| # | Гіпотеза | Що зроблено | До | Після | Висновок |
|---|---|---|---:|---:|---|
| 1 | `pickle.load` гальмує через `fields()` на кожному `Posting` | `Posting.__reduce__` → відновлення одним викликом конструктора | load 1.96 с, файл 14.6 МБ; `findex search` 2.3 с | load 1.22 с, файл 12.6 МБ; `findex search` 1.5 с | Підтвердилось: 1.6× (у мікротесті без позицій 2.5×). Решту часу забирають кортежі позицій. Постинги як масиви `.npz` (`ArrayIndex.save/load`, без позицій) вантажаться за **0.03 с** — у 40 разів швидше, ніж pickle. |
| 2 | Сніпети повторно читають і декодують JSONL | `lru_cache(512)` на текст документа | 2.67 мс на пошук (NumPy-скорер) | **1.07 мс** | Підтвердилось: 2.5× на повному шляху пошуку. Це не NumPy, а звичайний кеш: native-код не прискорити векторизацією, його можна лише не викликати. |
| 3 | Скоринг: Python-виклик на кожен постинг | BM25/TF-IDF як вирази над `int32`-масивами, top-k через `np.argpartition` | `python`: 0.65 мс | 0.07 мс | Підтвердилось для довгих списків (9× на 744 документах, 18–25× на 10× корпусі), але на повному пошуку виграш малий (2.79 → 2.67 мс), бо на цьому корпусі скоринг — лише ~20 % часу. Деталі: [`lab08_numpy_scorer.md`](lab08_numpy_scorer.md). |

Повний шлях пошуку (парсинг + Boolean + ранжування + сніпети; середнє з 600 пошуків, `benchmarks/benchmark_search_path.py`):

| Варіант | мс на пошук |
|---|---:|
| Lab 3 скорер, без кешу тексту (= `lab-07`) | 2.79 |
| NumPy скорер, без кешу тексту | 2.67 |
| NumPy скорер + кеш тексту | 1.07 |

Мораль профілювання: найбільший виграш дав не NumPy, а те, на що вказав профайлер (кеш для native-коду і менше Python-викликів при розпаковуванні). Векторизація окупається там, де Python виконує одну й ту саму дрібну операцію тисячі разів поспіль.
