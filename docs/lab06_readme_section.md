## Лабораторна робота №6

**Тема:** асинхронний краулер — `asyncio`, HTTP-запити, `TaskGroup`, черги, семафори та потоковий JSONL.

### Реалізовано

- `findex crawl` — асинхронний генератор сторінок на основі `asyncio.Queue` і `asyncio.TaskGroup`;
- один глобальний семафор, окремі семафори на origin і мінімальний інтервал між запитами до того самого origin;
- HTTP-клієнт `httpx.AsyncClient`, `asyncio.timeout`, обмежені повтори для `429`, `5xx` і таймаутів, exponential backoff, jitter та `Retry-After`;
- ручна обробка редиректів із блокуванням циклів і переходів за межі allowlist;
- `RobotsCache` — кеш robots policy для кожного origin; недоступний/помилковий `robots.txt` блокує краул, `404/410` означає, що правил немає;
- canonicalization URL: lower-case схема/host, прибрані fragment і trailing slash, відсортовані query-параметри;
- `seen` для запобігання дублям і верхня межа `--max-pages`;
- витягування title, тексту й посилань через стандартний `html.parser`;
- JSONL записується рядок за рядком; серіалізація/запис виводяться в `asyncio.to_thread`, а індексація залишається окремою командою;
- `Rich` live counters: сторінки, pages/s, запити in-flight, помилки й глибина черги;
- `crawl.log` містить статус, кількість байтів, час запиту та номер спроби;
- тести для HTTP виконуються через `httpx.MockTransport`, без живої мережі.

### Дозвіл на краул і User-Agent

Для перевірки та benchmark використовується **локальне дзеркало, створене `benchmarks/local_mirror.py` і прив'язане лише до `127.0.0.1`**. Воно знаходиться під контролем автора проєкту; його `robots.txt` дозволяє обхід. Зовнішній сайт для benchmark не використовується. User-Agent вказує репозиторій та контакт, переданий через обов'язковий `--contact`. Якщо запускаєте краулер на будь-якому публічному сайті, робіть це лише з дозволом, перевірте його robots.txt і вкажіть власний реальний email.

### Запуск краулера

У першому PowerShell вікні запустіть локальне дзеркало щонайменше на 250 сторінок:

```powershell
uv run python benchmarks/local_mirror.py --pages 250 --delay 0.01
```

Дзеркало навмисно має невелику затримку відповіді, щоб було видно різницю між конкурентністю 1, 5 і 20. Воно слухає лише loopback-інтерфейс.

У другому вікні:

```powershell
uv run findex crawl http://127.0.0.1:8765/page/0 --contact YOUR_REAL_EMAIL@example.com --max-pages 200 --concurrency 20 --per-host 20 --per-host-delay 0 --out data/crawl.jsonl --log-file crawl.log
```

Замініть `YOUR_REAL_EMAIL@example.com` на свою справжню контактну адресу. Для звичайного краулу defaults залишаються обережнішими: `--per-host 2` та `--per-host-delay 0.2`. У benchmark локального дзеркала per-host limit навмисно послаблюється, бо цей сервер належить автору проєкту.

Після завершення краулу індекс будується **поза event loop**. JSONL обробляється по одному рядку як по одному документу:

```powershell
uv run findex index data/crawl.jsonl --out index-crawl.bin --workers 4 --executor processes
uv run findex search index-crawl.bin "asyncio" -k 5
```

### Benchmark: concurrency 1 / 5 / 20

Залиште локальне дзеркало запущеним. Команда використовує той самий seed, той самий `max_pages=200`, три повтори й ті самі per-host параметри для всіх варіантів:

```powershell
uv run python benchmarks/benchmark_crawl.py --contact YOUR_REAL_EMAIL@example.com --max-pages 200 --repeats 3 --per-host 20 --per-host-delay 0
```

Скрипт вимірює wall time, сторінки за секунду, помилки з URL-логів і Peak RSS. Markdown-таблиця також записується до `data/lab06-benchmark/table.md`; після запуску перенесіть **реальні рядки з вашої машини** в таблицю нижче.

| Concurrency | Pages (median) | Wall (s, median) | Pages/s | Errors (median) | Peak RSS (MiB, median) |
|---:|---:|---:|---:|---:|---:|
| 1 | — | — | — | — | — |
| 5 | — | — | — | — | — |
| 20 | — | — | — | — | — |

Не підставляйте в цю таблицю числа з іншого комп'ютера — результати залежать від затримки дзеркала, Python, ОС і навантаження машини.

### Інтерпретація

Краулер майже весь час очікує мережеву відповідь. `asyncio` дає одному потоку перемикатися на інші задачі в `await`, поки сокет чекає відповіді. Це відрізняється від Lab 5: індексація CPU-bound, тому там процеси допомагають виконувати Python-код паралельно; у краулері процеси не потрібні для перекриття мережевих очікувань.

Обмеження на host зменшує максимальну швидкість, особливо коли сторінки беруться з одного домену. Воно залишається заради ввічливого навантаження на сервер і дотримання robots policy. Значення `--per-host 20 --per-host-delay 0` використовуються **тільки** для локального дзеркала в контрольованому benchmark, а не як рекомендовані налаштування для публічних сайтів.

Якби токенізація та індексація запускалися всередині корутини синхронним CPU-кодом, event loop не міг би перемикатися на інші запити до завершення цієї роботи. Тому краулер витягує текст, пише JSONL потоково, а `findex index` запускається окремо після завершення мережевого обходу.

### Приклад обробленого збою: HTTP 429

Локальне дзеркало має endpoint `/flaky429`: він повертає один `429` з `Retry-After: 0`, а наступний запит успішний. Для відтворення виконайте:

```powershell
uv run findex crawl http://127.0.0.1:8765/flaky429 --contact YOUR_REAL_EMAIL@example.com --max-pages 2 --concurrency 2 --per-host 2 --per-host-delay 0 --out data/failure-example.jsonl --log-file docs/lab06_failure.log
```

Приклад фактичних рядків з локального тестового запуску (час і `elapsed` у вашому запуску відрізнятимуться):

```text
INFO url=http://127.0.0.1:8765/flaky429 status=429 bytes=0 elapsed=0.004s attempt=1
WARNING url=http://127.0.0.1:8765/flaky429 status=429 action=retry delay=0.000s attempt=1/3
INFO url=http://127.0.0.1:8765/flaky429 status=200 bytes=105 elapsed=0.004s attempt=2
```

Тобто `429` не завершив увесь краул: клієнт виконав одну повторну спробу згідно з `Retry-After`, записав фінальну успішну сторінку, а лог залишив обидва статуси.

### Перевірки Lab 6

```powershell
uv run ruff check .
uv run ruff format --check .
uv run pyright
uv run pytest --basetemp .pytest_tmp -p no:cacheprovider
```

Методичка: [Lab 6 — Asyncio crawler](https://github.com/rmalkevy/Programming-Practice-Projects/blob/main/courses/python/lab-06-asyncio-crawler.md) · [Нотатки](https://github.com/rmalkevy/Programming-Practice-Projects/blob/main/courses/python/lab-06-asyncio-crawler.notes.md)

### Reflection — Lab 6

1. Що відбувається в event loop, коли корутина доходить до `await` на читанні сокета?
2. Чому `time.sleep()` блокує інші задачі, а `asyncio.sleep()` — ні?
3. Чому `TaskGroup` краще пасує до керованих worker-задач, ніж fire-and-forget?
4. Чому треба одночасно обмежувати глобальну конкурентність і кількість запитів до одного host?
5. Як `Retry-After` змінює звичайний exponential backoff?
6. Чому `counter += 1` і спільний змінюваний стан вимагають синхронізації, навіть якщо програма асинхронна?
7. Чому індексація має виконуватися після краулу, а не всередині корутини?
