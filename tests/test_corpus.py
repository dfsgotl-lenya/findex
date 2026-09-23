import inspect

from findex.corpus import iter_documents


def test_iter_documents_is_generator_function() -> None:
    assert inspect.isgeneratorfunction(iter_documents)


def test_iter_documents_yields_document(tmp_path) -> None:
    source = tmp_path / "note.txt"
    source.write_text("Hello Привіт", encoding="utf-8")

    documents = iter_documents(tmp_path)
    document = next(documents)

    assert document.doc_id == "note.txt"
    assert document.path == source
    assert document.text == "Hello Привіт"
