import socket

import pytest

from ragly_backend import offline
from ragly_backend.answer import build_messages, check_citations, is_refusal
from ragly_backend.chunker import chunk_pages
from ragly_backend.config import NOT_FOUND
from ragly_backend.retriever import Hit, fts_query


def words(t):
    return len(t.split())


def split_long(t, size, overlap):
    w = t.split()
    return [" ".join(w[i:i + size]) for i in range(0, len(w), size - overlap)]


def test_chunks_respect_size_and_pages():
    text = " ".join(f"Sentence number {i} is here." for i in range(200))
    chunks = chunk_pages([(1, text), (2, "Short page two.")], words, split_long, max_tokens=50, overlap=10)
    assert all(words(c.text) <= 50 for c in chunks)
    assert {c.page for c in chunks} == {1, 2}
    assert chunks[-1].text == "Short page two."
    # overlap: consecutive chunks on page 1 share text
    p1 = [c for c in chunks if c.page == 1]
    assert p1[0].text.split(".")[-2] in p1[1].text


def test_long_sentence_is_split():
    chunks = chunk_pages([(1, "word " * 300)], words, split_long, max_tokens=100, overlap=20)
    assert len(chunks) >= 3 and all(words(c.text) <= 100 for c in chunks)


def test_citation_check_drops_invalid():
    cleaned, valid, invalid = check_citations("Fee is 5 [1][7]. Term is 2 years [2, 9].", 3)
    assert valid == [1, 2] and invalid == [7, 9]
    assert "[7]" not in cleaned and "[1]" in cleaned and "[2]" in cleaned


def test_refusal_detection():
    assert is_refusal(NOT_FOUND)
    assert is_refusal("not found in your documents")
    assert not is_refusal("The notice period is 60 days [1].")


def test_fts_query_sanitises():
    q = fts_query('What is the "termination" notice? DROP TABLE; *')
    assert '"termination"' in q and '"what"' not in q and "*" not in q.replace('"', "")


def test_prompt_numbers_sources():
    h = Hit(1, 1, "a.pdf", 3, "TERMS", "Notice is 60 days.", 0.1, 0.8, 0)
    msgs = build_messages("notice?", [h])
    assert "[1] (a.pdf, page 3" in msgs[1]["content"]


def test_offline_guard_blocks_remote():
    offline.install()
    with pytest.raises(ConnectionRefusedError):
        socket.create_connection(("1.1.1.1", 443), timeout=1)
    with pytest.raises(ValueError):
        offline.assert_local_url("https://example.com/v1")
    offline.assert_local_url("http://127.0.0.1:8080/v1")


def test_a_citation_range_is_expanded():
    from ragly_backend.answer import normalise_citations

    assert normalise_citations("It is ₹57,530 [1-3].") == "It is ₹57,530 [1][2][3]."
    assert normalise_citations("See [2-2].") == "See [2]."


def test_a_sentence_never_starts_against_the_previous_full_stop():
    from ragly_backend.answer import tidy

    assert tidy("₹57,530 [1].The tenure is 36 months.") == "₹57,530 [1]. The tenure is 36 months."
    assert tidy("Mr. Iyer approved it [1].") == "Mr. Iyer approved it [1]."


def test_the_winrt_speech_modules_are_spelled_the_way_winrt_spells_them():
    """Regression: voice reported "not installed" on a machine where it WAS installed, because
    the probe imported winrt.windows.media.speech_recognition and the package is
    winrt.windows.media.speechrecognition."""
    from ragly_backend.voice import STT_MODULES, TTS_MODULES

    assert STT_MODULES[0] == "winrt.windows.media.speechrecognition"
    assert TTS_MODULES[0] == "winrt.windows.media.speechsynthesis"
    assert all("speech_recognition" in m or "speechrecognition" in m for m in STT_MODULES)


def test_the_speech_privacy_hresult_is_the_one_windows_actually_raises():
    # 0x80045509, as winrt hands it back through OSError.winerror
    from ragly_backend.voice import SPEECH_PRIVACY_HRESULT

    assert SPEECH_PRIVACY_HRESULT == -2147199735
    assert (SPEECH_PRIVACY_HRESULT + 2 ** 32) == 0x80045509


def test_speech_privacy_is_reported_as_unknown_off_windows_not_as_refused():
    # A Linux dev box must not be told its Windows consent flag is missing.
    from ragly_backend.voice import Voice

    import sys as _sys

    if _sys.platform != "win32":
        assert Voice.privacy_accepted() is None


def test_an_instruction_is_never_quoted_as_an_answer():
    """Regression: "What is CAR?" was answered with "5. Find a silver car" - a line from a
    test script in the PDF, not a fact about anything."""
    from ragly_backend.answer import is_instruction

    assert is_instruction("5. Find a silver car")
    assert is_instruction("Find a silver car")
    assert is_instruction("3) Upload the invoice and ask for the total")
    assert not is_instruction("The car was delivered on 4 May 2026.")
    assert not is_instruction("Ananya Rao drives a silver car.")


def test_a_file_name_in_a_question_is_a_scope_not_a_fact_to_verify():
    """"Summarize invoice-2026.pdf" must not be refused because the document does not
    contain its own file name."""
    from ragly_backend.evidence import classify_question

    shape = classify_question("Summarize Invoice-2026.pdf")
    assert shape.intent == "summary"
    assert not any("pdf" in e.lower() for e in shape.entities)
