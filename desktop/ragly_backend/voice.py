"""Push-to-talk voice, fully offline, using the speech engines built into Windows.

  speech -> text : Windows.Media.SpeechRecognition   (no model download, x64 + ARM64)
  text -> speech : Windows.Media.SpeechSynthesis     (returns a WAV the browser plays)

Recognised text is repaired against the auto-tuned glossary, so "e forty seven"
and "are em you" become "E-47" and "RMU" before the search runs.
"""
from __future__ import annotations

import asyncio
import logging
import re
import sys
import threading

from .autotune import Pack, normalise_code

log = logging.getLogger("ragly.voice")

NUMBER_WORDS = {
    "zero": "0", "one": "1", "two": "2", "three": "3", "four": "4", "five": "5", "six": "6", "seven": "7", "eight": "8", "nine": "9", "ten": "10", "eleven": "11",
    "twelve": "12", "thirteen": "13", "fourteen": "14", "fifteen": "15", "sixteen": "16", "seventeen": "17",
    "eighteen": "18", "nineteen": "19", "twenty": "20", "thirty": "30", "forty": "40", "fourty": "40",
    "fifty": "50", "sixty": "60", "seventy": "70", "eighty": "80", "ninety": "90", "hundred": "100",
}
LETTER_WORDS = {"ay": "a", "bee": "b", "see": "c", "dee": "d", "ee": "e", "eff": "f", "gee": "g", "aitch": "h",
                "jay": "j", "kay": "k", "el": "l", "em": "m", "en": "n", "pee": "p", "cue": "q", "are": "r",
                "ess": "s", "tee": "t", "you": "u", "vee": "v", "double you": "w", "ex": "x", "why": "y", "zed": "z"}


class VoiceError(RuntimeError):
    pass


#: Windows refuses every SpeechRecognizer call until the account has accepted the speech
#: privacy policy once. The HRESULT is 0x80045509, which surfaces through winrt as this
#: signed WinError. Accepting it is a per-user registry flag, not an admin action.
SPEECH_PRIVACY_HRESULT = -2147199735
SPEECH_PRIVACY_KEY = r"SOFTWARE\Microsoft\Speech_OneCore\Settings\OnlineSpeechPrivacy"
#: Windows keeps the per-app microphone consent here. "Deny" means every desktop app is
#: refused the microphone, which SpeechRecognizer reports only as USER_CANCELED.
MIC_CONSENT_KEY = r"SOFTWARE\\Microsoft\\Windows\\CurrentVersion\\CapabilityAccessManager\\ConsentStore\\microphone"
MIC_HELP = (
    "Windows cancelled the recording, which almost always means the microphone is blocked or "
    "busy. Check Settings > Privacy & security > Microphone: turn on 'Microphone access' and "
    "'Let desktop apps access your microphone', then close anything else using the mic "
    "(Teams, Zoom, Voice Recorder) and try again."
)
SPEECH_PRIVACY_HELP = (
    "Windows has not been given permission to use speech on this account. "
    "Run scripts\\enable_speech.ps1 once (no admin needed) and restart the app, or turn on "
    "Settings > Privacy & security > Speech > Speech recognition. "
    "It is a consent flag on the account, not a download."
)


#: winrt spells these without underscores; earlier releases used the underscored form, so both
#: are tried. Getting this wrong made voice report "not installed" on a machine where it was.
STT_MODULES = ("winrt.windows.media.speechrecognition", "winrt.windows.media.speech_recognition")
TTS_MODULES = ("winrt.windows.media.speechsynthesis", "winrt.windows.media.speech_synthesis")


def _words_to_digits(text: str) -> str:
    """Convert spoken numbers to digits, but only next to a code-ish token or another number."""
    words = text.split()
    out = []
    for i, w in enumerate(words):
        key = re.sub(r"[^a-z]", "", w.lower())
        prev = re.sub(r"[^a-z0-9]", "", words[i - 1].lower()) if i else ""
        nxt = re.sub(r"[^a-z0-9]", "", words[i + 1].lower()) if i + 1 < len(words) else ""
        near_code = (len(prev) <= 4 and prev.isalpha()) or prev in NUMBER_WORDS or nxt in NUMBER_WORDS \
            or any(c.isdigit() for c in prev + nxt)
        out.append(NUMBER_WORDS[key] if key in NUMBER_WORDS and near_code else w)
    # "40 7" spoken as "forty seven" -> "47"
    joined = " ".join(out)
    joined = re.sub(r"\b(\d0) (\d)\b", lambda m: str(int(m.group(1)) + int(m.group(2))), joined)
    return joined


def repair(text: str, pack: Pack | None) -> str:
    """Fix codes and abbreviations that speech recognition mangles."""
    if not text:
        return text
    fixed = _words_to_digits(text)
    fixed = re.sub(r"\b([A-Za-z])\s*[- ]\s*(\d{2,4})\b", r"\1-\2", fixed)   # "e 47" -> "e-47"
    if not pack:
        return fixed
    # snap tokens onto known codes ("e47", "e 47", "E-47" -> the code as written in the manual)
    known = {normalise_code(c): c for c in pack.sample_codes}
    if known:
        def snap(m: re.Match) -> str:
            return known.get(normalise_code(m.group(0)), m.group(0))

        fixed = re.sub(r"\b[A-Za-z]{1,4}[- ]?\d{2,6}\b", snap, fixed)
    # spelled-out abbreviations: "are em you" -> "RMU"
    low = fixed.lower()
    for abbr in pack.glossary:
        spelled = " ".join(next((k for k, v in LETTER_WORDS.items() if v == ch.lower()), ch.lower()) for ch in abbr)
        if spelled in low:
            fixed = re.sub(re.escape(spelled), abbr, fixed, flags=re.I)
        expansion = pack.glossary.get(abbr) or ""
        if expansion and expansion.lower() in low and abbr not in fixed:
            fixed += f" ({abbr})"
    return fixed.strip()


class Voice:
    """Lazy wrapper around the Windows speech engines."""

    _lock = threading.Lock()
    _stt_error: str | None = None
    _tts_error: str | None = None
    _checked = False

    # ---------------- availability ----------------
    @classmethod
    def status(cls) -> dict:
        with cls._lock:
            if not cls._checked:
                cls._stt_error = cls._probe(STT_MODULES, "SpeechRecognizer")
                cls._tts_error = cls._probe(TTS_MODULES, "SpeechSynthesizer")
                cls._checked = True
        consented = cls.privacy_accepted()
        mic = cls.mic_allowed()
        stt_error = cls._stt_error
        if stt_error is None and consented is False:
            stt_error = SPEECH_PRIVACY_HELP
        elif stt_error is None and mic is False:
            stt_error = MIC_HELP
        return {
            "stt": cls._stt_error is None and consented is not False and mic is not False,
            "tts": cls._tts_error is None,
            "stt_error": stt_error,
            "tts_error": cls._tts_error,
            "stt_installed": cls._stt_error is None,
            "speech_consent": consented,
            "microphone": mic,
            "platform": sys.platform,
        }

    @staticmethod
    def mic_allowed() -> bool | None:
        """Is any desktop app allowed to use the microphone? None when it cannot be read."""
        if sys.platform != "win32":
            return None
        try:
            import winreg

            with winreg.OpenKey(winreg.HKEY_CURRENT_USER, MIC_CONSENT_KEY) as key:
                value, _ = winreg.QueryValueEx(key, "Value")
                return str(value).lower() == "allow"
        except Exception:
            return None

    @staticmethod
    def privacy_accepted() -> bool | None:
        """Has this Windows account accepted the speech privacy policy?

        None on non-Windows or when the key cannot be read, so an unreadable registry never
        turns a working microphone off.
        """
        if sys.platform != "win32":
            return None
        try:
            import winreg

            with winreg.OpenKey(winreg.HKEY_CURRENT_USER, SPEECH_PRIVACY_KEY) as key:
                value, _ = winreg.QueryValueEx(key, "HasAccepted")
                return bool(value)
        except FileNotFoundError:
            return False
        except Exception:
            return None

    @staticmethod
    def _probe(modules: tuple[str, ...], attr: str) -> str | None:
        """Try each known module spelling. winrt names the speech modules without underscores
        (winrt.windows.media.speechrecognition); older releases used speech_recognition."""
        if sys.platform != "win32":
            return "Voice needs Windows (speech runs through the built-in Windows engines)"
        last = "not installed"
        for module in modules:
            try:
                mod = __import__(module, fromlist=[attr])
                getattr(mod, attr)
                return None
            except Exception as exc:  # missing package or missing speech feature
                last = f"{type(exc).__name__}: {exc}"
        return last

    @staticmethod
    def _import(modules: tuple[str, ...], attr: str):
        for module in modules:
            try:
                return getattr(__import__(module, fromlist=[attr]), attr)
            except Exception:
                continue
        raise VoiceError(f"{attr} is not available from any known winrt module")


    @staticmethod
    def _run(coro_factory):
        """Run a winrt coroutine on a thread of its own.

        FastAPI may call us from a thread that already drives an event loop; asyncio.run()
        would raise there and the failure used to surface as silence instead of an error.
        """
        import concurrent.futures

        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
            return pool.submit(lambda: asyncio.run(coro_factory())).result()

    # ---------------- speech -> text ----------------
    @classmethod
    def listen(cls, timeout_s: float = 8.0, pack: Pack | None = None) -> dict:
        """One utterance, heard through the Windows speech engine.

        Two modes are tried, in this order:

        1. a continuous recognition session, which reports what it heard as it hears it;
        2. one-shot ``recognize_async``.

        The one-shot call rejects anything it is not confident about and then returns an
        empty string with confidence REJECTED -- which is what an Indian-English speaker on
        a laptop microphone gets most of the time, and it reads as "the microphone is
        broken" when it is nothing of the sort. The continuous session hands back its
        hypothesis instead of discarding it, so low-confidence speech still becomes a
        question. Confidence is returned with the text, so a poor recognition is visible
        rather than silently trusted.
        """
        st = cls.status()
        if not st["stt"]:
            raise VoiceError(st["stt_error"] or "speech recognition unavailable")
        SpeechRecognizer = cls._import(STT_MODULES, "SpeechRecognizer")

        def build(recognizer):
            """Ask for free dictation explicitly; the default constraint set is stricter."""
            try:
                Topic = cls._import(STT_MODULES, "SpeechRecognitionTopicConstraint")
                Scenario = cls._import(STT_MODULES, "SpeechRecognitionScenario")
                recognizer.constraints.append(Topic(Scenario.DICTATION, "dictation"))
            except Exception as exc:
                log.debug("dictation constraint unavailable, using defaults: %s", exc)

        async def compile_or_fail(recognizer):
            compiled = await recognizer.compile_constraints_async()
            state = getattr(getattr(compiled, "status", None), "name", "")
            if state and state.lower() not in ("success", ""):
                raise VoiceError(
                    f"Windows dictation is not ready ({state}). Open Settings > Time & language > "
                    "Speech and install the English speech pack."
                )

        async def continuous():
            """Listen until something is heard or the window closes."""
            recognizer = SpeechRecognizer()
            build(recognizer)
            await compile_or_fail(recognizer)
            heard: list[tuple[str, str]] = []

            def on_result(_session, args):
                try:
                    result = args.result
                    text = (result.text or "").strip()
                    if text:
                        heard.append((text, getattr(result.confidence, "name", "")))
                except Exception:                     # a bad event must not kill the session
                    pass

            session = recognizer.continuous_recognition_session
            token = session.add_result_generated(on_result)
            await session.start_async()
            try:
                waited = 0.0
                while waited < timeout_s and not heard:
                    await asyncio.sleep(0.2)
                    waited += 0.2
                # let a final word land once speech has started
                if heard:
                    await asyncio.sleep(0.4)
            finally:
                try:
                    await session.stop_async()
                except Exception:
                    pass
                try:
                    session.remove_result_generated(token)
                except Exception:
                    pass
            if not heard:
                return None
            text, conf = max(heard, key=lambda pair: len(pair[0]))
            return {"text": text, "confidence": conf or "UNKNOWN", "status": "CONTINUOUS"}

        async def one_shot():
            recognizer = SpeechRecognizer()
            build(recognizer)
            await compile_or_fail(recognizer)
            try:
                from datetime import timedelta

                recognizer.timeouts.initial_silence_timeout = timedelta(seconds=min(timeout_s, 10))
                recognizer.timeouts.end_silence_timeout = timedelta(seconds=1.5)
            except Exception:
                pass
            result = await recognizer.recognize_async()
            status_name = getattr(result.status, "name", str(result.status))
            text = (result.text or "").strip()
            if not text:
                return {"text": "", "confidence": getattr(result.confidence, "name", ""),
                        "status": status_name}
            return {"text": text, "confidence": getattr(result.confidence, "name", ""),
                    "status": status_name}

        def attempt(coro_factory):
            try:
                return cls._run(coro_factory), None
            except VoiceError as exc:
                return None, exc
            except OSError as exc:
                if getattr(exc, "winerror", None) == SPEECH_PRIVACY_HRESULT \
                        or str(SPEECH_PRIVACY_HRESULT) in str(exc):
                    return None, VoiceError(SPEECH_PRIVACY_HELP)
                return None, VoiceError(f"Windows speech recognition failed: {type(exc).__name__}: {exc}")
            except Exception as exc:
                if str(SPEECH_PRIVACY_HRESULT) in str(exc):
                    return None, VoiceError(SPEECH_PRIVACY_HELP)
                return None, VoiceError(f"Windows speech recognition failed: {type(exc).__name__}: {exc}")

        got, failure = attempt(continuous)
        if not (got and got["text"]):
            fallback, second = attempt(one_shot)
            if fallback and fallback["text"]:
                got = fallback
            elif failure is None and second is not None and fallback is None:
                failure = second
            else:
                got = got or fallback

        if not (got and got.get("text")):
            if failure is not None:
                raise failure
            status_name = (got or {}).get("status", "UNKNOWN")
            if "cancel" in status_name.lower():
                raise VoiceError(MIC_HELP)
            raise VoiceError(
                "Windows heard the microphone but could not make out any words. Speak a full "
                "sentence a little louder, right after clicking, and check that the microphone "
                "Windows is set to use is the one you are speaking into "
                "(Settings > System > Sound > Input)."
            )

        heard_text = got["text"]
        return {"heard": heard_text, "text": repair(heard_text, pack),
                "confidence": got.get("confidence", ""), "status": got.get("status", "")}

    # ---------------- text -> speech ----------------
    @classmethod
    def speak(cls, text: str) -> bytes:
        st = cls.status()
        if not st["tts"]:
            raise VoiceError(st["tts_error"] or "speech synthesis unavailable")
        SpeechSynthesizer = cls._import(TTS_MODULES, "SpeechSynthesizer")
        from winrt.windows.storage.streams import DataReader

        async def run() -> bytes:
            synth = SpeechSynthesizer()
            stream = await synth.synthesize_text_to_stream_async(text[:1200])
            size = int(stream.size)
            reader = DataReader(stream.get_input_stream_at(0))
            await reader.load_async(size)
            return bytes(reader.read_buffer(size))

        try:
            return cls._run(run)
        except Exception as exc:
            raise VoiceError(f"Windows speech synthesis failed: {type(exc).__name__}: {exc}") from exc


def spoken_summary(answer: str, citations: list[dict]) -> str:
    """Short spoken version: warnings first, then the answer, then the source."""
    text = re.sub(r"\[\d+\]", "", answer).strip()
    warn = [ln for ln in text.splitlines() if ln.strip().upper().startswith(("WARNING", "CAUTION", "DANGER"))]
    body = " ".join(ln for ln in text.splitlines() if ln not in warn).strip()
    if len(body) > 600:
        body = body[:600].rsplit(".", 1)[0] + "."
    parts = warn + [body]
    if citations:
        c = citations[0]
        parts.append(f"Source: {c['doc_name']}, page {c['page']}.")
    return " ".join(p for p in parts if p)
