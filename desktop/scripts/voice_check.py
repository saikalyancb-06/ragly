"""Why is dictation not hearing anything? Run this and read the last line.

    .venv\\Scripts\\python.exe scripts\\voice_check.py

It prints what Windows actually offers (speech languages, the microphone, the consent
flags), then tries one real recognition. Nothing here touches the network.
"""
from __future__ import annotations

import asyncio
import platform
import sys


def head(text: str) -> None:
    print("\n" + text)
    print("-" * len(text))


def main() -> int:
    head("machine")
    print("python", platform.python_version(), "|", platform.machine(), "|", platform.platform())

    head("consent flags")
    try:
        import winreg

        for name, key, value in (
            ("speech privacy", r"SOFTWARE\Microsoft\Speech_OneCore\Settings\OnlineSpeechPrivacy", "HasAccepted"),
            ("microphone", r"SOFTWARE\Microsoft\Windows\CurrentVersion\CapabilityAccessManager\ConsentStore\microphone", "Value"),
        ):
            try:
                with winreg.OpenKey(winreg.HKEY_CURRENT_USER, key) as k:
                    print(f"{name:16} {winreg.QueryValueEx(k, value)[0]}")
            except Exception as exc:
                print(f"{name:16} unreadable ({exc})")
    except Exception as exc:
        print("registry unavailable:", exc)

    head("winrt speech module")
    SpeechRecognizer = None
    for module in ("winrt.windows.media.speechrecognition", "winrt.windows.media.speech_recognition"):
        try:
            SpeechRecognizer = getattr(__import__(module, fromlist=["SpeechRecognizer"]), "SpeechRecognizer")
            print("imported", module)
            break
        except Exception as exc:
            print("failed  ", module, "->", type(exc).__name__, exc)
    if SpeechRecognizer is None:
        print("\nVERDICT: the speech packages are not installed. Run scripts\\doctor.ps1.")
        return 1

    head("languages Windows can recognise")
    for attr in ("system_speech_language", "supported_topic_languages", "supported_grammar_languages"):
        try:
            value = getattr(SpeechRecognizer, attr)
            if hasattr(value, "language_tag"):
                print(f"{attr:28} {value.language_tag}")
            else:
                tags = [getattr(v, "language_tag", str(v)) for v in (value or [])]
                print(f"{attr:28} {tags or 'NONE'}")
        except Exception as exc:
            print(f"{attr:28} unreadable ({type(exc).__name__}: {exc})")

    head("recording device")
    try:
        from winrt.windows.devices.enumeration import DeviceClass, DeviceInformation

        async def mics():
            return await DeviceInformation.find_all_async_device_class(DeviceClass.AUDIO_CAPTURE)

        found = asyncio.run(mics())
        if not found:
            print("NO CAPTURE DEVICE - Windows sees no microphone at all")
        for d in found:
            print(("default " if d.is_default else "        ") + d.name + ("  [enabled]" if d.is_enabled else "  [DISABLED]"))
    except Exception as exc:
        print("could not enumerate microphones:", type(exc).__name__, exc)

    def dictation(recognizer):
        try:
            module = sys.modules[SpeechRecognizer.__module__]
            recognizer.constraints.append(
                module.SpeechRecognitionTopicConstraint(module.SpeechRecognitionScenario.DICTATION, "dictation"))
            return "dictation constraint"
        except Exception as exc:
            return f"default constraints ({type(exc).__name__})"

    head("test 1 of 2: continuous session - SPEAK NOW, a full sentence, for about four seconds")

    async def continuous():
        recognizer = SpeechRecognizer()
        print("using", dictation(recognizer))
        compiled = await recognizer.compile_constraints_async()
        print("constraint compile:", getattr(getattr(compiled, "status", None), "name", "?"))
        heard = []
        session = recognizer.continuous_recognition_session
        session.add_result_generated(
            lambda _s, args: heard.append((args.result.text or "",
                                           getattr(args.result.confidence, "name", "?"))))
        await session.start_async()
        waited = 0.0
        while waited < 8 and not any(t for t, _ in heard):
            await asyncio.sleep(0.2)
            waited += 0.2
        try:
            await session.stop_async()
        except Exception:
            pass
        return heard

    try:
        heard = asyncio.run(continuous())
        print("results:", heard or "none")
    except Exception as exc:
        heard = []
        print("continuous session raised", type(exc).__name__, exc)

    head("test 2 of 2: one-shot recognition - SPEAK AGAIN")

    async def run():
        recognizer = SpeechRecognizer()
        dictation(recognizer)
        await recognizer.compile_constraints_async()
        try:
            from datetime import timedelta

            recognizer.timeouts.initial_silence_timeout = timedelta(seconds=8)
            recognizer.timeouts.end_silence_timeout = timedelta(seconds=1.5)
        except Exception as exc:
            print("timeouts not set:", exc)
        return await recognizer.recognize_async()

    try:
        result = asyncio.run(run())
        status = getattr(getattr(result, "status", None), "name", "?")
        text = (result.text or "").strip()
        conf = getattr(getattr(result, "confidence", None), "name", "?")
        print(f"status={status}  confidence={conf}  text={text!r}")
    except Exception as exc:
        status, text, conf = "EXCEPTION", "", str(exc)
        print("the recogniser raised", type(exc).__name__, exc)

    head("verdict")
    if any(t for t, _ in heard):
        print("Continuous listening works; one-shot dictation is what was throwing your words")
        print("away. The app uses continuous listening first, so the mic button will work.")
        return 0
    if text:
        print("One-shot works. If the app still hears nothing, the app is the problem.")
        return 0
    if status.lower().startswith("topiclanguage"):
        print("The English speech pack is missing. In an ADMIN PowerShell:")
        print('  Add-WindowsCapability -Online -Name "Language.Speech~~~en-US~0.0.1.0"')
    elif "cancel" in status.lower():
        print("Windows cancelled it: the microphone is blocked or in use by another app.")
    elif conf.upper() == "REJECTED" or status.lower() == "timeoutexceeded":
        print("Windows heard audio but rejected every hypothesis. That is a signal problem, not")
        print("a permission one: check Settings > System > Sound > Input - the device shown there")
        print("is the one it listens to. Raise the input volume, speak a full sentence, and make")
        print("sure it is not a disconnected headset mic.")
    else:
        print(f"Status {status}, confidence {conf}, no text.")
    return 1


if __name__ == "__main__":
    sys.exit(main())
