"""
Mia CLI chat — thin wrapper around MiaEngine (same path as the API).
"""

from .engine import EXIT_PHRASES, MiaEngine


def main():
    engine = MiaEngine()
    print("Choose chat model: 1 = 7B (safer)   2 = 14B (smarter, tighter VRAM)")
    choice = input("Model [1]: ").strip() or "1"
    model_key = "14b" if choice == "2" else "7b"
    engine.load(model_key)
    print("Type a message and press Enter. Say good night / 晚安 / exit to leave.")
    print(f"Chat log session: {engine.session_id[:8]}…")

    while True:
        user_text = input("\nYou: ").strip()
        if not user_text:
            continue
        try:
            result = engine.chat(user_text)
        except (ValueError, RuntimeError) as e:
            print(f"[error] {e}")
            break

        hits = result.get("recalled") or []
        if hits:
            preview = ", ".join(
                f"{h['text'][:24]}…({h['score']:.2f})" for h in hits if h.get("text")
            )
            print(f"[memory] recalled: {preview}")
        else:
            print("[memory] recalled: (none)")

        print(f"Mia: {result['reply']}")

        if result.get("ended") or user_text.lower() in EXIT_PHRASES:
            print("[organize] done." if result.get("ended") else "")
            break


if __name__ == "__main__":
    main()
