"""Manual smoke check for the configured LLM provider (not collected by pytest).

Run: .venv/Scripts/python.exe -m tests.check_llm
"""

from app.llm import get_llm


def main() -> None:
    llm = get_llm()
    reply = llm.chat([{"role": "user", "content": "Say hello in one word"}])
    print(f"provider: {type(llm).__name__}")
    print(f"reply:    {reply.text!r}")


if __name__ == "__main__":
    main()
