"""Небольшой тест генерации текста через Chat Completions API.

Chat Completions не хранит диалог на сервере: каждый запрос независимый.
Память — это список messages в программе. На каждом ходе в него добавляются
реплика пользователя и ответ модели, и весь список уходит в следующий запрос.
Пока список жив, чат продолжается и модель видит предыдущий контекст.
"""

import json
import os
import sys
import urllib.error
import urllib.request
from pathlib import Path

from dotenv import load_dotenv
from openai import APIError, OpenAI

load_dotenv(Path(__file__).resolve().parent / ".env")

REGULAR_MODEL = os.environ.get("OPENAI_MODEL", "openai/gpt-5.6-luna")
THINKING_MODEL = "anthropic/claude-sonnet-5-5"
# Поле max_tokens у Claude обязательное. 128000 — потолок самой модели, не отдельный лимит.
THINKING_MAX_TOKENS = 128000
ANTHROPIC_VERSION = "2026-10-21"
EXIT_COMMANDS = {"/exit", "/quit", "exit", "quit", "/выход", "выход"}
RESET_COMMANDS = {"/reset", "/сброс"}

SYSTEM_PROMPT = (
    "Ты дружелюбный собеседник. Отвечай кратко. "
    "Если пользователь пишет по-русски, отвечай по-русски."
)


def create_client() -> OpenAI:
    """Клиент читает ключ из .env или из переменной окружения OPENAI_API_KEY.

    Если задан OPENAI_BASE_URL, запросы идут на этот адрес, а не на api.openai.com.
    """
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        print(
            "Не найден OPENAI_API_KEY. Добавьте его в файл .env и запустите скрипт снова.",
            file=sys.stderr,
        )
        sys.exit(1)

    base_url = os.environ.get("OPENAI_BASE_URL")
    if base_url:
        return OpenAI(api_key=api_key, base_url=base_url)
    return OpenAI(api_key=api_key)


def new_dialog(system_prompt: str = SYSTEM_PROMPT) -> list[dict[str, str]]:
    """Начинает диалог: в истории только системная инструкция."""
    return [{"role": "system", "content": system_prompt}]


def choose_mode() -> str:
    """Спрашивает режим в начале. Пустой ввод оставляет думающую модель."""
    print("Выберите режим:")
    print("  1 — думающая модель, Claude Sonnet 5.5 (по умолчанию)")
    print("  2 — обычная модель")

    while True:
        choice = input("Режим [1]: ").strip()
        if choice in {"", "1"}:
            return "thinking"
        if choice == "2":
            return "regular"
        print("Введите 1 или 2.")


def request_thinking(client: OpenAI, messages: list[dict[str, str]]) -> tuple[str, str]:
    """Запрос к Claude через /v1/messages: Chat Completions отбрасывает thinking.

    Возвращает ответ и текст размышления. В истории диалога остаётся только ответ.
    """
    system_text = "\n".join(
        message["content"] for message in messages if message["role"] == "system"
    )
    dialog = [message for message in messages if message["role"] != "system"]
    payload: dict = {
        "model": THINKING_MODEL,
        "max_tokens": THINKING_MAX_TOKENS,
        "thinking": {"type": "adaptive", "display": "summarized"},
        "output_config": {"effort": "max"},
        "messages": dialog,
    }
    if system_text:
        payload["system"] = system_text

    base_url = str(client.base_url).rstrip("/")
    request = urllib.request.Request(
        f"{base_url}/messages",
        data=json.dumps(payload).encode("utf-8"),
        headers={
            "x-api-key": str(client.api_key),
            "anthropic-version": ANTHROPIC_VERSION,
            "content-type": "application/json",
        },
    )
    with urllib.request.urlopen(request, timeout=90) as response:
        body = json.loads(response.read().decode("utf-8"))

    thoughts: list[str] = []
    answers: list[str] = []
    for block in body.get("content") or []:
        if block.get("type") == "thinking":
            text = (block.get("thinking") or "").strip()
            if text:
                thoughts.append(text)
        elif block.get("type") == "text":
            text = (block.get("text") or "").strip()
            if text:
                answers.append(text)
    return "\n\n".join(answers), "\n\n".join(thoughts)


def print_reply(reply: str, thinking: str = "") -> None:
    """Печатает размышление и ответ отдельными блоками."""
    if thinking:
        print()
        print("Размышление")
        print("=" * 40)
        print(thinking)
    print()
    print("Ответ")
    print("=" * 40)
    print(reply)


def generate_text(
    client: OpenAI,
    messages: list[dict[str, str]],
    user_text: str,
    model: str = REGULAR_MODEL,
    *,
    thinking: bool = False,
) -> tuple[str, str]:
    """Генерирует ответ и дописывает обе реплики в историю диалога.

    messages меняется на месте: после вызова в нём есть и вопрос, и ответ.
    Следующий вызов с тем же списком продолжает тот же разговор.
    Возвращает текст ответа и, если он пришёл, текст размышления.
    """
    messages.append({"role": "user", "content": user_text})
    try:
        if thinking:
            reply, reasoning = request_thinking(client, messages)
        else:
            completion = client.chat.completions.create(
                model=model,
                messages=messages,
            )
            reply = completion.choices[0].message.content or ""
            reasoning = ""
    except (APIError, urllib.error.HTTPError, urllib.error.URLError, TimeoutError):
        messages.pop()
        raise

    messages.append({"role": "assistant", "content": reply})
    return reply, reasoning


def run_dialog(client: OpenAI, model: str = REGULAR_MODEL, *, thinking: bool = False) -> None:
    """Крутит чат, пока пользователь сам не выйдет."""
    messages = new_dialog()

    print(f"\nДиалог запущен. Модель: {model}")
    print("Контекст сохраняется между сообщениями.")
    print("Команды: /reset — очистить историю, /exit — выйти.")

    while True:
        try:
            user_text = input("\nВы: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nДиалог завершён.")
            return

        if not user_text:
            continue

        command = user_text.lower()
        if command in EXIT_COMMANDS:
            print("Диалог завершён.")
            return
        if command in RESET_COMMANDS:
            messages = new_dialog()
            print("История очищена. Можно начать новый разговор.")
            continue

        try:
            reply, reasoning = generate_text(
                client,
                messages,
                user_text,
                model,
                thinking=thinking,
            )
        except APIError as error:
            print(f"Ошибка API: {error}")
            continue
        except urllib.error.HTTPError as error:
            detail = error.read().decode("utf-8", errors="replace")
            print(f"Ошибка API: {error.code} {detail}")
            continue
        except urllib.error.URLError as error:
            print(f"Ошибка сети: {error.reason}")
            continue
        except TimeoutError:
            print("Ошибка сети: превышено время ожидания.")
            continue

        print_reply(reply, reasoning)


def main() -> None:
    thinking = choose_mode() == "thinking"
    model = THINKING_MODEL if thinking else REGULAR_MODEL
    run_dialog(create_client(), model, thinking=thinking)


if __name__ == "__main__":
    main()
