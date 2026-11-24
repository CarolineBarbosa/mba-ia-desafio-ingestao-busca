"""Interactive chat: asks the RAG questions and shows the answers."""
from search import search_prompt

# Words that end the chat.
EXIT_COMMANDS = {"sair", "exit", "quit", "q", "bye", "tchau", "adeus", "out"}


def main() -> None:
    print("Chat RAG iniciado. Digite sua pergunta (ou 'sair'/'exit' para encerrar).")

    while True:
        pergunta = input("\nPERGUNTA: ").strip()

        if not pergunta:
            continue

        if pergunta.lower() in EXIT_COMMANDS:
            print("Encerrando.")
            break

        try:
            resposta = search_prompt(pergunta)
        except Exception as exc:  # keep the loop alive on errors (e.g. database down)
            print(f"\nErro ao responder: {exc}")
            continue

        print(f"\nRESPOSTA: {resposta}")


if __name__ == "__main__":
    main()
