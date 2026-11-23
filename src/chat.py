"""Chat interativo: faz perguntas ao RAG e exibe as respostas."""
from search import search_prompt

# Palavras que encerram o chat.
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
        except Exception as exc:  # mantém o loop vivo em erros (ex.: banco fora do ar)
            print(f"\nErro ao responder: {exc}")
            continue

        print(f"\nRESPOSTA: {resposta}")


if __name__ == "__main__":
    main()
