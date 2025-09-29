import os
import re
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st
import tempfile

# --- LangChain Imports ---
from langchain.chains import ConversationChain
from langchain.memory import ConversationBufferMemory
from langchain_google_genai import ChatGoogleGenerativeAI


# --- Funções auxiliares ---
def log_message(text: str):
    """Substitui prints por exibição no Streamlit"""
    st.write(text)


def gerar_prompt_python(csv_text: str, question: str, csv_path: str) -> str:
    """Monta o prompt que será enviado para o LLM"""
    prompt = f"""
    You have received a CSV file of financial transactions with approximately 284,808 rows and 31 columns.
    Here is a preview of the first lines to help you understand the data structure:
    {csv_text}

    Columns:
    - Time: number of seconds since the first transaction;
    - V1 to V28: variables resulting from PCA dimensionality reduction;
    - Amount: transaction value;
    - Class: indicates if the transaction is fraudulent (1) or normal (0).

    User question: "{question}"

    Your task:
    1. Provide a clear, small, simple and concise answer to the user question.
    2. Always base your reasoning and answers on the CSV dataset structure and content (csv_text is only a preview, but the full file should be used in the code).
    3. When generating Python code, always load the dataset using exactly:
       df = pd.read_csv("{csv_path}")
    5. Output only executable Python code (using pandas, numpy, matplotlib, or seaborn).
    6. In case the python code uses the print(), replace it with log_message()
    """
    return prompt.strip()


def executar_codigo_da_resposta(resposta: str):
    """Executa o código Python retornado pelo LLM"""
    blocos = re.findall(r"```python(.*?)```", resposta, re.DOTALL)

    if not blocos:
        log_message("ℹ️ Nenhum código Python detectado na resposta.\n")
        log_message(resposta)
        return

    for i, codigo in enumerate(blocos, 1):
        log_message(f"💻 Executando código Python:\n")
        try:
            local_vars = {}
            exec(codigo, globals(), local_vars)

            # --- Captura TODAS as figuras abertas ---
            figs = [plt.figure(n) for n in plt.get_fignums()]
            for fig in figs:
                st.pyplot(fig)

            # Limpa TODAS as figuras depois de exibir
            plt.close('all')

        except Exception as e:
            log_message(f"⚠️ Erro ao executar o código: {e}")

    # Texto que não está em blocos de código
    texto_fora = re.sub(r"```python.*?```", "", resposta, flags=re.DOTALL).strip()
    if texto_fora:
        log_message("📝 Resposta do agente:")
        log_message(texto_fora)



def enviar_pergunta(question: str, csv_text: str, csv_path: str, agent_chain):
    """Envia a pergunta do usuário ao LLM"""
    if not question.strip():
        log_message("⚠️ Digite uma pergunta válida.")
        return

    try:
        prompt = gerar_prompt_python(csv_text, question, csv_path)

        # Spinner para feedback durante a chamada ao LLM
        with st.spinner("🤖 O agente está pensando..."):
            answer = agent_chain.run(prompt)

        executar_codigo_da_resposta(answer)
    except Exception as e:
        log_message(f"📝 Ocorreu um erro: {e}")


# --- Interface Streamlit ---
st.title("🤖 Agente I2A2 - Individual")

# Campo para o usuário inserir a API Key
api_key = st.text_input("🔑 Insira sua Google API Key:", type="password")

if api_key:
    os.environ["GOOGLE_API_KEY"] = api_key

    # Criando o modelo de linguagem conectado ao Google Generative AI via LangChain
    model = ChatGoogleGenerativeAI(
        model="models/gemini-2.5-flash",
        temperature=0
    )

    # Memória da conversa
    memory = ConversationBufferMemory(memory_key="history", return_messages=True)

    # Criação da cadeia de conversação
    agent_chain = ConversationChain(llm=model, memory=memory, verbose=True)

    # Upload de CSV
    uploaded_file = st.file_uploader("📂 Faça upload do arquivo CSV", type=["csv"])

    if uploaded_file is not None:
        with st.spinner("📂 Processando o arquivo CSV..."):
            df = pd.read_csv(uploaded_file)

            tmp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".csv")
            tmp_file.write(uploaded_file.getbuffer())
            tmp_file.close()

            csv_path = tmp_file.name  # ✅ real path accessible on the server
            csv_text = "\n".join(df.to_csv(index=False).splitlines()[:5])

        st.success(f"✅ Arquivo '{uploaded_file.name}' carregado com sucesso!")
        
        pergunta = st.text_input("❓ Digite sua pergunta:")

        if st.button("Enviar pergunta"):
            enviar_pergunta(pergunta, csv_text, csv_path, agent_chain)
    else:
        st.info("⬆️ Por favor, faça upload de um arquivo CSV para continuar.")
else:
    st.warning("⚠️ Insira sua API Key para continuar.")
