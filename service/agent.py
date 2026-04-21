import os
from typing import Annotated, TypedDict
from dotenv import load_dotenv

from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage
from langchain_groq import ChatGroq
from langgraph.graph import StateGraph, END
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode

# Importamos a ferramenta que criamos no passo anterior
from service.agent_tools import run_vision_segmentation

load_dotenv()

# --- 1. O PROMPT DO ESPECIALISTA (SYSTEM PROMPT) ---
SYSTEM_PROMPT = """Você é um Engenheiro de IA Geoespacial Sênior.
Sua função é orquestrar um motor de Visão Computacional (U-Net) e apresentar os resultados com excelência e precisão.

REGRAS DE OPERAÇÃO:
1. Acione a ferramenta `run_vision_segmentation` APENAS UMA VEZ. Ao receber os dados, pare e formule a resposta final.
2. NUNCA invente métricas, tempos ou diretórios.

FORMATO OBRIGATÓRIO DE RESPOSTA:
Quando a ferramenta retornar sucesso, você DEVE estruturar sua resposta exatamente assim (usando Markdown):

**Relatório de Análise Geoespacial: [Nome da Fazenda]**

* **Status da Inferência:** Concluída com sucesso.
* **Tempo de Processamento:** [Exato tempo retornado]
* **Arquivo de Saída:** `[Exato path retornado]`
* **Área Detectada:** [Número de Pixels] pixels de infraestrutura/pivô identificados.
* **NDVI Médio:** [Valor do NDVI]

**Análise Agronômica:**
[Aqui você explica o que o NDVI significa, além de considerações sobre as áreas analisadas. Lembre-se: NDVI entre 0.0 e 0.15 significa solo exposto ou pousio dentro da área do pivô. NDVI acima de 0.5 significa cultura verde e ativa. Seja analítico e técnico.]
"""

# --- 2. O ESTADO (A Memória) ---
class AgentState(TypedDict):
    messages: Annotated[list, add_messages]

# --- 3. CONFIGURAÇÃO DO LLM (GROQ / LLAMA 3) ---
# Usamos o modelo de 70B parâmetros, que tem excelente raciocínio lógico para Tool Calling
llm = ChatGroq(
    model="llama-3.3-70b-versatile", 
    temperature=0.1, # Temperatura baixa = respostas mais precisas e menos "criativas"
    api_key=os.getenv("GROQ_API_KEY")
)

# Conectamos a ferramenta ao "Cérebro"
tools = [run_vision_segmentation]
llm_with_tools = llm.bind_tools(tools)

# --- 4. OS NÓS (Quem Trabalha) ---
def assistant_node(state: AgentState):
    """Nó do LLM: Pensa, avalia o histórico e decide o que falar ou fazer."""
    messages = state["messages"]
    
    # Injeta a persona Sênior caso seja o início da conversa
    if not any(isinstance(m, SystemMessage) for m in messages):
        messages = [SystemMessage(content=SYSTEM_PROMPT)] + messages
        
    response = llm_with_tools.invoke(messages)
    return {"messages": [response]}

# O nó de ferramentas já vem pronto do LangGraph (ToolNode)
tool_node = ToolNode(tools)

# --- 5. AS ARESTAS (Regras de Trânsito) ---
def should_continue(state: AgentState):
    """Verifica se o LLM decidiu usar uma ferramenta ou se quer responder ao usuário."""
    last_message = state["messages"][-1]
    
    if last_message.tool_calls:
        return "tools"
    return END

# --- 6. CONSTRUINDO O GRAFO ---
workflow = StateGraph(AgentState)

workflow.add_node("assistant", assistant_node)
workflow.add_node("tools", tool_node)

workflow.set_entry_point("assistant")
workflow.add_conditional_edges("assistant", should_continue)
workflow.add_edge("tools", "assistant")

# Compila o grafo em um aplicativo executável
spatial_agent = workflow.compile()

# --- 7. TESTE LOCAL ---
if __name__ == "__main__":
    print("🌍 [Spatial Agent] Online. Motor Groq/Llama Ativado.")
    
    # Simulando o usuário iniciando a conversa
    user_input = "Olá. Preciso que você faça a análise de risco e segmente a fazenda_teste_001. Pode me ajudar?"
    print(f"\n👨‍🌾 Usuário: {user_input}")
    
    # Iniciando o estado vazio
    initial_state = {"messages": [HumanMessage(content=user_input)]}
    
    # Rodando o Grafo
    for event in spatial_agent.stream(initial_state):
        for key, value in event.items():
            if key == "assistant":
                # Se for o assistente apenas pensando/falando
                if not value["messages"][0].tool_calls:
                    print(f"\n🤖 Agente: {value['messages'][0].content}")
                else:
                    print(f"\n🧠 [Agente Pensando...] Decidiu acionar a ferramenta de visão computacional.")
            elif key == "tools":
                # Se for a ferramenta sendo executada
                print(f"⚙️ [Sistema] Ferramenta executada. Retorno recebido.")