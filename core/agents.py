import os
from dotenv import load_dotenv
from langchain_ollama import ChatOllama, OllamaEmbeddings
from langchain_chroma import Chroma
from langchain_core.prompts import ChatPromptTemplate
from tavily import TavilyClient
from duckduckgo_search import DDGS

load_dotenv()

# 1. 初始化本地Ollama对话模型
llm = ChatOllama(
    base_url=os.getenv("OLLAMA_BASE_URL"),
    model=os.getenv("LLM_MODEL"),
    temperature=0.1
)

# 2. 初始化本地Ollama嵌入模型
embedding = OllamaEmbeddings(
    base_url=os.getenv("OLLAMA_BASE_URL"),
    model=os.getenv("EMBED_MODEL")
)

# 联网搜索客户端
tavily = TavilyClient(api_key=os.getenv("TAVILY_API_KEY", ""))

# 向量库初始化（本地知识库）
vector_store = Chroma(persist_directory="./chroma_db", embedding_function=embedding)
retriever = vector_store.as_retriever(search_kwargs={"k": 3})

# --------------------------
# Agent1：路由调度智能体
# --------------------------
def route_agent(state) -> dict:
    prompt = ChatPromptTemplate.from_template("""
    你是任务调度专家，根据用户问题判断交给哪个子Agent处理，只返回对应单词：
    可选：rag / search / both / direct
    rag：需要查询本地文档、私有知识库
    search：需要实时网络信息、新闻、最新数据
    both：既需要本地文档又需要联网
    direct：问题简单，直接回答无需检索
    用户问题：{query}
    仅输出一个单词，不要额外文字
    """)
    chain = prompt | llm
    res = chain.invoke({"query": state["user_query"]})
    route = res.content.strip().lower()
    return {"route_target": route}

# --------------------------
# Agent2：RAG知识库智能体
# --------------------------
def rag_agent(state):
    docs = retriever.get_relevant_documents(state["user_query"])
    doc_text = "\n".join([d.page_content for d in docs])
    prompt = ChatPromptTemplate.from_template("""
    根据下面知识库内容回答用户问题，只提炼关键信息：
    知识库：{docs}
    用户问题：{query}
    """)
    chain = prompt | llm
    ans = chain.invoke({"docs": doc_text, "query": state["user_query"]})
    return {"rag_result": ans.content}

# --------------------------
# Agent3：联网搜索智能体
# --------------------------
def search_agent(state):
    query = state["user_query"]
    results = []
    with DDGS() as ddgs:
        resp = ddgs.text(query, max_results=3)
        for r in resp:
            results.append(f"{r['title']}: {r['body']}")
    search_text = "\n".join(results)
    prompt = ChatPromptTemplate.from_template("""
    整理以下搜索结果，提炼有效信息：
    搜索内容：{search_text}
    用户问题：{query}
    """)
    chain = prompt | llm
    ans = chain.invoke({"search_text": search_text, "query": state["user_query"]})
    return {"search_result": ans.content}
# def search_agent(state):
#     # 无key时跳过搜索
#     if not tavily.api_key:
#         return {"search_result": "未配置搜索API，无法联网查询"}
#     search_data = tavily.search(query=state["user_query"], max_results=3)
#     search_text = "\n".join([f"{item['title']}:{item['content']}" for item in search_data["results"]])
#     prompt = ChatPromptTemplate.from_template("""
#     整理以下搜索结果，提炼有效信息：
#     搜索内容：{search_text}
#     用户问题：{query}
#     """)
#     chain = prompt | llm
#     ans = chain.invoke({"search_text": search_text, "query": state["user_query"]})
#     return {"search_result": ans.content}

# --------------------------
# Agent4：汇总合成智能体
# --------------------------
def summary_agent(state):
    rag_info = state.get("rag_result", "无本地知识库信息")
    search_info = state.get("search_result", "无网络搜索信息")
    prompt = ChatPromptTemplate.from_template("""
    整合下面两部分信息，完整、通顺回答用户问题，不要冗余：
    本地知识库信息：{rag_info}
    网络实时信息：{search_info}
    用户提问：{query}
    """)
    chain = prompt | llm
    ans = chain.invoke({
        "rag_info": rag_info,
        "search_info": search_info,
        "query": state["user_query"]
    })
    return {"final_answer": ans.content}