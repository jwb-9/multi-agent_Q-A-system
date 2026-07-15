from core.logger import logger
from langchain_ollama import ChatOllama, OllamaEmbeddings
from langchain_chroma import Chroma
from langchain_core.prompts import ChatPromptTemplate
from tavily import TavilyClient
from ddgs import DDGS
from ddgs.exceptions import DDGSException

from core.config import settings

# 初始化Ollama LLM
llm = ChatOllama(
    base_url=settings.OLLAMA_BASE_URL,
    model=settings.LLM_MODEL,
    temperature=0.1
)

# 嵌入模型
embedding = OllamaEmbeddings(
    base_url=settings.OLLAMA_BASE_URL,
    model=settings.EMBED_MODEL
)

# 向量库
vector_store = Chroma(persist_directory=settings.CHROMA_PERSIST_DIR, embedding_function=embedding)
retriever = vector_store.as_retriever(search_kwargs={"k": settings.RETRIEVE_TOP_K})

# 搜索客户端
tavily = TavilyClient(api_key=settings.TAVILY_API_KEY)

# --------------------------
# Agent1：路由调度智能体
# --------------------------
def route_agent(state) -> dict:
    query = state["user_query"]
    logger.info(f"路由Agent开始处理问题：{query}")
    prompt = ChatPromptTemplate.from_template("""
    你是任务调度专家，仅输出单个单词：rag / search / both / direct，禁止多余文字。
    分类严格规则：
    1. rag：问题和【多智能体、LangGraph、集中式路由、RAG向量检索】相关，本地知识库存有专业文档；
    2. search：需要2024~2026最新资讯、外网行业动态、实时数据；
    3. both：同时需要本地多智能体知识 + 外网最新落地信息；
    4. direct：数学、通用基础常识，和多智能体技术无关。
    
    示例：
    问：LangGraph集中路由架构优势 → rag
    问：2026最新大模型 → search
    问：LangGraph最新工业落地方案 → both
    问：1+1等于几 → direct
    
    用户问题：{query}
    仅输出标签单词
    """)
    chain = prompt | llm
    res = chain.invoke({"query": state["user_query"]})
    route = res.content.strip().lower()
    logger.info(f"路由判定结果：{route}")
    return {"route_target": route}

# --------------------------
# Agent2：RAG知识库智能体
# --------------------------
def rag_agent(state):
    logger.info("执行RAG知识库检索")
    docs = retriever.invoke(state["user_query"])
    logger.info(f"检索匹配文档数量：{len(docs)}")
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
    logger.info("执行联网搜索工具")
    query = state["user_query"]
    results = []
    try:
        with DDGS() as ddgs:
            resp = ddgs.text(query, max_results=3)
            for r in resp:
                results.append(f"{r['title']}: {r['body']}")
    except DDGSException as e:
        logger.error(f"联网搜索失败：{str(e)}")
        results.append("当前网络环境无法访问搜索引擎，无实时网络资讯")
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
    logger.info("执行结果汇总Agent")
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
    logger.info("答案生成完成")
    return {"final_answer": ans.content}