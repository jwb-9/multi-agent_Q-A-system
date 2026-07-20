# 生产级多智能体Agent RAG问答系统
## 项目简介
基于 Python + FastAPI + LangGraph 构建企业私有化多智能体问答平台，融合私有知识库RAG与实时联网搜索，解决单一大模型上下文不足、信息滞后、幻觉严重问题。
项目完整落地工业级Agent核心技术：任务规划Plan-and-Execute、反思自校验Reflection、三层分层记忆、标准化工具调用、LangGraph断点续跑、Agentic多跳检索；配套完整工程化能力：容器兼容、Redis缓存记忆、全链路日志追踪、统一异常处理、增量向量库、会话隔离，可直接用于内网私有化部署。

## 核心技术栈
### 后端框架
FastAPI、uvicorn、Pydantic v2
### Agent & LLM生态
LangChain、LangGraph、LangChain-Ollama、Chroma向量库
### 中间件
Redis（记忆存储、Graph Checkpoint断点）
### 工具能力
DDGS开源联网搜索、Ollama本地开源大模型
### 工程配套
结构化日志、滚动日志、全局异常捕获、入参校验、健康探针

## 项目架构说明
### 多智能体工作流程
1. **路由Agent**：结合历史对话记忆判定问题类型（rag/search/both/direct）
2. **规划Agent**：自动拆解复杂问题，生成多工具执行任务列表
3. **工具执行Agent**：统一调用私有知识库检索/联网搜索，记录工具审计日志
4. **反思校验Agent**：评估信息完整度，信息不足自动重试检索（限制最大循环次数）
5. **汇总合成Agent**：整合本地知识库、外网实时资讯、历史对话记忆，输出带来源溯源的最终答案

### 核心Agent能力亮点
1. Plan-and-Execute任务自动拆解，支持多工具链式调用
2. Reflection自校验循环，实现多跳Agentic RAG，抑制模型幻觉
3. 统一BaseTool工具框架，可快速扩展代码解释器、SQL查询、PDF解析等工具
4. 三层记忆机制：短期会话上下文、长期历史问答经验，自动摘要压缩长对话
5. LangGraph Redis Checkpoint：流程断点续跑，中断请求可恢复执行
6. 结构化JSON输出，LLM脏文本自动容错解析，接口稳定性大幅提升

## 环境准备
### 1. 本地中间件启动
1. Redis 本地服务（默认端口6379）
2. Ollama 本地服务，拉取模型：
```bash
ollama pull qwen2.5:7b
ollama pull bge-m3
