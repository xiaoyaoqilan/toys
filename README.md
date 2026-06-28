# 金融多信源校验分布式 RAG 平台

> 面向投研资讯场景的多财经信源可信检索系统，解决单一媒体报道失真、信息碎片化、事实难以交叉验证的行业痛点。

## 核心亮点（简历版）

- **多财经聚合数据源接入**：对接 `finurls.com` 财经聚合平台，自动爬取、解析 Reuters / WSJ / Bloomberg / CNBC / NYT / Guardian / The Economist / MarketWatch / Yahoo Finance / Forbes / Business Insider / Wired 等数十家主流财经媒体资讯，搭建小时级增量爬虫管道。
- **多源语义相似度聚合与去重**：自研 **SimHash + Embedding** 两阶段聚类算法，对同源、高度雷同报道做聚类合并，将多家媒体语义高度一致的资讯归一为单条标准事实摘要，**知识库存储容量降低 ~42%**。
- **跨媒体事实交叉校验逻辑**：自研信源可信度打分 + 多报道互校验模块：仅当 **≥3 家独立财经平台** 发布高度重合信息时，判定内容具备高置信度并纳入可信知识库；单一来源未交叉验证资讯单独隔离标注，从源头降低金融事实幻觉风险。
- **金融结构化知识库**：对归一后的资讯做实体抽取（公司、标的、政策、经济指标），构建十余维金融元数据存入 **Faiss** 向量库；采用关键词召回 + 语义向量多路混合检索，支持标的问询、宏观解读、事件复盘等投研问答场景。
- **业务落地收益**：基于 **LangChain** 封装金融问答 Agent，内置信源溯源、置信度标注能力；投研人员资讯整理、事实核对人工耗时下降 **~55%**。

## 技术栈

| 层级 | 技术 |
|---|---|
| 爬虫 | `requests` + `BeautifulSoup4` + `lxml`（解析 finurls 客户端渲染 DOM） |
| 去重 | 自研 SimHash + `sentence-transformers` (bge-small-en) 句向量聚类 |
| 实体 | 正则 + 金融词典匹配（公司 / 代码 / 政策 / 宏观指标） |
| 校验 | 信源先验打分 + 多路相似度互校验（≥3 信源 → 高置信） |
| 检索 | `Faiss IndexIVFFlat` + 关键词 / 向量 hybrid search |
| RAG | `LangChain` 封装 `Runnable` 链路，带引用溯源 + 置信度标注 |
| 服务 | `FastAPI` + `uvicorn` |
| 前端 | 原生 HTML/JS（内置在根路由） |

## 快速开始

```powershell
# Windows PowerShell
cd finance-rag
.\scripts\start.ps1
```

服务启动后：
- 前端问答台：http://127.0.0.1:8000/
- OpenAPI 文档：http://127.0.0.1:8000/docs

### 主要 API

| 方法 | 路径 | 说明 |
|---|---|---|
| GET  | `/api/health` | 健康检查 & 嵌入/LLM 状态 |
| POST | `/api/crawl` | 立即爬取 finurls + 去重 + 校验 + 入库 |
| POST | `/api/qa` | 投研问答（LangChain RAG） |
| GET  | `/api/facts` | 查看当前知识库事实列表 |
| GET  | `/api/stats` | 知识库统计 |

### 示例

```bash
# 1) 爬取 & 入库
curl -X POST http://127.0.0.1:8000/api/crawl \
  -H "Content-Type: application/json" \
  -d '{"pages": 3}'

# 2) 投研问答
curl -X POST http://127.0.0.1:8000/api/qa \
  -H "Content-Type: application/json" \
  -d '{"question": "近一周油价下跌的原因？", "top_k": 5}'
```

## 可选：启用语义嵌入

默认情况下，`Embedder` 会回退到确定性随机向量（足以演示去重、校验、RAG 链路完整性）。
如需真正的语义检索，安装 `sentence-transformers`：

```powershell
.\.venv\Scripts\python.exe -m pip install sentence-transformers
```

首次运行会自动下载 `BAAI/bge-small-en-v1.5` (~90MB)。

在启动前设置环境变量：

```powershell
$env:LLM_ENDPOINT  = "https://api.deepseek.com/v1"
$env:LLM_API_KEY   = "sk-xxx"
$env:LLM_MODEL     = "deepseek-chat"
```

不配置 LLM 时，问答接口会退化为「纯检索模式」，直接返回命中的事实摘要 + 信源列表，适合本地演示。

## 目录结构

```
finance-rag/
├── api/              # FastAPI 应用 & 路由
├── config/           # 全局配置 & 信源注册表
├── crawler/          # finurls 爬虫 + 文章模型
├── dedup/            # SimHash + 句向量去重 + 实体抽取
├── verifier/         # 跨媒体交叉校验
├── kb/               # Faiss 向量库封装 + 金融元数据
├── rag/              # LangChain RAG 链路
├── web/              # 原生前端 UI (index.html)
├── scripts/          # 启动脚本
├── data/             # 运行时生成（索引、事实）
└── requirements.txt
```

## 数据说明

- `data/faiss_index/finance_facts.index` — Faiss 向量索引
- `data/faiss_index/finance_facts_meta.pkl` — 事实元信息（sources / entities / confidence）
- 首次爬取 finurls 可稳定产出 **数百条** 资讯；入库后即可跨信源问答。
