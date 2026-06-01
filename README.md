# AgenticBI_Final_Olist

一个面向 **Brazilian E-Commerce Public Dataset by Olist** 的 Agentic BI 课程项目基础实现。

它实现了：

- Olist 9 张业务表导入与清洗入口
- MySQL 查询引擎支持，同时提供 SQLite demo 方便本地快速运行
- 6 张预聚合表/视图：`mv_monthly_sales`、`mv_state_sales`、`mv_category_sales`、`mv_delivery_perf`、`mv_seller_perf`、`mv_payment_dist`
- 多 Agent 协作：协调器、数据分析 Agent、可视化 Agent、NLP 评论洞察 Agent、预测 Agent、决策智能 Agent
- Streamlit Web 界面：自然语言提问、SQL 查询结果、图表展示、决策建议
- 6 类可视化：折线图、地图气泡图、柱状图、热力图、散点图、词云/文本主题
- 预测分析：基于历史 GMV 的未来 6 周简单趋势预测
- 可选 LLM 接入：支持 OpenAI-compatible API，例如 DeepSeek、Qwen、GPT 等；无 API Key 时使用规则模板兜底

---

## 1. 快速运行 Demo

本项目自带小样本数据生成脚本，不需要先下载 Kaggle 数据也能跑通界面。

```bash
cd AgenticBI_Final_Olist
pip install -r requirements.txt
python utils/init_sample_db.py
streamlit run app.py
```

浏览器打开 Streamlit 页面后，可以直接问：

```text
2017 年 GMV 是多少？按月和各州排名的趋势怎样？
平台整体准时交付率是多少？哪些州延迟最严重？
哪种支付方式最受欢迎？平均分期数是多少？
产品的重量、尺寸与运费之间有什么关系？
Top 10 差评品类及其主要差评原因是什么？
根据历史订单趋势，预测未来 6 周的销售额，并给出趋势解读。
基于全部分析结果，给出平台 3 个月内的三大优先改进策略。
```

---

## 2. 使用真实 Olist 数据

从 Kaggle 下载 **Brazilian E-Commerce Public Dataset by Olist**，把 CSV 放到：

```text
data/raw/
```

需要的文件名：

```text
olist_orders_dataset.csv
olist_order_items_dataset.csv
olist_products_dataset.csv
olist_customers_dataset.csv
olist_sellers_dataset.csv
olist_order_payments_dataset.csv
olist_order_reviews_dataset.csv
olist_geolocation_dataset.csv
product_category_name_translation.csv
```

### SQLite 本地导入

```bash

python -m utils.load_olist_csvs --db-url sqlite:///data/olist.db --data-dir data/raw
```

然后运行：

```bash
DB_URL=sqlite:///data/olist.db streamlit run app.py
```

### MySQL 导入

先创建数据库，例如：

```sql
CREATE DATABASE olist_agentic_bi DEFAULT CHARACTER SET utf8mb4;
```

然后执行：

```bash
pip install pymysql
python utils/load_olist_csvs.py \
  --db-url mysql+pymysql://root:你的密码@localhost:3306/olist_agentic_bi \
  --data-dir data/raw
```

运行系统：

```bash
DB_URL=mysql+pymysql://root:你的密码@localhost:3306/olist_agentic_bi streamlit run app.py
```

---

## 3. 可选 LLM 配置

复制环境变量示例：

```bash
cp .env.example .env
```

然后填写：

```bash
LLM_API_KEY=你的key
LLM_BASE_URL=https://api.deepseek.com/v1
LLM_MODEL=deepseek-chat
```

没有 LLM_API_KEY 也能运行，系统会用规则模板生成分析和建议。

---

## 4. 目录结构

```text
AgenticBI_Final_Olist/
├── agents/                         # 多智能体实现
├── config/                         # 数据字典、Prompt、设置
├── dashboard/                      # Streamlit 页面辅助函数
├── data/
│   ├── raw/                        # 放 Kaggle 原始 CSV
│   └── sample/                     # 小样本说明
├── models/                         # 预测与评论分析模型
├── outputs/charts/                 # 图表输出目录
├── report/report_template.md       # 报告模板
├── utils/                          # 数据导入、数据库、预聚合 SQL
├── app.py                          # Streamlit 入口
├── requirements.txt
└── README.md
```

---

## 5. 预聚合表说明

为了加速 Agent 高频查询，本项目把常用跨表 JOIN 提前算好：

| 名称 | 粒度 | 用途 |
|---|---|---|
| mv_monthly_sales | 年-月 | 月度 GMV、订单数、客单价、运费 |
| mv_state_sales | 年-月-州 | 各州 GMV、订单数、客户数 |
| mv_category_sales | 年-月-品类 | 品类销售趋势 |
| mv_delivery_perf | 年-月-州 | 平均配送时长、准时率、延迟订单 |
| mv_seller_perf | 年-月-卖家 | 卖家 GMV、订单数、平均评分 |
| mv_payment_dist | 年-月-支付方式 | 支付偏好、平均分期数 |

SQL 脚本在：

```text
utils/create_preaggregation_tables_mysql.sql
```

Python 刷新函数在：

```text
utils/preaggregation.py
```

---

## 6. 说明

这是课程作业的基础完整实现，重点保证结构、流程和功能覆盖。真实提交时建议补充：

1. 用真实 Olist 全量 CSV 运行截图。
2. 报告中加入预聚合前后查询耗时对比截图。
3. 根据老师要求替换成自己的 LLM Key 和模型。
4. 对图表样式、报告排版、小组分工进行个性化修改。

---

## 7. 预聚合性能对比截图

运行下面命令，把终端输出截图放进报告：

```bash
python utils/performance_compare.py --db-url sqlite:///data/sample_olist.db
```

如果使用 MySQL：

```bash
python utils/performance_compare.py --db-url mysql+pymysql://root:你的密码@localhost:3306/olist_agentic_bi
```
