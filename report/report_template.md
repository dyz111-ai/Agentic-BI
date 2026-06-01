# Agentic BI 驱动的多表电商运营分析与决策智能系统报告模板

## 1. 项目背景与目标

本项目面向 Olist 巴西电商公开数据集，构建 Agentic BI 系统，使非技术业务人员能够通过自然语言提问，自动获得跨表查询结果、可视化图表、预测结果和运营决策建议。

## 2. 系统架构设计

建议放架构图：

```text
用户问题
  ↓
协调器 Agent
  ├── 数据分析 Agent：自然语言 → SQL，优先查询 mv_* 预聚合表
  ├── 可视化 Agent：自动选择图表并展示
  ├── NLP 评论洞察 Agent：差评情感分析、关键词提取
  ├── 预测 Agent：未来 6 周 GMV 预测
  └── 决策智能 Agent：综合分析并生成运营建议
  ↓
Streamlit Web Dashboard
```

## 3. 数据集说明与预处理

- 数据集：Brazilian E-Commerce Public Dataset by Olist
- 表数量：9 张业务表
- 核心实体：订单、商品、顾客、卖家、支付、评论、地理位置、品类翻译
- 预处理步骤：
  1. 读取 CSV；
  2. 处理时间字段；
  3. 导入 MySQL；
  4. 创建预聚合表；
  5. Agent 查询时优先使用 mv_* 表。

## 4. 预聚合视图设计

### 4.1 预聚合表列表

| 表名 | 粒度 | 字段 | 用途 |
|---|---|---|---|
| mv_monthly_sales | 年-月 | year_month, total_gmv, total_orders, avg_basket, total_freight | 月度销售趋势、预测 |
| mv_state_sales | 年-月-州 | year_month, customer_state, total_gmv, total_orders, unique_customers | 区域销售分析 |
| mv_category_sales | 年-月-品类 | year_month, product_category_english, total_gmv, total_orders, avg_price | 品类表现分析 |
| mv_delivery_perf | 年-月-州 | avg_delivery_days, on_time_rate, delayed_orders | 配送诊断 |
| mv_seller_perf | 年-月-卖家 | seller_id, seller_state, total_gmv, avg_review_score | 卖家绩效 |
| mv_payment_dist | 年-月-支付方式 | payment_type, total_transactions, avg_installments, total_value | 支付偏好 |

### 4.2 SQL 定义

SQL 文件位置：

```text
utils/create_preaggregation_tables_mysql.sql
```

### 4.3 性能对比

请放截图：

- 查询 1：不使用预聚合表，实时 JOIN orders + order_items + payments + customers。
- 查询 2：使用 mv_state_sales。
- 对比查询耗时，证明预聚合表明显更快。

## 5. 多智能体设计

| Agent | 职责 |
|---|---|
| 协调器 Agent | 解析问题，调度其他 Agent |
| 数据分析 Agent | 生成 SQL，优先命中预聚合表，必要时回退基础表 |
| 可视化 Agent | 自动生成折线图、柱状图、热力图、散点图、地图、关键词图 |
| NLP 评论洞察 Agent | 评论情感分析、差评关键词、差评品类 |
| 预测 Agent | 未来 6 周销售额预测 |
| 决策智能 Agent | 输出具体运营建议 |

## 6. 分析任务实现

### 6.1 描述性分析

示例：2017 年 GMV、各州销售额、支付方式分布。

### 6.2 诊断性分析

示例：配送延迟州、低评分卖家、差评品类。

### 6.3 预测性分析

基于 mv_monthly_sales 的历史 GMV 生成未来 6 周预测。

### 6.4 规范性分析

综合销售、配送、支付、评论、预测结果，给出三个月运营改进建议。

## 7. 可视化结果

至少放 6 种图：

1. 月度 GMV 折线图 + 预测区间；
2. 巴西州销售气泡图；
3. Top 品类柱状图；
4. 支付方式热力图；
5. 商品重量 vs 运费散点图；
6. 差评关键词/词云。

## 8. 技术挑战与解决方案

- 多表 JOIN 慢：使用预聚合表。
- 自然语言问题类型多：使用协调器 Agent 先做任务分类。
- 评论文本为葡萄牙语：使用评分 + 关键词情感词典作为轻量兜底。
- LLM Key 不稳定：系统支持规则模板兜底。

## 9. 小组分工

| 成员 | 工作 | 比例 |
|---|---|---|
| A | 数据库、预聚合视图 | 25% |
| B | Agent 实现 | 25% |
| C | 可视化与 Streamlit | 25% |
| D | 报告与演示 | 25% |

## 10. 总结

本项目完成了基于 Olist 多表数据的 Agentic BI 分析系统，实现自然语言查询、预聚合加速、多 Agent 协作、预测分析、评论洞察和运营决策建议。
