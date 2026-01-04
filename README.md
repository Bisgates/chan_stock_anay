# 洛阳铲个股分析

专业交易员级别的股票技术分析系统，提供 K 线图、技术指标和分析报告展示。

## 功能特点

- **K 线图表**：基于 TradingView Lightweight Charts，支持缩放、拖拽
- **技术指标**：MA (5/10/20/60)、MACD、KDJ、成交量
- **关键价格线**：加仓/减仓/止损价格可视化显示
- **分析报告**：趋势结构、关键 K 线、成交量分析等
- **评分系统**：5 星评分分组展示
- **响应式设计**：支持桌面和移动端
- **高性能**：多 Worker 进程、数据预加载、30 分钟缓存

## 项目结构

```
chan_stock_anay/
├── app.py              # FastAPI 后端服务
├── stock_analyzer.py   # 技术分析引擎
├── templates/
│   └── index.html      # 前端页面
├── static/             # 静态文件
├── reports/            # CSV 报告文件目录（需手动添加）
└── README.md
```

## 安装依赖

```bash
pip install fastapi uvicorn pandas
```

## CSV 报告文件

将分析报告 CSV 文件放入 `reports/` 目录。

CSV 文件格式要求（列名）：
- `股票代码` - 股票 Symbol
- `股票名称` - 股票名称
- `评分` - 评分（如 "5.0"、"4 星"）
- `交易动作` - 多头/空头/观望
- `当前股价` - 当前价格
- `涨跌幅` - 涨跌幅（如 "+5.23%"）
- `加仓价格` - 加仓价位
- `减仓价格` - 减仓价位
- `止损价格` - 止损价位
- `趋势结构` - 趋势分析描述
- `关键K线` - K 线形态分析
- `成交量分析` - 成交量分析
- `MACD状态` - MACD 指标状态
- `KDJ状态` - KDJ 指标状态
- `原因` - 综合分析原因

## 启动服务

### 本地运行

```bash
python app.py
```

服务默认运行在 http://localhost:8888

### 外网访问（通过 Cloudflare Tunnel）

```bash
# 安装 cloudflared
brew install cloudflared

# 启动隧道
cloudflared tunnel --url http://localhost:8888
```

## K 线数据

系统需要股票历史数据来绘制 K 线图。数据目录配置在 `stock_analyzer.py` 中的 `StockDataLoader` 类：

```python
self.data_dir = "/Volumes/ssd/us_stock_data/1d"
```

数据格式：每日 CSV 文件，包含 `date`, `open`, `high`, `low`, `close`, `volume`, `symbol` 列。

如果没有 K 线数据，系统会显示 "无数据"，但仍可查看 CSV 中的分析报告。

## 使用说明

1. 打开页面后，左侧选择 CSV 报告文件
2. 股票按评分分组展示（5 星/4 星/...）
3. 点击股票查看 K 线图和分析报告
4. 工具栏可切换时间周期（90D/60D/30D）和指标显示
5. "关键价格" 按钮可在图表上显示加仓/减仓/止损价格线

## 移动端

- 默认显示分析报告
- 底部导航切换：图表 / 报告 / 列表
- 点击列表中的股票自动跳转到报告页

## 技术栈

- **后端**：FastAPI + Uvicorn（多 Worker）
- **前端**：原生 HTML/CSS/JS
- **图表**：TradingView Lightweight Charts
- **数据**：Pandas

## License

MIT
