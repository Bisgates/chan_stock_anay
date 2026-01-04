#!/usr/bin/env python3
"""
股票分析Web应用 - FastAPI后端
专业交易员级别的K线图和分析报告展示
性能优化版本 - 添加缓存
"""

from fastapi import FastAPI, HTTPException, Request
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Dict, Any, Optional
from functools import lru_cache
import uvicorn
from pathlib import Path
import pandas as pd
from datetime import datetime
import time
import re
from collections import defaultdict
import json

from stock_analyzer import StockAnalyzer, StockDataLoader, STOCK_INFO, TechnicalIndicators

app = FastAPI(title="洛阳铲个股分析", version="1.0.0")

# ========== 访问统计 ==========
class VisitorStats:
    def __init__(self):
        self.stats_file = Path(__file__).parent / "visitor_stats.json"

    def _load(self):
        if self.stats_file.exists():
            try:
                return json.loads(self.stats_file.read_text())
            except:
                pass
        return {'unique_ips': [], 'total_visits': 0, 'page_views': {}}

    def _save(self, data):
        data['last_updated'] = datetime.now().isoformat()
        self.stats_file.write_text(json.dumps(data, ensure_ascii=False, indent=2))

    def record_visit(self, ip: str, path: str):
        try:
            data = self._load()
            data['total_visits'] = data.get('total_visits', 0) + 1
            unique_ips = set(data.get('unique_ips', []))
            unique_ips.add(ip)
            data['unique_ips'] = list(unique_ips)
            page_views = data.get('page_views', {})
            page_views[path] = page_views.get(path, 0) + 1
            data['page_views'] = page_views
            self._save(data)
        except Exception as e:
            print(f"Stats error: {e}")

    def get_stats(self):
        data = self._load()
        return {
            'unique_visitors': len(data.get('unique_ips', [])),
            'total_visits': data.get('total_visits', 0),
            'top_pages': dict(sorted(data.get('page_views', {}).items(), key=lambda x: -x[1])[:10])
        }

visitor_stats = VisitorStats()

# CORS配置
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 访问统计中间件
@app.middleware("http")
async def track_visits(request: Request, call_next):
    response = await call_next(request)

    # 获取真实IP（通过 Cloudflare 等代理）
    ip = request.headers.get("CF-Connecting-IP") or \
         request.headers.get("X-Forwarded-For", "").split(",")[0].strip() or \
         (request.client.host if request.client else "unknown")
    path = request.url.path

    # 只统计主页访问
    if path == "/" and response.status_code == 200:
        visitor_stats.record_visit(ip, path)

    return response

# 目录配置
BASE_DIR = Path(__file__).parent
REPORTS_DIR = BASE_DIR / "reports"
REPORTS_DIR.mkdir(exist_ok=True)

# 静态文件
static_dir = BASE_DIR / "static"
static_dir.mkdir(exist_ok=True)
app.mount("/static", StaticFiles(directory=str(static_dir)), name="static")

# 全局分析器实例
analyzer = StockAnalyzer()
loader = StockDataLoader()


# ========== 缓存层 ==========
class DataCache:
    """数据缓存管理器"""
    def __init__(self):
        self.stock_data: Dict[str, pd.DataFrame] = {}
        self.chart_cache: Dict[str, Dict] = {}
        self.report_cache: Dict[str, Dict] = {}
        self.csv_cache: Dict[str, List[Dict]] = {}  # csv文件缓存
        self.no_data_symbols: set = set()  # 无数据的股票缓存
        self.last_update: float = 0
        self.cache_ttl = 1800  # 缓存30分钟

    def is_stale(self) -> bool:
        return time.time() - self.last_update > self.cache_ttl

    def get_stock_df(self, symbol: str) -> pd.DataFrame:
        if symbol not in self.stock_data or self.is_stale():
            df = loader.load_stock_data(symbol, days=250)
            if not df.empty:
                df = TechnicalIndicators.calc_ma(df)
                df = TechnicalIndicators.calc_macd(df)
                df = TechnicalIndicators.calc_kdj(df)
                df = TechnicalIndicators.calc_volume_ma(df)
                df = TechnicalIndicators.calc_atr(df)
                self.stock_data[symbol] = df
                self.last_update = time.time()
        return self.stock_data.get(symbol, pd.DataFrame())

    def get_chart_data(self, symbol: str, days: int) -> Dict:
        # 快速返回已知无数据的股票
        if symbol in self.no_data_symbols:
            return {'error': f'无数据'}

        cache_key = f"{symbol}_{days}"
        if cache_key not in self.chart_cache or self.is_stale():
            df = self.get_stock_df(symbol)
            if df.empty or len(df) < 30:
                self.no_data_symbols.add(symbol)
                return {'error': f'数据不足'}
            df_slice = df.tail(days).reset_index(drop=True)
            chart_data = self._build_chart_data(symbol, df_slice)
            self.chart_cache[cache_key] = chart_data
        return self.chart_cache.get(cache_key, {})

    def get_report(self, symbol: str) -> Dict:
        # 快速返回已知无数据的股票
        if symbol in self.no_data_symbols:
            return {'error': '无数据', '股票代码': symbol}

        if symbol not in self.report_cache or self.is_stale():
            report = analyzer.analyze(symbol)
            if report.get('status') == 'error':
                self.no_data_symbols.add(symbol)
            self.report_cache[symbol] = report
        return self.report_cache.get(symbol, {})

    def get_csv_stocks(self, csv_name: str) -> List[Dict]:
        """从CSV文件加载股票列表"""
        if csv_name not in self.csv_cache:
            csv_path = REPORTS_DIR / csv_name
            if not csv_path.exists():
                return []
            try:
                df = pd.read_csv(csv_path, encoding='utf-8-sig')
                stocks = []
                for _, row in df.iterrows():
                    # 提取评分数字
                    rating_str = str(row.get('评分', '3'))
                    rating_match = re.search(r'(\d)', rating_str)
                    rating = int(rating_match.group(1)) if rating_match else 3

                    # 处理价格，确保是有效数字
                    price = row.get('当前股价', 0)
                    if pd.isna(price) or price == float('inf') or price == float('-inf'):
                        price = 0
                    else:
                        price = round(float(price), 2)

                    # 处理涨跌幅
                    change = row.get('涨跌幅', '0%')
                    if pd.isna(change):
                        change = '0%'
                    else:
                        change = str(change)

                    # 安全获取字符串字段
                    def safe_str(val, default=''):
                        if pd.isna(val):
                            return default
                        return str(val)

                    stocks.append({
                        'symbol': safe_str(row.get('股票代码', '')),
                        'name': safe_str(row.get('股票名称', '')),
                        'rating': rating,
                        'rating_text': rating_str,
                        'action': safe_str(row.get('交易动作', '观望')),
                        'price': price,
                        'change': change,
                        'sector': safe_str(row.get('所属板块', '')),
                        # 完整报告字段
                        '趋势结构': safe_str(row.get('趋势结构', '')),
                        '关键K线': safe_str(row.get('关键K线', '')),
                        '成交量分析': safe_str(row.get('成交量分析', '')),
                        'MACD状态': safe_str(row.get('MACD状态', '')),
                        'KDJ状态': safe_str(row.get('KDJ状态', '')),
                        '加仓价格': safe_str(row.get('加仓价格', '')),
                        '减仓价格': safe_str(row.get('减仓价格', '')),
                        '止损价格': safe_str(row.get('止损价格', '')),
                        '原因': safe_str(row.get('原因', '')),
                    })
                self.csv_cache[csv_name] = stocks
            except Exception as e:
                print(f"Error loading CSV {csv_name}: {e}")
                return []
        return self.csv_cache.get(csv_name, [])

    def clear_csv_cache(self):
        """清除CSV缓存"""
        self.csv_cache.clear()

    def _build_chart_data(self, symbol: str, df: pd.DataFrame) -> Dict:
        candlestick_data = []
        volume_data = []
        ma_data = {f'MA{p}': [] for p in [5, 10, 20, 60]}
        macd_data = {'DIF': [], 'DEA': [], 'MACD': []}
        kdj_data = {'K': [], 'D': [], 'J': []}

        for _, row in df.iterrows():
            time_str = row['date'].strftime('%Y-%m-%d')
            candlestick_data.append({
                'time': time_str,
                'open': round(row['open'], 2),
                'high': round(row['high'], 2),
                'low': round(row['low'], 2),
                'close': round(row['close'], 2)
            })
            is_up = row['close'] >= row['open']
            volume_data.append({
                'time': time_str,
                'value': int(row['volume']),
                'color': 'rgba(38, 166, 154, 0.6)' if is_up else 'rgba(239, 83, 80, 0.6)'
            })
            for period in [5, 10, 20, 60]:
                col = f'MA{period}'
                if col in df.columns and pd.notna(row[col]):
                    ma_data[col].append({'time': time_str, 'value': round(row[col], 2)})
            if 'DIF' in df.columns and pd.notna(row['DIF']):
                macd_data['DIF'].append({'time': time_str, 'value': round(row['DIF'], 4)})
                macd_data['DEA'].append({'time': time_str, 'value': round(row['DEA'], 4)})
                macd_data['MACD'].append({
                    'time': time_str,
                    'value': round(row['MACD'], 4),
                    'color': 'rgba(38, 166, 154, 0.8)' if row['MACD'] >= 0 else 'rgba(239, 83, 80, 0.8)'
                })
            if 'K' in df.columns and pd.notna(row['K']):
                kdj_data['K'].append({'time': time_str, 'value': round(row['K'], 2)})
                kdj_data['D'].append({'time': time_str, 'value': round(row['D'], 2)})
                kdj_data['J'].append({'time': time_str, 'value': round(row['J'], 2)})

        return {
            'symbol': symbol,
            'name': STOCK_INFO.get(symbol, {}).get('name', symbol),
            'candlestick': candlestick_data,
            'volume': volume_data,
            'ma': ma_data,
            'macd': macd_data,
            'kdj': kdj_data
        }

    def preload(self, symbols: List[str]):
        for symbol in symbols:
            self.get_stock_df(symbol)
            self.get_chart_data(symbol, 90)
            self.get_report(symbol)


cache = DataCache()


class AnalyzeRequest(BaseModel):
    symbols: List[str]


@app.on_event("startup")
async def startup_event():
    print("洛阳铲个股分析系统启动...")
    # 预加载所有CSV报告中的股票数据
    import threading
    def preload_all():
        try:
            for csv_file in REPORTS_DIR.glob("*.csv"):
                stocks = cache.get_csv_stocks(csv_file.name)
                symbols = [s['symbol'] for s in stocks if s.get('symbol')]
                print(f"预加载 {csv_file.name}: {len(symbols)} 只股票")
                for symbol in symbols:
                    try:
                        cache.get_chart_data(symbol, 90)
                    except Exception as e:
                        pass
            print("预加载完成!")
        except Exception as e:
            print(f"预加载错误: {e}")
    # 后台线程预加载，不阻塞启动
    threading.Thread(target=preload_all, daemon=True).start()


@app.get("/", response_class=HTMLResponse)
async def index():
    html_path = BASE_DIR / "templates" / "index.html"
    if html_path.exists():
        return html_path.read_text(encoding='utf-8')
    return HTMLResponse("<h1>Stock Analyzer</h1><p>Template not found</p>")


# ========== CSV报告API ==========
@app.get("/api/reports")
async def list_reports():
    """获取所有可用的CSV报告文件"""
    reports = []
    for csv_file in sorted(REPORTS_DIR.glob("*.csv"), reverse=True):
        # 解析文件名提取时间
        name = csv_file.stem
        # 尝试从文件名提取日期时间
        date_match = re.search(r'(\d{8})_(\d{6})', name)
        if date_match:
            date_str = date_match.group(1)
            time_str = date_match.group(2)
            display_time = f"{date_str[:4]}-{date_str[4:6]}-{date_str[6:8]} {time_str[:2]}:{time_str[2:4]}"
        else:
            display_time = ""

        # 获取文件内股票数量
        stocks = cache.get_csv_stocks(csv_file.name)

        reports.append({
            'filename': csv_file.name,
            'name': name,
            'display_time': display_time,
            'stock_count': len(stocks)
        })

    return {"reports": reports}


@app.get("/api/reports/{filename}/stocks")
async def get_report_stocks(filename: str):
    """获取指定报告中的股票列表（按评分分组）"""
    stocks = cache.get_csv_stocks(filename)
    if not stocks:
        raise HTTPException(status_code=404, detail="Report not found")

    # 按评分分组
    grouped = {5: [], 4: [], 3: [], 2: [], 1: []}
    for stock in stocks:
        rating = stock.get('rating', 3)
        if rating in grouped:
            grouped[rating].append(stock)

    return {
        "filename": filename,
        "total": len(stocks),
        "grouped": grouped
    }


# ========== 股票数据API ==========
@app.get("/api/symbols")
async def get_symbols():
    dates = loader.get_available_dates()
    if not dates:
        return {"symbols": [], "error": "No data available"}

    latest_date = dates[-1]
    csv_path = Path(loader.data_dir) / latest_date / f"{latest_date}.csv"

    try:
        df = pd.read_csv(csv_path)
        symbols = sorted(df['symbol'].unique().tolist())
        symbol_info = []
        for sym in symbols:
            info = STOCK_INFO.get(sym, {})
            symbol_info.append({
                'symbol': sym,
                'name': info.get('name', sym),
                'sector': info.get('sector', '')
            })
        return {"symbols": symbol_info, "count": len(symbols), "date": latest_date}
    except Exception as e:
        return {"symbols": [], "error": str(e)}


@app.get("/api/stock/{symbol}")
async def get_stock_data(symbol: str, days: int = 90):
    symbol = symbol.upper()
    chart_data = cache.get_chart_data(symbol, days)
    if 'error' in chart_data:
        # 返回带错误信息的响应，而不是抛出异常
        return {
            "chart": {"error": chart_data['error'], "symbol": symbol, "candlestick": []},
            "report": {"error": chart_data['error'], "股票代码": symbol}
        }
    report = cache.get_report(symbol)
    return {"chart": chart_data, "report": report}


@app.post("/api/analyze")
async def analyze_stocks(request: AnalyzeRequest):
    results = []
    for symbol in request.symbols:
        try:
            report = cache.get_report(symbol.upper())
            results.append(report)
        except Exception as e:
            results.append({'股票代码': symbol, 'error': str(e), 'status': 'error'})
    return {"results": results}


@app.get("/api/search/{query}")
async def search_symbols(query: str):
    query = query.upper()
    dates = loader.get_available_dates()
    if not dates:
        return {"results": []}

    latest_date = dates[-1]
    csv_path = Path(loader.data_dir) / latest_date / f"{latest_date}.csv"

    try:
        df = pd.read_csv(csv_path)
        symbols = df['symbol'].unique()
        matches = [s for s in symbols if query in s][:20]
        results = []
        for sym in matches:
            info = STOCK_INFO.get(sym, {})
            results.append({
                'symbol': sym,
                'name': info.get('name', sym),
                'sector': info.get('sector', '')
            })
        return {"results": results}
    except Exception as e:
        return {"results": [], "error": str(e)}


@app.get("/api/preload")
async def preload_symbols(symbols: str):
    symbol_list = [s.strip().upper() for s in symbols.split(',')]
    cache.preload(symbol_list)
    return {"status": "ok", "preloaded": symbol_list}


@app.get("/api/stats")
async def get_visitor_stats():
    """获取访问统计"""
    return visitor_stats.get_stats()


if __name__ == "__main__":
    import multiprocessing
    workers = min(multiprocessing.cpu_count(), 4)  # 最多4个worker
    uvicorn.run("app:app", host="0.0.0.0", port=8888, workers=workers)
