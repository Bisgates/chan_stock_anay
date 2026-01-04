#!/usr/bin/env python3
"""
股票综合分析脚本
逆向美股盘中分析报告，基于日级别数据生成类似的技术分析报告
"""

import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# 股票基本信息映射
STOCK_INFO = {
    'TSLA': {'name': '特斯拉', 'sector': '新能源汽车/科技', 'asset_type': '正股'},
    'AAPL': {'name': '苹果', 'sector': '科技/消费电子', 'asset_type': '正股'},
    'SNDK': {'name': '闪迪(已被西数收购)', 'sector': '半导体/存储', 'asset_type': '正股'},
    'RKLB': {'name': 'Rocket Lab', 'sector': '航空航天/商业航天', 'asset_type': '正股'},
}


class StockDataLoader:
    """股票数据加载器"""

    def __init__(self, data_dir: str = '/Volumes/ssd/us_stock_data/1d'):
        self.data_dir = Path(data_dir)

    def get_available_dates(self) -> list:
        """获取所有可用日期"""
        dates = sorted([d.name for d in self.data_dir.iterdir()
                       if d.is_dir() and d.name.isdigit()])
        return dates

    def load_stock_data(self, symbol: str, days: int = 250) -> pd.DataFrame:
        """加载指定股票的历史数据"""
        dates = self.get_available_dates()[-days:]
        records = []

        for date in dates:
            csv_path = self.data_dir / date / f"{date}.csv"
            if csv_path.exists():
                try:
                    df = pd.read_csv(csv_path)
                    stock_data = df[df['symbol'] == symbol]
                    if not stock_data.empty:
                        row = stock_data.iloc[0]
                        records.append({
                            'date': date,
                            'open': row['open'],
                            'high': row['high'],
                            'low': row['low'],
                            'close': row['close'],
                            'volume': row['volume'],
                            'amount': row['amount']
                        })
                except Exception as e:
                    continue

        if not records:
            return pd.DataFrame()

        df = pd.DataFrame(records)
        df['date'] = pd.to_datetime(df['date'], format='%Y%m%d')
        df = df.sort_values('date').reset_index(drop=True)
        return df


class TechnicalIndicators:
    """技术指标计算器"""

    @staticmethod
    def calc_ma(df: pd.DataFrame, periods: list = [5, 10, 20, 60, 120, 250]) -> pd.DataFrame:
        """计算移动平均线"""
        for period in periods:
            if len(df) >= period:
                df[f'MA{period}'] = df['close'].rolling(window=period).mean()
        return df

    @staticmethod
    def calc_macd(df: pd.DataFrame, fast: int = 12, slow: int = 26, signal: int = 9) -> pd.DataFrame:
        """计算MACD指标"""
        exp1 = df['close'].ewm(span=fast, adjust=False).mean()
        exp2 = df['close'].ewm(span=slow, adjust=False).mean()
        df['DIF'] = exp1 - exp2
        df['DEA'] = df['DIF'].ewm(span=signal, adjust=False).mean()
        df['MACD'] = 2 * (df['DIF'] - df['DEA'])
        return df

    @staticmethod
    def calc_kdj(df: pd.DataFrame, n: int = 9, m1: int = 3, m2: int = 3) -> pd.DataFrame:
        """计算KDJ指标"""
        low_min = df['low'].rolling(window=n).min()
        high_max = df['high'].rolling(window=n).max()

        rsv = (df['close'] - low_min) / (high_max - low_min) * 100
        rsv = rsv.fillna(50)

        df['K'] = rsv.ewm(com=m1-1, adjust=False).mean()
        df['D'] = df['K'].ewm(com=m2-1, adjust=False).mean()
        df['J'] = 3 * df['K'] - 2 * df['D']
        return df

    @staticmethod
    def calc_volume_ma(df: pd.DataFrame, periods: list = [5, 10, 20]) -> pd.DataFrame:
        """计算成交量均线"""
        for period in periods:
            if len(df) >= period:
                df[f'VOL_MA{period}'] = df['volume'].rolling(window=period).mean()
        return df

    @staticmethod
    def calc_atr(df: pd.DataFrame, period: int = 14) -> pd.DataFrame:
        """计算ATR指标"""
        high = df['high']
        low = df['low']
        close = df['close'].shift(1)

        tr1 = high - low
        tr2 = abs(high - close)
        tr3 = abs(low - close)

        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        df['ATR'] = tr.rolling(window=period).mean()
        return df


class TrendAnalyzer:
    """趋势分析器"""

    def __init__(self, df: pd.DataFrame):
        self.df = df
        self.latest = df.iloc[-1] if len(df) > 0 else None
        self.prev = df.iloc[-2] if len(df) > 1 else None

    def analyze_ma_alignment(self) -> dict:
        """分析均线排列"""
        if self.latest is None:
            return {'alignment': '数据不足', 'description': ''}

        ma_cols = ['MA5', 'MA10', 'MA20', 'MA60']
        available_mas = [col for col in ma_cols if col in self.df.columns and pd.notna(self.latest[col])]

        if len(available_mas) < 3:
            return {'alignment': '数据不足', 'description': '均线数据不足'}

        ma_values = [self.latest[col] for col in available_mas]

        # 判断多头排列：短期均线在上
        is_bullish = all(ma_values[i] >= ma_values[i+1] for i in range(len(ma_values)-1))
        is_bearish = all(ma_values[i] <= ma_values[i+1] for i in range(len(ma_values)-1))

        # 判断均线斜率
        ma5_slope = (self.latest['MA5'] - self.df.iloc[-5]['MA5']) / 5 if 'MA5' in self.df.columns and len(self.df) >= 5 else 0

        if is_bullish:
            if ma5_slope > 0:
                return {'alignment': '多头排列', 'description': '均线系统呈现典型多头排列且斜率向上'}
            else:
                return {'alignment': '多头排列', 'description': '均线多头排列但斜率趋平'}
        elif is_bearish:
            return {'alignment': '空头排列', 'description': '均线系统呈现空头排列，短期承压'}
        else:
            return {'alignment': '震荡整理', 'description': '均线交织，处于震荡整理阶段'}

    def analyze_trend_structure(self) -> str:
        """分析趋势结构"""
        if len(self.df) < 20:
            return "数据不足以判断趋势结构"

        close = self.latest['close']
        ma_info = self.analyze_ma_alignment()

        # 检查是否突破
        high_20 = self.df['high'].iloc[-20:-1].max()
        low_20 = self.df['low'].iloc[-20:-1].min()

        # 计算涨跌幅
        change_pct = (close - self.prev['close']) / self.prev['close'] * 100 if self.prev is not None else 0

        if close > high_20 and change_pct > 2:
            if ma_info['alignment'] == '多头排列':
                return f"右侧确认突破；股价放量突破前期高位震荡箱体上沿，{ma_info['description']}；大趋势呈现加速上行阶段。"
            else:
                return f"右侧突破尝试；股价尝试突破前期高点{high_20:.2f}，关注突破有效性确认。"
        elif close < low_20 and change_pct < -2:
            return f"右侧破位下行；股价跌破前期低点支撑{low_20:.2f}，空头占优。"
        elif ma_info['alignment'] == '多头排列':
            return f"多头趋势延续；{ma_info['description']}；价格运行于均线系统上方。"
        elif ma_info['alignment'] == '空头排列':
            return f"空头趋势运行；{ma_info['description']}；价格承压于均线系统下方。"
        else:
            return f"横盘震荡整理；{ma_info['description']}；等待方向选择。"

    def analyze_kline(self) -> str:
        """分析关键K线形态"""
        if self.latest is None or self.prev is None:
            return "数据不足"

        open_p = self.latest['open']
        close = self.latest['close']
        high = self.latest['high']
        low = self.latest['low']

        prev_close = self.prev['close']
        prev_open = self.prev['open']
        prev_high = self.prev['high']
        prev_low = self.prev['low']

        body = abs(close - open_p)
        upper_shadow = high - max(open_p, close)
        lower_shadow = min(open_p, close) - low

        change_pct = (close - prev_close) / prev_close * 100

        # 判断K线类型
        is_bullish = close > open_p
        is_big_candle = body / prev_close > 0.02  # 实体大于2%

        # 检查是否跳空
        gap_up = open_p > prev_high
        gap_down = open_p < prev_low

        # 检查是否吞没
        engulfing = is_bullish and open_p < prev_low and close > prev_high

        if gap_up and is_bullish and is_big_candle:
            return f"今日出现跳空放量长阳线，实体饱满且收盘接近全天最高点；形成向上突破缺口，多头动能强劲。"
        elif engulfing:
            return f"今日出现放量反包大阳线，实体完全吞没前日的震荡区间；显示多头动能极强。"
        elif is_bullish and is_big_candle and upper_shadow < body * 0.3:
            return f"今日放量长阳突破；K线涨幅{change_pct:.2f}%，实体饱满无长上影，确立右侧进场点。"
        elif is_bullish and is_big_candle:
            return f"今日阳线上涨{change_pct:.2f}%；实体较大，显示买盘积极。"
        elif not is_bullish and is_big_candle:
            return f"今日阴线下跌{abs(change_pct):.2f}%；卖压较重，需关注支撑位。"
        elif upper_shadow > body * 2:
            return f"今日出现长上影线；上方抛压明显，短期存在回调压力。"
        elif lower_shadow > body * 2:
            return f"今日出现长下影线；下方有买盘支撑，显示抄底资金介入。"
        else:
            return f"今日小幅波动{change_pct:.2f}%；多空暂时平衡，等待方向选择。"

    def analyze_volume(self) -> str:
        """分析成交量"""
        if self.latest is None or 'VOL_MA5' not in self.df.columns:
            return "成交量数据不足"

        vol = self.latest['volume']
        vol_ma5 = self.latest['VOL_MA5']
        vol_ma20 = self.latest.get('VOL_MA20', vol_ma5)

        change_pct = (self.latest['close'] - self.prev['close']) / self.prev['close'] * 100 if self.prev is not None else 0

        vol_ratio = vol / vol_ma5 if vol_ma5 > 0 else 1

        if vol_ratio > 2:
            if change_pct > 0:
                return f"成交量呈现爆量状态，达到5日均量{vol_ratio:.1f}倍；典型的量增价涨，显示主力资金介入极深，突破有效性极高。"
            else:
                return f"放量下跌，成交量达5日均量{vol_ratio:.1f}倍；显示卖压沉重，需警惕进一步回调。"
        elif vol_ratio > 1.3:
            if change_pct > 0:
                return f"成交量显著放大至5日均量{vol_ratio:.1f}倍；呈现量增价涨态势，资金介入明显。"
            else:
                return f"成交量放大但价格下跌；量价背离需关注。"
        elif vol_ratio < 0.7:
            return f"成交量萎缩至5日均量{vol_ratio:.1f}倍；交投清淡，等待量能配合。"
        else:
            return f"成交量接近5日均量水平；量价配合正常。"

    def analyze_macd(self) -> str:
        """分析MACD状态"""
        if self.latest is None or 'DIF' not in self.df.columns:
            return "MACD数据不足"

        dif = self.latest['DIF']
        dea = self.latest['DEA']
        macd = self.latest['MACD']

        prev_dif = self.prev['DIF'] if self.prev is not None and 'DIF' in self.df.columns else dif
        prev_dea = self.prev['DEA'] if self.prev is not None and 'DEA' in self.df.columns else dea

        # 判断金叉死叉
        cross_up = prev_dif < prev_dea and dif > dea
        cross_down = prev_dif > prev_dea and dif < dea

        if dif > 0 and dea > 0:
            if cross_up or (dif > dea and macd > 0):
                if macd > self.df['MACD'].iloc[-5:-1].max():
                    return f"零轴上方形成强力金叉；DIF与DEA开口扩大；红色动能柱显著拉长，多头动能处于爆发期。"
                else:
                    return f"零轴上方金叉；DIFF与DEA在零轴上方交叉向上，红柱动能放大，属于强势区域信号。"
            elif dif < dea:
                return f"零轴上方但DIF下穿DEA；短期动能减弱，关注回调支撑。"
        elif dif < 0 and dea < 0:
            if cross_up:
                return f"零轴下方金叉；处于弱势区域的反弹信号，需观察能否突破零轴。"
            elif dif < dea:
                return f"零轴下方死叉运行；绿柱动能持续，空头趋势未改。"

        if dif > dea:
            return f"DIF位于DEA上方，多头动能占优；MACD红柱{'放大' if macd > 0 else '收窄'}。"
        else:
            return f"DIF位于DEA下方，短期承压；MACD绿柱{'放大' if macd < 0 else '收窄'}。"

    def analyze_kdj(self) -> str:
        """分析KDJ状态"""
        if self.latest is None or 'K' not in self.df.columns:
            return "KDJ数据不足"

        k = self.latest['K']
        d = self.latest['D']
        j = self.latest['J']

        prev_k = self.prev['K'] if self.prev is not None and 'K' in self.df.columns else k
        prev_d = self.prev['D'] if self.prev is not None and 'D' in self.df.columns else d

        # 判断金叉死叉
        cross_up = prev_k < prev_d and k > d
        cross_down = prev_k > prev_d and k < d

        if k > 80 and d > 80:
            if cross_down:
                return f"K、D、J三线进入超买区后死叉；高位钝化后出现拐头，需警惕回调风险。"
            else:
                return f"K、D、J三线进入超买区并向上钝化；强趋势下超买不代表反转，而是趋势极强的表现。"
        elif k < 20 and d < 20:
            if cross_up:
                return f"指标在超卖区形成金叉；底部反转信号出现，关注反弹机会。"
            else:
                return f"指标处于超卖区；空头动能衰竭，但仍需等待企稳信号。"
        elif k > 50:
            if cross_up or k > d:
                return f"指标金叉向上穿越50轴，处于强势进攻区，尚未触及超买区，短期仍有上行空间。"
            else:
                return f"指标位于50轴上方但动能减弱；关注能否继续维持强势。"
        else:
            if cross_up:
                return f"指标在50轴下方形成金叉；反弹动能启动，关注能否突破50轴。"
            else:
                return f"指标位于50轴下方；弱势震荡，等待企稳。"


class PriceCalculator:
    """价格计算器"""

    def __init__(self, df: pd.DataFrame):
        self.df = df
        self.latest = df.iloc[-1] if len(df) > 0 else None

    def calc_add_position_price(self) -> tuple:
        """计算加仓价格和逻辑"""
        if self.latest is None:
            return 0, "数据不足"

        close = self.latest['close']
        high = self.latest['high']

        # 前期高点
        high_20 = self.df['high'].iloc[-20:-1].max() if len(self.df) > 20 else high

        # 加仓价格：突破前高或当前高点上方
        add_price = max(high_20, high) * 1.005  # 前高上方0.5%

        if add_price > close:
            logic = f"若股价放量突破前期高点{high_20:.2f}并站稳，则确认开启新一轮主升浪，可于回踩确认时加仓。"
        else:
            logic = f"若股价回踩{close * 0.97:.2f}附近获得支撑后反弹，可考虑加仓。"
            add_price = close * 0.97

        return round(add_price, 2), logic

    def calc_reduce_position_price(self) -> tuple:
        """计算减仓价格和逻辑"""
        if self.latest is None:
            return 0, "数据不足"

        close = self.latest['close']
        atr = self.latest.get('ATR', close * 0.03)

        # 减仓价格：当前价格上方1.5-2个ATR
        reduce_price = close + atr * 1.5

        # 取整数或心理价位
        reduce_price = round(reduce_price, 0) if reduce_price > 10 else round(reduce_price, 2)

        logic = f"接近上方心理压力位或出现K线高位放量长上影线时，可先行兑现部分利润以控制乖离率风险。"

        return reduce_price, logic

    def calc_stop_loss_price(self) -> tuple:
        """计算止损价格和逻辑"""
        if self.latest is None or len(self.df) < 2:
            return 0, "数据不足"

        prev = self.df.iloc[-2]
        close = self.latest['close']
        low = self.latest['low']

        # 止损价格：前日收盘价下方或今日低点下方
        stop_price = min(prev['close'] * 0.97, low * 0.99)

        # 确保止损不会太远
        max_loss = close * 0.08  # 最大8%止损
        if close - stop_price > max_loss:
            stop_price = close - max_loss

        logic = f"跌破今日低点{low:.2f}及前日收盘{prev['close']:.2f}，若放量跌破{stop_price:.2f}则视为突破失败，需无条件止损。"

        return round(stop_price, 2), logic


class StockAnalyzer:
    """股票综合分析器"""

    def __init__(self, data_dir: str = '/Volumes/ssd/us_stock_data/1d'):
        self.loader = StockDataLoader(data_dir)

    def analyze(self, symbol: str) -> dict:
        """分析单只股票"""
        # 加载数据
        df = self.loader.load_stock_data(symbol, days=250)

        if df.empty or len(df) < 30:
            return {
                '股票代码': symbol,
                '股票名称': STOCK_INFO.get(symbol, {}).get('name', symbol),
                '市场': '美股',
                'error': f'数据不足，仅有{len(df)}条记录',
                'status': 'error'
            }

        # 计算技术指标
        df = TechnicalIndicators.calc_ma(df)
        df = TechnicalIndicators.calc_macd(df)
        df = TechnicalIndicators.calc_kdj(df)
        df = TechnicalIndicators.calc_volume_ma(df)
        df = TechnicalIndicators.calc_atr(df)

        # 趋势分析
        trend_analyzer = TrendAnalyzer(df)
        price_calc = PriceCalculator(df)

        latest = df.iloc[-1]
        prev = df.iloc[-2]

        # 计算涨跌幅
        change_pct = (latest['close'] - prev['close']) / prev['close'] * 100

        # 判断交易动作和评分
        ma_info = trend_analyzer.analyze_ma_alignment()
        is_bullish = ma_info['alignment'] == '多头排列'
        is_breakout = latest['close'] > df['high'].iloc[-20:-1].max()
        vol_ratio = latest['volume'] / latest['VOL_MA5'] if 'VOL_MA5' in df.columns else 1

        if is_bullish and is_breakout and vol_ratio > 1.3 and change_pct > 2:
            score = "5-强烈可交易"
            action = "多头"
        elif is_bullish and change_pct > 0:
            score = "4"
            action = "多头"
        elif is_bullish:
            score = "3"
            action = "观望"
        elif ma_info['alignment'] == '空头排列':
            score = "2"
            action = "空头"
        else:
            score = "3"
            action = "观望"

        # 计算价格
        add_price, add_logic = price_calc.calc_add_position_price()
        reduce_price, reduce_logic = price_calc.calc_reduce_position_price()
        stop_price, stop_logic = price_calc.calc_stop_loss_price()

        # 生成分析原因
        reasons = []
        if is_breakout:
            reasons.append("股价突破前期高点")
        if vol_ratio > 1.5:
            reasons.append("成交量显著放大")
        if is_bullish:
            reasons.append("均线多头排列")
        if latest['DIF'] > latest['DEA'] and latest['DIF'] > 0:
            reasons.append("MACD零轴上方金叉")
        if latest['K'] > 50:
            reasons.append("KDJ指标强势")

        reason = "；".join(reasons) if reasons else "技术面中性，等待方向选择"

        # 获取股票信息
        stock_info = STOCK_INFO.get(symbol, {})

        return {
            '股票代码': symbol,
            '股票名称': stock_info.get('name', symbol),
            '市场': '美股',
            '评分': score,
            '交易动作': action,
            '趋势结构': trend_analyzer.analyze_trend_structure(),
            '关键K线': trend_analyzer.analyze_kline(),
            '成交量分析': trend_analyzer.analyze_volume(),
            'MACD状态': trend_analyzer.analyze_macd(),
            'KDJ状态': trend_analyzer.analyze_kdj(),
            '加仓价格': add_price,
            '加仓逻辑': add_logic,
            '减仓价格': reduce_price,
            '减仓逻辑': reduce_logic,
            '止损价格': stop_price,
            '止损逻辑': stop_logic,
            '原因': reason,
            '资产类型': stock_info.get('asset_type', '正股'),
            '所属板块': stock_info.get('sector', '未分类'),
            '当前股价': round(latest['close'], 2),
            '涨跌幅': f"{'+' if change_pct > 0 else ''}{change_pct:.2f}%",
            'raw_response': '',
            'image_path': '',
            'image_name': '',
            'analysis_time': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'error': '',
            'status': 'success'
        }

    def analyze_batch(self, symbols: list) -> pd.DataFrame:
        """批量分析股票"""
        results = []
        for symbol in symbols:
            print(f"分析 {symbol}...")
            result = self.analyze(symbol)
            results.append(result)

        return pd.DataFrame(results)

    def save_report(self, df: pd.DataFrame, output_path: str):
        """保存分析报告"""
        # 确保列顺序与原报告一致
        columns = [
            '股票代码', '股票名称', '市场', '评分', '交易动作', '趋势结构', '关键K线',
            '成交量分析', 'MACD状态', 'KDJ状态', '加仓价格', '加仓逻辑', '减仓价格',
            '减仓逻辑', '止损价格', '止损逻辑', '原因', '资产类型', '所属板块',
            '当前股价', '涨跌幅', 'raw_response', 'image_path', 'image_name',
            'analysis_time', 'error', 'status'
        ]

        # 只保留存在的列
        existing_cols = [col for col in columns if col in df.columns]
        df = df[existing_cols]

        df.to_csv(output_path, index=False, encoding='utf-8-sig')
        print(f"报告已保存至: {output_path}")

    def get_chart_data(self, symbol: str, days: int = 90) -> dict:
        """获取图表数据，用于Web展示"""
        df = self.loader.load_stock_data(symbol, days=days + 60)  # 额外加载数据计算指标

        if df.empty or len(df) < 30:
            return {'error': f'数据不足，仅有{len(df)}条记录'}

        # 计算技术指标
        df = TechnicalIndicators.calc_ma(df)
        df = TechnicalIndicators.calc_macd(df)
        df = TechnicalIndicators.calc_kdj(df)
        df = TechnicalIndicators.calc_volume_ma(df)

        # 只返回最近N天的数据
        df = df.tail(days).reset_index(drop=True)

        # 转换为图表格式
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

            # 成交量颜色
            is_up = row['close'] >= row['open']
            volume_data.append({
                'time': time_str,
                'value': int(row['volume']),
                'color': 'rgba(38, 166, 154, 0.6)' if is_up else 'rgba(239, 83, 80, 0.6)'
            })

            # 均线数据
            for period in [5, 10, 20, 60]:
                col = f'MA{period}'
                if col in df.columns and pd.notna(row[col]):
                    ma_data[col].append({
                        'time': time_str,
                        'value': round(row[col], 2)
                    })

            # MACD数据
            if 'DIF' in df.columns and pd.notna(row['DIF']):
                macd_data['DIF'].append({'time': time_str, 'value': round(row['DIF'], 4)})
                macd_data['DEA'].append({'time': time_str, 'value': round(row['DEA'], 4)})
                macd_data['MACD'].append({
                    'time': time_str,
                    'value': round(row['MACD'], 4),
                    'color': 'rgba(38, 166, 154, 0.8)' if row['MACD'] >= 0 else 'rgba(239, 83, 80, 0.8)'
                })

            # KDJ数据
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


def main():
    """主函数"""
    # 待分析的股票列表
    symbols = ['TSLA', 'AAPL', 'SNDK', 'RKLB']

    # 创建分析器
    analyzer = StockAnalyzer()

    # 批量分析
    results_df = analyzer.analyze_batch(symbols)

    # 保存报告
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_path = f'/Users/han/project/chan_stock_anay/stock_analysis_results_{timestamp}.csv'
    analyzer.save_report(results_df, output_path)

    # 打印结果预览
    print("\n分析结果预览:")
    print(results_df[['股票代码', '股票名称', '评分', '交易动作', '当前股价', '涨跌幅']].to_string())

    return output_path


if __name__ == '__main__':
    main()
