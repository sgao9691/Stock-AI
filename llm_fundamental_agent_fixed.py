# =====================
# 0. 标准库 & 第三方库导入
# =====================
import os
import sys
import json
import argparse
import logging
import hashlib
from datetime import datetime, timedelta
from typing import Dict, List, Optional
from pathlib import Path
import functools

# 加载环境变量
from dotenv import load_dotenv
load_dotenv()

import numpy as np
import pandas as pd

# 核心依赖
import yfinance as yf
import vectorbt as vbt

# LangChain
try:
    from langchain_openai import ChatOpenAI
    from langchain.schema import HumanMessage
except ImportError:
    print("请安装最新版本的 langchain 和 langchain-openai:")
    print("pip install langchain langchain-openai")
    sys.exit(1)

# =====================
# 1. 日志配置
# =====================
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('llm_agent_simplified.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# =====================
# 2. 全局配置
# =====================
BASE_DIR = Path(__file__).parent.absolute()
DATA_DIR = BASE_DIR / 'data'
PRICE_DIR = DATA_DIR / 'prices'
SIGNAL_DIR = DATA_DIR / 'signals'
BT_DIR = DATA_DIR / 'backtest'

# 环境变量
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY', '')
MODEL_NAME = os.getenv('MODEL_NAME', 'gpt-4o-mini')
START_DATE = os.getenv('START_DATE', '2020-01-01')
END_DATE = os.getenv('END_DATE', '2025-01-01')
FEE_BPS = float(os.getenv('FEE_BPS', '10'))

# 数据切片限制
MAX_LLM_INPUT_CHARS = 200000

# =====================
# 3. LLM Prompt模板
# =====================
COMBINED_ANALYSIS_PROMPT = """
你是一名资深投资分析师。请基于以下财务数据和技术分析进行综合分析：

{combined_data}

请从以下角度进行分析：

1. **基本面分析** (150字以内):
   - 财务健康状况 (收入增长、盈利能力、负债水平)
   - 估值水平 (PE、PB比率是否合理)
   - 行业地位和竞争优势

2. **技术面分析** (100字以内):
   - 当前趋势方向和强度
   - 关键技术位和支撑阻力
   - 短期动量和风险

3. **综合投资建议** (100字以内):
   - 明确给出 Buy/Hold/Sell 建议
   - 主要理由和风险提示
   - 建议持有期和目标价位(如适用)

请确保分析客观、简洁，重点突出。
"""

SIGNAL_PROMPT = """
基于以下分析结果，生成交易信号：

{analysis_result}

要求：
- 给出明确的 Buy/Hold/Sell 信号
- 100字内说明理由
- 指出主要风险
- 输出JSON格式：{{"signal": "Buy", "reason": "...", "risk": "..."}}
"""

# =====================
# 4. 配置验证
# =====================
def validate_config():
    """验证必需的配置项"""
    errors = []
    
    if not OPENAI_API_KEY:
        errors.append("OPENAI_API_KEY 环境变量未设置")
    
    if errors:
        logger.error("配置验证失败:")
        for error in errors:
            logger.error(f"  - {error}")
        sys.exit(1)
    
    logger.info("配置验证通过")

def ensure_dirs():
    """创建必需的目录"""
    for d in [DATA_DIR, PRICE_DIR, SIGNAL_DIR, BT_DIR]:
        d.mkdir(parents=True, exist_ok=True)

# =====================
# 5. 财务数据获取（使用yfinance）
# =====================
def get_financial_data_yfinance(ticker: str) -> dict:
    """使用yfinance获取财报数据"""
    try:
        logger.info(f"获取 {ticker} 的财务数据...")
        stock = yf.Ticker(ticker)
        
        # 获取各种财务报表
        financials = stock.financials          # 年度利润表
        quarterly_financials = stock.quarterly_financials  # 季度利润表
        balance_sheet = stock.balance_sheet    # 年度资产负债表
        quarterly_balance_sheet = stock.quarterly_balance_sheet  # 季度资产负债表
        cashflow = stock.cashflow             # 年度现金流量表
        quarterly_cashflow = stock.quarterly_cashflow  # 季度现金流量表
        
        # 获取关键指标和公司信息
        info = stock.info  # 包含PE、PB、ROE等关键指标
        
        data = {
            'info': info,
            'financials': financials,
            'quarterly_financials': quarterly_financials,
            'balance_sheet': balance_sheet,
            'quarterly_balance_sheet': quarterly_balance_sheet,
            'cashflow': cashflow,
            'quarterly_cashflow': quarterly_cashflow
        }
        
        logger.info(f"{ticker} 财务数据获取完成")
        return data
        
    except Exception as e:
        logger.error(f"获取 {ticker} 财务数据失败: {e}")
        return {}

def extract_key_financials(financial_data: dict) -> dict:
    """提取关键财务指标"""
    try:
        info = financial_data.get('info', {})
        financials = financial_data.get('financials', pd.DataFrame())
        balance_sheet = financial_data.get('balance_sheet', pd.DataFrame())
        
        key_metrics = {}
        
        # 从info中提取关键指标
        key_fields = [
            'longName', 'industry', 'sector', 'marketCap', 'enterpriseValue',
            'trailingPE', 'forwardPE', 'priceToBook', 'priceToSalesTrailing12Months',
            'returnOnEquity', 'returnOnAssets', 'debtToEquity', 'currentRatio',
            'grossMargins', 'operatingMargins', 'profitMargins',
            'revenueGrowth', 'earningsGrowth', 'dividendYield',
            'beta', 'fiftyTwoWeekHigh', 'fiftyTwoWeekLow'
        ]
        
        for field in key_fields:
            key_metrics[field] = info.get(field, 'N/A')
        
        # 从财务报表中提取最新数据
        if not financials.empty:
            latest_year = financials.columns[0]
            key_metrics['latest_revenue'] = financials.loc['Total Revenue', latest_year] if 'Total Revenue' in financials.index else 'N/A'
            key_metrics['latest_net_income'] = financials.loc['Net Income', latest_year] if 'Net Income' in financials.index else 'N/A'
        
        if not balance_sheet.empty:
            latest_year = balance_sheet.columns[0]
            key_metrics['latest_total_assets'] = balance_sheet.loc['Total Assets', latest_year] if 'Total Assets' in balance_sheet.index else 'N/A'
            key_metrics['latest_total_debt'] = balance_sheet.loc['Total Debt', latest_year] if 'Total Debt' in balance_sheet.index else 'N/A'
        
        return key_metrics
        
    except Exception as e:
        logger.error(f"提取关键财务指标失败: {e}")
        return {}

# =====================
# 6. 价格数据和技术指标
# =====================
def fetch_prices_enhanced(ticker: str, start: str = START_DATE, end: str = END_DATE) -> Path:
    """获取价格数据"""
    path = PRICE_DIR / f"{ticker}_prices.parquet"
    
    # 检查缓存
    if path.exists():
        try:
            df = pd.read_parquet(path)
            latest_date = df.index.max()
            if latest_date >= pd.Timestamp(end) - timedelta(days=7):
                logger.info(f"使用缓存的价格数据: {ticker}")
                return path
        except Exception:
            logger.warning(f"缓存文件损坏，重新下载: {ticker}")
    
    try:
        logger.info(f"下载价格数据: {ticker} ({start} 到 {end})")
        
        # 使用yfinance获取价格数据
        stock = yf.Ticker(ticker)
        df = stock.history(
            start=start,
            end=end,
            interval='1d',
            auto_adjust=True,
            back_adjust=True
        )
        
        if df.empty:
            raise ValueError(f"未获取到 {ticker} 的价格数据")
        
        logger.info(f"获取到 {len(df)} 天的价格数据")
        
        # 保存数据
        df.to_parquet(path)
        logger.info(f"价格数据保存完成")
        return path
        
    except Exception as e:
        logger.error(f"获取价格数据失败 {ticker}: {e}")
        raise

def add_basic_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """计算基础技术指标"""
    try:
        df = df.copy()
        
        # 基础验证
        if 'Close' not in df.columns or df['Close'].isna().all():
            logger.error("无效的价格数据")
            return df
        
        if len(df) < 5:
            logger.warning("数据点太少，技术指标可能不准确")
        
        # 1. 移动平均线
        df['SMA5'] = df['Close'].rolling(window=5, min_periods=1).mean()
        df['SMA20'] = df['Close'].rolling(window=20, min_periods=1).mean()
        df['SMA50'] = df['Close'].rolling(window=50, min_periods=1).mean()
        df['SMA200'] = df['Close'].rolling(window=200, min_periods=1).mean()
        
        # 2. 指数移动平均线
        df['EMA12'] = df['Close'].ewm(span=12, adjust=False).mean()
        df['EMA26'] = df['Close'].ewm(span=26, adjust=False).mean()
        
        # 3. 价格变化和收益率
        df['Price_Change'] = df['Close'].diff()
        df['Price_Change_Pct'] = df['Close'].pct_change() * 100
        df['Return_5D'] = df['Close'].pct_change(5) * 100
        df['Return_20D'] = df['Close'].pct_change(20) * 100
        
        # 4. RSI
        price_diff = df['Close'].diff()
        gains = price_diff.where(price_diff > 0, 0)
        losses = -price_diff.where(price_diff < 0, 0)
        
        avg_gain = gains.rolling(window=14, min_periods=1).mean()
        avg_loss = losses.rolling(window=14, min_periods=1).mean()
        
        # 避免除零错误
        avg_loss = avg_loss.replace(0, 0.0001)
        rs = avg_gain / avg_loss
        df['RSI'] = 100 - (100 / (1 + rs))
        
        # 5. 波动率
        df['Volatility_5D'] = df['Price_Change_Pct'].rolling(5).std()
        df['Volatility_20D'] = df['Price_Change_Pct'].rolling(20).std()
        
        # 6. 布林带
        sma20 = df['SMA20']
        std20 = df['Close'].rolling(20).std()
        df['BB_Upper'] = sma20 + (std20 * 2)
        df['BB_Lower'] = sma20 - (std20 * 2)
        df['BB_Position'] = (df['Close'] - df['BB_Lower']) / (df['BB_Upper'] - df['BB_Lower'])
        
        # 7. MACD
        df['MACD'] = df['EMA12'] - df['EMA26']
        df['MACD_Signal'] = df['MACD'].ewm(span=9).mean()
        df['MACD_Histogram'] = df['MACD'] - df['MACD_Signal']
        
        # 8. 成交量相关指标
        if 'Volume' in df.columns and not df['Volume'].isna().all():
            df['Avg_Volume_20D'] = df['Volume'].rolling(20, min_periods=1).mean()
            df['Volume_Ratio'] = df['Volume'] / df['Avg_Volume_20D']
        
        # 填充NaN值
        df = df.fillna(method='ffill').fillna(method='bfill')
        
        logger.info("技术指标计算完成")
        return df
        
    except Exception as e:
        logger.error(f"技术指标计算失败: {e}")
        return df

def generate_technical_summary(df: pd.DataFrame, ticker: str) -> str:
    """生成技术分析摘要"""
    try:
        if df.empty:
            return f"{ticker}: 无价格数据"
        
        latest = df.iloc[-1]
        prev = df.iloc[-2] if len(df) > 1 else latest
        
        summary_parts = []
        
        # 价格信息
        price = latest['Close']
        change = latest['Close'] - prev['Close'] if len(df) > 1 else 0
        change_pct = (change / prev['Close'] * 100) if len(df) > 1 and prev['Close'] != 0 else 0
        
        summary_parts.append(f"当前价格: ${price:.2f} ({change_pct:+.2f}%)")
        
        # 趋势分析
        if pd.notna(latest['SMA20']) and pd.notna(latest['SMA200']):
            trend = "上升趋势" if latest['SMA20'] > latest['SMA200'] else "下降趋势"
            position = "上方" if price > latest['SMA20'] else "下方"
            summary_parts.append(f"趋势: {trend}, 价格在20日均线{position}")
        
        # RSI
        if pd.notna(latest['RSI']):
            rsi = latest['RSI']
            if rsi > 70:
                rsi_signal = "超买区域"
            elif rsi < 30:
                rsi_signal = "超卖区域"
            else:
                rsi_signal = "中性区域"
            summary_parts.append(f"RSI({rsi:.1f}): {rsi_signal}")
        
        # 波动率
        if pd.notna(latest['Volatility_20D']):
            vol = latest['Volatility_20D']
            vol_level = "高波动" if vol > 3 else "中波动" if vol > 1.5 else "低波动"
            summary_parts.append(f"20日波动率: {vol:.1f}%({vol_level})")
        
        # MACD信号
        if pd.notna(latest['MACD']) and pd.notna(latest['MACD_Signal']):
            macd_signal = "金叉" if latest['MACD'] > latest['MACD_Signal'] else "死叉"
            summary_parts.append(f"MACD: {macd_signal}")
        
        # 布林带位置
        if pd.notna(latest['BB_Position']):
            bb_pos = latest['BB_Position']
            if bb_pos > 0.8:
                bb_signal = "接近上轨"
            elif bb_pos < 0.2:
                bb_signal = "接近下轨"
            else:
                bb_signal = "中轨附近"
            summary_parts.append(f"布林带: {bb_signal}")
        
        return "; ".join(summary_parts)
        
    except Exception as e:
        logger.error(f"生成技术摘要失败: {e}")
        return f"{ticker}: 技术分析数据不足"

# =====================
# 7. 数据切片和LLM输入准备
# =====================
def slice_data_for_llm(data: str, max_chars: int = MAX_LLM_INPUT_CHARS) -> str:
    """将数据切片以适应LLM输入限制"""
    if len(data) <= max_chars:
        return data
    
    logger.info(f"数据过长({len(data)}字符)，进行智能切片...")
    
    # 智能切片：保留开头、关键部分和结尾
    start_chars = max_chars // 4
    end_chars = max_chars // 4
    middle_chars = max_chars - start_chars - end_chars - 200  # 留出标记空间
    
    start_part = data[:start_chars]
    end_part = data[-end_chars:]
    
    # 从中间部分选择最重要的数据（寻找包含数字和关键词的部分）
    middle_start = len(data) // 2 - middle_chars // 2
    middle_end = len(data) // 2 + middle_chars // 2
    middle_part = data[middle_start:middle_end]
    
    sliced_data = f"{start_part}\n\n... [数据已截取，包含{len(data)}字符] ...\n\n{middle_part}\n\n... [数据已截取] ...\n\n{end_part}"
    
    logger.info(f"切片完成，输出{len(sliced_data)}字符")
    return sliced_data

def prepare_llm_input(financial_data: dict, key_metrics: dict, technical_summary: str) -> str:
    """准备LLM输入数据"""
    try:
        # 构建完整的分析输入
        analysis_text = "=== 综合投资分析报告 ===\n\n"
        
        # 1. 公司基本信息
        analysis_text += "=== 公司基本信息 ===\n"
        if key_metrics:
            analysis_text += f"公司名称: {key_metrics.get('longName', 'N/A')}\n"
            analysis_text += f"行业: {key_metrics.get('industry', 'N/A')}\n"
            analysis_text += f"板块: {key_metrics.get('sector', 'N/A')}\n"
            analysis_text += f"市值: {key_metrics.get('marketCap', 'N/A')}\n"
            analysis_text += f"企业价值: {key_metrics.get('enterpriseValue', 'N/A')}\n\n"
        
        # 2. 估值指标
        analysis_text += "=== 估值指标 ===\n"
        if key_metrics:
            analysis_text += f"市盈率(TTM): {key_metrics.get('trailingPE', 'N/A')}\n"
            analysis_text += f"预期市盈率: {key_metrics.get('forwardPE', 'N/A')}\n"
            analysis_text += f"市净率: {key_metrics.get('priceToBook', 'N/A')}\n"
            analysis_text += f"市销率: {key_metrics.get('priceToSalesTrailing12Months', 'N/A')}\n"
            analysis_text += f"股息收益率: {key_metrics.get('dividendYield', 'N/A')}\n\n"
        
        # 3. 盈利能力指标
        analysis_text += "=== 盈利能力指标 ===\n"
        if key_metrics:
            analysis_text += f"净资产收益率: {key_metrics.get('returnOnEquity', 'N/A')}\n"
            analysis_text += f"总资产收益率: {key_metrics.get('returnOnAssets', 'N/A')}\n"
            analysis_text += f"毛利率: {key_metrics.get('grossMargins', 'N/A')}\n"
            analysis_text += f"营业利润率: {key_metrics.get('operatingMargins', 'N/A')}\n"
            analysis_text += f"净利润率: {key_metrics.get('profitMargins', 'N/A')}\n\n"
        
        # 4. 成长性指标
        analysis_text += "=== 成长性指标 ===\n"
        if key_metrics:
            analysis_text += f"收入增长率: {key_metrics.get('revenueGrowth', 'N/A')}\n"
            analysis_text += f"盈利增长率: {key_metrics.get('earningsGrowth', 'N/A')}\n\n"
        
        # 5. 财务健康指标
        analysis_text += "=== 财务健康指标 ===\n"
        if key_metrics:
            analysis_text += f"负债权益比: {key_metrics.get('debtToEquity', 'N/A')}\n"
            analysis_text += f"流动比率: {key_metrics.get('currentRatio', 'N/A')}\n"
            analysis_text += f"Beta系数: {key_metrics.get('beta', 'N/A')}\n\n"
        
        # 6. 最新财务数据
        if 'financials' in financial_data and not financial_data['financials'].empty:
            analysis_text += "=== 最新年度财务报表摘要 ===\n"
            financials = financial_data['financials']
            # 只取最近2年的数据以节省空间
            recent_financials = financials.iloc[:, :2] if financials.shape[1] >= 2 else financials
            analysis_text += recent_financials.to_string() + "\n\n"
        
        # 7. 技术分析
        analysis_text += f"=== 技术分析摘要 ===\n{technical_summary}\n\n"
        
        # 8. 52周价格区间
        if key_metrics:
            analysis_text += "=== 价格区间 ===\n"
            analysis_text += f"52周最高价: {key_metrics.get('fiftyTwoWeekHigh', 'N/A')}\n"
            analysis_text += f"52周最低价: {key_metrics.get('fiftyTwoWeekLow', 'N/A')}\n\n"
        
        # 应用数据切片
        return slice_data_for_llm(analysis_text, MAX_LLM_INPUT_CHARS)
        
    except Exception as e:
        logger.error(f"准备LLM输入失败: {e}")
        return f"数据准备失败: {str(e)}"

# =====================
# 8. LLM分析
# =====================
class LLMRunner:
    """LLM运行器"""
    def __init__(self, api_key: str, model_name: str):
        if not api_key:
            raise ValueError("OPENAI_API_KEY 未设置")
        
        try:
            self.llm = ChatOpenAI(
                api_key=api_key,
                model=model_name,
                temperature=0.2,
                max_tokens=2000
            )
            logger.info(f"初始化 LLM: {model_name}")
        except Exception as e:
            logger.error(f"初始化 LLM 失败: {e}")
            raise

    @functools.lru_cache(maxsize=32)
    def _cached_run(self, prompt_hash: str, prompt_content: str) -> str:
        """缓存的 LLM 调用"""
        try:
            messages = [HumanMessage(content=prompt_content)]
            response = self.llm.invoke(messages)
            return response.content
        except Exception as e:
            logger.error(f"LLM 调用失败: {e}")
            raise

    def run(self, template_str: str, **kwargs) -> str:
        """运行 LLM 推理"""
        try:
            # 格式化模板
            prompt_content = template_str.format(**kwargs)
            
            # 检查输入长度
            if len(prompt_content) > MAX_LLM_INPUT_CHARS:
                logger.warning(f"输入过长({len(prompt_content)}字符)，可能影响分析质量")
            
            # 生成缓存键
            prompt_hash = hashlib.md5(prompt_content.encode()).hexdigest()
            
            # 使用缓存调用
            result = self._cached_run(prompt_hash, prompt_content)
            logger.debug("LLM 推理完成")
            return result
        except Exception as e:
            logger.error(f"LLM 推理失败: {e}")
            return f"分析失败: {str(e)}"

def generate_signal_from_analysis(llm_runner: LLMRunner, analysis_result: str, ticker: str) -> dict:
    """基于分析结果生成交易信号"""
    try:
        # 调用LLM生成信号
        raw_response = llm_runner.run(SIGNAL_PROMPT, analysis_result=analysis_result)
        
        # 尝试解析JSON响应
        try:
            # 清理响应文本
            cleaned_response = raw_response.strip()
            if cleaned_response.startswith('```json'):
                cleaned_response = cleaned_response[7:]
            if cleaned_response.endswith('```'):
                cleaned_response = cleaned_response[:-3]
            cleaned_response = cleaned_response.strip()
            
            signal_json = json.loads(cleaned_response)
            
            # 验证和标准化信号
            signal = signal_json.get('signal', 'Hold').strip()
            if signal.lower() in ['buy', 'purchase', '买入']:
                signal = 'Buy'
            elif signal.lower() in ['sell', '卖出']:
                signal = 'Sell'
            else:
                signal = 'Hold'
            
            result = {
                "signal": signal,
                "reason": signal_json.get('reason', '无具体原因')[:200],
                "risk": signal_json.get('risk', '无明确风险')[:200],
                "raw_response": raw_response
            }
            
            logger.info(f"{ticker}: 信号生成完成 - {signal}")
            return result
            
        except json.JSONDecodeError:
            logger.warning(f"{ticker}: JSON解析失败，使用文本分析")
            
            # 从原始响应中提取信息
            signal = 'Hold'
            if any(word in raw_response.lower() for word in ['buy', 'purchase', '买入']):
                signal = 'Buy'
            elif any(word in raw_response.lower() for word in ['sell', '卖出']):
                signal = 'Sell'
            
            return {
                "signal": signal,
                "reason": raw_response[:200],
                "risk": "无法解析具体风险",
                "raw_response": raw_response
            }
    
    except Exception as e:
        logger.error(f"{ticker}: 信号生成失败: {e}")
        return {
            "signal": "Hold",
            "reason": f"信号生成失败: {str(e)}",
            "risk": "系统错误",
            "raw_response": ""
        }

# =====================
# 9. 回测功能
# =====================
def create_signal_dataframe(signal_result: dict, ticker: str) -> pd.DataFrame:
    """创建信号DataFrame用于回测"""
    try:
        signal_numeric = 1 if signal_result['signal'] == 'Buy' else (-1 if signal_result['signal'] == 'Sell' else 0)
        
        signal_data = {
            "date": datetime.now().strftime('%Y-%m-%d'),
            "ticker": ticker,
            "signal": signal_numeric,
            "confidence": None,
            "reason": signal_result['reason'],
            "risk": signal_result['risk'],
            "raw_signal": signal_result['signal']
        }
        
        df = pd.DataFrame([signal_data])
        df['date'] = pd.to_datetime(df['date'])
        
        return df
        
    except Exception as e:
        logger.error(f"创建信号DataFrame失败: {e}")
        return pd.DataFrame()

def backtest_single_enhanced(signals_df: pd.DataFrame, price_df: pd.DataFrame, fee_bps: float = FEE_BPS) -> dict:
    """简化版单资产回测"""
    try:
        if signals_df.empty or price_df.empty:
            return {"error": "数据不足，无法回测"}
        
        # 数据预处理
        signals_df = signals_df.copy()
        if 'date' in signals_df.columns:
            signals_df['date'] = pd.to_datetime(signals_df['date'])
            signals_df = signals_df.set_index('date')
        
        price_df = price_df.copy()
        if not isinstance(price_df.index, pd.DatetimeIndex):
            price_df.index = pd.to_datetime(price_df.index)
        
        # 对齐数据
        signals_df = signals_df.reindex(price_df.index, method='ffill').fillna(0)
        
        # 生成交易信号
        entries = signals_df['signal'] == 1
        exits = signals_df['signal'] == -1
        
        if not entries.any() and not exits.any():
            return {"error": "无有效交易信号"}
        
        # 执行回测
        pf = vbt.Portfolio.from_signals(
            price_df['Close'],
            entries,
            exits,
            fees=fee_bps / 10000,
            freq='D'
        )
        
        # 计算基础统计指标
        total_return = pf.total_return()
        sharpe_ratio = pf.sharpe_ratio()
        max_drawdown = pf.max_drawdown()
        
        # 保存权益曲线
        equity_curve = pf.value()
        eq_path = BT_DIR / 'equity_curve_single.csv'
        equity_curve.to_csv(eq_path)
        
        result = {
            "total_return": float(total_return) if pd.notna(total_return) else 0.0,
            "sharpe_ratio": float(sharpe_ratio) if pd.notna(sharpe_ratio) else 0.0,
            "max_drawdown": float(max_drawdown) if pd.notna(max_drawdown) else 0.0,
            "equity_curve_path": str(eq_path),
            "total_trades": len(pf.orders.records_readable)
        }
        
        logger.info("单资产回测完成")
        return result
        
    except Exception as e:
        logger.error(f"单资产回测失败: {e}")
        return {"error": str(e)}

# =====================
# 10. 主要处理流程
# =====================
def orchestrate_single_simplified(ticker: str) -> dict:
    """简化版单公司分析流程"""
    try:
        logger.info(f"开始分析: {ticker}")
        
        # 1. 获取财务数据
        financial_data = get_financial_data_yfinance(ticker)
        if not financial_data:
            raise ValueError(f"无法获取 {ticker} 的财务数据")
        
        # 2. 提取关键财务指标
        key_metrics = extract_key_financials(financial_data)
        
        # 3. 获取价格数据和技术指标
        price_path = fetch_prices_enhanced(ticker, START_DATE, END_DATE)
        price_df = pd.read_parquet(price_path)
        price_df = add_basic_indicators(price_df)
        
        # 4. 生成技术分析摘要
        technical_summary = generate_technical_summary(price_df, ticker)
        
        # 5. 准备LLM输入（结合财务+技术数据，并切片）
        llm_input = prepare_llm_input(financial_data, key_metrics, technical_summary)
        
        # 6. LLM综合分析
        llm_runner = LLMRunner(OPENAI_API_KEY, MODEL_NAME)
        analysis_result = llm_runner.run(COMBINED_ANALYSIS_PROMPT, combined_data=llm_input)
        
        # 7. 生成交易信号
        signal_result = generate_signal_from_analysis(llm_runner, analysis_result, ticker)
        
        # 8. 回测
        signal_df = create_signal_dataframe(signal_result, ticker)
        backtest_result = backtest_single_enhanced(signal_df, price_df)
        
        # 9. 保存信号
        save_json([{
            "date": datetime.now().strftime('%Y-%m-%d'),
            "ticker": ticker,
            "signal": signal_result['signal'],
            "reason": signal_result['reason'],
            "risk": signal_result['risk']
        }], SIGNAL_DIR / f"{ticker}_signals.json")
        
        result = {
            "ticker": ticker,
            "processing_time": datetime.now().isoformat(),
            "analysis": analysis_result,
            "signal": signal_result,
            "backtest_results": backtest_result,
            "data_quality": {
                "input_chars": len(llm_input),
                "price_data_points": len(price_df),
                "financial_data_available": bool(financial_data),
                "key_metrics_count": len(key_metrics)
            }
        }
        
        logger.info(f"{ticker}: 分析完成")
        return result
        
    except Exception as e:
        logger.error(f"{ticker}: 分析失败: {e}")
        return {
            "ticker": ticker,
            "error": str(e),
            "processing_time": datetime.now().isoformat()
        }

def orchestrate_batch_simplified(tickers: List[str]) -> dict:
    """简化版批量处理流程"""
    try:
        logger.info(f"开始批量分析: {len(tickers)} 个股票")
        
        all_results = {}
        all_signals = []
        
        for i, ticker in enumerate(tickers):
            logger.info(f"处理进度: {i+1}/{len(tickers)} - {ticker}")
            
            try:
                result = orchestrate_single_simplified(ticker)
                all_results[ticker] = result
                
                if 'signal' in result:
                    signal_record = {
                        "date": datetime.now().strftime('%Y-%m-%d'),
                        "ticker": ticker,
                        "signal": result['signal']['signal'],
                        "reason": result['signal']['reason'],
                        "risk": result['signal']['risk']
                    }
                    all_signals.append(signal_record)
                    
            except Exception as e:
                logger.error(f"{ticker} 处理失败: {e}")
                all_results[ticker] = {"error": str(e)}
                continue
        
        # 生成批量结果摘要
        successful_tickers = [t for t, r in all_results.items() if "error" not in r]
        failed_tickers = [t for t, r in all_results.items() if "error" in r]
        
        # 统计信号分布
        buy_signals = len([s for s in all_signals if s["signal"] == "Buy"])
        sell_signals = len([s for s in all_signals if s["signal"] == "Sell"])
        hold_signals = len([s for s in all_signals if s["signal"] == "Hold"])
        
        batch_result = {
            "processing_time": datetime.now().isoformat(),
            "requested_tickers": tickers,
            "successful_tickers": successful_tickers,
            "failed_tickers": failed_tickers,
            "success_rate": len(successful_tickers) / len(tickers) if tickers else 0,
            "total_signals": len(all_signals),
            "signal_distribution": {
                "buy": buy_signals,
                "sell": sell_signals,
                "hold": hold_signals
            },
            "signals": all_signals,
            "detailed_results": all_results
        }
        
        # 保存批量结果
        save_json(batch_result, SIGNAL_DIR / "batch_results.json")
        
        logger.info(f"批量分析完成: {len(successful_tickers)}/{len(tickers)} 成功")
        return batch_result
        
    except Exception as e:
        logger.error(f"批量分析失败: {e}")
        return {
            "error": str(e),
            "processing_time": datetime.now().isoformat(),
            "requested_tickers": tickers
        }

# =====================
# 11. 工具函数
# =====================
def save_json(obj, path: Path):
    """安全保存JSON文件"""
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(obj, f, ensure_ascii=False, indent=2, default=str)
        logger.debug(f"保存JSON文件: {path}")
    except Exception as e:
        logger.error(f"保存JSON文件失败 {path}: {e}")
        raise

def system_health_check() -> dict:
    """系统健康检查"""
    health = {
        "timestamp": datetime.now().isoformat(),
        "status": "healthy",
        "checks": {}
    }
    
    try:
        # 检查环境变量
        health["checks"]["openai_api"] = bool(OPENAI_API_KEY)
        
        # 检查目录
        health["checks"]["directories"] = all(d.exists() for d in [DATA_DIR, PRICE_DIR, SIGNAL_DIR])
        
        # 检查依赖库
        try:
            import yfinance
            health["checks"]["yfinance"] = True
        except ImportError:
            health["checks"]["yfinance"] = False
        
        try:
            import vectorbt
            health["checks"]["vectorbt"] = True
        except ImportError:
            health["checks"]["vectorbt"] = False
        
        try:
            from langchain_openai import ChatOpenAI
            health["checks"]["langchain"] = True
        except ImportError:
            health["checks"]["langchain"] = False
        
        # 测试yfinance功能
        try:
            test_stock = yf.Ticker("AAPL")
            test_info = test_stock.info
            health["checks"]["yfinance_api"] = bool(test_info)
        except Exception:
            health["checks"]["yfinance_api"] = False
        
        # 综合状态
        failed_checks = [k for k, v in health["checks"].items() if not v]
        if failed_checks:
            health["status"] = "warning" if len(failed_checks) < 2 else "error"
            health["failed_checks"] = failed_checks
    
    except Exception as e:
        health["status"] = "error"
        health["error"] = str(e)
    
    return health

# =====================
# 12. CLI 入口
# =====================
def main():
    """主入口函数"""
    parser = argparse.ArgumentParser(
        description="简化版LLM基本面分析代理",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
使用示例:
  单公司分析: python %(prog)s --mode single --ticker AAPL
  批量分析: python %(prog)s --mode batch --tickers AAPL MSFT GOOGL
  健康检查: python %(prog)s --health-check

环境变量要求:
  OPENAI_API_KEY: OpenAI API密钥 (必需)

特性:
  - 使用yfinance统一获取财务和价格数据
  - 智能数据切片适应LLM 200k字符限制
  - 结合基本面和技术面综合分析
  - 简化依赖，提高稳定性
        """
    )
    
    parser.add_argument(
        '--mode',
        choices=['single', 'batch'],
        default='single',
        help='处理模式: single(单公司) 或 batch(批量)'
    )
    
    parser.add_argument(
        '--ticker',
        type=str,
        default='AAPL',
        help='单公司模式的股票代码'
    )
    
    parser.add_argument(
        '--tickers',
        nargs='+',
        help='批量模式的股票代码列表'
    )
    
    parser.add_argument(
        '--health-check',
        action='store_true',
        help='执行系统健康检查'
    )
    
    parser.add_argument(
        '--output',
        type=str,
        help='输出结果的JSON文件路径'
    )
    
    parser.add_argument(
        '--verbose',
        action='store_true',
        help='显示详细日志'
    )
    
    args = parser.parse_args()
    
    # 设置日志级别
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    try:
        # 健康检查
        if args.health_check:
            ensure_dirs()
            health = system_health_check()
            print("=== 系统健康检查 ===")
            print(json.dumps(health, ensure_ascii=False, indent=2))
            
            if health["status"] == "error":
                sys.exit(1)
            elif health["status"] == "warning":
                print("\n⚠️  发现一些问题，但系统仍可运行")
            else:
                print("\n✅ 系统状态良好")
            return
        
        # 验证配置
        validate_config()
        ensure_dirs()
        
        # 执行分析
        if args.mode == 'single':
            logger.info(f"开始单公司分析: {args.ticker}")
            result = orchestrate_single_simplified(args.ticker)
        else:
            tickers = args.tickers or ['AAPL', 'MSFT', 'GOOGL']
            logger.info(f"开始批量分析: {tickers}")
            result = orchestrate_batch_simplified(tickers)
        
        # 输出结果
        output_json = json.dumps(result, ensure_ascii=False, indent=2, default=str)
        
        if args.output:
            output_path = Path(args.output)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write(output_json)
            logger.info(f"结果已保存到: {output_path}")
        else:
            print(output_json)
        
        # 检查结果状态
        if "error" in result:
            logger.error("分析过程中发生错误")
            sys.exit(1)
        
        logger.info("分析完成")
    
    except KeyboardInterrupt:
        logger.info("用户中断操作")
        sys.exit(1)
    except Exception as e:
        logger.error(f"程序执行失败: {e}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        sys.exit(1)

if __name__ == '__main__':
    main()