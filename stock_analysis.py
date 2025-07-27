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

# 环境变量加载
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass  # dotenv是可选的

import numpy as np
import pandas as pd
import yfinance as yf

# LLM
try:
    from langchain_openai import ChatOpenAI
    from langchain.schema import HumanMessage
except ImportError:
    print("❌ 请安装LangChain: pip install langchain langchain-openai")
    sys.exit(1)

# 回测（可选）
try:
    import vectorbt as vbt
    VBT_AVAILABLE = True
except ImportError:
    print("⚠️  VectorBT未安装，回测功能将被禁用")
    print("   可选安装: pip install vectorbt")
    VBT_AVAILABLE = False

# =====================
# 1. 配置和日志
# =====================
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('financial_agent.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# 目录配置
BASE_DIR = Path(__file__).parent.absolute()
DATA_DIR = BASE_DIR / 'data'
PRICE_DIR = DATA_DIR / 'prices'
SIGNAL_DIR = DATA_DIR / 'signals'
BT_DIR = DATA_DIR / 'backtest'

# 环境变量配置
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY', '')
MODEL_NAME = os.getenv('MODEL_NAME', 'gpt-4o-mini')
START_DATE = os.getenv('START_DATE', '2020-01-01')
END_DATE = os.getenv('END_DATE', '2025-01-01')
FEE_BPS = float(os.getenv('FEE_BPS', '10'))
MAX_LLM_INPUT_CHARS = int(os.getenv('MAX_LLM_INPUT_CHARS', '150000'))

# =====================
# 2. LLM提示模板
# =====================
ANALYSIS_PROMPT = """
你是一名专业的投资分析师。请对以下股票进行全面分析：

{stock_data}

请按以下结构进行分析：

**1. 基本面分析 (200字以内)**
- 财务健康状况（收入、盈利、负债）
- 估值水平（PE、PB等指标）
- 行业地位和竞争优势
- 成长性评估

**2. 技术面分析 (150字以内)**
- 价格趋势和动量
- 关键技术指标解读
- 支撑阻力位分析

**3. 综合投资建议 (100字以内)**
- 明确的Buy/Hold/Sell建议
- 核心投资逻辑
- 主要风险提示
- 建议时间周期

请保持分析客观、专业，突出重点。
"""

SIGNAL_PROMPT = """
基于以下分析，生成标准化交易信号：

{analysis_text}

要求：
1. 给出明确的 Buy/Hold/Sell 信号
2. 简述核心理由（50字内）
3. 指出主要风险（50字内）
4. 如果是Buy/Sell，给出目标价或止损价

输出JSON格式：
{{"signal": "Buy", "reason": "核心理由", "risk": "主要风险", "target_price": 150.0}}
"""

# =====================
# 3. 工具函数
# =====================
def ensure_dirs():
    """创建必需目录"""
    for d in [DATA_DIR, PRICE_DIR, SIGNAL_DIR, BT_DIR]:
        d.mkdir(parents=True, exist_ok=True)

def validate_config():
    """验证配置"""
    if not OPENAI_API_KEY:
        logger.error("❌ OPENAI_API_KEY 环境变量未设置")
        print("请设置环境变量: export OPENAI_API_KEY='your-key-here'")
        sys.exit(1)
    logger.info("✅ 配置验证通过")

def save_json(obj: dict, path: Path):
    """保存JSON文件"""
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(obj, f, ensure_ascii=False, indent=2, default=str)
        logger.debug(f"保存文件: {path}")
    except Exception as e:
        logger.error(f"保存文件失败 {path}: {e}")

def slice_text_for_llm(text: str, max_chars: int = MAX_LLM_INPUT_CHARS) -> str:
    """智能文本切片"""
    if len(text) <= max_chars:
        return text
    
    logger.info(f"文本过长({len(text)}字符)，进行智能切片...")
    
    # 保留开头、关键部分和结尾
    start_size = max_chars // 3
    end_size = max_chars // 3
    middle_size = max_chars - start_size - end_size - 200
    
    start_part = text[:start_size]
    end_part = text[-end_size:]
    
    # 提取中间的关键数据（包含数字和重要关键词的部分）
    middle_start = len(text) // 2 - middle_size // 2
    middle_end = len(text) // 2 + middle_size // 2
    middle_part = text[middle_start:middle_end]
    
    result = f"{start_part}\n\n... [数据已优化，原长度{len(text)}字符] ...\n\n{middle_part}\n\n... [数据继续] ...\n\n{end_part}"
    
    logger.info(f"切片完成: {len(result)} 字符")
    return result

# =====================
# 4. 财务数据获取
# =====================
class YFinanceDataProvider:
    """YFinance数据提供器"""
    
    def __init__(self, ticker: str):
        self.ticker = ticker.upper()
        self.stock = yf.Ticker(self.ticker)
        logger.info(f"初始化数据提供器: {self.ticker}")
    
    def get_company_info(self) -> dict:
        """获取公司基本信息"""
        try:
            info = self.stock.info
            if not info:
                return {"error": "无法获取公司信息"}
            
            return {
                "symbol": info.get('symbol', self.ticker),
                "name": info.get('longName', 'Unknown'),
                "sector": info.get('sector', 'Unknown'),
                "industry": info.get('industry', 'Unknown'),
                "market_cap": info.get('marketCap', 0),
                "employees": info.get('fullTimeEmployees', 0),
                "description": info.get('longBusinessSummary', '')[:300] + '...' if info.get('longBusinessSummary') else '',
                "website": info.get('website', ''),
                "currency": info.get('currency', 'USD')
            }
        except Exception as e:
            logger.error(f"获取公司信息失败: {e}")
            return {"error": str(e)}
    
    def get_financial_data(self) -> dict:
        """获取财务数据"""
        try:
            data = {}
            
            # 获取财务报表（最近4期）
            try:
                data['income_stmt'] = self.stock.financials.iloc[:, :4]
                data['balance_sheet'] = self.stock.balance_sheet.iloc[:, :4]
                data['cash_flow'] = self.stock.cashflow.iloc[:, :4]
                
                # 季度数据
                data['quarterly_income'] = self.stock.quarterly_financials.iloc[:, :4]
                data['quarterly_balance'] = self.stock.quarterly_balance_sheet.iloc[:, :4]
                
            except Exception as e:
                logger.warning(f"获取部分财务报表失败: {e}")
            
            logger.info(f"财务数据获取完成: {len(data)} 个报表")
            return data
            
        except Exception as e:
            logger.error(f"获取财务数据失败: {e}")
            return {}
    
    def get_key_metrics(self) -> dict:
        """获取关键财务指标"""
        try:
            info = self.stock.info
            if not info:
                return {}
            
            # 提取关键指标
            metrics = {}
            
            # 估值指标
            valuation_metrics = {
                'pe_ratio': 'trailingPE',
                'forward_pe': 'forwardPE', 
                'pb_ratio': 'priceToBook',
                'ps_ratio': 'priceToSalesTrailing12Months',
                'peg_ratio': 'pegRatio',
                'ev_revenue': 'enterpriseToRevenue',
                'ev_ebitda': 'enterpriseToEbitda'
            }
            
            # 盈利能力
            profitability_metrics = {
                'roe': 'returnOnEquity',
                'roa': 'returnOnAssets', 
                'gross_margin': 'grossMargins',
                'operating_margin': 'operatingMargins',
                'net_margin': 'netIncomeToCommon'
            }
            
            # 财务健康
            financial_health = {
                'debt_to_equity': 'debtToEquity',
                'current_ratio': 'currentRatio',
                'quick_ratio': 'quickRatio',
                'interest_coverage': 'interestCoverage'
            }
            
            # 成长性
            growth_metrics = {
                'revenue_growth': 'revenueGrowth',
                'earnings_growth': 'earningsGrowth',
                'earnings_quarterly_growth': 'earningsQuarterlyGrowth'
            }
            
            # 股息和回报
            dividend_metrics = {
                'dividend_yield': 'dividendYield',
                'payout_ratio': 'payoutRatio',
                'dividend_rate': 'dividendRate'
            }
            
            # 其他重要指标
            other_metrics = {
                'beta': 'beta',
                'book_value': 'bookValue',
                'price_to_book': 'priceToBook',
                '52w_high': 'fiftyTwoWeekHigh',
                '52w_low': 'fiftyTwoWeekLow',
                'market_cap': 'marketCap',
                'enterprise_value': 'enterpriseValue'
            }
            
            # 合并所有指标
            all_metrics = {
                **valuation_metrics,
                **profitability_metrics, 
                **financial_health,
                **growth_metrics,
                **dividend_metrics,
                **other_metrics
            }
            
            # 提取数据
            for key, info_key in all_metrics.items():
                value = info.get(info_key)
                if value is not None:
                    metrics[key] = value
            
            logger.info(f"关键指标提取完成: {len(metrics)} 个指标")
            return metrics
            
        except Exception as e:
            logger.error(f"获取关键指标失败: {e}")
            return {}
    
    def get_price_data(self, start_date: str, end_date: str) -> pd.DataFrame:
        """获取价格数据"""
        try:
            df = self.stock.history(
                start=start_date,
                end=end_date,
                interval='1d',
                auto_adjust=True,
                back_adjust=True
            )
            
            if df.empty:
                raise ValueError("未获取到价格数据")
            
            logger.info(f"价格数据获取完成: {len(df)} 个交易日")
            return df
            
        except Exception as e:
            logger.error(f"获取价格数据失败: {e}")
            raise

# =====================
# 5. 技术指标计算
# =====================
def calculate_technical_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """计算技术指标"""
    try:
        if df.empty or 'Close' not in df.columns:
            logger.error("无效的价格数据")
            return df
        
        df = df.copy()
        
        # 移动平均线
        df['SMA_20'] = df['Close'].rolling(20, min_periods=1).mean()
        df['SMA_50'] = df['Close'].rolling(50, min_periods=1).mean() 
        df['SMA_200'] = df['Close'].rolling(200, min_periods=1).mean()
        
        # 指数移动平均
        df['EMA_12'] = df['Close'].ewm(span=12).mean()
        df['EMA_26'] = df['Close'].ewm(span=26).mean()
        
        # RSI
        delta = df['Close'].diff()
        gain = delta.where(delta > 0, 0).rolling(14, min_periods=1).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(14, min_periods=1).mean()
        rs = gain / loss.replace(0, np.inf)
        df['RSI'] = 100 - (100 / (1 + rs))
        
        # MACD
        df['MACD'] = df['EMA_12'] - df['EMA_26']
        df['MACD_Signal'] = df['MACD'].ewm(span=9).mean()
        df['MACD_Hist'] = df['MACD'] - df['MACD_Signal']
        
        # 布林带
        bb_period = 20
        bb_std = 2
        sma = df['Close'].rolling(bb_period, min_periods=1).mean()
        std = df['Close'].rolling(bb_period, min_periods=1).std()
        df['BB_Upper'] = sma + (std * bb_std)
        df['BB_Lower'] = sma - (std * bb_std)
        df['BB_Middle'] = sma
        
        # 布林带位置
        bb_range = df['BB_Upper'] - df['BB_Lower']
        df['BB_Position'] = (df['Close'] - df['BB_Lower']) / bb_range.replace(0, np.nan)
        df['BB_Position'] = df['BB_Position'].clip(0, 1)
        
        # 价格变化
        df['Price_Change'] = df['Close'].pct_change()
        df['Price_Change_5D'] = df['Close'].pct_change(5)
        df['Price_Change_20D'] = df['Close'].pct_change(20)
        
        # 波动率
        df['Volatility_20D'] = df['Price_Change'].rolling(20, min_periods=1).std() * np.sqrt(252)
        
        # 成交量指标
        if 'Volume' in df.columns:
            df['Volume_MA_20'] = df['Volume'].rolling(20, min_periods=1).mean()
            df['Volume_Ratio'] = df['Volume'] / df['Volume_MA_20']
        
        # 填充NaN
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        df[numeric_cols] = df[numeric_cols].fillna(method='ffill').fillna(0)
        
        logger.info("技术指标计算完成")
        return df
        
    except Exception as e:
        logger.error(f"技术指标计算失败: {e}")
        return df

def generate_technical_summary(df: pd.DataFrame, ticker: str) -> str:
    """生成技术分析摘要"""
    try:
        if df.empty:
            return f"{ticker}: 无技术数据"
        
        latest = df.iloc[-1]
        prev = df.iloc[-2] if len(df) > 1 else latest
        
        # 基本价格信息
        price = latest['Close']
        change = price - prev['Close'] if len(df) > 1 else 0
        change_pct = (change / prev['Close'] * 100) if len(df) > 1 and prev['Close'] != 0 else 0
        
        summary = f"当前价格: ${price:.2f} ({change_pct:+.2f}%)\n"
        
        # 趋势分析
        if pd.notna(latest['SMA_20']) and pd.notna(latest['SMA_200']):
            trend = "多头排列" if latest['SMA_20'] > latest['SMA_200'] else "空头排列"
            position = "上方" if price > latest['SMA_20'] else "下方"
            summary += f"趋势: {trend}, 价格在20日线{position}\n"
        
        # RSI
        if pd.notna(latest['RSI']):
            rsi = latest['RSI']
            rsi_status = "超买" if rsi > 70 else "超卖" if rsi < 30 else "中性"
            summary += f"RSI: {rsi:.1f} ({rsi_status})\n"
        
        # MACD
        if pd.notna(latest['MACD']) and pd.notna(latest['MACD_Signal']):
            macd_signal = "金叉" if latest['MACD'] > latest['MACD_Signal'] else "死叉"
            summary += f"MACD: {macd_signal}\n"
        
        # 波动率
        if pd.notna(latest['Volatility_20D']):
            vol = latest['Volatility_20D'] * 100
            vol_level = "高" if vol > 30 else "中" if vol > 15 else "低"
            summary += f"年化波动率: {vol:.1f}% ({vol_level})\n"
        
        # 布林带
        if pd.notna(latest['BB_Position']):
            bb_pos = latest['BB_Position']
            if bb_pos > 0.8:
                bb_status = "接近上轨(超买区域)"
            elif bb_pos < 0.2:
                bb_status = "接近下轨(超卖区域)"
            else:
                bb_status = "中轨区域"
            summary += f"布林带: {bb_status}\n"
        
        return summary.strip()
        
    except Exception as e:
        logger.error(f"技术分析摘要失败: {e}")
        return f"{ticker}: 技术分析失败"

# =====================
# 6. LLM分析器
# =====================
class LLMAnalyzer:
    """LLM分析器"""
    
    def __init__(self, api_key: str, model_name: str):
        try:
            self.llm = ChatOpenAI(
                api_key=api_key,
                model=model_name,
                temperature=0.1,
                max_tokens=2000
            )
            logger.info(f"✅ LLM初始化成功: {model_name}")
        except Exception as e:
            logger.error(f"❌ LLM初始化失败: {e}")
            raise
    
    @functools.lru_cache(maxsize=32)
    def _cached_analyze(self, prompt_hash: str, prompt: str) -> str:
        """缓存的LLM调用"""
        try:
            response = self.llm.invoke([HumanMessage(content=prompt)])
            return response.content
        except Exception as e:
            logger.error(f"LLM调用失败: {e}")
            return f"分析失败: {str(e)}"
    
    def analyze_stock(self, stock_data: str) -> str:
        """股票综合分析"""
        try:
            # 应用文本切片
            sliced_data = slice_text_for_llm(stock_data, MAX_LLM_INPUT_CHARS)
            
            prompt = ANALYSIS_PROMPT.format(stock_data=sliced_data)
            prompt_hash = hashlib.md5(prompt.encode()).hexdigest()
            
            return self._cached_analyze(prompt_hash, prompt)
            
        except Exception as e:
            logger.error(f"股票分析失败: {e}")
            return f"分析失败: {str(e)}"
    
    def generate_signal(self, analysis_text: str) -> dict:
        """生成交易信号"""
        try:
            prompt = SIGNAL_PROMPT.format(analysis_text=analysis_text)
            prompt_hash = hashlib.md5(prompt.encode()).hexdigest()
            
            raw_response = self._cached_analyze(prompt_hash, prompt)
            
            # 解析JSON响应
            try:
                # 清理响应
                cleaned = raw_response.strip()
                if cleaned.startswith('```json'):
                    cleaned = cleaned[7:]
                if cleaned.endswith('```'):
                    cleaned = cleaned[:-3]
                cleaned = cleaned.strip()
                
                signal_data = json.loads(cleaned)
                
                # 标准化信号
                signal = signal_data.get('signal', 'Hold').strip()
                if signal.lower() in ['buy', '买入']:
                    signal = 'Buy'
                elif signal.lower() in ['sell', '卖出']:
                    signal = 'Sell'
                else:
                    signal = 'Hold'
                
                return {
                    "signal": signal,
                    "reason": signal_data.get('reason', '未提供理由')[:100],
                    "risk": signal_data.get('risk', '未识别风险')[:100],
                    "target_price": signal_data.get('target_price'),
                    "confidence": "high",
                    "raw_response": raw_response
                }
                
            except json.JSONDecodeError:
                logger.warning("JSON解析失败，使用文本分析")
                
                # 从文本提取信号
                signal = 'Hold'
                if any(word in raw_response.lower() for word in ['buy', '买入', 'purchase']):
                    signal = 'Buy'
                elif any(word in raw_response.lower() for word in ['sell', '卖出']):
                    signal = 'Sell'
                
                return {
                    "signal": signal,
                    "reason": raw_response[:100],
                    "risk": "解析失败，请检查原始响应",
                    "target_price": None,
                    "confidence": "low",
                    "raw_response": raw_response
                }
                
        except Exception as e:
            logger.error(f"信号生成失败: {e}")
            return {
                "signal": "Hold",
                "reason": f"系统错误: {str(e)}",
                "risk": "系统不稳定",
                "target_price": None,
                "confidence": "none",
                "raw_response": ""
            }

# =====================
# 7. 回测系统
# =====================
def simple_backtest(signal_data: dict, price_df: pd.DataFrame) -> dict:
    """简化回测"""
    try:
        if not VBT_AVAILABLE:
            return {"error": "VectorBT未安装，回测功能不可用"}
        
        if price_df.empty:
            return {"error": "价格数据不足"}
        
        # 创建简单的买入持有策略进行对比
        initial_price = price_df['Close'].iloc[0]
        final_price = price_df['Close'].iloc[-1]
        buy_hold_return = (final_price / initial_price) - 1
        
        # 计算基本统计
        returns = price_df['Close'].pct_change().dropna()
        volatility = returns.std() * np.sqrt(252)
        
        result = {
            "buy_hold_return": float(buy_hold_return),
            "annualized_volatility": float(volatility),
            "sharpe_ratio": float(buy_hold_return / volatility) if volatility != 0 else 0,
            "max_drawdown": float((price_df['Close'] / price_df['Close'].cummax() - 1).min()),
            "total_days": len(price_df),
            "signal_generated": signal_data.get('signal', 'Hold'),
            "note": "简化回测：基于买入持有策略"
        }
        
        logger.info("简化回测完成")
        return result
        
    except Exception as e:
        logger.error(f"回测失败: {e}")
        return {"error": str(e)}

# =====================
# 8. 主要分析流程
# =====================
def analyze_single_stock(ticker: str) -> dict:
    """分析单个股票"""
    try:
        logger.info(f"🚀 开始分析: {ticker}")
        
        # 1. 初始化数据提供器
        data_provider = YFinanceDataProvider(ticker)
        
        # 2. 获取公司信息
        company_info = data_provider.get_company_info()
        if "error" in company_info:
            raise ValueError(f"无法获取公司信息: {company_info['error']}")
        
        # 3. 获取财务数据
        financial_data = data_provider.get_financial_data()
        key_metrics = data_provider.get_key_metrics()
        
        # 4. 获取价格数据
        price_df = data_provider.get_price_data(START_DATE, END_DATE)
        price_df = calculate_technical_indicators(price_df)
        
        # 5. 生成技术分析摘要
        technical_summary = generate_technical_summary(price_df, ticker)
        
        # 6. 准备LLM输入数据
        stock_data = prepare_stock_data(company_info, financial_data, key_metrics, technical_summary)
        
        # 7. LLM分析
        analyzer = LLMAnalyzer(OPENAI_API_KEY, MODEL_NAME)
        analysis_result = analyzer.analyze_stock(stock_data)
        
        # 8. 生成交易信号
        signal_result = analyzer.generate_signal(analysis_result)
        
        # 9. 回测
        backtest_result = simple_backtest(signal_result, price_df)
        
        # 10. 保存数据
        save_data(ticker, {
            'company_info': company_info,
            'key_metrics': key_metrics,
            'signal': signal_result,
            'analysis': analysis_result,
            'technical_summary': technical_summary
        }, price_df)
        
        # 11. 组装结果
        result = {
            "ticker": ticker,
            "timestamp": datetime.now().isoformat(),
            "company_info": company_info,
            "analysis": analysis_result,
            "technical_summary": technical_summary,
            "key_metrics": key_metrics,
            "signal": signal_result,
            "backtest": backtest_result,
            "data_stats": {
                "price_days": len(price_df),
                "metrics_count": len(key_metrics),
                "financial_reports": len(financial_data),
                "input_size": len(stock_data)
            }
        }
        
        logger.info(f"✅ {ticker} 分析完成: {signal_result['signal']}")
        return result
        
    except Exception as e:
        logger.error(f"❌ {ticker} 分析失败: {e}")
        return {
            "ticker": ticker,
            "timestamp": datetime.now().isoformat(),
            "error": str(e)
        }

def analyze_batch_stocks(tickers: List[str]) -> dict:
    """批量分析股票"""
    logger.info(f"🚀 开始批量分析: {len(tickers)} 个股票")
    
    results = {}
    signals = []
    
    for i, ticker in enumerate(tickers, 1):
        logger.info(f"📊 进度: {i}/{len(tickers)} - {ticker}")
        
        result = analyze_single_stock(ticker)
        results[ticker] = result
        
        if "error" not in result and "signal" in result:
            signals.append({
                "ticker": ticker,
                "signal": result["signal"]["signal"],
                "reason": result["signal"]["reason"],
                "confidence": result["signal"]["confidence"]
            })
    
    # 批量统计
    successful = [t for t, r in results.items() if "error" not in r]
    failed = [t for t, r in results.items() if "error" in r]
    
    signal_stats = {}
    for signal in signals:
        sig = signal["signal"]
        signal_stats[sig] = signal_stats.get(sig, 0) + 1
    
    batch_result = {
        "timestamp": datetime.now().isoformat(),
        "total_requested": len(tickers),
        "successful": len(successful),
        "failed": len(failed),
        "success_rate": len(successful) / len(tickers) if tickers else 0,
        "signal_distribution": signal_stats,
        "signals_summary": signals,
        "detailed_results": results,
        "successful_tickers": successful,
        "failed_tickers": failed
    }
    
    # 保存批量结果
    save_json(batch_result, SIGNAL_DIR / "batch_analysis.json")
    
    logger.info(f"✅ 批量分析完成: {len(successful)}/{len(tickers)} 成功")
    return batch_result

# =====================
# 9. 辅助函数
# =====================
def prepare_stock_data(company_info: dict, financial_data: dict, key_metrics: dict, technical_summary: str) -> str:
    """准备股票分析数据"""
    try:
        data_sections = []
        
        # 公司基本信息
        if company_info:
            data_sections.append("=== 公司基本信息 ===")
            data_sections.append(f"公司: {company_info.get('name', 'N/A')} ({company_info.get('symbol', 'N/A')})")
            data_sections.append(f"行业: {company_info.get('sector', 'N/A')} - {company_info.get('industry', 'N/A')}")
            data_sections.append(f"市值: {company_info.get('market_cap', 'N/A'):,}" if isinstance(company_info.get('market_cap'), (int, float)) else f"市值: {company_info.get('market_cap', 'N/A')}")
            data_sections.append(f"员工数: {company_info.get('employees', 'N/A'):,}" if isinstance(company_info.get('employees'), (int, float)) else f"员工数: {company_info.get('employees', 'N/A')}")
            data_sections.append("")
        
        # 关键财务指标
        if key_metrics:
            data_sections.append("=== 关键财务指标 ===")
            
            # 估值指标
            valuation = ["估值指标:"]
            for key in ['pe_ratio', 'forward_pe', 'pb_ratio', 'ps_ratio', 'peg_ratio']:
                if key in key_metrics:
                    valuation.append(f"  {key}: {key_metrics[key]}")
            data_sections.extend(valuation)
            
            # 盈利能力
            profitability = ["盈利能力:"]
            for key in ['roe', 'roa', 'gross_margin', 'operating_margin', 'net_margin']:
                if key in key_metrics:
                    profitability.append(f"  {key}: {key_metrics[key]}")
            data_sections.extend(profitability)
            
            # 财务健康
            health = ["财务健康:"]
            for key in ['debt_to_equity', 'current_ratio', 'quick_ratio']:
                if key in key_metrics:
                    health.append(f"  {key}: {key_metrics[key]}")
            data_sections.extend(health)
            
            # 成长性
            growth = ["成长性:"]
            for key in ['revenue_growth', 'earnings_growth']:
                if key in key_metrics:
                    growth.append(f"  {key}: {key_metrics[key]}")
            data_sections.extend(growth)
            data_sections.append("")
        
        # 财务报表摘要
        if financial_data:
            data_sections.append("=== 财务报表摘要 ===")
            for report_name, report_data in financial_data.items():
                if hasattr(report_data, 'head') and not report_data.empty:
                    data_sections.append(f"{report_name}:")
                    # 只显示最重要的几行数据
                    summary = report_data.head(8).to_string()
                    data_sections.append(summary)
                    data_sections.append("")
        
        # 技术分析
        data_sections.append("=== 技术分析 ===")
        data_sections.append(technical_summary)
        
        return "\n".join(data_sections)
        
    except Exception as e:
        logger.error(f"准备股票数据失败: {e}")
        return f"数据准备失败: {str(e)}"

def save_data(ticker: str, analysis_data: dict, price_df: pd.DataFrame):
    """保存分析数据"""
    try:
        # 保存分析结果
        save_json(analysis_data, SIGNAL_DIR / f"{ticker}_analysis.json")
        
        # 保存价格数据
        price_df.to_csv(PRICE_DIR / f"{ticker}_prices.csv")
        
        logger.debug(f"数据保存完成: {ticker}")
        
    except Exception as e:
        logger.error(f"保存数据失败: {e}")

def system_health_check() -> dict:
    """系统健康检查"""
    health = {
        "timestamp": datetime.now().isoformat(),
        "status": "healthy",
        "checks": {},
        "recommendations": []
    }
    
    try:
        # API密钥检查
        health["checks"]["openai_api_key"] = bool(OPENAI_API_KEY)
        if not OPENAI_API_KEY:
            health["recommendations"].append("设置OPENAI_API_KEY环境变量")
        
        # 依赖库检查
        dependencies = {
            "pandas": True,
            "numpy": True,
            "yfinance": True,
            "langchain": True
        }
        
        try:
            import pandas, numpy, yfinance
            from langchain_openai import ChatOpenAI
        except ImportError as e:
            dependencies[str(e).split("'")[1]] = False
        
        health["checks"]["dependencies"] = dependencies
        
        # VectorBT检查
        health["checks"]["vectorbt"] = VBT_AVAILABLE
        if not VBT_AVAILABLE:
            health["recommendations"].append("安装vectorbt以启用完整回测功能")
        
        # 网络连接检查
        try:
            test_stock = yf.Ticker("AAPL")
            test_info = test_stock.info
            health["checks"]["yfinance_api"] = bool(test_info)
        except Exception:
            health["checks"]["yfinance_api"] = False
            health["recommendations"].append("检查网络连接，无法访问Yahoo Finance")
        
        # 目录检查
        health["checks"]["directories"] = all(d.exists() for d in [DATA_DIR, PRICE_DIR, SIGNAL_DIR])
        
        # 综合评估
        failed_critical = []
        if not health["checks"]["openai_api_key"]:
            failed_critical.append("OpenAI API Key")
        if not health["checks"]["yfinance_api"]:
            failed_critical.append("YFinance API")
        
        if failed_critical:
            health["status"] = "error"
            health["critical_failures"] = failed_critical
        elif health["recommendations"]:
            health["status"] = "warning"
        
    except Exception as e:
        health["status"] = "error"
        health["error"] = str(e)
    
    return health

# =====================
# 10. CLI入口
# =====================
def main():
    parser = argparse.ArgumentParser(
        description="优化版LLM财务分析代理",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
使用示例:
  单股分析: python financial_agent.py --mode single --ticker AAPL
  批量分析: python financial_agent.py --mode batch --tickers AAPL MSFT GOOGL AMZN
  健康检查: python financial_agent.py --health-check

环境变量:
  OPENAI_API_KEY: OpenAI API密钥 (必需)
  MODEL_NAME: 模型名称 (可选，默认gpt-4o-mini)
  START_DATE: 分析开始日期 (可选，默认2020-01-01)
  MAX_LLM_INPUT_CHARS: LLM输入字符限制 (可选，默认150000)

特点:
  ✅ 完全基于YFinance，无需SEC数据
  ✅ 智能数据切片，适应LLM输入限制
  ✅ 结合基本面和技术面分析
  ✅ 简化依赖，提高稳定性
        """
    )
    
    parser.add_argument('--mode', choices=['single', 'batch'], default='single', help='分析模式')
    parser.add_argument('--ticker', default='AAPL', help='单股票代码')
    parser.add_argument('--tickers', nargs='+', help='批量股票代码')
    parser.add_argument('--health-check', action='store_true', help='系统健康检查')
    parser.add_argument('--output', help='输出文件路径')
    parser.add_argument('--verbose', action='store_true', help='详细日志')
    
    args = parser.parse_args()
    
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    try:
        # 健康检查
        if args.health_check:
            ensure_dirs()
            health = system_health_check()
            print("=" * 50)
            print("🏥 系统健康检查")
            print("=" * 50)
            print(json.dumps(health, ensure_ascii=False, indent=2))
            
            if health["status"] == "error":
                print("\n❌ 系统存在严重问题")
                sys.exit(1)
            elif health["status"] == "warning":
                print("\n⚠️  系统可用，但建议解决警告问题")
            else:
                print("\n✅ 系统状态良好")
            return
        
        # 配置验证
        validate_config()
        ensure_dirs()
        
        # 执行分析
        if args.mode == 'single':
            result = analyze_single_stock(args.ticker)
        else:
            tickers = args.tickers or ['AAPL', 'MSFT', 'GOOGL']
            result = analyze_batch_stocks(tickers)
        
        # 输出结果
        output_json = json.dumps(result, ensure_ascii=False, indent=2, default=str)
        
        if args.output:
            output_path = Path(args.output)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write(output_json)
            print(f"📄 结果已保存: {output_path}")
        else:
            print(output_json)
        
        # 状态检查
        if isinstance(result, dict) and "error" in result:
            logger.error("❌ 分析过程发生错误")
            sys.exit(1)
        
        logger.info("🎉 分析完成")
        
    except KeyboardInterrupt:
        logger.info("⚠️  用户中断")
        sys.exit(1)
    except Exception as e:
        logger.error(f"❌ 程序异常: {e}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        sys.exit(1)

if __name__ == '__main__':
    main()