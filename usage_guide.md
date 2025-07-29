### 1. 环境准备

```bash
# 创建虚拟环境
python -m venv financial_agent_env
source financial_agent_env/bin/activate  # Linux/Mac
# 或 financial_agent_env\Scripts\activate  # Windows

# 安装依赖
pip install -r requirements.txt

# 或者最小安装 (仅核心功能)
pip install pandas numpy yfinance langchain langchain-openai
```

### 2. 配置API密钥

```bash
# 方法1: 环境变量
export OPENAI_API_KEY="sk-your-openai-key-here"

# 方法2: .env文件
echo "OPENAI_API_KEY=sk-your-openai-key-here" > .env
```

### 3. 系统检查

```bash
# 验证系统配置
python financial_agent.py --health-check
```

### 4. 开始分析

```bash
# 单股票分析
python financial_agent.py --mode single --ticker AAPL

# 批量分析（含智能排名）
python financial_agent.py --mode batch --tickers AAPL MSFT GOOGL

# 显示排名摘要
python financial_agent.py --mode batch --tickers AAPL MSFT GOOGL --show-ranking

# 保存结果
python financial_agent.py --mode single --ticker AAPL --output results.json
```

## 新增排名功能

### 智能股票排名

批量分析时自动生成股票综合排名：

- **排名标准**: 基本面(40%) + 技术面(30%) + 风险(30%)
- **投资组合**: 自动推荐配置比例
- **市场展望**: 基于分析结果的投资建议

```bash
# 生成排名分析
python financial_agent.py --mode batch --tickers AAPL MSFT GOOGL AMZN --show-ranking
```

**输出示例:**
```
股票排名分析结果
 AAPL   | 评分: 9.2/10 | 基本面优秀，技术面强劲
 MSFT   | 评分: 8.8/10 | 估值合理，成长稳定
 GOOGL  | 评分: 8.1/10 | 创新能力强，估值略高

推荐投资组合配置:
AAPL: 40.0%  MSFT: 30.0%  GOOGL: 20.0%  现金: 10.0%
```

## 功能展示

### 单股票分析示例

```bash
python financial_agent.py --mode single --ticker AAPL --verbose
```

**输出示例：**
```json
{
  "ticker": "AAPL",
  "timestamp": "2025-01-26T10:30:00",
  "company_info": {
    "name": "Apple Inc.",
    "sector": "Technology",
    "market_cap": 3000000000000
  },
  "analysis": "基本面分析：苹果公司财务状况良好...",
  "signal": {
    "signal": "Buy",
    "reason": "估值合理，增长稳定，技术面向好",
    "risk": "市场波动和竞争加剧",
    "confidence": "high"
  },
  "backtest": {
    "buy_hold_return": 0.156,
    "sharpe_ratio": 1.23,
    "max_drawdown": -0.089
  }
}
```

### 批量分析示例

```bash
python financial_agent.py --mode batch --tickers AAPL MSFT GOOGL AMZN
```

**输出包含：**
- 每个股票的详细分析
- 信号分布统计
- 成功率报告
- 批量摘要

## 配置选项

### 环境变量

| 变量名 | 描述 | 默认值 | 必需 |
|--------|------|--------|------|
| `OPENAI_API_KEY` | OpenAI API密钥 | - | ✅ |
| `MODEL_NAME` | LLM模型名称 | `gpt-4o-mini` | ❌ |
| `START_DATE` | 分析开始日期 | `2020-01-01` | ❌ |
| `MAX_LLM_INPUT_CHARS` | LLM输入字符限制 | `150000` | ❌ |
| `FEE_BPS` | 交易费用(基点) | `10` | ❌ |

### CLI参数

```bash
python financial_agent.py [选项]

选项:
  --mode {single,batch}    分析模式
  --ticker TICKER          单股票代码 (默认: AAPL)
  --tickers TICKER [...]   批量股票代码
  --health-check          系统健康检查
  --output FILE           输出文件路径
  --verbose               详细日志输出
  --show-ranking          显示股票排名摘要 (仅批量模式)
  --help                  显示帮助信息
```

## 输出文件

系统会在以下目录保存分析结果：

```
data/
├── prices/              # 价格数据 (CSV格式)
├── signals/             # 分析结果 (JSON格式)
└── backtest/            # 回测结果
```

### 主要输出文件

- `{TICKER}_analysis.json` - 详细分析结果
- `{TICKER}_prices.csv` - 价格和技术指标数据
- `batch_analysis.json` - 批量分析汇总
- `financial_agent.log` - 系统日志

## 系统架构

```
YFinance数据获取 → 技术指标计算 → LLM智能分析 → 信号生成 → 股票排名 → 回测验证
```

### 核心模块

1. **YFinanceDataProvider** - 统一数据获取
   - 公司基本信息
   - 财务报表 (利润表、资产负债表、现金流)
   - 关键财务指标 (PE、ROE等)
   - 历史价格数据

2. **技术指标计算**
   - 移动平均线 (SMA、EMA)
   - 动量指标 (RSI、MACD)
   - 波动率指标 (布林带)
   - 成交量分析

3. **LLMAnalyzer** - AI分析引擎
   - 基本面分析
   - 技术面解读
   - 投资建议生成
   - 风险识别
   - **智能股票排名**

4. **排名系统** - 投资组合优化
   - 综合评分 (基本面40% + 技术面30% + 风险30%)
   - 投资组合配置建议
   - 市场环境分析

5. **回测系统** - 策略验证
   - 买入持有对比
   - 风险指标计算
   - 夏普比率等

## 故障排除

### 常见问题

#### 1. OpenAI API错误
```bash
# 检查API密钥
echo $OPENAI_API_KEY

# 检查余额和配额限制
# 尝试切换模型
export MODEL_NAME="gpt-3.5-turbo"
```

#### 2. YFinance连接失败
```bash
# 更新yfinance
pip install --upgrade yfinance

# 检查网络连接
ping finance.yahoo.com

# 使用代理 (如果需要)
export HTTP_PROXY=http://proxy:8080
```

#### 3. VectorBT安装失败
```bash
# VectorBT是可选的，可以跳过
pip install --no-deps vectorbt

# 或者完全跳过，系统会自动禁用回测功能
```

#### 4. 内存不足
```bash
# 减少LLM输入大小
export MAX_LLM_INPUT_CHARS="100000"

# 分批处理大量股票
python financial_agent.py --mode batch --tickers AAPL MSFT
python financial_agent.py --mode batch --tickers GOOGL AMZN
```

### 系统要求

- **Python**: 3.8+
- **内存**: 建议4GB+
- **网络**: 稳定的互联网连接
- **平台**: Windows/macOS/Linux

## 📈 性能优化建议

### 1. 成本控制
```bash
# 使用更便宜的模型
export MODEL_NAME="gpt-4o-mini"

# 利用缓存避免重复分析
# 批量分析时控制并发数量
```

### 2. 速度优化
```bash
# 减少历史数据范围
export START_DATE="2022-01-01"

# 使用SSD存储提高I/O性能
# 在高配置机器上运行批量任务
```

### 3. 稳定性提升
```bash
# 使用虚拟环境隔离依赖
# 定期更新依赖库
pip install --upgrade -r requirements.txt

# 监控日志文件
tail -f financial_agent.log
```

欢迎提交Issue和Pull Request！

### 开发环境设置
```bash
git clone <repository>
cd financial-agent
python -m venv dev_env
source dev_env/bin/activate
pip install -r requirements.txt
pip install pytest black flake8

# 运行测试
pytest tests/

# 代码格式化
black financial_agent.py

# 代码检查
flake8 financial_agent.py
```
## 🙏 致谢

- [YFinance](https://github.com/ranaroussi/yfinance) - 提供可靠的金融数据
- [LangChain](https://github.com/langchain-ai/langchain) - LLM集成框架
- [VectorBT](https://github.com/polakowo/vectorbt) - 回测框架
- [OpenAI](https://openai.com) - 强大的LLM能力

---

**📧 联系方式**
如有问题或建议，请提交Issue或通过邮件联系。

**⭐ 如果这个项目对你有帮助，请给个Star！**
