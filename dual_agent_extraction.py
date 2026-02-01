"""
双Agent并行抽取架构
- Agent A (宏观分析师): 负责定性分析 - TRL、P_Score、应用场景、问题、方法
- Agent B (数据挖掘师): 负责定量抽取 - results_extract 部分
- Merger: 合并两者的 JSON 结果

支持的 API 提供商：
- DeepSeek: 设置 DEEPSEEK_API_KEY 环境变量
- DashScope (阿里云通义千问): 设置 DASHSCOPE_API_KEY 环境变量
"""

import argparse
import json
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional

import requests


# ============================================================================
# API 提供商配置
# ============================================================================

class APIProvider(Enum):
    """支持的 API 提供商"""
    DEEPSEEK = "deepseek"
    DASHSCOPE = "dashscope"


@dataclass
class APIConfig:
    """API 配置"""
    provider: APIProvider
    api_key: str
    model: str
    endpoint: str
    supports_json_mode: bool = True
    
    @classmethod
    def from_provider(
        cls, 
        provider: APIProvider, 
        api_key: str, 
        model: Optional[str] = None
    ) -> "APIConfig":
        """根据提供商创建配置"""
        if provider == APIProvider.DEEPSEEK:
            return cls(
                provider=provider,
                api_key=api_key,
                model=model or "deepseek-reasoner",
                endpoint="https://api.deepseek.com/v1/chat/completions",
                supports_json_mode=True
            )
        elif provider == APIProvider.DASHSCOPE:
            return cls(
                provider=provider,
                api_key=api_key,
                model=model or "qwen-plus",
                endpoint="https://dashscope.aliyuncs.com/compatible-mode/v1/chat/completions",
                supports_json_mode=True
            )
        else:
            raise ValueError(f"不支持的提供商: {provider}")


# ============================================================================
# Agent A: 宏观分析师 - 专注语义理解、商业价值、技术成熟度
# ============================================================================

SYSTEM_PROMPT_A = '''
你是"产业问题解决力智能评价模型"的**宏观分析师**，专门从高校论文中抽取**定性信息**，包括：
- 研究问题与背景 (problem)
- 技术方法与创新点 (method)
- 应用场景 (application)
- 技术就绪度 (TRL)
- 专利-权益要求评分 (P_score)

### 核心原则

## 0. 语言一致性 (Language Consistency)
- **原文保留**：技术术语、实体名必须**严格保持论文原文语言**。
- **禁止翻译**：如果论文是英文，输出的 JSON 值必须是英文。
- **解释性字段**：仅 `rationale`, `why_industry_matters`, `p_reason` 等解释字段允许使用中文。

## 1. 证据优先 (Evidence First)
- **所有抽取必须有原文支撑**：严禁基于"领域常识"或"合理推测"进行提取。
- **Quote 必须精准**：evidence.quote 必须是论文中的原句或连续短语。

## 2. 应用场景严格性
- **禁止过度推断**：仅提取论文**明确提及**的应用场景。
- **时间维度划分**：若论文未明确区分近/中/远期，可根据技术成熟度合理推断，但必须在 evidence 中注明。

## 3. TRL 评估 (精确分级)
请严格对照以下定义，判断技术所处的**具体等级**（TRL 1 至 TRL 9）。

**一、基础研究阶段 (发现原理)**
- **TRL 1**：基本原理被观察到并被报告。
- **TRL 2**：技术概念或应用方案的形成。
- **TRL 3**：关键功能的概念验证（实验室）。(特征：**公开学术数据集**，Demo)

**二、技术开发阶段 (样机验证)**
- **TRL 4**：实验室环境下的组件验证。(特征：原型系统)
- **TRL 5**：相关环境下的组件/样机验证。(特征：**企业脱敏数据**，离线测试)
- **TRL 6**：相关环境下的系统级样机演示。(特征：工业现场，**半实物仿真**)

**三、系统验证阶段 (产业落地)**
- **TRL 7**：真实操作环境下的系统演示。(特征：**工厂产线/真实电网**，解决最后一公里)
- **TRL 8**：最终系统完成并经过测试定型。(特征：芯片流片，**批量测试**)
- **TRL 9**：实际系统在真实任务中被证明。(特征：**商业化规模应用**，采购目录)

**判定原则：**
1. 若证据处于两个等级之间，选择**较低**的那个等级。
2. 必须在 `rationale` 中明确指出判定依据。

## 4. P_score 专利-权益要求评分
- **P1 边缘性改进**：针对局部问题的小改动，不触及核心逻辑。关键词：外观优化、偶发问题、易替代。
- **P2 常规性优化**：沿用已知技术路线，做稳步性能提升，未改变架构。
- **P3 核心瓶颈突破**：解决行业共性痛点，指标显著改善（如功耗下降 ≥20%），触及核心逻辑。
- **P4 系统性创新**：提出系统级新方案，具备强技术壁垒，可支撑完整国产可控链条。
- **P5 颠覆式/引领性**：开辟全新技术赛道，成为行业基准或打破封锁。
输出 `p_score`（1-5）并在 `p_reason` 中用中文简述依据与对应关键特征。
'''

PROMPT_TEMPLATE_A = """
作为**宏观分析师**，请从给定的论文（Markdown 全文）中抽取定性信息。

**重要**：如果论文是英文，技术术语必须保持英文，不要翻译。

---

## 论文内容

{paper_content}

---

## 抽取任务

A. 问题抽取（problem）
1) **研究对象 (object)**：系统/算法对象是什么？
2) **应用场景 (scenario)**：面向什么具体行业或场景？
3) **约束条件 (constraints)**：关注数据隐私、实时性、算力成本、硬件限制等。
4) **关键难题 (key_questions)**：该行业痛点是什么？

B. 方法抽取（method）
1) **技术路线 (technical_route)**：主要步骤或架构。
2) **创新点 (innovations)**：核心贡献。
3) **假设与依赖 (assumptions_or_dependencies)**：算法运行依赖什么假设？

C. 应用场景（application）
- 从原文内容中提取出技术的近期、中期、远期应用方向。

D. 技术就绪度（TRL）
- **Level**："TRL 1" | ... | "TRL 9"
- **Rationale**：引用原文解释判定依据。
- **Uncertain**：如果证据不足以确信等级，设为 true。

E. 专利-权益要求评分（P_score）
- 输出 `p_score`（P1-P5）以及 `p_reason`。

## JSON 输出结构

```json
{
  "paper_title": "",
  "domain": "",
  "language": "",
  "problem": {
    "object": [{"name": "", "evidence": {"section": "", "quote": ""}}],
    "scenario": [{"name": "", "industry": "", "evidence": {"section": "", "quote": ""}}],
    "constraints": [{"constraint": "", "type": "", "evidence": {"section": "", "quote": ""}}],
    "key_questions": [{
      "question": "",
      "why_industry_matters": "",
      "evidence": {"section": "", "quote": ""}
    }]
  },
  "method": {
    "technical_route": [{"step": "", "evidence": {"section": "", "quote": ""}}],
    "innovations": [{"point": "", "category": "", "evidence": {"section": "", "quote": ""}}],
    "assumptions_or_dependencies": [{"item": "", "evidence": {"section": "", "quote": ""}}]
  },
  "application": {
    "near_term": [{"direction": "", "evidence": {"section": "", "quote": ""}}],
    "mid_term": [{"direction": "", "evidence": {"section": "", "quote": ""}}],
    "long_term": [{"direction": "", "evidence": {"section": "", "quote": ""}}]
  },
  "trl": {
    "level": "",
    "rationale": "",
    "evidence": [{"section": "", "quote": ""}],
    "uncertain": ""
  },
  "p_score": "",
  "p_reason": ""
}
```
## 补充规则
- 最终仅输出 JSON，不得输出任何说明文本。
- 确保 JSON 格式合法，确保没有尾部逗号错误，反斜杠转义错误等问题。
"""


# ============================================================================
# Agent B: 数据挖掘师 - 专注数字、表格、定量对比
# ============================================================================

SYSTEM_PROMPT_B = '''
你是"产业问题解决力智能评价模型"的**数据挖掘师**，专门从高校论文中抽取**定量数据**，包括：
- 性能指标 (Metrics)
- 实验结果 (Experiment Results)
- 对比方法 (Comparison Methods)

你不关心商业价值或应用场景，只关心**精确的数字、单位、实验条件**。

### 核心原则

## 0. 语言一致性 (Language Consistency) - HIGHEST PRIORITY
- **原文保留**：指标名、单位、对比算法名称必须**严格保持论文原文语言**。
- **禁止翻译**：如果论文是英文，严禁将 "Accuracy" 翻译为 "准确率"。

## 1. 数据精确性 (Precision of Data) - CRITICAL
- **Value 必须是量化数据**：严禁在 `value` 字段填写 "improved", "high", "low", "faster" 等形容词。
- **必须提取具体的数字**：如 "95.4", "15%", "120ms"。
- **无数字则不提取**：如果原文只提到"性能提升"但未给出具体数字，**请勿**将其作为一条 Metric 提取。

## 2. Unit 必须明确
- 如 "%", "ms", "FPS", "Joules", "dB", "mW", "GHz" 等。

## 3. 条件必须完整
- experimental_setting 必须包含供电电压、负载条件、温度、频率等关键参数。
- 示例："4.2V supply, 8Ω load, -60dBFS input"

## 4. 禁止虚构对比
- `comparison_methods` 仅在原文明确列出对比算法的**具体数值**时填写。
- 如果原文仅说 "outperforms state-of-the-art" 但没列出具体算法和数值，该列表必须为 `[]`。
- **必须验证**：comparison_methods 中的每个方法是否都有原文明确的具体数值？

## 5. 层级结构
- 结果需要按 proposed_methods -> experiment_results -> metrics 层级结构提取。
- 严禁扁平化处理。

## 6. 表格优先
- 优先从论文中的表格（Table）提取数据，表格数据通常更精确。
- 注意区分不同实验条件下的同一指标。
'''

PROMPT_TEMPLATE_B = """
作为**数据挖掘师**，请从论文中抽取所有定量实验结果。

**重要**：
1. 如果论文是英文，所有指标名、方法名必须保持英文。
2. 只提取有具体数字的指标，不要提取定性描述。
3. comparison_methods 必须有原文具体数值支持，否则保持为空数组 []。

---

## 论文内容

{paper_content}

---

## 抽取任务

从论文中提取 **results_extract** 部分。

**⚠️ 关键：必须严格按照以下层级结构输出，禁止扁平化！**

```
results_extract
└── proposed_methods (数组，每个元素是一个对象)
    └── method_name (字符串)
    └── experiment_results (数组，每个元素是一个对象)
        └── task_or_scenario (字符串)
        └── metrics (数组，每个元素是一个对象)
            └── metric_name (字符串)
            └── value (字符串，必须是数字)
            └── unit (字符串)
            └── experimental_setting (字符串)
            └── evidence (对象)
            └── comparison_methods (数组，每个元素是一个对象)
```

### 各层级说明：

1) **proposed_methods**：论文提出的方法/模型/芯片名称。
   - 类型：对象数组 `[{...}, {...}]`，**不是**字符串数组
   
2) **experiment_results**：该方法的具体实验任务或场景。
   - 类型：对象数组，嵌套在每个 proposed_method 内部
   
3) **metrics**：性能指标。
   - 类型：对象数组，嵌套在每个 experiment_result 内部
   - **value 必须是具体数字**（如 "16", "5.7", "32"），禁止填写 "improved" 等形容词

4) **comparison_methods**：对比方法。
   - 类型：对象数组，嵌套在每个 metric 内部
   - 如果没有具体对比数值，必须为空数组 `[]`

## JSON 输出结构（严格遵循此格式）

```json
{
  "results_extract": {
    "proposed_methods": [
      {
        "method_name": "方法名称",
        "experiment_results": [
          {
            "task_or_scenario": "实验任务描述",
            "metrics": [
              {
                "metric_name": "指标名",
                "value": "具体数字",
                "unit": "单位",
                "experimental_setting": "实验条件",
                "evidence": {"section": "章节", "quote": "原文引用"},
                "comparison_methods": [
                  {
                    "method_name": "对比方法名",
                    "value": "具体数字",
                    "experimental_setting": "实验条件",
                    "evidence": {"section": "章节", "quote": "原文引用"}
                  }
                ]
              }
            ]
          }
        ]
      }
    ]
  }
}
```

## 质量检查清单（输出前自检）

- [ ] proposed_methods 是**对象数组**吗？（不是字符串数组如 `["Zen 5"]`）
- [ ] experiment_results 嵌套在 proposed_methods 内部吗？
- [ ] metrics 嵌套在 experiment_results 内部吗？
- [ ] 每个 value 都是具体数字吗？（不是 "improved" 或 "high"）
- [ ] 每个 comparison_methods 中的方法都有原文具体数值支持吗？

## 补充规则
- 最终仅输出 JSON，不得输出任何说明文本。
- 确保 JSON 格式合法，确保没有尾部逗号错误，反斜杠转义错误等问题。
- **再次强调：严格保持层级嵌套结构，不要扁平化！**
"""


# ============================================================================
# Agent 类定义
# ============================================================================

class BaseAgent:
    """基础 Agent 类，支持多个 API 提供商"""
    
    def __init__(
        self, 
        config: APIConfig,
        name: str = "BaseAgent"
    ) -> None:
        self.config = config
        self.name = name
    
    def _call_api(self, system_prompt: str, user_prompt: str) -> Dict[str, Any]:
        """调用 API 并返回解析后的 JSON"""
        try:
            # 构建请求体
            request_body = {
                "model": self.config.model,
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                "temperature": 0.1,
            }
            
            # 如果支持 JSON 模式，添加 response_format
            if self.config.supports_json_mode:
                request_body["response_format"] = {"type": "json_object"}
            
            response = requests.post(
                self.config.endpoint,
                headers={
                    "Authorization": f"Bearer {self.config.api_key}",
                    "Content-Type": "application/json"
                },
                json=request_body,
                timeout=180,
            )
            response.raise_for_status()
            content = response.json()["choices"][0]["message"]["content"].strip()
            
            # 清理可能存在的 Markdown 标记
            if content.startswith("```"):
                lines = content.splitlines()
                if lines[0].startswith("```"): 
                    lines = lines[1:]
                if lines and lines[-1].strip() == "```": 
                    lines = lines[:-1]
                content = "\n".join(lines).strip()
            
            return json.loads(content)
        except json.JSONDecodeError as e:
            print(f"[{self.name}] JSON 解析错误: {e}")
            return {"error": f"JSON parse error: {e}", "raw": content if 'content' in locals() else ""}
        except Exception as e:
            print(f"[{self.name}] API 调用错误: {e}")
            return {"error": str(e)}


class MacroAnalystAgent(BaseAgent):
    """Agent A: 宏观分析师 - 负责定性分析"""
    
    def __init__(self, config: APIConfig) -> None:
        super().__init__(config, name="宏观分析师")
    
    def extract(self, paper_content: str) -> Dict[str, Any]:
        """抽取定性信息"""
        user_prompt = PROMPT_TEMPLATE_A.replace("{paper_content}", paper_content)
        print(f"  [{self.name}] 开始分析... (模型: {self.config.model})")
        result = self._call_api(SYSTEM_PROMPT_A, user_prompt)
        print(f"  [{self.name}] 分析完成")
        return result


class DataMinerAgent(BaseAgent):
    """Agent B: 数据挖掘师 - 负责定量抽取"""
    
    def __init__(self, config: APIConfig) -> None:
        super().__init__(config, name="数据挖掘师")
    
    def extract(self, paper_content: str) -> Dict[str, Any]:
        """抽取定量数据"""
        user_prompt = PROMPT_TEMPLATE_B.replace("{paper_content}", paper_content)
        print(f"  [{self.name}] 开始抽取... (模型: {self.config.model})")
        result = self._call_api(SYSTEM_PROMPT_B, user_prompt)
        print(f"  [{self.name}] 抽取完成")
        return result


# ============================================================================
# Merger: 合并器
# ============================================================================

class ResultMerger:
    """合并两个 Agent 的结果"""
    
    @staticmethod
    def merge(
        qualitative_result: Dict[str, Any], 
        quantitative_result: Dict[str, Any],
        source_filename: str
    ) -> Dict[str, Any]:
        """
        合并定性和定量结果
        
        Args:
            qualitative_result: Agent A 的定性分析结果
            quantitative_result: Agent B 的定量抽取结果
            source_filename: 源文件名
        
        Returns:
            合并后的完整 JSON
        """
        # 检查是否有错误
        has_error_a = "error" in qualitative_result
        has_error_b = "error" in quantitative_result
        
        # 基础结构来自定性分析
        if has_error_a:
            merged = {
                "paper_title": "",
                "domain": "",
                "language": "",
                "problem": {},
                "method": {},
                "application": {},
                "trl": {},
                "p_score": "",
                "p_reason": "",
                "_qualitative_error": qualitative_result.get("error", "")
            }
        else:
            merged = qualitative_result.copy()
        
        # 添加定量结果
        if has_error_b:
            merged["results_extract"] = {"proposed_methods": []}
            merged["_quantitative_error"] = quantitative_result.get("error", "")
        else:
            # 从定量结果中提取 results_extract
            merged["results_extract"] = quantitative_result.get(
                "results_extract", 
                {"proposed_methods": []}
            )
        
        # 添加元信息
        merged["source_filename"] = source_filename
        merged["extraction_timestamp"] = datetime.now().isoformat()
        merged["extraction_mode"] = "dual_agent_parallel"
        
        return merged


# ============================================================================
# 主协调器
# ============================================================================

class DualAgentOrchestrator:
    """双 Agent 并行抽取协调器"""
    
    def __init__(
        self, 
        config_a: APIConfig,
        config_b: Optional[APIConfig] = None
    ) -> None:
        """
        初始化协调器
        
        Args:
            config_a: Agent A (宏观分析师) 的 API 配置
            config_b: Agent B (数据挖掘师) 的 API 配置，如果为 None 则使用 config_a
        """
        self.agent_a = MacroAnalystAgent(config_a)
        self.agent_b = DataMinerAgent(config_b or config_a)
        self.merger = ResultMerger()
    
    def extract(self, paper_content: str, source_filename: str) -> Dict[str, Any]:
        """
        并行执行两个 Agent 的抽取任务
        
        Args:
            paper_content: 论文 Markdown 内容
            source_filename: 源文件名
        
        Returns:
            合并后的完整抽取结果
        """
        results = {}
        
        # 使用线程池并行执行
        with ThreadPoolExecutor(max_workers=2) as executor:
            future_a = executor.submit(self.agent_a.extract, paper_content)
            future_b = executor.submit(self.agent_b.extract, paper_content)
            
            # 等待结果
            for future in as_completed([future_a, future_b]):
                if future == future_a:
                    results["qualitative"] = future.result()
                else:
                    results["quantitative"] = future.result()
        
        # 合并结果
        merged = self.merger.merge(
            results["qualitative"],
            results["quantitative"],
            source_filename
        )
        
        return merged


# ============================================================================
# 辅助函数
# ============================================================================

def get_api_config(
    provider: str = "auto",
    model: Optional[str] = None
) -> APIConfig:
    """
    根据环境变量和参数获取 API 配置
    
    Args:
        provider: API 提供商 ("deepseek", "dashscope", "auto")
        model: 指定模型名，为 None 时使用默认模型
    
    Returns:
        APIConfig 实例
    """
    if provider == "auto":
        # 自动检测可用的 API Key
        if os.environ.get("DEEPSEEK_API_KEY"):
            provider = "deepseek"
        elif os.environ.get("DASHSCOPE_API_KEY"):
            provider = "dashscope"
        else:
            raise EnvironmentError(
                "未找到 API Key，请设置 DEEPSEEK_API_KEY 或 DASHSCOPE_API_KEY 环境变量"
            )
    
    if provider == "deepseek":
        api_key = os.environ.get("DEEPSEEK_API_KEY")
        if not api_key:
            raise EnvironmentError("请设置环境变量 DEEPSEEK_API_KEY")
        return APIConfig.from_provider(APIProvider.DEEPSEEK, api_key, model)
    
    elif provider == "dashscope":
        api_key = os.environ.get("DASHSCOPE_API_KEY")
        if not api_key:
            raise EnvironmentError("请设置环境变量 DASHSCOPE_API_KEY")
        return APIConfig.from_provider(APIProvider.DASHSCOPE, api_key, model)
    
    else:
        raise ValueError(f"不支持的提供商: {provider}")


# ============================================================================
# 主函数
# ============================================================================

def main() -> None:
    # 解析命令行参数
    parser = argparse.ArgumentParser(
        description="双 Agent 并行抽取系统",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  # 自动检测 API (优先使用 DeepSeek)
  python dual_agent_extraction.py

  # 指定使用 DashScope
  python dual_agent_extraction.py --provider dashscope

  # 指定模型
  python dual_agent_extraction.py --provider dashscope --model qwen-max

  # 两个 Agent 使用不同的提供商
  python dual_agent_extraction.py --provider-a deepseek --provider-b dashscope

环境变量:
  DEEPSEEK_API_KEY   - DeepSeek API 密钥
  DASHSCOPE_API_KEY  - 阿里云 DashScope API 密钥
        """
    )
    parser.add_argument(
        "--provider", "-p",
        choices=["deepseek", "dashscope", "auto"],
        default="auto",
        help="API 提供商 (默认: auto，自动检测)"
    )
    parser.add_argument(
        "--model", "-m",
        default=None,
        help="指定模型名 (默认: deepseek-reasoner 或 qwen-plus)"
    )
    parser.add_argument(
        "--provider-a",
        choices=["deepseek", "dashscope"],
        default=None,
        help="Agent A (宏观分析师) 使用的提供商"
    )
    parser.add_argument(
        "--provider-b",
        choices=["deepseek", "dashscope"],
        default=None,
        help="Agent B (数据挖掘师) 使用的提供商"
    )
    parser.add_argument(
        "--model-a",
        default=None,
        help="Agent A 使用的模型"
    )
    parser.add_argument(
        "--model-b",
        default=None,
        help="Agent B 使用的模型"
    )
    parser.add_argument(
        "--input-dir", "-i",
        default=None,
        help="输入目录 (默认: md/1.26/filtered)"
    )
    parser.add_argument(
        "--output-dir", "-o",
        default=None,
        help="输出目录 (默认: extraction_results/dual_agent)"
    )
    
    args = parser.parse_args()
    
    # 配置 API
    if args.provider_a or args.provider_b:
        # 两个 Agent 使用不同配置
        config_a = get_api_config(
            args.provider_a or args.provider,
            args.model_a or args.model
        )
        config_b = get_api_config(
            args.provider_b or args.provider,
            args.model_b or args.model
        )
    else:
        # 两个 Agent 使用相同配置
        config_a = get_api_config(args.provider, args.model)
        config_b = None  # 使用 config_a

    base_dir = Path(__file__).resolve().parent
    
    # 设置输入输出目录
    if args.input_dir:
        md_dir = Path(args.input_dir)
        if not md_dir.is_absolute():
            md_dir = base_dir / args.input_dir
    else:
        md_dir = base_dir / "md" / "1.26" / "test"
    
    if args.output_dir:
        results_dir = Path(args.output_dir)
        if not results_dir.is_absolute():
            results_dir = base_dir / args.output_dir
    else:
        results_dir = base_dir / "extraction_results" / "dual_agent" / "1.31" / "deepseek"
    
    results_dir.mkdir(parents=True, exist_ok=True)
    
    if not md_dir.exists():
        raise FileNotFoundError(f"目录未找到: {md_dir}")
        
    inputs = sorted(md_dir.glob("*.md"))
    if not inputs:
        raise FileNotFoundError(f"{md_dir} 目录下未找到 md 文件")

    orchestrator = DualAgentOrchestrator(config_a, config_b)
    
    print(f"{'=' * 60}")
    print(f"双 Agent 并行抽取系统")
    print(f"{'=' * 60}")
    print(f"Agent A (宏观分析师): {config_a.provider.value} / {config_a.model}")
    print(f"Agent B (数据挖掘师): {(config_b or config_a).provider.value} / {(config_b or config_a).model}")
    print(f"输入目录: {md_dir}")
    print(f"输出目录: {results_dir}")
    print(f"待处理文件数: {len(inputs)}")
    print(f"{'=' * 60}")
    
    for idx, file_path in enumerate(inputs, 1):
        print(f"\n[{idx}/{len(inputs)}] 正在处理: {file_path.name}")
        print("-" * 40)
        
        content = file_path.read_text(encoding="utf-8")
        
        # 并行抽取
        result = orchestrator.extract(content, file_path.name)
        
        # 保存结果
        output_path = results_dir / f"{file_path.stem}.json"
        output_path.write_text(
            json.dumps(result, ensure_ascii=False, indent=2),
            encoding="utf-8"
        )
        print(f"  [保存完成] {output_path.name}")
    
    print(f"\n{'=' * 60}")
    print(f"全部处理完成！结果已保存至: {results_dir}")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
