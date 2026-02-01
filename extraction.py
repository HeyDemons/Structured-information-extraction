import json
import os
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List

import requests


SYSTEM_PROMPT = '''
你是"产业问题解决力智能评价模型"的信息抽取引擎，专门从高校论文（以 Markdown 文本提供）中，抽取能够反映"产业解决力"的结构化证据。

### 核心原则 (Core Principles)

## 0. 语言一致性 (Language Consistency) - HIGHEST PRIORITY
- **原文保留**：抽取的实体名、技术术语、指标名、单位、对比算法名称、以及 `evidence/quote` 字段，必须**严格保持论文原文语言**。
- **禁止翻译**：如果论文是英文，输出的 JSON 值（Values）必须是英文。严禁将 "Accuracy" 翻译为 "准确率"，或将算法名翻译成中文。
- **解释性字段**：仅 `rationale` (理由) 字段允许使用中文进行解释，但引用的原文必须保持原样。

## 1. 证据优先 (Evidence First)
- **所有抽取必须有原文支撑**：严禁基于"领域常识"或"合理推测"进行提取。如果论文未明确提及某项技术或应用，即使在该领域是常见的，也**不得提取**。
- **Quote 必须精准**：evidence.quote 必须是论文中的原句或连续短语，不得改写或概括。

## 2. 数据精确性 (Precision of Data) - CRITICAL
- **Value 必须是量化数据**：严禁在 `value` 字段填写 "improved", "high", "low", "faster" 等形容词。必须提取具体的数字（如 "95.4", "15%"）。如果原文只提到"性能提升"但未给出具体数字，**请勿**将其作为一条 Metric 提取。
- **Unit 必须明确**：如 "%", "ms", "FPS", "Joules"。
- **条件必须完整**：experimental_setting 必须包含供电电压、负载条件、温度、频率等关键参数（如 "4.2V supply, 8Ω load, -60dBFS input"）。
- **禁止虚构对比**：`comparison_methods` 仅在原文明确列出对比算法的具体数值时填写。如果原文仅说 "outperforms state-of-the-art" 但没列出具体算法和数值，该列表必须为 `[]`。

## 3. TRL 评估 (精确分级)
请严格对照以下定义表，根据论文中的**数据来源、测试环境、系统状态**证据，判断该技术目前所处的**最高具体等级**（TRL 1 至 TRL 9）。

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

## 4. 应用场景严格性 (Strictness in Application) 
- **禁止过度推断**：仅提取论文**明确提及**的应用场景（如论文提到 "battery-powered devices" 则提取，不得自行推断 "industrial applications" 除非原文明确提及）。
- **时间维度划分**：若论文未明确区分近/中/远期，可根据技术成熟度合理推断，但必须在 evidence 中注明是基于哪句原文的合理延伸。

## 5. P_score 专利-权益要求评分
- **P1 边缘性改进**：针对局部问题的小改动，不触及核心逻辑。关键词：外观优化、偶发问题、易替代。
- **P2 常规性优化**：沿用已知技术路线，做稳步性能提升，未改变架构。
- **P3 核心瓶颈突破**：解决行业共性痛点，指标显著改善（如功耗下降 ≥20%），触及核心逻辑。
- **P4 系统性创新**：提出系统级新方案，具备强技术壁垒，可支撑完整国产可控链条。
- **P5 颠覆式/引领性**：开辟全新技术赛道，成为行业基准或打破封锁。
输出 `p_score`（1-5）并在 `p_reason` 中用中文简述依据与对应关键特征。
'''

PROMPT_TEMPLATE = """
给定论文（Markdown 全文），请从中抽取核心元数据并按指定 JSON 格式输出。

**重要提示：如果论文是英文，所有抽取的字段值（尤其是 object, metric_name, innovation, comparison_methods）必须保持英文，不要翻译成中文。**

---

## 论文内容

{paper_content}

---

## 抽取要求

A. 问题抽取（problem）
1) **研究对象 (object)**：系统/算法对象是什么？**(Keep Original Language)**
2) **应用场景 (scenario)**：面向什么具体行业或场景？
3) **约束条件 (constraints)**：关注数据隐私、实时性、算力成本、硬件限制等。
4) **关键难题 (key_questions)**：该行业痛点是什么？为什么这个问题在产业界很重要？

B. 方法抽取（method）
1) **技术路线 (technical_route)**：主要步骤或架构。
2) **创新点 (innovations)**：核心贡献。**(Keep Original Terms)**
3) **假设与依赖 (assumptions_or_dependencies)**：算法运行依赖什么假设？（如：假设光照均匀、假设拥有标注数据等）。

C. 结果抽取（results_extract）
**注意：结果需要按层级结构提取，严禁扁平化处理。**
1) **Proposed Methods**：论文提出的主要方法/模型/芯片名称。
2) **Experiment Results**：针对该方法进行的具体实验任务（Task）或场景（Scenario）。
3) **Metrics**：
    - **Metric Name**：指标名称（如 "Accuracy", "Latency"）。
    - **Value & Unit**：**必须提取具体的数字**（如 "0.45", "120"）。**如果原文是定性描述（如 "significantly improved"）且无具体数字支持，请不要提取该指标。**
    - **Condition/Setting**：数据来源必须具体（如 "COCO dataset" 或 "Real-world factory data"）。
    - **Comparison Methods**：
        - 仅提取原文中明确列出**具体数值**的对比方法（Baseline）。
        - 如果没有具体的对比数值，**保持该数组为空 []**。
        - **必须验证**：comparison_methods 中的每个方法是否都有原文明确的具体数值？
        - **禁止**：使用 "Previous work" 或 "Baseline" 作为 method_name，除非原文图表确实如此标记。
        - **格式**：method_name 应包含年份和参考文献编号（如 "ISSCC'22 [6]"），如果原文如此标注。

D. 应用场景（application）
- 从原文内容中提取出技术的近期、中期、远期应用方向。
- **Evidence**：必须给出具体的原文引用。

E. 技术就绪度（TRL）
**核心任务：给出具体的 TRL 等级 (1-9)**
- **Level**："TRL 1" | ... | "TRL 9"
- **Rationale**：引用原文解释判定依据（数据来源、环境）。**Rationale 可以用中文，但 Evidence 必须是原文。**
- **Uncertain**：如果证据不足以确信等级，设为 true。

F. 专利-权益要求评分（P_score）
- 根据系统提示中的 P_score 表，输出一个 `p_score`（P1-P5）以及 `p_reason`（中文说明，引用关键依据）。

## JSON 输出结构

```json
{
  "paper_title": "",
  "domain": "",
  "language": "",  // "en" or "zh"
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
  "results_extract": {
    "proposed_methods": [{
        "method_name": "",
        "experiment_results": [{
            "task_or_scenario": "",
            "metrics": [{
                "metric_name": "",
                "value": "", // Must be a number or specific quantitative string. NO adjectives like "improved".
                "unit": "",
                "experimental_setting": "",
                "evidence": {"section": "", "quote": ""},
                "comparison_methods": [{ // Keep empty [] if no specific numerical comparison exists
                    "method_name": "",
                    "value": "", // Must be a specific number
                    "experimental_setting": "",
                    "evidence": {"section": "", "quote": ""}
                }]
            }]
        }]
    }]
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
    "uncertain": "" // "true" or "false"
  },
  "p_score": "",
  "p_reason": ""
}
```
## 补充规则
- 最终仅输出 JSON，不得输出任何说明文本。
- 确保 JSON 格式合法，确保没有尾部逗号错误，反斜杠转义错误等问题。
- Value 字段检查：如果提取到的 value 是非数字的形容词，请直接丢弃该字段。
"""


class DeepSeekAgent:
    def __init__(self, api_key: str, model: str = "deepseek-reasoner") -> None:
        self.api_key = api_key
        self.model = model
        self.endpoint = "https://api.deepseek.com/v1/chat/completions"

    def _build_prompt(self, text: str) -> Dict[str, Any]:
        # 防止文本过长导致 Token 溢出，实际生产中可能需要在这里做截断
        # 但 DeepSeek Context 较长，一般论文全文都没问题
        prompt_content = PROMPT_TEMPLATE.replace("{paper_content}", text)
        return {
            "model": self.model,
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": prompt_content},
            ],
            "temperature": 0.1, # 降低温度以保证 JSON 格式稳定
            "response_format": {"type": "json_object"} # 强制 JSON 模式（DeepSeek 支持）
        }

    def extract_metadata(self, text: str) -> Dict[str, Any]:
        try:
            response = requests.post(
                self.endpoint,
                headers={"Authorization": f"Bearer {self.api_key}"},
                json=self._build_prompt(text),
                timeout=120, # 增加超时时间，因为思维链输出较长
            )
            response.raise_for_status()
            content = response.json()["choices"][0]["message"]["content"].strip()
            
            # 清理可能存在的 Markdown 标记 (虽然 Prompt 禁止了，但防万一)
            if content.startswith("```"):
                lines = content.splitlines()
                # 移除第一行 ```json 和最后一行 ```
                if lines[0].startswith("```"): lines = lines[1:]
                if lines and lines[-1].strip() == "```": lines = lines[:-1]
                content = "\n".join(lines).strip()
                
            return json.loads(content)
        except Exception as e:
            print(f"Error extracting metadata: {e}")
            # 返回空结构或错误信息，避免整个程序崩溃
            return {"error": str(e), "raw_response": content if 'content' in locals() else ""}

def main() -> None:
    api_key = os.environ.get("DEEPSEEK_API_KEY")
    if not api_key:
        # 为了演示，如果没设置环境变量，抛出错误
        raise EnvironmentError("请设置环境变量 DEEPSEEK_API_KEY")

    base_dir = Path(__file__).resolve().parent
    md_dir = base_dir / "md" / "1.26" / "filtered"
    results_dir = base_dir / "extraction_results" / "1-26"
    results_dir.mkdir(exist_ok=True)
    
    if not md_dir.exists():
        raise FileNotFoundError(f"目录未找到: {md_dir}")
        
    inputs = sorted(md_dir.glob("*.md"))
    if not inputs:
        raise FileNotFoundError(f"{md_dir} 目录下未找到 md 文件")

    agent = DeepSeekAgent(api_key=api_key)
    
    for file_path in inputs:
        print(f"正在处理: {file_path.name} ...")
        content = file_path.read_text(encoding="utf-8")
        
        # 调用抽取
        metadata = agent.extract_metadata(content)
        metadata["source_filename"] = file_path.name
        output_path = results_dir / f"{file_path.stem}.json"
        output_path.write_text(
            json.dumps(metadata, ensure_ascii=False, indent=2),
            encoding="utf-8"
        )
    print(f"处理完成，结果已保存至: {results_dir}")

if __name__ == "__main__":
    main()