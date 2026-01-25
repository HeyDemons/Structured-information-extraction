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
- **原文保留**：抽取的实体名、技术术语、指标名、单位、以及 `evidence/quote` 字段，必须**严格保持论文原文语言**。
- **禁止翻译**：如果论文是英文，输出的 JSON 值（Values）必须是英文。严禁将 "Accuracy" 翻译为 "准确率"，或将算法名翻译成中文。
- **解释性字段**：仅 `rationale` (理由) 字段允许使用中文进行解释，但引用的原文必须保持原样。

## 1. 证据优先
所有抽取结论必须尽量给出原文证据片段（quote）与其在 paper 中的定位信息。

## 2. 数值严谨与计算
结果抽取必须抓取具体数值。
**必须计算 P_score**：针对每一项有明确 Baseline 对比的指标，执行公式计算：
$$P_{score} = (Result_{paper} - Result_{baseline}) / Result_{baseline}$$
(确保在计算前统一单位，计算结果保留 4 位小数)

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
'''

PROMPT_TEMPLATE = """
给定论文（Markdown 全文），请从中抽取核心元数据。

**重要提示：如果论文是英文，所有抽取的字段值（尤其是 object, metric_name, innovation）必须保持英文，不要翻译成中文。**

---

## 论文内容

{paper_content}

---

## 抽取要求

A. 问题抽取（problem）
1) 研究对象（object）：**(Keep Original Language)** 系统 / 算法对象是什么？
2) 应用场景（scenario）：面向什么行业？
3) 约束条件（constraints）：关注数据隐私、实时性、成本等。

B. 方法抽取（method）
- 技术路线与创新点 (technical_route / innovations)
- **保持术语原文**，不要翻译专有名词。

C. 结果抽取（results）
- metric_name **(Keep Original English)**
- value / unit
- condition **(Data source must be specific, e.g., "COCO dataset" or "Real-world factory data")**

**计算规则：**
1. 找到 $Result_{paper}$ 和 $Result_{baseline}$。
2. 计算：$$P_{score} = (Result_{paper} - Result_{baseline}) / Result_{baseline}$$
3. 若无 Baseline，p_score 填 null。

D. 应用场景（application）
- near_term / mid_term / long_term

E. 技术就绪度（TRL）
**核心任务：给出具体的 TRL 等级 (1-9)**
- level："TRL 1" | ... | "TRL 9"
- rationale：引用原文解释判定依据（数据来源、环境）。**Rationale 可以用中文，但 Evidence 必须是原文。**

## JSON 输出结构

{
  "paper_id": "",
  "title": "",
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
  "results": {
    "items": [
      {
        "metric_name": "",        // Must be Original Text 
        "paper_value": "",        // Number 论文方法的数值(数字类型)
        "baseline_value": "",     // Number or null
        "unit": "",
        "p_score": null,          // Number (4 decimal places)
        "baseline_name": "",      // Original Text 对比的算法/系统名称
        "condition": "",          // Original Text 数据集/实验设置
        "significance": "",       // p值等
        "evidence": {"section": "", "quote": ""}
      }
    ],
    "results_kv": {}
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
    "uncertain": true
  }
}

## 补充规则
- 最终仅输出 JSON，不得输出任何说明文本。
"""


os.environ['DEEPSEEK_API_KEY'] = 'sk-feee15f8635f42db849c9137ec48d961'

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
    md_dir = base_dir 
    results_dir = base_dir / "extraction_results" / "1-24"
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