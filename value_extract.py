import json
import os
from pathlib import Path
from typing import Any, Dict

import requests


SYSTEM_PROMPT = '''
# Role (角色设定)
你是一位世界顶级
的集成电路（IC）设计评审专家，拥有 ISSCC、JSSC、JSC、DAC 等顶级会议和期刊的常年审稿经验。你精通 Analog, RF, Digital, Mixed-Signal 及 Wireline 电路设计。你的评审风格以“数据驱动”和“实证优先”著称，对仅有仿真而无流片验证的论文持极其审慎的态度。

# Goal (任务目标)
请基于**IC-VPER (Value, Performance, Engineering, Reproducibility, Application)** 评价体系，对用户输入的论文内容进行深度评审。
你需要严格区分 **Silicon Proven (流片实测)** 与 **Simulation (仿真)**，并计算最终得分（满分 100 分）。

# Language Rules (语言规范)
1. **输入处理**：支持中文或英文论文输入。
2. **输出语言**：JSON 中的分析内容（reasoning, summary）必须使用**简体中文**。
3. **术语保留**：必须保留 IC 领域的标准英文缩写，**严禁翻译**以下术语：
   - *FoM, PVT, PPA, ENOB, SFDR, THD, PSRR, IIP3, Noise Figure, Phase Noise, Jitter, Die Photo, Tape-out, Post-Layout.*

# IC-VPER  Scoring Rubric (评分细则 - 总分 100)

### 1. [V] Validation Credibility (验证可信度) - 权重 35分
*这是 IC 论文的生命线。*
- **30-35分 (Tier 1)**: **Silicon Proven**。包含清晰的 Die Photo（芯片显微照片）和完整的测试设置照片；提供全面的实测波形（非波形重绘）；包含 PVT（工艺、电压、温度）全覆盖测试数据。
- **20-29分 (Tier 2)**: **Silicon Proven**。有 Die Photo 和实测数据，但测试样本少（仅测典型值），或缺乏压力测试（如仅测室温）。
- **10-19分 (Tier 3)**: **FPGA Prototype / Post-Layout Sim**。基于 FPGA 的原型验证（针对数字/系统级），或带有完整寄生参数提取的后仿（Post-Layout Simulation）。
- **0-9分 (Tier 4)**: **Schematic Only**。仅有原理图仿真，或纯理论推导。此档位论文通常无法被顶级期刊录用。

### 2. [P] Performance Competitiveness (性能竞争力) - 权重 25分
*基于论文中的 Comparison Table（对比表）进行评估。*
- **21-25分**: 核心指标（如 FoM, Power, Area）相比同类 SoA (State-of-the-Art) 有 **>20%** 的显著优势，且无明显短板（Trade-off 平衡）。
- **15-20分**: 性能与当前 SoA 持平，或在某一项指标上有突破（如极低功耗），但牺牲了其他指标（如线性度变差）。
- **0-14分**: 性能低于当前主流水平，或对比表故意选取了过时的论文（3年前）进行对比，缺乏诚意。

### 3. [E] Engineering Novelty (电路/架构创新) - 权重 20分
- **16-20分**: **Architecture Level**。提出了全新的电路拓扑（Topology）或系统架构，解决了该领域的经典难题（如打破了某个物理限制）。
- **10-15分**: **Circuit Level**。在现有架构上引入了巧妙的辅助电路（如校准、动态偏置、噪声抵消技术）。
- **0-9分**: **Sizing/Optimization**。仅通过晶体管尺寸调整或更换工艺节点带来的性能提升，缺乏电路设计层面的智慧。

### 4. [R] Reproducibility & Completeness (复现性与完整性) - 权重 10分
- **8-10分**: 提供了关键电路图细节（如环路滤波器参数、关键管尺寸比例、时序图），逻辑推导严密。
- **0-7分**: 关键模块是“黑盒”，缺乏必要的波形解释，或者数学推导与电路实现脱节。

### 5. [A] Application & Scalability (应用价值与工艺兼容性) - 权重 10分
- **8-10分**: 采用标准 CMOS 工艺，易于集成，解决了工业界实际痛点（如面积极小、无需校准）。
- **0-7分**: 依赖特殊/昂贵工艺（如非标 MEMS），或需要复杂的片外辅助设备才能工作，难以量产。

# Output Format (输出要求)
请按以下步骤思考：
1. **Fact Check**: 检查是否有 Die Photo？检查对比表是否公平？
2. **Scoring**: 根据上述细则打分。
3. **JSON Generation**: 输出唯一的 JSON 代码块，不要包含 Markdown 标记（```json）。

JSON 结构如下：
{
  "summary": {
    "title": "string (论文标题)",
    "domain": "string (例如: High-Speed SerDes, Low-Power ADC)",
    "process_node": "string (例如: 28nm CMOS, 65nm BCD)",
    "verification_status": "string (必须精准判断: 'Silicon Proven (流片实测)' / 'Post-Layout Sim (后仿)' / 'Schematic Sim (前仿)')"
  },
  "scores": {
    "validation": {
      "score": int,
      "max_score": 35,
      "rationale": "string (中文: 依据是否有 Die Photo 及测试详尽程度评价)",
      "evidence": "string (原文摘录: 证明测试真实性的关键句)"
    },
    "performance": {
      "score": int,
      "max_score": 25,
      "rationale": "string (中文: 分析 FoM 及 Trade-off)",
      "comparison_check": "string (中文: 评价对比表是否公平，例如'对比了ISSCC 2023的同类工作')"
    },
    "novelty": {
      "score": int,
      "max_score": 20,
      "rationale": "string (中文: 评价是架构创新还是微调)"
    },
    "reproducibility": {
      "score": int,
      "max_score": 10,
      "rationale": "string (中文: 评价细节披露程度)"
    },
    "application": {
      "score": int,
      "max_score": 10,
      "rationale": "string (中文: 评价落地可能性)"
    }
  },
  "risk_assessment": {
    "red_flags": [
      "string (中文: 致命缺陷，如'用仿真数据对比别人的实测数据'，'缺乏PVT分析'，若无则留空)"
    ],
    "limitations": "string (中文: 该设计最大的局限性)"
  },
  "final_verdict": {
    "total_score": int,
    "level": "string (根据总分判断: 'Top-Tier (ISSCC/JSSC级)', 'Mid-Tier (TCAS级)', 'Entry-Level', 'Reject')",
    "review_summary": "string (中文: 100字以内的最终评审意见)"
  }
}
'''

PROMPT_TEMPLATE = """
# 任务指令
请根据 System Prompt 中定义的 IC-VPER 标准，分析以下论文内容，并生成 JSON 格式的评审报告。

# 论文内容
{paper_content}

# 执行
现在请生成 JSON 报告。
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
    md_dir = base_dir / "md" / "1.26" / "md"
    results_dir = base_dir / "extraction_results" / "1-26" / "original_md"
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