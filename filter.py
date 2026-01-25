import concurrent.futures
import json
import re
from pathlib import Path
import os
from typing import Dict, List, Tuple

import httpx


def split_text_smart(text: str, max_chunk_size=2000) -> List[str]:
    """
    按段落智能切分，尽量保持语义完整性。
    提高上限，减少表格/图注与上下文被拆散。
    """
    paragraphs = text.split('\n\n')
    chunks = []
    current_chunk = []
    current_len = 0

    for para in paragraphs:
        para_len = len(para)
        if para_len > max_chunk_size:
            if current_chunk:
                chunks.append("\n\n".join(current_chunk))
                current_chunk = []
                current_len = 0
            chunks.append(para)
        elif current_len + para_len < max_chunk_size:
            current_chunk.append(para)
            current_len += para_len
        else:
            chunks.append("\n\n".join(current_chunk))
            current_chunk = [para]
            current_len = para_len

    if current_chunk:
        chunks.append("\n\n".join(current_chunk))
    return chunks


def contains_markdown_table(text: str) -> bool:
    """
    更鲁棒的 Markdown/HTML 表格检测：
    - HTML <table
    - 典型对齐分隔线 | --- | :---: |
    - 连续多行以 | 分割的表格行
    """
    lower = text.lower()
    if "<table" in lower or "<tr" in lower and "<td" in lower:
        return True

    lines = [ln for ln in text.splitlines() if not ln.strip().startswith("```")]
    sep_line = re.compile(r'^\s*\|?(?:\s*:?-{3,}:?\s*\|)+\s*:?-{3,}:?\s*\|?\s*$')
    pipe_row = re.compile(r'^\s*\|.*\|\s*$')

    for i, ln in enumerate(lines):
        if sep_line.match(ln):
            return True

    # 连续两行以上是“管道行”，或单行管道数较多
    consecutive = 0
    for ln in lines:
        if pipe_row.match(ln) or ln.count('|') >= 3:
            consecutive += 1
            if consecutive >= 2:
                return True
        else:
            consecutive = 0

    return False


def remove_image_only_lines(text: str) -> str:
    """
    删除整行仅为图片链接的内容（不处理代码块内部）：
    - Markdown: ![alt](url "title")
    - 嵌套点击图片: [![alt](url)](link)
    - HTML: <img src="...">
    """
    img_md = re.compile(r'^\s*!\[[^\]]*\]\([^)]+(?:\s+"[^"]*")?\)\s*$')
    img_nested = re.compile(r'^\s*\[!\[[^\]]*\]\([^)]+\)\]\([^)]+\)\s*$')
    img_html = re.compile(r'^\s*<img\b[^>]*\bsrc\s*=\s*["\'][^"\']+["\'][^>]*>\s*$', re.IGNORECASE)

    out_lines = []
    in_fence = False
    for ln in text.splitlines():
        stripped = ln.strip()
        if stripped.startswith("```"):
            in_fence = not in_fence
            out_lines.append(ln)
            continue
        if not in_fence and (img_md.match(ln) or img_nested.match(ln) or img_html.match(ln)):
            continue
        out_lines.append(ln)
    return "\n".join(out_lines)

 


def call_llm(prompt: str) -> Dict[str, str]:
    api_key = os.environ.get("DASHSCOPE_API_KEY", "")
    if not api_key:
        return {"decision": "KEEP", "reason": "No API Key"}

    try:
        response = httpx.post(
            "https://dashscope.aliyuncs.com/compatible-mode/v1/chat/completions",
            headers={"Authorization": f"Bearer {api_key}"},
            json={
                "model": "qwen-plus",
                "messages": [{"role": "user", "content": prompt}],
                "temperature": 0.01,
            },
            timeout=30,
        )
        response.raise_for_status()
        content = response.json()["choices"][0]["message"]["content"].strip()
        content = content.replace("```json", "").replace("```", "")
        return json.loads(content)
    except Exception:
        # 保守策略：异常则默认 KEEP，防网络问题导致漏保关键证据
        return {"decision": "KEEP", "reason": "fallback_error"}


def semantic_filter_paper(markdown_text: str) -> Tuple[str, str]:
    """
    返回：保留后的文本、被移除的文本
    """
    # 启用图片预清洗：移除整行图片链接，保留图注/说明文字
    markdown_text = remove_image_only_lines(markdown_text)

    chunks = split_text_smart(markdown_text, max_chunk_size=2000)

    # 双语且覆盖抽取要件的提示词
    SCREENER_PROMPT = """
任务/Task: 判断以下文本块是否包含【结构化信息抽取】所需的有效数据。
Decide if the chunk contains information needed for structured extraction.

【必须丢弃 / DROP】：
- 参考文献/致谢/基金/版权/利益冲突/作者信息/附录/补充材料 (References/Bibliography, Acknowledgements, Funding, Copyright,
  Conflict of Interest, Author Contributions, Appendix, Supplementary)。
- 纯目录、纯公式列表（无实验指标/条件）。

【必须保留 / KEEP】：
- 论文标题（Paper Title），即使仅为概括性描述，也必须保留。
- 含具体实验数据或单位数值（如 "accuracy 95%", "power 4 mW"）。
- Baseline/对比描述（"baseline", "vs.", "compared with", “对比/基线”）。
- 数据集与实验设置/测试环境（"COCO", "ImageNet", "real factory data", "measurement setup"）。
- 硬件/流片/现场系统演示/产线/批量测试等 TRL 证据（"silicon", "tape-out", "field trial", “流片/产线/现场演示/批量测试”）。
- 方法/工艺/系统架构；图表与图注（Markdown/HTML Tables, Figure Captions）。
- Abstract/Introduction/Results/Conclusion/Discussion/Future Work/Prospects/Limitations（即使无新指标也要保留）。

Input Chunk:
{text_chunk}

请仅输出 JSON / Output JSON only:
{{"decision": "KEEP" | "DROP", "reason": "..."}}
    """.strip()

    kept_chunks = []
    dropped_chunks = []

    with concurrent.futures.ThreadPoolExecutor(max_workers=8) as pool:
        future_map = {}

        for idx, chunk in enumerate(chunks):
            # 第一段必须保留（通常为论文标题），不调用 LLM
            if idx == 0:
                kept_chunks.append((idx, chunk))
                continue
            # 完全交由 LLM：移除本地规则丢弃
            # if is_high_confidence_reference(chunk):
            #     dropped_chunks.append((idx, chunk))
            #     continue

            # 直接交给 LLM 判别
            future = pool.submit(call_llm, SCREENER_PROMPT.format(text_chunk=chunk))
            future_map[future] = (idx, chunk)

        # 仅依据 LLM 结果，不再做二次本地规则覆盖
        for future in concurrent.futures.as_completed(future_map):
            idx, chunk = future_map[future]
            try:
                res = future.result()
                decision = res.get("decision", "KEEP").upper()

                if decision == "KEEP":
                    kept_chunks.append((idx, chunk))
                else:
                    dropped_chunks.append((idx, chunk))
            except Exception:
                kept_chunks.append((idx, chunk))

    kept_chunks.sort(key=lambda x: x[0])

    final_text = "\n\n".join([c[1] for c in kept_chunks])
    removed_text = "\n\n".join([c[1] for c in sorted(dropped_chunks, key=lambda x: x[0])])

    return final_text, removed_text

markdown_file_path = Path("output-1-23/md/3.1_A_121.3dB-DR_115dB-PSNR_Digital-Input_Capacitive-Feedback_Class-D_Audio_Amplifier_with_Double-Sided_Voltage-Boosting_DSVB_Modulation.md")
markdown_text = markdown_file_path.read_text(encoding="utf-8")
semantic_text, removed_text = semantic_filter_paper(markdown_text)
with open("semantic_filtered_output.md", "w", encoding="utf-8") as f:
    f.write(semantic_text)
if removed_text.strip():
    with open("semantic_filtered_removed.md", "w", encoding="utf-8") as f:
        f.write(removed_text)
    print("=== Removed Section ===")
    print(removed_text.strip())
