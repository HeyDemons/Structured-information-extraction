import concurrent.futures
import json
import re
from pathlib import Path
import os
from typing import Dict, List, Tuple

import httpx


def split_text_smart(text: str, max_chunk_size=1000) -> List[str]:
    """
    仅按段落（两个换行符）切分文本，逐段返回。
    """
    return [para for para in text.split("\n\n") if para.strip()]


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
                "model": "deepseek-v3.2",
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
    任务/Task: 这里的文本块来自一篇学术/技术论文。请判断该文本块是否包含用于【下游结构化抽取】的有价值信息。
    Decide if the chunk contains valuable information for downstream structured extraction (Problem, Method, Results, TRL).

    ### 判别标准 (Criteria)

    【必须丢弃 / DROP】 (Pure Noise):
    - **参考文献列表** (References/Bibliography)。
    - **作者/作者单位信息** (Author names, affiliations, emails)。
    - **致谢/版权/基金/作者简介** (Acknowledgements, Funding, Copyright, Author Bios, Contact Info)。
    - **纯粹的页眉/页脚/页码** (Repeated headers/footers, page numbers)。
    - **目录/附录索引** (Table of Contents, Appendix Index)。
    - **利益冲突声明** (Conflict of Interest)。
    - **无意义的文本碎片** (Gibberish, random symbols)。例如：'(a)'。
    
    【必须保留 / KEEP】 (Valuable Signal):
    请检查文本是否包含以下任一维度的信息：

    1. **核心元数据**: 论文标题 (Paper Title)、摘要 (Abstract)。
    2. **问题背景 (Problem & Constraints)**:
    - 行业痛点/关键难题 (pain points, "key challenges")。
    - 约束条件描述 (constraints: 隐私, 实时性, 成本, 功耗等文字描述)。
    - 应用场景定义 (scenario, industry application)。
    3. **方法论 (Methodology)**:
    - 算法/系统架构描述 (System architecture, pipeline)。
    - 核心创新点声明 (Innovations, contributions)。
    - 假设与依赖条件 (Assumptions, dependencies)。
    4. **定量结果 (Quantitative Results)**:
    - **任何**数字指标、单位、性能数值 (metrics, values, units)。
    - 对比实验、Baseline 提及 (Comparison, SOTA, "outperforms").
    - 图表标题或图注 (Figure captions, Table headers)。
    5. **TRL 关键证据 (TRL Evidence)**:
    - 测试环境描述 (Testbed, "real-world", "simulation").
    - 硬件/落地关键词 ("FPGA", "chip", "deployment", "mass production", "field trial").
    6. **未来展望 (Application)**:
    - 结论与未来工作 (Conclusion, Future work, Limitations)。


    Input Chunk:
    {text_chunk}

    请仅输出 JSON / Output JSON only:
    {{"decision": "KEEP" | "DROP", "reason": "Briefly explain why (e.g., 'Contains metrics', 'Describes method', 'Reference list')"}}
    """

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

def process_markdown_directory(directory: Path) -> None:
    directory = directory.expanduser().resolve()
    if not directory.is_dir():
        raise ValueError(f"Markdown directory not found: {directory}")

    filtered_root = directory.parent / "filtered"
    filtered_root.mkdir(parents=True, exist_ok=True)

    for md_file in sorted(directory.rglob("*.md")):
        rel_path = md_file.relative_to(directory)
        target_path = filtered_root / rel_path
        target_path.parent.mkdir(parents=True, exist_ok=True)

        markdown_text = md_file.read_text(encoding="utf-8")
        semantic_text, _ = semantic_filter_paper(markdown_text)

        target_path.write_text(semantic_text, encoding="utf-8")
        print(f"[filtered] {md_file} -> {target_path}")

if __name__ == "__main__":
    target_dir = Path(os.environ.get("FILTER_MD_DIR", "md/1.26/md"))
    process_markdown_directory(target_dir)
