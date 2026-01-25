from pathlib import Path
from paddleocr import PaddleOCRVL
import shutil

input_dir = Path("./pdfs")
output_path = Path("./output")
backup_dir = output_path / "md"
backup_dir.mkdir(parents=True, exist_ok=True)

# 英伟达 GPU
pipeline = PaddleOCRVL()
# 昆仑芯 XPU
# pipeline = PaddleOCRVL(device="xpu")
# 海光 DCU
# pipeline = PaddleOCRVL(device="dcu")
# 沐曦 GPU
# pipeline = PaddleOCRVL(device="metax_gpu")

if not input_dir.exists():
    raise FileNotFoundError(f"指定文件夹不存在: {input_dir}")

pdf_files = sorted(input_dir.glob("*.pdf"))
if not pdf_files:
    raise FileNotFoundError(f"未在 {input_dir} 中找到任何 PDF 文件")

for input_file in pdf_files:
    output = pipeline.predict(input=str(input_file))

    markdown_list = []
    markdown_images = []

    for res in output:
        md_info = res.markdown
        markdown_list.append(md_info)
        markdown_images.append(md_info.get("markdown_images", {}))

    markdown_texts = pipeline.concatenate_markdown_pages(markdown_list)

    mkd_file_path = output_path / f"{input_file.stem}.md"
    mkd_file_path.parent.mkdir(parents=True, exist_ok=True)

    with open(mkd_file_path, "w", encoding="utf-8") as f:
        f.write(markdown_texts)

    backup_path = backup_dir / mkd_file_path.name
    shutil.copy2(mkd_file_path, backup_path)

    for item in markdown_images:
        if item:
            for path, image in item.items():
                file_path = output_path / path
                file_path.parent.mkdir(parents=True, exist_ok=True)
                image.save(file_path)