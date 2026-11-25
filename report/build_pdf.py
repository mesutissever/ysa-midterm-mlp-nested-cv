"""report/rapor.md -> report.pdf

Markdown HTML'e çevrilir; "**Şekil N.** ... `figures/x.png`" satırlarında anılan şekiller base64 olarak
gömülür; PDF, Chrome/Edge'in headless yazdırma özelliğiyle üretilir.

Kullanım:  python report/build_pdf.py        (gerekli: pip install markdown; Chrome veya Edge)
"""
import base64
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import markdown

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "report" / "rapor.md"
OUT = ROOT / "report.pdf"
BROWSERS = [
    r"C:\Program Files\Google\Chrome\Application\chrome.exe",
    r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe",
    r"C:\Program Files\Microsoft\Edge\Application\msedge.exe",
    "google-chrome", "chromium", "chromium-browser", "msedge",
]

WIDE = ("flow", "categorical", "training_curves", "search_comparison")
STACK = ("numeric_hist", "numeric_box")   # alt alta yerleştirilen şekil çiftleri
CSS = """
@page { size: A4; margin: 15mm 15mm 16mm 15mm;
        @bottom-center { content: counter(page); font: 9pt 'Segoe UI', Arial, sans-serif; color: #555; } }
@page :first { @bottom-center { content: none; } }
body { font-family: 'Segoe UI', Calibri, Arial, sans-serif; font-size: 9.8pt; line-height: 1.28; color: #111; }
h1 { font-size: 17pt; text-align: center; margin: 0 0 14pt; line-height: 1.3; }
h2 { font-size: 13pt; border-bottom: 1px solid #999; padding-bottom: 2pt; margin: 12pt 0 6pt; break-after: avoid; }
h3 { font-size: 11pt; margin: 9pt 0 4pt; break-after: avoid; }
p, li { text-align: justify; margin: 4pt 0; }
table { border-collapse: collapse; margin: 4pt auto 8pt; font-size: 8.2pt; break-inside: avoid; }
th, td { border: 1px solid #bbb; padding: 1.5pt 4pt; vertical-align: top; }
th { background: #eef1f5; }
tr:nth-child(even) td { background: #fafafa; }
code { font-family: Consolas, 'Courier New', monospace; font-size: 9pt; background: #f3f3f3; padding: 0 2px; }
figure { margin: 4pt 0 2pt; text-align: center; break-inside: avoid; }
figure img { max-width: 100%; max-height: 54mm; }
figure.multi img { max-width: 49%; max-height: 70mm; }
figure.wide img { max-height: 68mm; }
figure.stack img { display: block; margin: 0 auto 2pt; max-width: 88%; max-height: 62mm; }
hr { display: none; }
p:has(+ table) { break-after: avoid; }
ol li { text-align: left; }
.cover { height: 250mm; display: flex; flex-direction: column; justify-content: center; text-align: center;
         break-after: page; }
.cover h1 { font-size: 20pt; }
.cover p { text-align: center; font-size: 11.5pt; margin: 3pt 0; }
.cover .course { font-size: 12.5pt; margin-bottom: 30pt; }
"""


def find_browser():
    for b in BROWSERS:
        p = shutil.which(b) or (b if Path(b).exists() else None)
        if p:
            return p
    sys.exit("Chrome/Edge bulunamadı.")


def img_tag(rel):
    data = base64.b64encode((ROOT / rel).read_bytes()).decode()
    return f'<img src="data:image/png;base64,{data}" alt="{rel}">'


def embed_figures(md_text):
    """Şekil başlığı satırlarının altına, satırda anılan figures/*.png dosyalarını ekler."""
    out = []
    for line in md_text.splitlines():
        if line.startswith("|"):                      # tablo hücrelerinde "a ± b" satır sonunda bölünmesin
            line = line.replace(" ± ", " ± ")
        out.append(line)
        if line.startswith("**Şekil"):
            figs = re.findall(r"`(figures/[\w\-]+\.png)`", line)
            if figs:
                out[-1] = re.sub(r"\s*`figures/[\w\-]+\.png`(,\s*)?", " ", line).rstrip()
                cls = "multi" if len(figs) > 1 else ("wide" if any(k in figs[0] for k in WIDE) else "")
                if any(k in f for f in figs for k in STACK):
                    cls = "stack"
                imgs = "".join(img_tag(f) for f in figs)
                out[-1] = f'<figure class="{cls}">{imgs}</figure>\n\n' + out[-1]
    return "\n".join(out)


def cover_and_body(md_text):
    """Başlık bloğunu (ilk '---' öncesi) kapak sayfasına dönüştürür."""
    head, body = md_text.split("\n---\n", 1)
    lines = [l for l in head.splitlines() if l.strip()]
    title = lines[0].lstrip("# ").strip()
    meta = [markdown.markdown(l).replace("<p>", "").replace("</p>", "") for l in lines[1:]]
    course = meta[0] if meta else ""
    rest = "".join(f"<p>{m}</p>" for m in meta[1:])
    cover = f'<div class="cover"><h1>{title}</h1><p class="course">{course}</p>{rest}</div>'
    return cover, body


def main():
    md_text = SRC.read_text(encoding="utf-8")
    cover, body = cover_and_body(md_text)
    body_html = markdown.markdown(embed_figures(body), extensions=["tables", "sane_lists"])
    html = (f'<!doctype html><html lang="tr"><head><meta charset="utf-8"><title>Rapor</title>'
            f"<style>{CSS}</style></head><body>{cover}{body_html}</body></html>")
    with tempfile.TemporaryDirectory() as tmp:
        page = Path(tmp) / "rapor.html"
        page.write_text(html, encoding="utf-8")
        subprocess.run([find_browser(), "--headless=new", "--disable-gpu", "--no-pdf-header-footer",
                        f"--print-to-pdf={OUT}", page.as_uri()], check=True, capture_output=True)
    print(f"Yazıldı: {OUT}  ({OUT.stat().st_size / 1024:.0f} KB)")


if __name__ == "__main__":
    main()
