# analyze_results.py
import json, sys
from pathlib import Path

# Intentamos usar matplotlib; si falla por problemas de NumPy o no está
# disponible, hacemos un fallback a Pillow para dibujar el histograma sin depender de NumPy.
try:
    import matplotlib.pyplot as plt
    _HAS_MPL = True
except Exception:
    _HAS_MPL = False

def analyze_summary(summary_path):
    with open(summary_path) as f:
        results = json.load(f)
    # Si el JSON contiene un solo resultado (dict), convertirlo en lista para tratarlo igual
    if isinstance(results, dict):
        results = [results]
    # tomar el primer resultado exitoso o el primero
    selected = None
    for r in results:
        if r.get('factors'):
            selected = r
            break
    if selected is None:
        selected = results[0]
    counts = selected['counts']
    items = sorted(counts.items(), key=lambda kv: kv[1], reverse=True)[:50]
    labels = [str(k) for k,_ in items]
    vals = [v for _,v in items]
    outp = Path(summary_path).parent / f"hist_N{selected['N']}_a{selected['a']}.png"

    if _HAS_MPL:
        plt.figure(figsize=(10,5))
        plt.bar(labels, vals)
        plt.xticks(rotation=90)
        plt.xlabel('Measured phase value (integer)')
        plt.ylabel('Counts (shots)')
        plt.title(f"Histogram: N={selected['N']} a={selected['a']}")
        plt.tight_layout()
        plt.savefig(outp)
        print("Saved histogram:", outp)
        return

    # Fallback sencillo con Pillow (no requiere NumPy)
    try:
        from PIL import Image, ImageDraw, ImageFont
    except Exception:
        print("Neither matplotlib nor Pillow are available. Please install one: e.g. pip install matplotlib or pip install pillow")
        return

    # Parámetros de la imagen
    n = len(labels)
    width = max(800, 30 * n)
    height = 480
    margin = 60
    bar_area_w = width - 2 * margin
    bar_area_h = height - 2 * margin

    maxv = max(vals) if vals else 1
    img = Image.new('RGB', (width, height), 'white')
    draw = ImageDraw.Draw(img)
    try:
        font = ImageFont.truetype("arial.ttf", 12)
    except Exception:
        font = ImageFont.load_default()

    # Dibujar ejes
    x0 = margin
    y0 = margin
    x1 = width - margin
    y1 = height - margin
    draw.rectangle([x0, y0, x1, y1], outline='black')

    # Dibujar barras
    if n > 0:
        bw = bar_area_w / n
        for i, v in enumerate(vals):
            bx = x0 + i * bw
            bh = (v / maxv) * (bar_area_h - 20)
            top = y1 - bh
            draw.rectangle([bx + 2, top, bx + bw - 2, y1 - 1], fill='#4C72B0', outline='black')
            # etiquetas (rotadas es costoso; colocamos etiquetas verticales abreviadas)
            label = labels[i]
            # truncar si muy largo
            if len(label) > 10:
                lab = label[:10] + '...'
            else:
                lab = label
            draw.text((bx + 4, y1 + 4), lab, fill='black', font=font)

    # Títulos y ejes
    title = f"Histogram: N={selected['N']} a={selected['a']}"
    draw.text((margin, 8), title, fill='black', font=font)
    draw.text((8, margin), 'Counts', fill='black', font=font)
    draw.text((width//2 - 40, height - 20), 'Measured phase value (integer)', fill='black', font=font)

    img.save(outp)
    print("Saved histogram (Pillow):", outp)

if __name__ == "__main__":
    if len(sys.argv)<2:
        print("Usage: python analyze_results.py path/to/summary_N{N}.json")
        sys.exit(1)
    analyze_summary(sys.argv[1])
