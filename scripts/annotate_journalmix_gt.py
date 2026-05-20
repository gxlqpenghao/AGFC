#!/usr/bin/env python3
"""Local browser-based GT annotation tool for JournalMix dataset.

Opens a web UI at http://localhost:8765 where you can:
1. Browse all 84 pages with their page images
2. See current AGFC-predicted bboxes (green rectangles)
3. Draw corrected bboxes by click-dragging
4. Mark pages as confirmed / needs-fix / skip
5. Save corrections back to gt/*.json and meta/*.json

Usage:
    python3 scripts/annotate_journalmix_gt.py \
        --dataset-root data/private/journalmix_v1
"""
from __future__ import annotations

import argparse
import json
import sys
from http.server import HTTPServer, SimpleHTTPRequestHandler
from pathlib import Path
from urllib.parse import parse_qs, urlparse

PROJECT_ROOT = Path(__file__).resolve().parents[1]

# ---------------------------------------------------------------------------
# HTML / JS for the annotation UI
# ---------------------------------------------------------------------------

HTML_TEMPLATE = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>JournalMix GT Annotator</title>
<style>
  * { margin: 0; padding: 0; box-sizing: border-box; }
  body {
    font-family: 'Inter', -apple-system, sans-serif;
    background: #0f1117;
    color: #e0e0e0;
    display: flex;
    height: 100vh;
  }
  #sidebar {
    width: 280px;
    background: #161822;
    border-right: 1px solid #2a2d3a;
    overflow-y: auto;
    flex-shrink: 0;
  }
  #sidebar h2 {
    padding: 16px;
    font-size: 14px;
    color: #8b8fa3;
    text-transform: uppercase;
    letter-spacing: 1px;
    border-bottom: 1px solid #2a2d3a;
    position: sticky;
    top: 0;
    background: #161822;
  }
  .page-item {
    padding: 10px 16px;
    cursor: pointer;
    border-bottom: 1px solid #1e2030;
    transition: background 0.15s;
    display: flex;
    justify-content: space-between;
    align-items: center;
  }
  .page-item:hover { background: #1e2030; }
  .page-item.active { background: #252840; border-left: 3px solid #4f8cff; }
  .page-item .id { font-weight: 600; font-size: 13px; }
  .page-item .info { font-size: 11px; color: #6b6f83; }
  .page-item .badge {
    font-size: 10px;
    padding: 2px 6px;
    border-radius: 4px;
    font-weight: 600;
  }
  .badge-confirmed { background: #1a3a2a; color: #4ade80; }
  .badge-prelabeled { background: #3a2a1a; color: #fbbf24; }
  .badge-needs_fix { background: #3a1a1a; color: #f87171; }

  #main {
    flex: 1;
    display: flex;
    flex-direction: column;
    overflow: hidden;
  }
  #toolbar {
    padding: 12px 20px;
    background: #161822;
    border-bottom: 1px solid #2a2d3a;
    display: flex;
    gap: 12px;
    align-items: center;
  }
  #toolbar .title {
    font-weight: 700;
    font-size: 16px;
    margin-right: auto;
  }
  button {
    padding: 8px 16px;
    border: none;
    border-radius: 6px;
    cursor: pointer;
    font-size: 13px;
    font-weight: 600;
    transition: all 0.15s;
  }
  .btn-confirm { background: #166534; color: #4ade80; }
  .btn-confirm:hover { background: #15803d; }
  .btn-fix { background: #7c2d12; color: #fb923c; }
  .btn-fix:hover { background: #9a3412; }
  .btn-clear { background: #1e293b; color: #94a3b8; }
  .btn-clear:hover { background: #334155; }
  .btn-nav { background: #1e293b; color: #94a3b8; }
  .btn-nav:hover { background: #334155; }
  .btn-save { background: #1e40af; color: #93c5fd; }
  .btn-save:hover { background: #1d4ed8; }

  #canvas-container {
    flex: 1;
    overflow: auto;
    display: flex;
    justify-content: center;
    align-items: flex-start;
    padding: 20px;
    background: #1a1a2e;
  }
  canvas {
    cursor: crosshair;
    box-shadow: 0 4px 24px rgba(0,0,0,0.5);
  }
  #status {
    padding: 8px 20px;
    background: #161822;
    border-top: 1px solid #2a2d3a;
    font-size: 12px;
    color: #6b6f83;
  }
  #info-panel {
    padding: 8px 20px;
    background: #1a1c2e;
    border-top: 1px solid #2a2d3a;
    font-size: 12px;
    color: #8b8fa3;
    display: flex;
    gap: 24px;
  }
</style>
</head>
<body>
<div id="sidebar">
  <h2>Pages (%%PAGE_COUNT%%)</h2>
  <div id="page-list"></div>
</div>
<div id="main">
  <div id="toolbar">
    <span class="title" id="page-title">Select a page</span>
    <button class="btn-nav" onclick="navigate(-1)">← Prev</button>
    <button class="btn-nav" onclick="navigate(1)">Next →</button>
    <button class="btn-clear" onclick="clearDrawn()">Clear Drawn</button>
    <button class="btn-confirm" onclick="confirmPage()">✓ Confirm</button>
    <button class="btn-fix" onclick="markNeedsFix()">✗ Needs Fix</button>
    <button class="btn-save" onclick="savePage()">💾 Save</button>
  </div>
  <div id="canvas-container">
    <canvas id="canvas"></canvas>
  </div>
  <div id="info-panel">
    <span id="info-family"></span>
    <span id="info-difficulty"></span>
    <span id="info-figs"></span>
    <span id="info-doc"></span>
  </div>
  <div id="status">Ready. Click and drag on the image to draw figure bboxes. Green = existing, Blue = your correction.</div>
</div>

<script>
const pages = %%PAGES_JSON%%;
let currentIdx = -1;
let pageImg = null;
let drawnBoxes = [];
let existingBoxes = [];
const canvas = document.getElementById('canvas');
const ctx = canvas.getContext('2d');

const HANDLE_SIZE = 8;
let mode = 'idle';
let selectedBoxIdx = -1;
let startX, startY;
let dragEdge = null;
let dragStartBox = null;

const list = document.getElementById('page-list');
pages.forEach((p, i) => {
  const div = document.createElement('div');
  div.className = 'page-item';
  div.id = 'pitem-' + i;
  div.innerHTML = '<div><div class="id">' + p.page_id + '</div><div class="info">' + p.family + ' · ' + p.difficulty + '</div></div><span class="badge badge-' + p.status + '">' + p.status + '</span>';
  div.onclick = () => loadPage(i);
  list.appendChild(div);
});

function loadPage(idx) {
  if (currentIdx >= 0) document.getElementById('pitem-' + currentIdx)?.classList.remove('active');
  currentIdx = idx;
  document.getElementById('pitem-' + idx)?.classList.add('active');
  document.getElementById('pitem-' + idx)?.scrollIntoView({block: 'nearest'});
  const p = pages[idx];
  document.getElementById('page-title').textContent = p.page_id + ' — ' + p.family;
  document.getElementById('info-family').textContent = 'Family: ' + p.family;
  document.getElementById('info-difficulty').textContent = 'Difficulty: ' + p.difficulty;
  document.getElementById('info-figs').textContent = 'Figures: ' + p.figures.length;
  document.getElementById('info-doc').textContent = 'Doc: ' + p.doc_id;
  existingBoxes = p.figures.map(function(f){ return Object.assign({}, f); });
  drawnBoxes = [];
  selectedBoxIdx = -1;
  mode = 'idle';
  pageImg = new Image();
  pageImg.onload = function() { canvas.width = pageImg.width; canvas.height = pageImg.height; redraw(); };
  pageImg.src = '/image?path=' + encodeURIComponent(p.page_image);
}

function redraw() {
  if (!pageImg) return;
  ctx.clearRect(0, 0, canvas.width, canvas.height);
  ctx.drawImage(pageImg, 0, 0);
  var dpiScale = pages[currentIdx].dpi_scale;
  ctx.setLineDash([8, 4]);
  existingBoxes.forEach(function(fig) {
    var coords = fig.bbox.map(function(v){ return v * dpiScale; });
    ctx.strokeStyle = '#00ff00'; ctx.lineWidth = 3;
    ctx.strokeRect(coords[0], coords[1], coords[2]-coords[0], coords[3]-coords[1]);
    ctx.fillStyle = 'rgba(0,255,0,0.08)';
    ctx.fillRect(coords[0], coords[1], coords[2]-coords[0], coords[3]-coords[1]);
    ctx.fillStyle = '#00ff00'; ctx.font = '14px monospace';
    ctx.fillText(fig.figure_id, coords[0]+4, coords[1]-4);
  });
  ctx.setLineDash([]);
  drawnBoxes.forEach(function(box, i) {
    var sel = (i === selectedBoxIdx);
    ctx.strokeStyle = sel ? '#60a5fa' : '#4f8cff';
    ctx.lineWidth = sel ? 4 : 3;
    ctx.strokeRect(box.x, box.y, box.w, box.h);
    ctx.fillStyle = sel ? 'rgba(96,165,250,0.15)' : 'rgba(79,140,255,0.08)';
    ctx.fillRect(box.x, box.y, box.w, box.h);
    ctx.fillStyle = '#4f8cff'; ctx.font = '14px monospace';
    ctx.fillText('fig_' + (i+1), box.x+4, box.y-6);
    if (sel) {
      getHandles(box).forEach(function(h) {
        ctx.fillStyle = '#60a5fa';
        ctx.fillRect(h.x-HANDLE_SIZE, h.y-HANDLE_SIZE, HANDLE_SIZE*2, HANDLE_SIZE*2);
        ctx.strokeStyle = '#1e40af'; ctx.lineWidth = 1;
        ctx.strokeRect(h.x-HANDLE_SIZE, h.y-HANDLE_SIZE, HANDLE_SIZE*2, HANDLE_SIZE*2);
      });
    }
  });
}

function getHandles(b) {
  var cx=b.x+b.w/2, cy=b.y+b.h/2;
  return [
    {t:'tl',x:b.x,y:b.y},{t:'tr',x:b.x+b.w,y:b.y},
    {t:'bl',x:b.x,y:b.y+b.h},{t:'br',x:b.x+b.w,y:b.y+b.h},
    {t:'top',x:cx,y:b.y},{t:'bottom',x:cx,y:b.y+b.h},
    {t:'left',x:b.x,y:cy},{t:'right',x:b.x+b.w,y:cy}
  ];
}

function hitHandle(mx, my) {
  for (var i=drawnBoxes.length-1; i>=0; i--) {
    var hs = getHandles(drawnBoxes[i]);
    for (var j=0; j<hs.length; j++) {
      if (Math.abs(mx-hs[j].x)<=HANDLE_SIZE+2 && Math.abs(my-hs[j].y)<=HANDLE_SIZE+2)
        return {idx:i, edge:hs[j].t};
    }
  }
  return null;
}

function hitBox(mx, my) {
  for (var i=drawnBoxes.length-1; i>=0; i--) {
    var b=drawnBoxes[i];
    if (mx>=b.x && mx<=b.x+b.w && my>=b.y && my<=b.y+b.h) return i;
  }
  return -1;
}

var cursorMap = {top:'n-resize',bottom:'s-resize',left:'w-resize',right:'e-resize',tl:'nw-resize',tr:'ne-resize',bl:'sw-resize',br:'se-resize'};

canvas.addEventListener('mousedown', function(e) {
  var r=canvas.getBoundingClientRect(), mx=e.clientX-r.left, my=e.clientY-r.top;
  if (selectedBoxIdx>=0) {
    var hh=hitHandle(mx,my);
    if (hh) { mode='edge_drag'; dragEdge=hh.edge; selectedBoxIdx=hh.idx; dragStartBox=Object.assign({},drawnBoxes[selectedBoxIdx]); startX=mx; startY=my; return; }
  }
  var bi=hitBox(mx,my);
  if (bi>=0) { selectedBoxIdx=bi; mode='move'; dragStartBox=Object.assign({},drawnBoxes[bi]); startX=mx; startY=my; redraw(); return; }
  selectedBoxIdx=-1; mode='drawing'; startX=mx; startY=my;
});

canvas.addEventListener('mousemove', function(e) {
  var r=canvas.getBoundingClientRect(), mx=e.clientX-r.left, my=e.clientY-r.top;
  if (mode==='drawing') { redraw(); ctx.setLineDash([4,4]); ctx.strokeStyle='#60a5fa'; ctx.lineWidth=2; ctx.strokeRect(startX,startY,mx-startX,my-startY); ctx.setLineDash([]); return; }
  if (mode==='edge_drag' && selectedBoxIdx>=0) {
    var b=drawnBoxes[selectedBoxIdx], dx=mx-startX, dy=my-startY, s=dragStartBox;
    if (dragEdge==='left'){b.x=s.x+dx;b.w=s.w-dx;} if (dragEdge==='right'){b.w=s.w+dx;}
    if (dragEdge==='top'){b.y=s.y+dy;b.h=s.h-dy;} if (dragEdge==='bottom'){b.h=s.h+dy;}
    if (dragEdge==='tl'){b.x=s.x+dx;b.w=s.w-dx;b.y=s.y+dy;b.h=s.h-dy;}
    if (dragEdge==='tr'){b.w=s.w+dx;b.y=s.y+dy;b.h=s.h-dy;}
    if (dragEdge==='bl'){b.x=s.x+dx;b.w=s.w-dx;b.h=s.h+dy;}
    if (dragEdge==='br'){b.w=s.w+dx;b.h=s.h+dy;}
    redraw();
    document.getElementById('status').textContent='Resizing fig_'+(selectedBoxIdx+1)+': ['+Math.round(b.x)+','+Math.round(b.y)+','+Math.round(b.x+b.w)+','+Math.round(b.y+b.h)+']';
    return;
  }
  if (mode==='move' && selectedBoxIdx>=0) {
    var b=drawnBoxes[selectedBoxIdx]; b.x=dragStartBox.x+(mx-startX); b.y=dragStartBox.y+(my-startY); redraw(); return;
  }
  if (selectedBoxIdx>=0) { var hh=hitHandle(mx,my); if (hh){canvas.style.cursor=cursorMap[hh.edge]||'move';return;} }
  canvas.style.cursor = hitBox(mx,my)>=0 ? 'move' : 'crosshair';
});

canvas.addEventListener('mouseup', function(e) {
  var r=canvas.getBoundingClientRect(), mx=e.clientX-r.left, my=e.clientY-r.top;
  if (mode==='drawing') {
    var x=Math.min(startX,mx), y=Math.min(startY,my), w=Math.abs(mx-startX), h=Math.abs(my-startY);
    if (w>10&&h>10) { drawnBoxes.push({x:x,y:y,w:w,h:h}); selectedBoxIdx=drawnBoxes.length-1;
      document.getElementById('status').textContent='Drew fig_'+drawnBoxes.length+'. Drag handles to adjust edges.'; }
  }
  if ((mode==='edge_drag'||mode==='move') && selectedBoxIdx>=0) {
    var b=drawnBoxes[selectedBoxIdx];
    if (b.w<0){b.x+=b.w;b.w=-b.w;} if (b.h<0){b.y+=b.h;b.h=-b.h;}
  }
  mode='idle'; dragEdge=null; dragStartBox=null; redraw();
});

function clearDrawn() { drawnBoxes=[]; selectedBoxIdx=-1; mode='idle'; redraw(); document.getElementById('status').textContent='Cleared all drawn boxes.'; }

function deleteSelected() {
  if (selectedBoxIdx>=0 && selectedBoxIdx<drawnBoxes.length) { drawnBoxes.splice(selectedBoxIdx,1); selectedBoxIdx=-1; redraw(); document.getElementById('status').textContent='Deleted selected box.'; }
}

function navigate(dir) { var n=currentIdx+dir; if (n>=0&&n<pages.length) loadPage(n); }

function confirmPage() {
  if (currentIdx<0) return;
  var p=pages[currentIdx], newFigures;
  if (drawnBoxes.length>0) {
    var scale=1.0/p.dpi_scale;
    newFigures=drawnBoxes.map(function(box,i){return{
      figure_id:'gt_'+(i+1),
      bbox:[Math.round(box.x*scale*10)/10,Math.round(box.y*scale*10)/10,Math.round((box.x+box.w)*scale*10)/10,Math.round((box.y+box.h)*scale*10)/10],
      logical_group_id:'group_'+(i+1),panel_bboxes:[],caption_bbox:null,body_exclusion_bboxes:[]
    };});
  } else { newFigures=existingBoxes; }
  fetch('/save',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({page_id:p.page_id,status:'confirmed',figures:newFigures})})
  .then(function(r){return r.json();}).then(function(d){
    p.status='confirmed'; p.figures=newFigures;
    existingBoxes=newFigures.map(function(f){return Object.assign({},f);});
    drawnBoxes=[]; selectedBoxIdx=-1;
    updateBadge(currentIdx,'confirmed');
    document.getElementById('status').textContent='✓ '+p.page_id+' confirmed with '+newFigures.length+' figures.';
    redraw();
  });
}

function markNeedsFix() {
  if (currentIdx<0) return;
  var p=pages[currentIdx];
  fetch('/save',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({page_id:p.page_id,status:'needs_fix',figures:p.figures})})
  .then(function(r){return r.json();}).then(function(d){ p.status='needs_fix'; updateBadge(currentIdx,'needs_fix'); document.getElementById('status').textContent='✗ '+p.page_id+' marked as needs_fix.'; });
}

function savePage() { if (currentIdx<0) return; confirmPage(); }

function updateBadge(idx,status) {
  var item=document.getElementById('pitem-'+idx), badge=item.querySelector('.badge');
  badge.className='badge badge-'+status; badge.textContent=status;
}

document.addEventListener('keydown', function(e) {
  if (e.key==='ArrowRight'||e.key==='n') navigate(1);
  if (e.key==='ArrowLeft'||e.key==='p') navigate(-1);
  if (e.key==='c') confirmPage();
  if (e.key==='x') markNeedsFix();
  if (e.key==='Escape'){clearDrawn();e.preventDefault();}
  if (e.key==='Delete'||e.key==='Backspace'){deleteSelected();e.preventDefault();}
});

if (pages.length>0) loadPage(0);
</script>
</body>
</html>"""


def build_pages_data(dataset_root: Path) -> list[dict]:
    """Collect page data for the annotation UI."""
    gt_dir = dataset_root / "gt"
    meta_dir = dataset_root / "meta"
    pages = []

    for f in sorted(gt_dir.glob("*.json")):
        gt = json.loads(f.read_text(encoding="utf-8"))
        meta_path = meta_dir / f.name
        meta = json.loads(meta_path.read_text(encoding="utf-8")) if meta_path.exists() else {}

        page_dir = meta.get("page_dir", "")
        page_image = str(Path(page_dir) / "page.png") if page_dir else ""

        # Determine DPI scale: bboxes are in PDF points (72 DPI),
        # images rendered at RENDER_DPI (144).  scale = 144/72 = 2.0
        dpi_scale = 2.0  # RENDER_DPI / 72
        # Try to read actual render_dpi from summary.json
        if page_dir:
            summary_path = Path(page_dir).parent.parent / "summary.json"
            if summary_path.exists():
                try:
                    s = json.loads(summary_path.read_text(encoding="utf-8"))
                    rdpi = s.get("render_dpi", 144)
                    if rdpi:
                        dpi_scale = float(rdpi) / 72.0
                except Exception:
                    pass

        pages.append({
            "page_id": f.stem,
            "family": meta.get("figure_family", ""),
            "difficulty": meta.get("difficulty", ""),
            "doc_id": meta.get("doc_id", ""),
            "status": meta.get("review_status", "prelabeled"),
            "page_image": page_image,
            "dpi_scale": dpi_scale,
            "figures": gt.get("figures", []),
        })

    return pages


class AnnotationHandler(SimpleHTTPRequestHandler):
    """Serve the annotation UI and handle save requests."""

    dataset_root: Path
    pages_data: list[dict]

    def do_GET(self):
        parsed = urlparse(self.path)

        if parsed.path == "/":
            html = HTML_TEMPLATE.replace(
                "%%PAGES_JSON%%", json.dumps(self.pages_data)
            ).replace(
                "%%PAGE_COUNT%%", str(len(self.pages_data))
            )
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.end_headers()
            self.wfile.write(html.encode("utf-8"))

        elif parsed.path == "/image":
            params = parse_qs(parsed.query)
            img_path = params.get("path", [""])[0]
            if img_path and Path(img_path).exists():
                self.send_response(200)
                self.send_header("Content-Type", "image/png")
                self.end_headers()
                self.wfile.write(Path(img_path).read_bytes())
            else:
                self.send_response(404)
                self.end_headers()
        else:
            self.send_response(404)
            self.end_headers()

    def do_POST(self):
        if self.path == "/save":
            length = int(self.headers.get("Content-Length", 0))
            body = json.loads(self.rfile.read(length).decode("utf-8"))

            page_id = body["page_id"]
            status = body["status"]
            figures = body["figures"]

            # Update GT
            gt_path = self.dataset_root / "gt" / f"{page_id}.json"
            gt = json.loads(gt_path.read_text(encoding="utf-8"))
            gt["figures"] = figures
            gt_path.write_text(
                json.dumps(gt, indent=2, ensure_ascii=False) + "\n",
                encoding="utf-8",
            )

            # Update meta
            meta_path = self.dataset_root / "meta" / f"{page_id}.json"
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
            meta["review_status"] = status
            meta_path.write_text(
                json.dumps(meta, indent=2, ensure_ascii=False) + "\n",
                encoding="utf-8",
            )

            # Update in-memory data
            for p in self.pages_data:
                if p["page_id"] == page_id:
                    p["status"] = status
                    p["figures"] = figures
                    break

            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(json.dumps({"ok": True}).encode("utf-8"))
            print(f"  [SAVED] {page_id} → {status}, {len(figures)} figures")
        else:
            self.send_response(404)
            self.end_headers()

    def log_message(self, format, *args):
        pass  # suppress default logging


def main() -> int:
    parser = argparse.ArgumentParser(description="Visual GT annotation tool for JournalMix.")
    parser.add_argument(
        "--dataset-root", type=Path,
        default=Path("data/private/journalmix_v1"),
    )
    parser.add_argument("--port", type=int, default=8765)
    args = parser.parse_args()

    dataset_root = args.dataset_root.resolve()
    print(f"[annotator] Loading pages from {dataset_root}...")
    pages_data = build_pages_data(dataset_root)
    print(f"[annotator] Loaded {len(pages_data)} pages")

    # Attach data to handler class
    AnnotationHandler.dataset_root = dataset_root
    AnnotationHandler.pages_data = pages_data

    server = HTTPServer(("localhost", args.port), AnnotationHandler)
    print(f"[annotator] Open http://localhost:{args.port} in your browser")
    print(f"[annotator] Shortcuts: ←/→ navigate, C confirm, X needs_fix, Esc clear")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\n[annotator] Stopped.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
