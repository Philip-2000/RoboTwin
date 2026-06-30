from __future__ import annotations

import argparse
import json
import mimetypes
import re
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, quote, unquote, urlparse


DEFAULT_WORLD_ROOT = Path("/home/users/liang01.yue/D/WorldArena_Robotwin2.0")


def _safe_resolve(path: str | Path, roots: tuple[Path, ...]) -> Path:
    resolved = Path(path).expanduser().resolve()
    for root in roots:
        try:
            resolved.relative_to(root.resolve())
            return resolved
        except ValueError:
            continue
    raise PermissionError(f"Path is outside allowed roots: {resolved}")


def _read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _episode_from_name(path: Path) -> int | None:
    match = re.search(r"episode(\d+)", path.name)
    return None if match is None else int(match.group(1))


def _score_from_report(data: dict) -> float | None:
    try:
        return float(data["candidates"][0]["parameters"]["render_search"]["score"])
    except Exception:
        return None


def _task_from_report(data: dict) -> str:
    return str(data.get("task_name") or data.get("candidates", [{}])[0].get("task_name") or "unknown")


def _image_url(path: str | Path | None) -> str | None:
    if path is None:
        return None
    return "/file?path=" + quote(str(path))


def _attempts(report: dict) -> list[dict]:
    try:
        attempts = report["candidates"][0]["parameters"]["render_search"]["sapien_search"]["attempts"]
    except Exception:
        return []
    out = []
    for item in attempts:
        score = None
        try:
            score = float(item["sapien_score"]["score"])
        except Exception:
            pass
        out.append(
            {
                "attempt_index": item.get("attempt_index"),
                "tag": item.get("tag"),
                "score": score,
                "objectwise_target": item.get("objectwise_target"),
                "objectwise_score": item.get("objectwise_score"),
                "comparison_url": _image_url(item.get("comparison_path")),
                "render_url": _image_url(item.get("render_path")),
                "parameters": item.get("parameters", {}),
                "per_object": item.get("sapien_score", {}).get("per_object", []),
                "error": item.get("error"),
            }
        )
    return out


def _best_attempt(report: dict) -> dict | None:
    attempts = _attempts(report)
    if not attempts:
        return None
    try:
        best_index = report["candidates"][0]["parameters"]["render_search"]["sapien_search"]["best_attempt_index"]
    except Exception:
        best_index = None
    for item in attempts:
        if item["attempt_index"] == best_index:
            return item
    return max(attempts, key=lambda item: -1 if item["score"] is None else item["score"])


def _related_images(scene_root: Path, split: str, episode: int) -> dict[str, list[dict]]:
    pattern = f"episode{episode}.*"
    images = []
    for path in scene_root.rglob(pattern):
        if path.suffix.lower() not in {".jpg", ".jpeg", ".png"}:
            continue
        if split not in str(path):
            continue
        images.append(
            {
                "name": path.name,
                "group": path.parent.parent.parent.name if len(path.parts) >= 3 else path.parent.name,
                "path": str(path),
                "url": _image_url(path),
            }
        )
    images.sort(key=lambda item: (item["group"], item["name"]))
    return {"images": images}


def build_index(world_root: Path) -> dict:
    scene_root = world_root / "scene_recon"
    reports = []
    for path in scene_root.rglob("episode*.scene_recon.json"):
        try:
            data = _read_json(path)
        except Exception:
            continue
        episode = _episode_from_name(path)
        if episode is None:
            continue
        try:
            split = data["episode"]["split"]
        except Exception:
            split = "unknown"
        rel = path.relative_to(scene_root)
        report_set = rel.parts[0] if rel.parts else "scene_recon"
        reports.append(
            {
                "episode": episode,
                "split": split,
                "task": _task_from_report(data),
                "score": _score_from_report(data),
                "report_set": report_set,
                "report_path": str(path),
                "report_url": "/api/report?path=" + quote(str(path)),
            }
        )
    reports.sort(key=lambda item: (item["split"], item["task"], item["episode"], item["report_set"]))
    return {
        "world_root": str(world_root),
        "scene_root": str(scene_root),
        "reports": reports,
        "report_sets": sorted({item["report_set"] for item in reports}),
        "tasks": sorted({item["task"] for item in reports}),
        "splits": sorted({item["split"] for item in reports}),
    }


def report_payload(report_path: Path, world_root: Path) -> dict:
    data = _read_json(report_path)
    episode = int(data["episode"]["episode"])
    split = str(data["episode"]["split"])
    candidate = data.get("candidates", [{}])[0]
    render_search = candidate.get("parameters", {}).get("render_search", {})
    best = _best_attempt(data)
    scene_root = world_root / "scene_recon"
    detection_path = scene_root / "detections_simple_cv" / split / "fixed_scene_task" / f"episode{episode}.detections.jpg"
    related = _related_images(scene_root, split, episode)
    return {
        "summary": {
            "episode": episode,
            "split": split,
            "task": _task_from_report(data),
            "instruction": data["episode"].get("instruction"),
            "score": render_search.get("score"),
            "renderer": render_search.get("renderer"),
            "strategy": render_search.get("sapien_search", {}).get("strategy"),
            "attempt_count": render_search.get("sapien_search", {}).get("attempt_count"),
            "report_path": str(report_path),
        },
        "images": {
            "first_frame": _image_url(data["episode"].get("first_frame_path")),
            "detections": _image_url(detection_path) if detection_path.exists() else None,
            "render_search": _image_url(render_search.get("visualization_path")),
            "best_comparison": None if best is None else best.get("comparison_url"),
        },
        "attempts": _attempts(data),
        "best_attempt": best,
        "candidate_parameters": candidate.get("parameters", {}),
        "related": related,
        "raw": data,
    }


HTML = r"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>SceneRecon Viewer</title>
  <style>
    :root {
      --bg: #f6f7f8;
      --panel: #ffffff;
      --ink: #1d252c;
      --muted: #67727e;
      --line: #d8dee5;
      --accent: #0c7c59;
      --bad: #a43838;
      --soft: #eaf3ef;
    }
    * { box-sizing: border-box; }
    body {
      margin: 0;
      font-family: ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
      background: var(--bg);
      color: var(--ink);
    }
    header {
      height: 54px;
      display: flex;
      align-items: center;
      gap: 16px;
      padding: 0 18px;
      border-bottom: 1px solid var(--line);
      background: #fff;
      position: sticky;
      top: 0;
      z-index: 10;
    }
    header h1 {
      margin: 0;
      font-size: 18px;
      font-weight: 700;
      letter-spacing: 0;
    }
    header .meta { color: var(--muted); font-size: 13px; }
    main {
      display: grid;
      grid-template-columns: 360px minmax(0, 1fr);
      min-height: calc(100vh - 54px);
    }
    aside {
      border-right: 1px solid var(--line);
      background: #fff;
      padding: 14px;
      overflow: auto;
      max-height: calc(100vh - 54px);
    }
    .filters {
      display: grid;
      grid-template-columns: 1fr 1fr;
      gap: 8px;
      margin-bottom: 12px;
    }
    select, input {
      width: 100%;
      border: 1px solid var(--line);
      border-radius: 6px;
      padding: 8px 9px;
      background: #fff;
      color: var(--ink);
      font-size: 13px;
    }
    .list {
      display: flex;
      flex-direction: column;
      gap: 6px;
    }
    .row {
      border: 1px solid var(--line);
      border-radius: 7px;
      padding: 9px;
      cursor: pointer;
      background: #fff;
    }
    .row:hover { border-color: #9bb9ad; background: #fbfdfc; }
    .row.active { border-color: var(--accent); background: var(--soft); }
    .row-top { display: flex; justify-content: space-between; gap: 8px; align-items: center; }
    .episode { font-weight: 700; font-size: 14px; }
    .task { color: var(--muted); font-size: 12px; margin-top: 4px; overflow-wrap: anywhere; }
    .score {
      font-variant-numeric: tabular-nums;
      color: var(--accent);
      font-weight: 700;
      font-size: 13px;
    }
    section.content {
      padding: 16px;
      overflow: auto;
      max-height: calc(100vh - 54px);
    }
    .summary {
      display: grid;
      grid-template-columns: minmax(0, 1fr) auto;
      gap: 16px;
      align-items: start;
      margin-bottom: 14px;
    }
    .summary h2 { margin: 0 0 5px; font-size: 20px; }
    .instruction { color: var(--muted); font-size: 13px; line-height: 1.4; max-width: 900px; }
    .stats {
      display: grid;
      grid-template-columns: repeat(4, minmax(86px, auto));
      gap: 8px;
    }
    .stat {
      border: 1px solid var(--line);
      border-radius: 7px;
      background: #fff;
      padding: 8px;
    }
    .stat .label { color: var(--muted); font-size: 11px; }
    .stat .value { font-weight: 700; margin-top: 3px; font-size: 14px; }
    .image-grid {
      display: grid;
      grid-template-columns: repeat(2, minmax(0, 1fr));
      gap: 12px;
      margin-bottom: 14px;
    }
    figure {
      margin: 0;
      background: #fff;
      border: 1px solid var(--line);
      border-radius: 7px;
      overflow: hidden;
      min-height: 160px;
    }
    figcaption {
      padding: 8px 10px;
      font-size: 12px;
      color: var(--muted);
      border-bottom: 1px solid var(--line);
      display: flex;
      justify-content: space-between;
      gap: 8px;
    }
    figure img {
      display: block;
      width: 100%;
      max-height: 520px;
      object-fit: contain;
      background: #f1f2f1;
    }
    .empty {
      padding: 24px;
      color: var(--muted);
      font-size: 13px;
    }
    .attempts {
      display: grid;
      grid-template-columns: minmax(360px, 520px) minmax(0, 1fr);
      gap: 12px;
      align-items: start;
    }
    table {
      width: 100%;
      border-collapse: collapse;
      background: #fff;
      border: 1px solid var(--line);
      border-radius: 7px;
      overflow: hidden;
      font-size: 12px;
    }
    th, td { padding: 7px 8px; border-bottom: 1px solid var(--line); text-align: left; }
    th { color: var(--muted); background: #fbfbfb; font-weight: 600; position: sticky; top: 0; }
    tr { cursor: pointer; }
    tr.selected { background: var(--soft); }
    .bad { color: var(--bad); }
    pre {
      white-space: pre-wrap;
      overflow-wrap: anywhere;
      background: #fff;
      border: 1px solid var(--line);
      border-radius: 7px;
      padding: 10px;
      font-size: 12px;
      line-height: 1.4;
      max-height: 340px;
      overflow: auto;
    }
    .thumbs {
      display: grid;
      grid-template-columns: repeat(auto-fill, minmax(220px, 1fr));
      gap: 10px;
      margin-top: 14px;
    }
    @media (max-width: 1100px) {
      main { grid-template-columns: 1fr; }
      aside { max-height: 360px; border-right: 0; border-bottom: 1px solid var(--line); }
      section.content { max-height: none; }
      .image-grid, .attempts, .summary { grid-template-columns: 1fr; }
      .stats { grid-template-columns: repeat(2, minmax(86px, auto)); }
    }
  </style>
</head>
<body>
  <header>
    <h1>SceneRecon Viewer</h1>
    <div class="meta" id="rootMeta"></div>
  </header>
  <main>
    <aside>
      <div class="filters">
        <select id="splitFilter"></select>
        <select id="taskFilter"></select>
        <select id="setFilter"></select>
        <input id="searchBox" placeholder="episode or task" />
      </div>
      <div class="meta" id="countMeta"></div>
      <div class="list" id="reportList"></div>
    </aside>
    <section class="content" id="content">
      <div class="empty">Loading reports...</div>
    </section>
  </main>
  <script>
    const state = { index: null, selected: null, report: null, attemptIndex: null };
    const $ = (id) => document.getElementById(id);
    const fmt = (v) => v === null || v === undefined || Number.isNaN(v) ? "n/a" : Number(v).toFixed(3);
    const esc = (s) => String(s ?? "").replace(/[&<>"']/g, c => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));

    async function loadIndex() {
      const res = await fetch('/api/index');
      state.index = await res.json();
      $('rootMeta').textContent = state.index.scene_root;
      fillSelect('splitFilter', ['all', ...state.index.splits]);
      fillSelect('taskFilter', ['all', ...state.index.tasks]);
      fillSelect('setFilter', ['all', ...state.index.report_sets]);
      ['splitFilter','taskFilter','setFilter','searchBox'].forEach(id => $(id).addEventListener('input', renderList));
      renderList();
    }

    function fillSelect(id, values) {
      $(id).innerHTML = values.map(v => `<option value="${esc(v)}">${esc(v)}</option>`).join('');
    }

    function filteredReports() {
      const split = $('splitFilter').value;
      const task = $('taskFilter').value;
      const set = $('setFilter').value;
      const q = $('searchBox').value.trim().toLowerCase();
      return state.index.reports.filter(r => {
        if (split !== 'all' && r.split !== split) return false;
        if (task !== 'all' && r.task !== task) return false;
        if (set !== 'all' && r.report_set !== set) return false;
        if (q && !(`${r.episode} ${r.task} ${r.report_set}`.toLowerCase().includes(q))) return false;
        return true;
      });
    }

    function renderList() {
      const reports = filteredReports();
      $('countMeta').textContent = `${reports.length} reports`;
      $('reportList').innerHTML = reports.map((r, i) => `
        <div class="row ${state.selected && state.selected.report_path === r.report_path ? 'active' : ''}" data-path="${esc(r.report_path)}">
          <div class="row-top"><div class="episode">episode${r.episode}</div><div class="score">${fmt(r.score)}</div></div>
          <div class="task">${esc(r.task)} · ${esc(r.report_set)}</div>
        </div>
      `).join('');
      document.querySelectorAll('.row').forEach(row => row.addEventListener('click', () => selectReport(row.dataset.path)));
      if (!state.selected && reports.length) selectReport(reports[0].report_path);
    }

    async function selectReport(path) {
      const item = state.index.reports.find(r => r.report_path === path);
      state.selected = item;
      renderList();
      $('content').innerHTML = '<div class="empty">Loading report...</div>';
      const res = await fetch('/api/report?path=' + encodeURIComponent(path));
      state.report = await res.json();
      const attempts = state.report.attempts || [];
      const best = state.report.best_attempt;
      state.attemptIndex = best ? best.attempt_index : (attempts[0] ? attempts[0].attempt_index : null);
      renderReport();
    }

    function imageFigure(label, url) {
      if (!url) return `<figure><figcaption>${esc(label)}</figcaption><div class="empty">No image</div></figure>`;
      return `<figure><figcaption><span>${esc(label)}</span><a href="${url}" target="_blank">open</a></figcaption><img src="${url}" /></figure>`;
    }

    function renderReport() {
      const r = state.report;
      const s = r.summary;
      const attempts = r.attempts || [];
      const selectedAttempt = attempts.find(a => a.attempt_index === state.attemptIndex) || attempts[0];
      $('content').innerHTML = `
        <div class="summary">
          <div>
            <h2>episode${s.episode} · ${esc(s.task)}</h2>
            <div class="instruction">${esc(s.instruction)}</div>
          </div>
          <div class="stats">
            <div class="stat"><div class="label">score</div><div class="value">${fmt(s.score)}</div></div>
            <div class="stat"><div class="label">strategy</div><div class="value">${esc(s.strategy || 'n/a')}</div></div>
            <div class="stat"><div class="label">attempts</div><div class="value">${esc(s.attempt_count || attempts.length)}</div></div>
            <div class="stat"><div class="label">renderer</div><div class="value">${esc(s.renderer || 'n/a')}</div></div>
          </div>
        </div>
        <div class="image-grid">
          ${imageFigure('first frame', r.images.first_frame)}
          ${imageFigure('detections', r.images.detections)}
          ${imageFigure('best comparison', r.images.best_comparison)}
          ${imageFigure('render-search summary', r.images.render_search)}
        </div>
        <div class="attempts">
          <div>
            ${attemptTable(attempts)}
          </div>
          <div>
            ${selectedAttempt ? imageFigure(`attempt ${selectedAttempt.attempt_index} · ${fmt(selectedAttempt.score)}`, selectedAttempt.comparison_url) : '<div class="empty">No attempts</div>'}
            <pre>${esc(JSON.stringify(selectedAttempt || {}, null, 2))}</pre>
          </div>
        </div>
        <div class="thumbs">${(r.related.images || []).map(img => imageFigure(`${img.group} · ${img.name}`, img.url)).join('')}</div>
      `;
      document.querySelectorAll('[data-attempt]').forEach(row => row.addEventListener('click', () => {
        state.attemptIndex = Number(row.dataset.attempt);
        renderReport();
      }));
    }

    function attemptTable(attempts) {
      if (!attempts.length) return '<div class="empty">No SAPIEN attempts in this report.</div>';
      return `<table>
        <thead><tr><th>#</th><th>score</th><th>tag</th><th>object</th></tr></thead>
        <tbody>
          ${attempts.map(a => `
            <tr data-attempt="${a.attempt_index}" class="${a.attempt_index === state.attemptIndex ? 'selected' : ''}">
              <td>${esc(a.attempt_index)}</td>
              <td class="${a.error ? 'bad' : ''}">${a.error ? 'error' : fmt(a.score)}</td>
              <td>${esc(a.tag || '')}</td>
              <td>${esc(a.objectwise_target || '')}</td>
            </tr>
          `).join('')}
        </tbody>
      </table>`;
    }

    loadIndex().catch(err => {
      $('content').innerHTML = `<div class="empty bad">${esc(err.stack || err)}</div>`;
    });
  </script>
</body>
</html>
"""


class ViewerHandler(BaseHTTPRequestHandler):
    world_root: Path = DEFAULT_WORLD_ROOT
    allowed_roots: tuple[Path, ...] = (DEFAULT_WORLD_ROOT, Path.cwd())

    def log_message(self, fmt: str, *args) -> None:
        return

    def _send_json(self, data: dict) -> None:
        body = json.dumps(data, ensure_ascii=False).encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _send_error(self, status: int, message: str) -> None:
        body = json.dumps({"error": message}, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self) -> None:
        parsed = urlparse(self.path)
        try:
            if parsed.path == "/":
                body = HTML.encode("utf-8")
                self.send_response(200)
                self.send_header("Content-Type", "text/html; charset=utf-8")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)
                return
            if parsed.path == "/api/index":
                self._send_json(build_index(self.world_root))
                return
            if parsed.path == "/api/report":
                query = parse_qs(parsed.query)
                raw_path = query.get("path", [None])[0]
                if raw_path is None:
                    self._send_error(400, "Missing path")
                    return
                report_path = _safe_resolve(unquote(raw_path), self.allowed_roots)
                self._send_json(report_payload(report_path, self.world_root))
                return
            if parsed.path == "/file":
                query = parse_qs(parsed.query)
                raw_path = query.get("path", [None])[0]
                if raw_path is None:
                    self._send_error(400, "Missing path")
                    return
                path = _safe_resolve(unquote(raw_path), self.allowed_roots)
                if not path.exists() or not path.is_file():
                    self._send_error(404, "File not found")
                    return
                body = path.read_bytes()
                ctype = mimetypes.guess_type(str(path))[0] or "application/octet-stream"
                self.send_response(200)
                self.send_header("Content-Type", ctype)
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)
                return
            self._send_error(404, "Not found")
        except Exception as exc:
            self._send_error(500, repr(exc))


def serve(world_root: Path, host: str, port: int) -> None:
    world_root = world_root.expanduser().resolve()
    ViewerHandler.world_root = world_root
    ViewerHandler.allowed_roots = (world_root, Path.cwd().resolve())
    server = ThreadingHTTPServer((host, port), ViewerHandler)
    print(f"SceneRecon viewer: http://{host}:{port}")
    print(f"WorldArena root: {world_root}")
    server.serve_forever()


def main() -> None:
    parser = argparse.ArgumentParser(description="Serve a local SceneRecon visualization dashboard.")
    parser.add_argument("--worldarena-root", default=str(DEFAULT_WORLD_ROOT))
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    args = parser.parse_args()
    serve(Path(args.worldarena_root), args.host, args.port)


if __name__ == "__main__":
    main()
