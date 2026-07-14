"""
bitmaptodxf.com -> DXFstudio.com redirect service.

Replaces the old Streamlit app entirely. Behavior:

  * Homepage ("/"): shows a short branded "We've moved" notice, then
    auto-forwards to DXFstudio after REDIRECT_DELAY_SECONDS (default 6).
    Set REDIRECT_DELAY_SECONDS=0 in Railway to skip the notice and
    issue an instant 301 instead.
  * Every other path (old bookmarks, guide pages, etc.): instant
    HTTP 301 permanent redirect to DXFstudio. 301s tell Google to
    transfer bitmaptodxf's search ranking to the new domain.

Environment variables (all optional):
  TARGET_URL               default "https://www.dxfstudio.com"
  REDIRECT_DELAY_SECONDS   default "6"  (0 = instant 301 everywhere)
"""

import os
from flask import Flask, redirect, request

app = Flask(__name__)

TARGET = os.environ.get("TARGET_URL", "https://www.dxfstudio.com").rstrip("/")
DELAY = int(os.environ.get("REDIRECT_DELAY_SECONDS", "6"))

NOTICE_HTML = """<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<meta http-equiv="refresh" content="{delay};url={target}">
<title>Bitmap to DXF has moved to DXF Studio</title>
<link rel="canonical" href="{target}">
<style>
  body {{
    margin: 0; min-height: 100vh; display: flex;
    align-items: center; justify-content: center;
    background: #0e1117; color: #fafafa;
    font-family: -apple-system, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
    text-align: center;
  }}
  .card {{
    max-width: 560px; padding: 48px 36px; border-radius: 16px;
    background: #161b24; box-shadow: 0 8px 40px rgba(0,0,0,.45);
  }}
  h1 {{ font-size: 1.6rem; margin: 0 0 12px; }}
  p  {{ color: #b9c2d0; line-height: 1.55; margin: 0 0 24px; }}
  a.btn {{
    display: inline-block; padding: 13px 30px; border-radius: 10px;
    background: #2457e6; color: #fff; text-decoration: none;
    font-weight: 600; font-size: 1.02rem;
  }}
  a.btn:hover {{ background: #1d47c2; }}
  .count {{ margin-top: 18px; font-size: .85rem; color: #7a8698; }}
</style>
</head>
<body>
  <div class="card">
    <h1>Bitmap&nbsp;to&nbsp;DXF is now part of <span style="color:#5b8bff">DXF&nbsp;Studio</span></h1>
    <p>Same conversion engine, plus DXF hatching, toolpath optimization,
       and cleanup tools &mdash; all in one place at our new home.</p>
    <a class="btn" href="{target}">Continue to DXFstudio.com &rarr;</a>
    <div class="count">Redirecting automatically in <span id="s">{delay}</span>&nbsp;seconds&hellip;</div>
  </div>
  <script>
    var s = {delay}, el = document.getElementById('s');
    var t = setInterval(function () {{
      s -= 1;
      if (s <= 0) {{ clearInterval(t); window.location.href = "{target}"; }}
      else el.textContent = s;
    }}, 1000);
  </script>
</body>
</html>"""


@app.route("/", defaults={"path": ""})
@app.route("/<path:path>")
def catch_all(path):
    # Instant permanent redirect for deep links / bookmarks, or for
    # everything when the delay is set to 0.
    if path or DELAY <= 0:
        return redirect(TARGET, code=301)
    # Homepage: friendly notice with countdown, then auto-forward.
    return NOTICE_HTML.format(target=TARGET, delay=DELAY), 200


@app.route("/healthz")
def healthz():
    return "ok", 200


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 8501)))
