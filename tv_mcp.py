"""MCP server that lets an AI agent see and control an Android / Google TV over ADB.

Configure with env vars:
  TV_IP    (default 192.168.1.10)
  TV_PORT  (default 5555)
  ADB_PATH (default: `adb` on PATH)
"""

from __future__ import annotations

import asyncio
import gzip
import io
import os
import re
import shlex
import shutil
import xml.etree.ElementTree as ET
from urllib.parse import quote

from mcp.server.mcpserver import Image, MCPServer
from mcp.server.mcpserver.exceptions import ToolError
from mcp.types import ToolAnnotations
from PIL import Image as PILImage

TV_IP = os.environ.get("TV_IP", "192.168.1.10")
TV_PORT = os.environ.get("TV_PORT", "5555")
SERIAL = f"{TV_IP}:{TV_PORT}"
ADB = os.environ.get("ADB_PATH") or shutil.which("adb") or "/opt/homebrew/bin/adb"

INSTRUCTIONS = """\
Controls an Android/Google TV over ADB. Work like a computer-use agent: observe -> act -> verify.
1. Observe: `screenshot` shows the screen; `get_ui` lists elements, which one is FOCUSED, and coordinates.
   YouTube (and some other web-based apps) expose no UI tree, so use screenshots there.
2. Act: TVs are D-pad driven. `select_element` moves focus to an element by text and presses OK (most reliable
   in native apps); `press_keys` for manual D-pad/back/home/media keys; `tap`/`tap_element` only where touch works.
   Jump straight to content with `launch_app`, `play_youtube`, `open_url` (deep links), `search_tv`.
3. Verify with `screenshot`, `get_ui` or `now_playing` after each meaningful step; retry differently if it didn't work.
Playback: video frames appear BLACK in screenshots (hardware video layer), so use `now_playing` for title/state/position
and `media` for play/pause/seek. Netflix BLOCKS screenshots and exposes no UI tree: use the `netflix` tool (deep links by title id; find ids via web
search for "<title> netflix.com/title"), verify with `now_playing`, and use `media` for play/pause/seek.
Coordinates for `tap`/`swipe` are native screen pixels; screenshots state the scale factor.
"""

mcp = MCPServer("android-tv", instructions=INSTRUCTIONS)

# Friendly names -> Android keycodes.
KEYS = {
    "up": "KEYCODE_DPAD_UP", "down": "KEYCODE_DPAD_DOWN", "left": "KEYCODE_DPAD_LEFT",
    "right": "KEYCODE_DPAD_RIGHT", "ok": "KEYCODE_DPAD_CENTER", "select": "KEYCODE_DPAD_CENTER",
    "center": "KEYCODE_DPAD_CENTER", "enter": "KEYCODE_ENTER", "back": "KEYCODE_BACK",
    "home": "KEYCODE_HOME", "menu": "KEYCODE_MENU", "settings": "KEYCODE_SETTINGS",
    "search": "KEYCODE_SEARCH", "voice": "KEYCODE_VOICE_ASSIST", "assistant": "KEYCODE_ASSIST",
    "apps": "KEYCODE_ALL_APPS", "recents": "KEYCODE_APP_SWITCH", "notifications": "KEYCODE_NOTIFICATION",
    "play_pause": "KEYCODE_MEDIA_PLAY_PAUSE", "play": "KEYCODE_MEDIA_PLAY", "pause": "KEYCODE_MEDIA_PAUSE",
    "stop": "KEYCODE_MEDIA_STOP", "next": "KEYCODE_MEDIA_NEXT", "previous": "KEYCODE_MEDIA_PREVIOUS",
    "rewind": "KEYCODE_MEDIA_REWIND", "fast_forward": "KEYCODE_MEDIA_FAST_FORWARD",
    "volume_up": "KEYCODE_VOLUME_UP", "volume_down": "KEYCODE_VOLUME_DOWN", "mute": "KEYCODE_VOLUME_MUTE",
    "power": "KEYCODE_POWER", "sleep": "KEYCODE_SLEEP", "wakeup": "KEYCODE_WAKEUP",
    "input": "KEYCODE_TV_INPUT", "hdmi1": "KEYCODE_TV_INPUT_HDMI_1", "hdmi2": "KEYCODE_TV_INPUT_HDMI_2",
    "hdmi3": "KEYCODE_TV_INPUT_HDMI_3", "hdmi4": "KEYCODE_TV_INPUT_HDMI_4", "guide": "KEYCODE_GUIDE",
    "info": "KEYCODE_INFO", "channel_up": "KEYCODE_CHANNEL_UP", "channel_down": "KEYCODE_CHANNEL_DOWN",
    "captions": "KEYCODE_CAPTIONS", "delete": "KEYCODE_DEL", "backspace": "KEYCODE_DEL",
    "forward_delete": "KEYCODE_FORWARD_DEL", "space": "KEYCODE_SPACE", "tab": "KEYCODE_TAB",
    "escape": "KEYCODE_ESCAPE", "page_up": "KEYCODE_PAGE_UP", "page_down": "KEYCODE_PAGE_DOWN",
    "move_home": "KEYCODE_MOVE_HOME", "move_end": "KEYCODE_MOVE_END",
    **{str(d): f"KEYCODE_{d}" for d in range(10)},
}

# Friendly app names -> packages (anything else is matched against installed packages).
APP_ALIASES = {
    "netflix": "com.netflix.ninja", "youtube": "com.google.android.youtube.tv",
    "youtube music": "com.google.android.youtube.tvmusic", "prime video": "com.amazon.amazonvideo.livingroom",
    "prime": "com.amazon.amazonvideo.livingroom", "amazon": "com.amazon.amazonvideo.livingroom",
    "spotify": "com.spotify.tv.android", "hotstar": "in.startv.hotstar", "disney": "in.startv.hotstar",
    "jiohotstar": "in.startv.hotstar", "sonyliv": "com.sonyliv", "sony liv": "com.sonyliv",
    "zee5": "com.graymatrix.did", "jiocinema": "com.jio.media.stb.ondemand", "apple tv": "com.apple.atve.androidtv.appletv",
    "mx player": "com.mxtech.videoplayer.television", "smarttube": "com.teamsmart.videomanager.tv",
    "cloudstream": "com.lagradost.cloudstream3.prerelease", "browser": "com.tcl.browser",
    "play store": "com.android.vending", "google play": "com.android.vending", "settings": "com.android.tv.settings",
    "live tv": "com.tcl.tv", "tv": "com.tcl.tv", "google tv": "com.google.android.videos",
    "airscreen": "com.ionitech.airscreen", "gallery": "com.tcl.gallery", "media center": "com.tcl.ui_mediaCenter",
}


# --------------------------------------------------------------------------- adb plumbing

async def _exec(*args: str, timeout: float = 20) -> tuple[int, bytes, bytes]:
    proc = await asyncio.create_subprocess_exec(
        ADB, *args, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE
    )
    try:
        out, err = await asyncio.wait_for(proc.communicate(), timeout)
    except asyncio.TimeoutError:
        proc.kill()
        raise ToolError(f"adb {' '.join(args[:3])}... timed out after {timeout}s")
    return proc.returncode, out, err


async def _connect() -> str:
    try:
        _, out, err = await _exec("connect", SERIAL, timeout=10)
    except ToolError:
        raise ToolError(f"TV at {SERIAL} is unreachable (off, asleep on Wi-Fi, or IP changed). "
                        "Turn it on, check the IP, then call reconnect.")
    return (out + err).decode(errors="replace").strip()


_DISCONNECTED = ("not found", "offline", "no devices", "unauthorized", "closed", "failed to connect")


async def adb(*args: str, timeout: float = 20, binary: bool = False, check: bool = True):
    """Run `adb -s SERIAL ...`, reconnecting once if the TV dropped off."""
    for attempt in range(2):
        rc, out, err = await _exec("-s", SERIAL, *args, timeout=timeout)
        msg = err.decode(errors="replace").strip()
        if rc != 0 and attempt == 0 and any(s in msg.lower() for s in _DISCONNECTED):
            await _connect()
            continue
        if rc != 0 and check:
            if "unauthorized" in msg.lower():
                raise ToolError("TV rejected ADB: accept the 'Allow USB debugging' prompt on the TV, then retry.")
            raise ToolError(f"adb {' '.join(args)} failed: {msg or out.decode(errors='replace')[:500]}")
        if binary:
            return out
        text = out.decode(errors="replace").strip()
        # Unchecked callers inspect the text for failures (am start reports "Error:" on stderr).
        return text if check or not msg else f"{text}\n{msg}".strip()
    raise ToolError(f"Cannot reach TV at {SERIAL}. Is it on and on the same network? Last error: {msg}")


async def shell(cmd: str, timeout: float = 20, check: bool = True) -> str:
    return await adb("shell", cmd, timeout=timeout, check=check)


async def screen_size() -> tuple[int, int]:
    m = re.search(r"(\d+)x(\d+)", await shell("wm size"))
    return (int(m[1]), int(m[2])) if m else (1920, 1080)


async def foreground() -> str:
    for cmd in ("dumpsys activity activities | grep -m1 -E 'mResumedActivity|topResumedActivity'",
                "dumpsys window | grep -m1 -E 'mCurrentFocus|mFocusedApp'"):
        m = re.search(r"u0 ([\w.]+)/([\w.$]+)", await shell(cmd, check=False))
        if m:
            return f"{m[1]}/{m[2]}"
    return "unknown (transitioning)"


def _key(name: str) -> str:
    k = name.strip().lower().replace(" ", "_").replace("-", "_")
    if k in KEYS:
        return KEYS[k]
    if name.upper().startswith("KEYCODE_") or name.isdigit():
        return name.upper()
    raise ToolError(f"Unknown key '{name}'. Use one of: {', '.join(sorted(KEYS))}, or any KEYCODE_* name.")


# --------------------------------------------------------------------------- seeing

async def _capture() -> PILImage.Image:
    # Raw RGBA gzipped on the TV is ~2x faster over Wi-Fi than PNG.
    data = await adb("exec-out", "screencap | gzip -1", binary=True, timeout=30)
    try:
        raw = gzip.decompress(data)
        w, h, fmt = (int.from_bytes(raw[i:i + 4], "little") for i in (0, 4, 8))
        header = len(raw) - w * h * 4
        if fmt == 1 and header in (12, 16):
            return PILImage.frombuffer("RGBA", (w, h), raw[header:], "raw", "RGBA", 0, 1).convert("RGB")
    except (OSError, ValueError):
        pass
    png = await adb("exec-out", "screencap -p", binary=True, timeout=30)
    if not png:
        raise ToolError(f"The foreground app ({await foreground()}) blocks screen capture (secure window, e.g. Netflix). "
                        "You are blind here: use `netflix` deep links, `now_playing`, and press_keys.")
    return PILImage.open(io.BytesIO(png)).convert("RGB")


@mcp.tool(annotations=ToolAnnotations(title="Screenshot", read_only_hint=True))
async def screenshot(width: int = 1280, settle_ms: int = 400) -> list:
    """Capture what is currently on the TV screen.

    Args:
        width: Output width in pixels (aspect kept). 1280 is a good balance; use 1920 for small text.
        settle_ms: Wait this long first so focus animations finish.
    """
    if settle_ms:
        await asyncio.sleep(settle_ms / 1000)
    img = await _capture()
    sw, sh = img.size
    if width and width < sw:
        img = img.resize((width, round(sh * width / sw)), PILImage.LANCZOS)
    buf = io.BytesIO()
    img.save(buf, "JPEG", quality=80)
    scale = sw / img.size[0]
    note = (f"Screenshot {img.size[0]}x{img.size[1]} of a {sw}x{sh} screen. "
            f"Multiply image coordinates by {scale:.3g} for tap/swipe. Foreground: {await foreground()}")
    return [Image(data=buf.getvalue(), format="jpeg"), note]


def _center(bounds: str) -> tuple[int, int] | None:
    m = re.findall(r"\d+", bounds)
    if len(m) != 4:
        return None
    x1, y1, x2, y2 = map(int, m)
    return (x1 + x2) // 2, (y1 + y2) // 2


async def _ui_nodes() -> list[dict]:
    for _ in range(3):
        out = await adb("exec-out", "uiautomator dump /dev/tty", timeout=25, check=False)
        start, end = out.find("<?xml"), out.rfind("</hierarchy>")
        if start != -1 and end != -1:
            root = ET.fromstring(out[start:end + len("</hierarchy>")])
            nodes = []
            for el in root.iter("node"):
                n = dict(el.attrib)
                if not (n.get("text") or n.get("content-desc")):
                    # Containers (tabs, cards) often carry their label on children.
                    kids = [k.get("text") or k.get("content-desc") for k in el.iter("node") if k is not el]
                    n["label"] = " / ".join(dict.fromkeys(k for k in kids if k))[:100]
                nodes.append(n)
            return nodes
        await asyncio.sleep(0.7)  # "could not get idle state" while video/animations run
    raise ToolError(f"UI dump failed (screen may be playing video or animating): {out[-300:]}")


def _describe(n: dict) -> str:
    parts = []
    if n.get("text"):
        parts.append(f'"{n["text"][:80]}"')
    if n.get("content-desc"):
        parts.append(f'desc="{n["content-desc"][:80]}"')
    if n.get("label"):
        parts.append(f'label="{n["label"]}"')
    rid = n.get("resource-id", "").split("/")[-1]
    if rid and "obfuscated" not in rid:
        parts.append(f"id={rid}")
    parts.append(n.get("class", "").split(".")[-1])
    flags = [f for f in ("focused", "selected", "checked") if n.get(f) == "true"]
    if n.get("clickable") == "true":
        flags.append("clickable")
    if n.get("scrollable") == "true":
        flags.append("scrollable")
    if flags:
        parts.append("[" + ",".join(flags) + "]")
    c = _center(n.get("bounds", ""))
    if c:
        parts.append(f"@({c[0]},{c[1]})")
    return " ".join(parts).replace("\n", " ")


@mcp.tool(annotations=ToolAnnotations(title="Read on-screen UI", read_only_hint=True))
async def get_ui(filter: str = "", max_items: int = 120) -> str:
    """List on-screen UI elements (text, descriptions, ids, FOCUSED state, tap center coordinates).

    Cheaper and more precise than a screenshot for reading text and finding what is focused.

    Args:
        filter: Only include elements whose text/description/id contains this (case-insensitive).
        max_items: Cap on elements returned.
    """
    nodes = await _ui_nodes()
    q = filter.lower()
    lines, focused = [], None
    for n in nodes:
        meaningful = (n.get("text") or n.get("content-desc") or n.get("focused") == "true"
                      or n.get("selected") == "true" or (n.get("clickable") == "true" and n.get("label")))
        if not meaningful:
            continue
        hay = " ".join(n.get(k, "") for k in ("text", "content-desc", "label", "resource-id")).lower()
        if q and q not in hay:
            continue
        if n.get("focused") == "true":
            focused = _describe(n)
        lines.append(_describe(n))
    head = f"Foreground: {await foreground()}\nFocused: {focused or 'none reported'}\n"
    if len(lines) > max_items:
        lines = lines[:max_items] + [f"... {len(lines) - max_items} more (use filter)"]
    return head + "\n".join(lines or ["(no matching elements)"])


@mcp.tool(annotations=ToolAnnotations(title="TV status", read_only_hint=True))
async def tv_status() -> str:
    """Connection, power/screen state, foreground app, volume, screen size and device info."""
    model = await shell("getprop ro.product.manufacturer; getprop ro.product.model; getprop ro.build.version.release")
    power = await shell("dumpsys power | grep -E 'mWakefulness=|Display Power: state='", check=False)
    w, h = await screen_size()
    vol = await _volume()
    lines = [
        f"Connected: {SERIAL}",
        "Device: " + " ".join(model.split("\n")[:2]) + f" (Android {model.split()[-1]})",
        "Power: " + " | ".join(l.strip() for l in power.splitlines()),
        f"Screen: {w}x{h}",
        f"Foreground: {await foreground()}",
        f"Volume (media): {vol['level']}/{vol['max']} muted={vol['muted']} output={vol['output']}",
    ]
    return "\n".join(lines)


# --------------------------------------------------------------------------- acting

@mcp.tool(annotations=ToolAnnotations(title="Press remote keys"))
async def press_keys(keys: list[str], repeat: int = 1, delay_ms: int = 150) -> str:
    """Press remote-control buttons in order, like a remote.

    Names: up, down, left, right, ok, back, home, menu, settings, search, voice, apps,
    play_pause, play, pause, stop, next, previous, rewind, fast_forward, volume_up, volume_down,
    mute, power, sleep, wakeup, input, hdmi1-4, guide, info, channel_up, channel_down, captions,
    0-9, delete, space, enter. Any raw KEYCODE_* name also works.

    Args:
        keys: e.g. ["down", "down", "right", "ok"].
        repeat: Repeat the whole sequence N times (e.g. keys=["right"], repeat=5).
        delay_ms: Pause between presses so the UI keeps up.
    """
    codes = [_key(k) for k in keys] * max(1, repeat)
    if delay_ms <= 0:
        await shell("input keyevent " + " ".join(codes))
    else:
        for c in codes:
            await shell(f"input keyevent {c}")
            await asyncio.sleep(delay_ms / 1000)
    return f"Pressed {len(codes)} key(s): {' '.join(k.lower() for k in keys)}" + (f" x{repeat}" if repeat > 1 else "")


@mcp.tool(annotations=ToolAnnotations(title="Long-press key"))
async def long_press_key(key: str) -> str:
    """Long-press a remote button (e.g. 'ok' for context menus, 'home' for quick settings)."""
    await shell(f"input keyevent --longpress {_key(key)}")
    return f"Long-pressed {key}"


@mcp.tool(annotations=ToolAnnotations(title="Type text"))
async def type_text(text: str, submit: bool = False, clear_first: bool = False) -> str:
    """Type text into the focused text field (open a search box / keyboard first).

    Args:
        text: ASCII text to type. Non-ASCII characters are not supported by Android's input command.
        submit: Press Enter afterwards.
        clear_first: Select-all and delete existing text first.
    """
    if not text.isascii():
        raise ToolError("Only ASCII text can be typed over ADB. Try `open_url` with a search deep link instead.")
    if clear_first:
        await shell("input keyevent KEYCODE_MOVE_END " + " ".join(["KEYCODE_DEL"] * 60))
    # `input text` treats %s as a space; quote everything else for the device shell.
    for chunk in re.findall(r".{1,60}", text, re.S):
        await shell("input text " + shlex.quote(chunk.replace("%", "\\%").replace(" ", "%s")))
    if submit:
        await shell("input keyevent KEYCODE_ENTER")
    return f"Typed {len(text)} chars" + (" and pressed Enter" if submit else "")


@mcp.tool(annotations=ToolAnnotations(title="Tap"))
async def tap(x: int, y: int, long_press: bool = False) -> str:
    """Tap at native screen coordinates (from get_ui centers, or screenshot coords x scale factor)."""
    if long_press:
        await shell(f"input swipe {x} {y} {x} {y} 800")
    else:
        await shell(f"input tap {x} {y}")
    return f"{'Long-pressed' if long_press else 'Tapped'} ({x},{y})"


def _bounds(n: dict) -> tuple[int, int, int, int] | None:
    b = [int(v) for v in re.findall(r"\d+", n.get("bounds", ""))]
    return tuple(b) if len(b) == 4 else None


def _find(nodes: list[dict], text: str, exact: bool = False) -> list[dict]:
    """Elements matching text. Ranked: exact own text > partial own text > container label; then smallest."""
    q = text.lower()
    scored = []
    for n in nodes:
        b = _bounds(n)
        if not b:
            continue
        own = [n.get("text", ""), n.get("content-desc", ""), n.get("resource-id", "").split("/")[-1]]
        label = n.get("label", "")
        if any(f.lower() == q for f in own):
            rank = 0
        elif not exact and any(q in f.lower() for f in own if f):
            rank = 1
        elif label.lower() == q or (not exact and q in label.lower() and label.count(" / ") < 3):
            rank = 2
        else:
            continue
        scored.append((rank, (b[2] - b[0]) * (b[3] - b[1]), n))
    return [n for _, _, n in sorted(scored, key=lambda s: s[:2])]


def _pick(matches: list[dict], text: str, index: int) -> dict:
    if not matches:
        raise ToolError(f"No element matching '{text}'. Use get_ui (or a screenshot; some apps like YouTube "
                        "expose no UI tree) to see what's on screen.")
    if index >= len(matches):
        raise ToolError(f"Only {len(matches)} match(es) for '{text}'.")
    return matches[index]


@mcp.tool(annotations=ToolAnnotations(title="Tap element by text"))
async def tap_element(text: str, index: int = 0, exact: bool = False) -> str:
    """Find an on-screen element by its text/description/id and TAP its center.

    Many TV screens (e.g. Settings, launchers) ignore touch; if nothing happens use select_element.

    Args:
        text: Text to match (case-insensitive substring unless exact=True).
        index: Which match to use if several.
        exact: Require exact text/description match.
    """
    matches = _find(await _ui_nodes(), text, exact)
    n = _pick(matches, text, index)
    x, y = _center(n["bounds"])
    await shell(f"input tap {x} {y}")
    others = f" ({len(matches)} matches; others via index)" if len(matches) > 1 else ""
    return f"Tapped {_describe(n)}{others}"


def _contains(outer: tuple, x: int, y: int) -> bool:
    return outer[0] <= x <= outer[2] and outer[1] <= y <= outer[3]


@mcp.tool(annotations=ToolAnnotations(title="Select element with D-pad"))
async def select_element(text: str, index: int = 0, exact: bool = False, press_ok: bool = True,
                         max_steps: int = 30) -> str:
    """Move D-pad focus onto the element matching `text`, then press OK (like a person with a remote).

    The most reliable way to activate items on TV screens. Works when the target and the
    focused element both appear in get_ui.

    Args:
        text: Text to match (case-insensitive substring unless exact=True).
        index: Which match to use if several.
        exact: Require exact text/description match.
        press_ok: Press OK once focused (False = only move focus).
        max_steps: Give up after this many key presses.
    """
    path: list[str] = []
    last_focus, stuck, settle = None, set(), 0
    last_center, last_move, last_count, step = None, None, 1, {}
    for _ in range(max_steps + 3):
        nodes = await _ui_nodes()
        target = _pick(_find(nodes, text, exact), text, index)
        tb = _bounds(target)
        tx, ty = _center(target["bounds"])
        focused = [n for n in nodes if n.get("focused") == "true" and _bounds(n)]
        if not focused:
            raise ToolError(f"No focused element reported; can't navigate. Moves so far: {path}. "
                            "Try press_keys or tap_element.")
        f = min(focused, key=lambda n: (lambda b: (b[2] - b[0]) * (b[3] - b[1]))(_bounds(n)))
        fb = _bounds(f)
        fx, fy = _center(f["bounds"])
        # A focused list/grid (not an item inside it) means focus hasn't landed yet.
        container = f.get("scrollable") == "true" or f.get("label", "").count(" / ") >= 3
        if container and settle < 2:
            settle += 1
            await asyncio.sleep(0.6)
            continue
        if container:
            fb = (fx, fy, fx, fy)  # steer from its center
        # Focused item is (or sits inside / wraps) the target.
        elif _contains(fb, tx, ty) or _contains(tb, fx, fy):
            if press_ok:
                await shell("input keyevent KEYCODE_DPAD_CENTER")
                await asyncio.sleep(0.8)
            return (f"{'Selected' if press_ok else 'Focused'} {_describe(target)} after {len(path)} move(s)"
                    f"{': ' + ' '.join(path) if path else ''}. Foreground: {await foreground()}")
        if f.get("bounds") == last_focus and path:
            stuck.add(path[-1])  # last move didn't change focus
        vertical = "down" if ty > fb[3] else "up" if ty < fb[1] else None
        horizontal = "right" if tx > fb[2] else "left" if tx < fb[0] else None
        options = [d for d in (vertical, horizontal) if d and d not in stuck]
        if not options:
            raise ToolError(f"Focus is stuck on {_describe(f)} and can't reach {_describe(target)}. "
                            f"Moves: {path}. Try press_keys manually.")
        move = options[0]
        # Learn how far one press moves focus, then batch presses toward the target.
        if path and last_move == path[-1] and f.get("bounds") != last_focus and last_center:
            moved = abs(fx - last_center[0]) if move in ("left", "right") else abs(fy - last_center[1])
            if moved:
                step[move] = moved / last_count
        dist = abs(tx - fx) if move in ("left", "right") else abs(ty - fy)
        count = max(1, min(8, int(dist / step[move]) if step.get(move) else 1))
        last_focus, last_center, last_move, last_count = f.get("bounds"), (fx, fy), move, count
        await shell("input keyevent " + " ".join([KEYS[move]] * count))
        path.extend([move] * count)
        await asyncio.sleep(0.25)
    raise ToolError(f"Gave up after {max_steps} moves ({' '.join(path)}). Target: {text}")


@mcp.tool(annotations=ToolAnnotations(title="Swipe / scroll"))
async def swipe(x1: int, y1: int, x2: int, y2: int, duration_ms: int = 300) -> str:
    """Swipe between native screen coordinates (scroll lists in touch-friendly apps)."""
    await shell(f"input swipe {x1} {y1} {x2} {y2} {duration_ms}")
    return f"Swiped ({x1},{y1}) -> ({x2},{y2})"


# --------------------------------------------------------------------------- apps & content

async def _launchable() -> list[str]:
    out = await shell("cmd package query-activities --brief -a android.intent.action.MAIN "
                      "-c android.intent.category.LEANBACK_LAUNCHER")
    return sorted({l.strip().split("/")[0] for l in out.splitlines() if "/" in l})


async def _resolve_package(app: str) -> str:
    key = app.strip().lower()
    if key in APP_ALIASES:
        return APP_ALIASES[key]
    pkgs = await _launchable()
    if app in pkgs:
        return app
    hits = [p for p in pkgs if key.replace(" ", "") in p.lower()]
    if len(hits) == 1:
        return hits[0]
    if not hits:
        all_pkgs = [l.removeprefix("package:") for l in (await shell("pm list packages")).splitlines()]
        if app in all_pkgs:
            return app
    raise ToolError(f"Can't resolve app '{app}'" + (f"; candidates: {hits}" if hits else "") +
                    ". Use list_apps for package names.")


@mcp.tool(annotations=ToolAnnotations(title="List apps", read_only_hint=True))
async def list_apps() -> str:
    """List TV apps that can be launched, with friendly aliases accepted by launch_app."""
    pkgs = await _launchable()
    rev: dict[str, list[str]] = {}
    for alias, p in APP_ALIASES.items():
        rev.setdefault(p, []).append(alias)
    return "\n".join(p + (f"  (aliases: {', '.join(rev[p])})" if p in rev else "") for p in pkgs)


@mcp.tool(annotations=ToolAnnotations(title="Launch app"))
async def launch_app(app: str) -> str:
    """Open an app by friendly name (netflix, youtube, prime video, spotify, hotstar, settings...) or package name."""
    pkg = await _resolve_package(app)
    comp = ""
    for category in ("LEANBACK_LAUNCHER", "LAUNCHER"):
        act = (await shell(f"cmd package resolve-activity --brief -a android.intent.action.MAIN "
                           f"-c android.intent.category.{category} {pkg}", check=False)).splitlines()
        if act and "/" in act[-1]:
            comp = act[-1].strip()
            break
    if comp:
        out = await shell(f"am start -n {comp}", check=False)
    elif pkg == "com.android.tv.settings":
        out = await shell("am start -a android.settings.SETTINGS", check=False)
    else:
        out = await shell(f"monkey -p {pkg} -c android.intent.category.LEANBACK_LAUNCHER 1", check=False)
    if "Error" in out or "No activities" in out:
        raise ToolError(f"Failed to launch {pkg}: {out[-300:]}")
    await asyncio.sleep(1.5)
    return f"Launched {pkg}. Foreground now: {await foreground()}"


@mcp.tool(annotations=ToolAnnotations(title="Close app", destructive_hint=False))
async def close_app(app: str) -> str:
    """Force-stop an app (friendly name or package)."""
    pkg = await _resolve_package(app)
    await shell(f"am force-stop {pkg}")
    return f"Stopped {pkg}. Foreground now: {await foreground()}"


@mcp.tool(annotations=ToolAnnotations(title="Open URL / deep link"))
async def open_url(url: str, package: str = "") -> str:
    """Open a URL or deep link via an Android VIEW intent.

    Examples: https://www.youtube.com/watch?v=ID, https://www.netflix.com/title/80057281,
    https://www.primevideo.com/detail/..., spotify:search:artist, any https:// page (opens browser).

    Args:
        url: The URL / URI.
        package: Optional app package (or alias) to force, e.g. "youtube".
    """
    pkg = await _resolve_package(package) if package else ""
    cmd = f"am start -a android.intent.action.VIEW -d {shlex.quote(url)}" + (f" -p {pkg}" if pkg else "")
    out = await shell(cmd, check=False)
    if "Error" in out:
        raise ToolError(f"Could not open {url}: {out[-300:]}")
    await asyncio.sleep(1.5)
    return f"Opened {url}. Foreground now: {await foreground()}"


@mcp.tool(annotations=ToolAnnotations(title="Play on YouTube"))
async def play_youtube(query_or_id: str) -> str:
    """Open YouTube on the TV: an 11-char video id / youtube URL plays it, anything else opens search results."""
    s = query_or_id.strip()
    m = re.search(r"(?:v=|youtu\.be/|shorts/)([\w-]{11})", s)
    vid = m[1] if m else (s if re.fullmatch(r"[\w-]{11}", s) and not s.isalpha() else None)
    url = f"https://www.youtube.com/watch?v={vid}" if vid else f"https://www.youtube.com/results?search_query={quote(s)}"
    return await open_url(url, "com.google.android.youtube.tv")


@mcp.tool(annotations=ToolAnnotations(title="Search"))
async def search_tv(query: str, app: str = "") -> str:
    """Search for content.

    With `app` (e.g. "hotstar", "play store", "spotify", "mx player", "smarttube", "youtube"),
    the query goes straight into that app's own search. Without it, Google TV global search is used
    (on some TVs it hangs without results; then pass an app).
    """
    if app:
        pkg = await _resolve_package(app)
        if pkg == "com.google.android.youtube.tv":
            return await play_youtube(query)
        acts = await shell(f"cmd package query-activities --brief -a android.intent.action.SEARCH {pkg}", check=False)
        comp = next((l.strip() for l in acts.splitlines() if l.strip().startswith(pkg + "/")), "")
        target = f"-n {comp}" if comp else f"-p {pkg}"
        out = await shell(f"am start -a android.intent.action.SEARCH --es query {shlex.quote(query)} {target}",
                          check=False)
        if "Error" in out:
            raise ToolError(f"{pkg} doesn't accept search intents; launch it and use type_text instead.")
    else:
        await shell(f"am start -a android.search.action.GLOBAL_SEARCH --es query {shlex.quote(query)}", check=False)
    await asyncio.sleep(2.5)
    return f"Searched '{query}'{' in ' + app if app else ''}. Foreground: {await foreground()}. Screenshot to see results."


NETFLIX = "com.netflix.ninja/.MainActivity"


@mcp.tool(annotations=ToolAnnotations(title="Netflix"))
async def netflix(action: str, value: str = "") -> str:
    """Drive Netflix by deep link. Netflix blocks screenshots and exposes no UI tree, so this is the reliable path.

    Actions:
      play   - value = Netflix title/episode id or netflix.com URL; starts playback (for series: resumes/next episode).
      title  - value = id or URL; opens the title's details page.
      search - value = query; opens Netflix search with the query filled in.
      home   - opens Netflix home.
    Find ids with a web search ("<show> site:netflix.com/title" -> netflix.com/title/<id>).
    Verify playback with now_playing (Netflix reports state/position but never the title).
    """
    m = re.search(r"(\d{6,})", value)
    if action in ("play", "title") and not m:
        raise ToolError("value must be a Netflix id like 80057281 or a netflix.com/title/<id> URL.")
    url = {
        "play": f"https://www.netflix.com/watch/{m[1] if m else ''}",
        "title": f"https://www.netflix.com/title/{m[1] if m else ''}",
        "search": f"https://www.netflix.com/search?q={quote(value)}",
        "home": "https://www.netflix.com/browse",
    }.get(action)
    if not url:
        raise ToolError("action must be play, title, search or home")
    out = await shell(f"am start -n {NETFLIX} -a android.intent.action.VIEW -d {shlex.quote(url)} -e source 30",
                      check=False)
    if "Error" in out:
        raise ToolError(f"Netflix rejected {url}: {out[-300:]}")
    if action != "play":
        await asyncio.sleep(3)
        return f"Opened {url}. Netflix screens can't be captured; navigate with press_keys if needed."
    for _ in range(10):  # wait for playback to actually start
        await asyncio.sleep(2)
        np = await now_playing()
        if "com.netflix.ninja: playing" in np:
            return f"Requested {url}. Playback started: {np}"
    return f"Requested {url}, but playback not confirmed yet (profile picker? press ok). now_playing: {await now_playing()}"


# --------------------------------------------------------------------------- media, volume, power

async def _volume() -> dict:
    out = await shell("dumpsys audio", check=False)
    sect = out.split("- STREAM_MUSIC:", 1)[-1].split("- STREAM_", 1)[0]
    g = lambda pat, d="?": (re.search(pat, sect) or [None, d])[1]
    return {"level": g(r"streamVolume:\s*(\d+)"), "max": g(r"Max:\s*(\d+)"),
            "muted": g(r"Muted:\s*(\w+)"), "output": g(r"Devices:\s*(\S+)")}


@mcp.tool(annotations=ToolAnnotations(title="Set volume"))
async def set_volume(level: int | None = None, mute: bool | None = None) -> str:
    """Set media volume to an absolute level and/or mute/unmute. Call with no args to just read it.

    Args:
        level: Absolute volume (0..max, see tv_status; usually 0-100 on this TV).
        mute: True to mute, False to unmute.
    """
    v = await _volume()
    if level is not None:
        mx = int(v["max"]) if v["max"].isdigit() else 100
        await shell(f"cmd media_session volume --stream 3 --set {max(0, min(mx, level))}")
    if mute is not None and str(mute).lower() != v["muted"]:
        await shell("input keyevent KEYCODE_VOLUME_MUTE")
    v = await _volume()
    return f"Volume {v['level']}/{v['max']} muted={v['muted']} output={v['output']}"


@mcp.tool(annotations=ToolAnnotations(title="Now playing", read_only_hint=True))
async def now_playing() -> str:
    """What media is playing: active media sessions with state, title and position."""
    out = await shell("dumpsys media_session; echo UPTIME=$(cat /proc/uptime)", check=False)
    up = re.search(r"UPTIME=([\d.]+)", out)
    now_ms = float(up[1]) * 1000 if up else None
    states = {"0": "none", "1": "stopped", "2": "paused", "3": "playing", "4": "fast-forwarding",
              "5": "rewinding", "6": "buffering", "7": "error", "8": "connecting"}
    res = []
    for block in re.split(r"\n\s{4}(?=\S.*\n\s+ownerPid)", out):
        pkg = re.search(r"package=(\S+)", block)
        st = re.search(r"state=PlaybackState \{state=(\d+), position=(\d+).*?speed=([\d.]+), updated=(\d+)", block)
        if not (pkg and st) or pkg[1] == "com.android.bluetooth":
            continue
        meta = re.search(r"metadata:.*?description=(.+)", block)
        active = re.search(r"active=(\w+)", block)
        pos_ms = int(st[2])
        if st[1] == "3" and now_ms:  # apps only report position on state changes; extrapolate while playing
            pos_ms += max(0, now_ms - int(st[4])) * float(st[3])
        pos = int(pos_ms) // 1000
        res.append(f"{pkg[1]}: {states.get(st[1], st[1])} at {pos // 60}:{pos % 60:02d}"
                   + (f" | {meta[1].strip()}" if meta else "") + (f" | active={active[1]}" if active else ""))
    return "\n".join(res) or f"No media sessions. Foreground: {await foreground()}"


@mcp.tool(annotations=ToolAnnotations(title="Media control"))
async def media(action: str) -> str:
    """Control playback: play, pause, play_pause, stop, next, previous, rewind, fast_forward."""
    if action not in {"play", "pause", "play_pause", "stop", "next", "previous", "rewind", "fast_forward"}:
        raise ToolError("action must be play, pause, play_pause, stop, next, previous, rewind or fast_forward")
    await shell(f"input keyevent {KEYS[action]}")
    await asyncio.sleep(0.8)
    return f"Sent {action}.\n" + await now_playing()


@mcp.tool(annotations=ToolAnnotations(title="Power"))
async def power(action: str = "status") -> str:
    """Screen power: 'on' (wake), 'off' (standby; ADB normally stays reachable), 'toggle', or 'status'."""
    keys = {"on": "KEYCODE_WAKEUP", "off": "KEYCODE_SLEEP", "toggle": "KEYCODE_POWER"}
    if action != "status":
        if action not in keys:
            raise ToolError("action must be on, off, toggle or status")
        await shell(f"input keyevent {keys[action]}")
        await asyncio.sleep(1.5)
    out = await shell("dumpsys power | grep -E 'mWakefulness=|Display Power: state='", check=False)
    return " | ".join(l.strip() for l in out.splitlines())


# --------------------------------------------------------------------------- escape hatches

@mcp.tool(annotations=ToolAnnotations(title="Run ADB shell command", destructive_hint=True))
async def adb_shell(command: str, timeout_s: int = 30) -> str:
    """Run an arbitrary `adb shell` command on the TV (settings, dumpsys, pm, am, etc.). Output is truncated to 20k chars."""
    out = await shell(command, timeout=timeout_s, check=False)
    return out[:20000] or "(no output)"


@mcp.tool(annotations=ToolAnnotations(title="Reconnect to TV"))
async def reconnect(ip: str = "", port: str = "") -> str:
    """(Re)connect ADB to the TV, optionally switching to a different IP/port."""
    global SERIAL
    if ip:
        SERIAL = f"{ip}:{port or TV_PORT}"
    result = await _connect()
    rc, out, _ = await _exec("devices")
    state = next((l.split()[1] for l in out.decode().splitlines() if l.startswith(SERIAL)), "missing")
    if state != "device":
        raise ToolError(f"{result}. Device state: {state}. Enable ADB/network debugging on the TV "
                        "and accept the authorization prompt.")
    return f"{result}. State: {state}"


def main() -> None:
    mcp.run("stdio")


if __name__ == "__main__":
    main()
