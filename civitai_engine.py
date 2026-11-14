#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
civitai_engine —— CivitAI 断点续传下载引擎（供 CLI 与 GUI 共用）

核心特性（针对"下载一小点就断开"）：
  * 分片多连接：每条连接只负责自己那段，断了只丢它当前那一小片
  * 断点续传：进度落盘，进程被杀 / 机器重启后继续
  * 自动换票：CivitAI 的下载地址是 307 跳转到 R2 签名地址，会过期，
    每次重连都重新换取新地址
  * 无限重试 + 指数退避
  * 单文件互斥锁：避免两个进程同时写同一个 .part 把文件写坏
  * 完成后校验字节数与 safetensors 头，再改名为正式文件
"""
from __future__ import annotations

import hashlib
import json
import os
import re
import struct
import sys
import threading
import time
import urllib.error
import urllib.request
from collections import deque

try:
    import msvcrt
except ImportError:  # 非 Windows 平台退化为无锁
    msvcrt = None

try:
    from hf_source import HfError, HfSource, is_hf_url
except ImportError:            # 同目录导入失败时兜底（例如被当包导入）
    import os as _os
    import sys as _sys
    _sys.path.insert(0, _os.path.dirname(_os.path.abspath(__file__)))
    from hf_source import HfError, HfSource, is_hf_url

HERE = os.path.dirname(os.path.abspath(__file__))
TOKEN_FILE = os.path.join(HERE, ".civitai_token")
CONFIG_FILE = os.path.join(HERE, "civitai_gui_config.json")

DEFAULT_MODELS_ROOT = r"D:\AI\ComfyUI-aki-v3.2\ComfyUI\models"
UA = ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
      "(KHTML, like Gecko) Chrome/131.0.0.0 Safari/537.36")

READ_TILE = 512 * 1024        # 单次写入粒度
SPAN = 32 * 1024 * 1024       # 单次 HTTP Range 最大跨度
SOCK_TIMEOUT = 45

# CivitAI 模型类型 → ComfyUI 模型子目录
TYPE_TO_SUBFOLDER = {
    "Checkpoint": "checkpoints",
    "LORA": "loras",
    "LoCon": "loras",
    "DoRA": "loras",
    "TextualInversion": "embeddings",
    "VAE": "vae",
    "Controlnet": "controlnet",
    "Upscaler": "upscale_models",
    "MotionModule": "animatediff_models",
    "Hypernetwork": "hypernetworks",
}

SUBFOLDERS = ["checkpoints", "loras", "vae", "upscale_models", "controlnet",
              "diffusion_models", "text_encoders", "clip", "unet", "embeddings",
              "model_patches", "style_models", "photomaker", "hypernetworks",
              "gligen", "animatediff_models", "sams", "LLM"]

# 一个 CivitAI 类型可能落在多个类别目录里，按顺序探测「已有基模子目录」，
# 这样能跟着用户自己的归类走（例如 Krea 2 的 checkpoint 放在 diffusion_models/Krea2）
TYPE_TO_SUBFOLDER_CANDIDATES = {
    "Checkpoint": ["checkpoints", "diffusion_models"],
    "LORA": ["loras", "models/Lora"],
    "LoCon": ["loras"],
    "DoRA": ["loras"],
    "TextualInversion": ["embeddings"],
    "VAE": ["vae"],
    "Controlnet": ["controlnet"],
    "Upscaler": ["upscale_models"],
    "MotionModule": ["animatediff_models"],
    "Hypernetwork": ["hypernetworks"],
}

# CivitAI 基模名 -> 常见目录名（仅在目录里没有现成同名子目录时用于新建）
BASE_MODEL_FOLDER_HINTS = {
    "krea 2": "Krea2",
    "krea2": "Krea2",
    "sdxl 1.0": "SDXL",
    "sdxl": "SDXL",
    "sd 1.5": "SD15",
    "sd 2.0": "SD20",
    "sd 3.5": "SD35",
    "flux.1 d": "Flux",
    "flux.1 s": "Flux",
    "flux.1 krea": "FluxKrea",
    "pony": "Pony",
    "illustrious": "Illustrious",
    "noobai": "NoobAI",
    "hunyuan": "Hunyuan",
    "wan video": "WanVideo",
    "anima": "Anima",
    "qwen image": "QwenImage",
}


def _norm_key(s: str) -> str:
    """目录名/基模名归一化：忽略大小写、空格、点、横杠。'Krea 2' 与 'krea2' 等价。"""
    return re.sub(r"[^a-z0-9]", "", (s or "").lower())


def _pretty_folder(base_model: str) -> str:
    """没有现成目录也没有提示时，把基模名转成一个像样的目录名。"""
    if not base_model:
        return ""
    hit = BASE_MODEL_FOLDER_HINTS.get(base_model.strip().lower())
    if hit:
        return hit
    return re.sub(r"[^A-Za-z0-9]", "", base_model) or ""


def list_existing_subfolders(models_root: str) -> list:
    """列出 models 目录下所有 <类别> 和 <类别>/<子目录> 组合，供界面下拉选择。"""
    out = []
    if not os.path.isdir(models_root):
        return out
    for cat in sorted(os.listdir(models_root)):
        cpath = os.path.join(models_root, cat)
        if not os.path.isdir(cpath) or cat.startswith("."):
            continue
        out.append(cat)
        for sub in sorted(os.listdir(cpath)):
            if sub.startswith("."):
                continue
            if os.path.isdir(os.path.join(cpath, sub)):
                out.append(cat + "/" + sub)
    return out


def resolve_target_folder(models_root: str, model_type: str, base_model: str,
                          fallback: str = "") -> str:
    """
    按「<类别>/<基模简称>」给出保存子目录。

    规则（data-driven，优先跟随用户既有习惯）：
      1. 在候选类别目录里找**已存在**的基模子目录（忽略大小写/空格/点号）→ 直接复用
      2. 找不到时，用 BASE_MODEL_FOLDER_HINTS 的约定名新建
      3. 都没有就退到 fallback（或第一个候选类别根目录）
    """
    cands = TYPE_TO_SUBFOLDER_CANDIDATES.get(model_type or "") \
        or ([fallback] if fallback else ["checkpoints"])
    key = _norm_key(base_model)

    if key and os.path.isdir(models_root):
        for cat in cands:
            cpath = os.path.join(models_root, cat)
            if not os.path.isdir(cpath):
                continue
            try:
                entries = sorted(os.listdir(cpath))
            except OSError:
                continue
            for name in entries:
                if name.startswith("."):
                    continue
                if os.path.isdir(os.path.join(cpath, name)) and _norm_key(name) == key:
                    return cat + "/" + name      # 复用你已有的目录名（保留原大小写）

    name = _pretty_folder(base_model)
    base_cat = cands[0] if cands else "checkpoints"
    return base_cat + "/" + name if name else (fallback or base_cat)


# ----------------------------------------------------------------- 工具
def read_token() -> str:
    if os.path.exists(TOKEN_FILE):
        return open(TOKEN_FILE, "r", encoding="utf-8").read().strip()
    return ""


def save_token(token: str) -> None:
    with open(TOKEN_FILE, "w", encoding="utf-8") as f:
        f.write(token.strip())


def load_config() -> dict:
    if os.path.exists(CONFIG_FILE):
        try:
            return json.load(open(CONFIG_FILE, "r", encoding="utf-8"))
        except Exception:
            pass
    return {}


def save_config(cfg: dict) -> None:
    tmp = CONFIG_FILE + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(cfg, f, ensure_ascii=False, indent=2)
    os.replace(tmp, CONFIG_FILE)


def human(n: float) -> str:
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if abs(n) < 1024.0 or unit == "TB":
            return ("%.1f %s" % (n, unit)) if unit != "B" else ("%d B" % n)
        n /= 1024.0
    return "%.1f TB" % n


def gib(n: float) -> str:
    return "%.2f GiB" % (n / 1073741824)


def hms(sec: float) -> str:
    if not sec or sec <= 0 or sec > 86400 * 7:
        return "--:--:--"
    sec = int(sec)
    return "%02d:%02d:%02d" % (sec // 3600, (sec % 3600) // 60, sec % 60)


def read_partial_info(dest: str):
    """
    不启动引擎，只读 .part.json 估算进度。
    用于界面重启后展示历史任务（否则队列是空的，看不到旧任务）。
    返回 {"total": 字节, "done": 字节} 或 None。
    """
    state = dest + ".part" + ".json"
    if not os.path.exists(state):
        return None
    try:
        with open(state, "r", encoding="utf-8") as f:
            st = json.load(f)
        total = int(st.get("total") or 0)
        if total <= 0:
            return None
        done = int(st.get("base_done") or 0) + sum(
            max(0, s["next"] - s["start"]) for s in st.get("segments", []))
        return {"total": total, "done": min(done, total)}
    except Exception:
        return None


def parse_civitai_ref(url: str) -> dict:
    """
    解析任意 civitai 链接，区分两种形态：

      API 地址   /api/download/models/2982178?fileId=xxxx
                 → 2982178 是「版本 ID」
      页面地址   /models/2655820/baka-skin?modelVersionId=2982178
                 → 2655820 是「模型 ID」，2982178 才是「版本 ID」

    以前把页面地址里的模型 ID 当成版本 ID 用，会静默查到另一个无关模型
    （版本号空间是共用的），所以这里必须分开处理。
    """
    url = (url or "").strip()
    ref = {"raw": url, "model_id": None, "version_id": None, "file_id": None,
           "is_api": False}
    m = re.search(r"/api/download/models/(\d+)", url)
    if m:
        ref["version_id"] = int(m.group(1))
        ref["is_api"] = True
    else:
        m = re.search(r"/models/(\d+)", url)
        if m:
            ref["model_id"] = int(m.group(1))        # 页面地址里这段是模型 ID
        m2 = re.search(r"[?&]modelVersionId=(\d+)", url)
        if m2:
            ref["version_id"] = int(m2.group(1))
    m3 = re.search(r"[?&]fileId=(\d+)", url)
    if m3:
        ref["file_id"] = int(m3.group(1))
    return ref


def parse_civitai_url(url: str):
    """兼容旧调用：返回 (version_id, file_id)。页面地址请改用 resolve_version_id()。"""
    r = parse_civitai_ref(url)
    return r["version_id"], r["file_id"]


def resolve_version_id(url: str, token: str = "", host="https://civitai.red") -> dict:
    """
    把任意 civitai 链接补全成带 version_id 的 ref。
    页面地址若没带 modelVersionId，就查该模型取最新版本（并标记 assumed_latest）。
    """
    ref = parse_civitai_ref(url)
    if not ref["version_id"] and ref["model_id"]:
        d = _http_json("%s/api/v1/models/%d" % (host, ref["model_id"]), token)
        vers = d.get("modelVersions") or []
        if vers:
            ref["version_id"] = vers[0].get("id")
            ref["assumed_latest"] = True
    return ref


def build_api_url(version_id: int, file_id=None, host="https://civitai.red") -> str:
    u = "%s/api/download/models/%d" % (host, version_id)
    if file_id:
        u += "?fileId=%d" % file_id
    return u


class _NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        return None


# ----------------------------------------------------------------- 限速
class RateLimiter:
    """
    全局令牌桶限速，多个任务的所有连接共享同一个桶。
    bytes_per_sec <= 0 表示不限速。

    容量刻意做小（0.2 秒的量，但不小于单块大小），并且**初始为空**，
    避免"一上来就把 1 秒额度全部打出去"造成的瞬时超速。
    """

    def __init__(self, bytes_per_sec: float = 0):
        self._lock = threading.Lock()
        self._rate = float(bytes_per_sec or 0)
        self._cap = self._capacity_for(self._rate)
        self._allowance = 0.0
        self._last = time.monotonic()

    @staticmethod
    def _capacity_for(rate: float) -> float:
        # 容量必须 >= 单块大小，否则 consume(块) 永远凑不满会卡死
        return max(float(READ_TILE), rate * 0.2) if rate > 0 else 0.0

    @property
    def rate(self) -> float:
        return self._rate

    def set_rate(self, bytes_per_sec: float):
        with self._lock:
            self._rate = float(bytes_per_sec or 0)
            self._cap = self._capacity_for(self._rate)
            self._allowance = 0.0        # 重新计时，不继承旧额度
            self._last = time.monotonic()

    def consume(self, n: int):
        """取走 n 字节的配额，不够就睡到够为止。"""
        if self._rate <= 0 or n <= 0:
            return
        while True:
            with self._lock:
                now = time.monotonic()
                elapsed = now - self._last
                self._last = now
                self._allowance = min(self._cap, self._allowance + elapsed * self._rate)
                if self._allowance >= n:
                    self._allowance -= n
                    return
                wait = (n - self._allowance) / self._rate
            time.sleep(min(wait, 0.2))


# ----------------------------------------------------------------- 元数据
class PaidModelError(RuntimeError):
    """付费/受限模型。在请求下载地址「之前」抛出，绝不触发任何扣费。"""

    def __init__(self, info=None):
        self.info = info or {}
        super().__init__(
            "这是付费模型（%s），已在请求下载地址之前拦截，未产生任何费用。"
            "确实要下载请在界面里勾选确认后重试。" % paid_reason(self.info))


def paid_reason(info) -> str:
    if not info:
        return "付费/受限模型"
    if not info.get("paid"):
        return "免费公开"
    if info.get("permanent"):
        return "永久付费访问"
    if info.get("endsAt"):
        return "早期访问（到 %s 后转公开）" % str(info["endsAt"])[:10]
    av = info.get("availability")
    if av and av != "Public":
        return "受限访问（%s）" % av
    return "付费/受限模型"


def _http_json(url, token="", tries=6, timeout=45):
    """读 JSON 接口，链路瞬断自动重试（这条链路会随机重置 TLS）。"""
    last = None
    for i in range(tries):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": UA,
                                                       "Accept": "application/json"})
            if token:
                req.add_header("Authorization", "Bearer " + token)
            with urllib.request.urlopen(req, timeout=timeout) as r:
                return json.loads(r.read().decode("utf-8", "replace"))
        except urllib.error.HTTPError as e:
            if e.code in (401, 403, 404):
                raise
            last = e
        except Exception as e:
            last = e
        time.sleep(min(2 ** i, 8))
    raise last


def _paid_from_version(d: dict) -> dict:
    """
    从 model-version JSON 判定是否付费/受限。

    实测 CivitAI 的真实字段：
      免费版本  3404343 -> paidAccess = None,              availability = None
      付费版本  3404368 -> paidAccess = {permanent: true, endsAt: null}
      付费版本  3353829 -> paidAccess = {permanent: true, endsAt: null}
    注意：API **不返回金额**（价格只在站内页面显示），所以这里只判定"要不要拦"。
    """
    pa = d.get("paidAccess") or {}
    av = d.get("availability")
    cfg = d.get("earlyAccessConfig") or {}
    ends = pa.get("endsAt") or d.get("earlyAccessEndsAt")
    permanent = bool(pa.get("permanent")) or bool(d.get("earlyAccessPermanent"))
    restricted = av not in (None, "Public")
    paid = bool(pa) or restricted or bool(cfg)
    return {
        "paid": bool(paid),
        "permanent": permanent,
        "endsAt": ends,
        "availability": av,
        "early_access": bool(cfg) or bool(ends),
        "charge_for_download": cfg.get("chargeForDownload"),
        "download_price": cfg.get("downloadPrice"),
        "has_active_paid_access": bool((d.get("model") or {}).get("hasActivePaidAccess")),
        "model_id": d.get("modelId"),
        "version_id": d.get("id"),
        "page_url": ("https://civitai.com/models/%s?modelVersionId=%s"
                     % (d.get("modelId"), d.get("id"))),
    }


def probe_paid(api_url: str, token: str = "", host="https://civitai.red") -> dict:
    """
    只查元数据判定是否付费。**不会碰下载端点**，因此不会产生扣费。
    拿不到元数据时返回 paid=False + unknown=True（不拦，但会提示）。
    """
    ref = resolve_version_id(api_url, token, host)
    vid = ref.get("version_id")
    if not vid:
        return {"paid": False, "unknown": True,
                "error": "链接里没有可识别的版本 ID（页面地址请带 ?modelVersionId=…）"}
    try:
        d = _http_json("%s/api/v1/model-versions/%d" % (host, vid), token)
    except Exception as e:
        return {"paid": False, "unknown": True, "error": str(e)}
    info = _paid_from_version(d)
    info["unknown"] = False
    return info


def fetch_model_meta(version_id: int, token: str = "", host="https://civitai.red") -> dict:
    """查模型版本信息：名称、底模、类型、可选文件列表（含大小/精度）、付费状态。"""
    d = _http_json("%s/api/v1/model-versions/%d" % (host, version_id), token)
    files = []
    for f in d.get("files") or []:
        meta = f.get("metadata") or {}
        files.append({
            "id": f.get("id"),
            "name": f.get("name") or "model.safetensors",
            "bytes": int(round((f.get("sizeKB") or 0) * 1024)),
            "fp": meta.get("fp") or "",
            "format": meta.get("format") or "",
        })
    model = d.get("model") or {}
    return {
        "version_id": version_id,
        "version_name": d.get("name") or "",
        "model_id": d.get("modelId"),
        "model_name": model.get("name") or "",
        "base_model": d.get("baseModel") or "",
        "type": model.get("type") or "",
        "nsfw": bool(model.get("nsfw")),
        "files": files,
        "subfolder": TYPE_TO_SUBFOLDER.get(model.get("type") or "", ""),
        "paid": _paid_from_version(d),
    }


# ----------------------------------------------------------------- 引擎
class DownloadEngine:
    """一个文件一个实例。线程安全地对外暴露 snapshot()。"""

    def __init__(self, api_url: str, dest_path: str, segments: int = 8,
                 token: str = "", emit=None, allow_paid: bool = False,
                 verify_hash: bool = True, limiter: "RateLimiter" = None,
                 hf_token: str = "", hf_mirror: str = ""):
        self.api_url = api_url
        self.dest = dest_path
        self.allow_paid = bool(allow_paid)
        self.verify_hash = bool(verify_hash)
        self.limiter = limiter          # 全局共享限速器（None = 不限速）
        self.paid_info = None
        self.hf_token = hf_token or ""
        self.hf_mirror = hf_mirror or ""
        # 来源识别：HuggingFace（含镜像）走 HfSource，其余按 CivitAI 处理
        self.hf = None
        if is_hf_url(api_url):
            try:
                self.hf = HfSource(api_url, token=self.hf_token, mirror=self.hf_mirror)
            except HfError as e:
                self.hf = None
                self._hf_parse_error = str(e)
            else:
                self._hf_parse_error = ""
        else:
            self._hf_parse_error = ""
        self.part = dest_path + ".part"
        self.state_path = self.part + ".json"
        self.segments_n = max(1, min(32, int(segments)))
        self.token = token
        self._emit_fn = emit

        self._pause = threading.Event()
        self._cancel = threading.Event()
        self._mu = threading.Lock()
        self._thread = None
        self._lockfile = None
        self._lockpath = None

        self.st = None
        self.status = "idle"           # idle|preparing|downloading|paused|done|cancelled|error
        self.error = ""
        self.hash_mismatch = False     # 完成后 SHA256 不符时为 True
        self.needs_auth = False        # HF 返回 401/403（需登录/同意条款）时为 True
        self.auth_hits = 0
        self.retries = 0
        self.resolves = 0
        self._samples = deque(maxlen=16)
        self._last_emit = 0.0
        self._url = None
        self._url_born = 0.0

    # ---------- 事件 ----------
    def _emit(self, kind, payload=None):
        if self._emit_fn:
            try:
                self._emit_fn(kind, payload)
            except Exception:
                pass

    def log(self, msg):
        self._emit("log", msg)

    # ---------- 单文件互斥锁 ----------
    def _acquire_lock(self):
        if msvcrt is None:
            return
        path = self.dest + ".lock"
        f = open(path, "a+b")
        try:
            msvcrt.locking(f.fileno(), msvcrt.LK_NBLCK, 1)
        except OSError:
            f.close()
            raise RuntimeError(
                "另一个下载进程正在写这个文件（%s）。\n请先关掉它，再点开始。" % os.path.basename(path))
        self._lockfile = f
        self._lockpath = path

    def _release_lock(self):
        if self._lockfile and msvcrt is not None:
            try:
                msvcrt.locking(self._lockfile.fileno(), msvcrt.LK_UNLCK, 1)
            except Exception:
                pass
            try:
                self._lockfile.close()
            except Exception:
                pass
        self._lockfile = None
        # 解锁后把锁文件删掉，不留垃圾
        if self._lockpath:
            try:
                os.remove(self._lockpath)
            except OSError:
                pass
        self._lockpath = None

    # ---------- 签名地址 ----------
    def _api_url_effective(self):
        """允许直接粘页面地址：自动换成 API 下载地址（模型 ID 不当版本 ID 用）。"""
        if self.hf is not None:
            return self.hf.download_url
        if self._hf_parse_error:
            raise RuntimeError(self._hf_parse_error)
        url = self.api_url
        if "/api/download/models/" in url:
            return url
        ref = resolve_version_id(url, self.token)
        vid = ref.get("version_id")
        if not vid:
            raise RuntimeError(
                "无法从链接识别版本：如果是模型页面地址，请带上 ?modelVersionId=… ")
        if ref.get("assumed_latest"):
            self.log("链接未指定版本，已按该模型的最新版本处理（version %d）" % vid)
        return build_api_url(vid, ref.get("file_id"))

    def _resolve_url(self):
        # HuggingFace：不需要换票，直链本身就是稳定的（307 由 urlopen 自己跟随）
        if self.hf is not None:
            return self.hf.download_url
        req = urllib.request.Request(
            self._api_url_effective(),
            headers={"Authorization": "Bearer " + self.token, "User-Agent": UA})
        opener = urllib.request.build_opener(_NoRedirect)
        try:
            opener.open(req, timeout=SOCK_TIMEOUT)
        except urllib.error.HTTPError as e:
            if e.code in (301, 302, 303, 307, 308):
                loc = e.headers.get("Location")
                if not loc:
                    raise RuntimeError("重定向缺少 Location")
                self.resolves += 1
                return loc
            if e.code in (401, 403):
                raise RuntimeError("鉴权失败 (HTTP %s)：检查 CivitAI API Key 是否有效/该模型是否需授权" % e.code)
            raise RuntimeError("换取下载地址失败 HTTP %s" % e.code)
        raise RuntimeError("接口没有返回重定向：token 可能已失效")

    def _url_get(self, force=False):
        with self._mu:
            if force or not self._url or time.time() - self._url_born > 240:
                self._url = self._resolve_url()
                self._url_born = time.time()
            return self._url

    # ---------- 状态 ----------
    def _save_state(self):
        if not self.st:
            return
        # 先把 .part 里已写的数据真正刷到磁盘，再落进度；
        # 这样状态文件永远不会"领先"于磁盘数据（进程崩溃/断电也不会多报）
        try:
            with open(self.part, "r+b") as pf:
                os.fsync(pf.fileno())
        except Exception:
            pass
        tmp = self.state_path + ".tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(self.st, f)
        os.replace(tmp, self.state_path)

    def _done_bytes(self):
        if not self.st:
            return 0
        # base_done：重切分片时已经躺在磁盘上的字节数（新分片只覆盖缺口，不再包含它们）
        return int(self.st.get("base_done", 0) or 0) + \
            sum(max(0, s["next"] - s["start"]) for s in self.st["segments"])

    def _touched_bytes(self):
        """磁盘上真正写过的字节（含已下载但未提交到状态的部分），用于估算进度。"""
        if not self.st:
            return 0
        return sum(s["next"] - s["start"] for s in self.st["segments"])

    def _probe_total(self):
        backoff = 1.0
        auth_hits = 0
        while not self._cancel.is_set():
            try:
                req = urllib.request.Request(
                    self._url_get(), headers={"User-Agent": UA, "Range": "bytes=0-0"})
                with urllib.request.urlopen(req, timeout=SOCK_TIMEOUT) as r:
                    cr = r.headers.get("Content-Range") or ""
                    if "/" in cr:
                        return int(cr.rsplit("/", 1)[1])
                    cl = r.headers.get("Content-Length")
                    if cl:
                        return int(cl)
            except urllib.error.HTTPError as e:
                # HuggingFace 的 401/403 不是"换票过期"，重试没意义
                if self.hf is not None and e.code in (401, 403):
                    auth_hits += 1
                    if auth_hits >= 3:
                        self.needs_auth = True
                        raise RuntimeError(
                            "HTTP %s：HuggingFace 要求登录授权。"
                            "请填入 HF Token，并先到仓库页面点同意条款。" % e.code)
                    self.log("探测被拒绝 (HTTP %s)，可能是限流，%.0fs 后重试"
                             % (e.code, backoff))
                    time.sleep(backoff)
                    backoff = min(backoff * 2, 30)
                    continue
                self.log("探测文件长度失败 (HTTP %s)，%.0fs 后重试" % (e.code, backoff))
                time.sleep(backoff)
                backoff = min(backoff * 2, 30)
            except Exception as e:
                self.log("探测文件长度失败 (%s)，%.0fs 后重试" % (type(e).__name__, backoff))
                try:
                    self._url_get(force=True)
                except Exception:
                    pass
                time.sleep(backoff)
                backoff = min(backoff * 2, 30)
        raise RuntimeError("已取消")

    def _build_segments(self, total):
        n = self.segments_n
        base = total // n
        segs = []
        for i in range(n):
            start = i * base
            end = total - 1 if i == n - 1 else (i + 1) * base - 1
            segs.append({"start": start, "end": end, "next": start})
        return segs

    def _load_or_init(self):
        have = os.path.getsize(self.part) if os.path.exists(self.part) else 0
        if os.path.exists(self.state_path):
            try:
                st = json.load(open(self.state_path, "r", encoding="utf-8"))
                ident_ok = True
                # 身份绑定：.part 若被删除重建（ctime 变化），旧进度一律不能信，
                # 否则会出现"状态说已下载、文件其实是预分配的零"的空洞。
                if st.get("part_ctime") and os.path.exists(self.part):
                    try:
                        if abs(os.stat(self.part).st_ctime - float(st["part_ctime"])) > 0.5:
                            ident_ok = False
                            self.log("检测到 .part 被重建（创建时间变化），旧进度作废，从头下载")
                    except OSError:
                        ident_ok = False
                if ident_ok and st.get("total") and os.path.exists(self.part) and have == st["total"]:
                    self.st = st
                    # 分片数可能被用户改过
                    if len(st["segments"]) != self.segments_n:
                        self._resegment()
                    self.log("发现未完成的任务，从 %.2f%% 继续" %
                             (self._done_bytes() * 100.0 / st["total"]))
                    return
            except Exception:
                pass
        total = self._probe_total()
        self.log("文件总长度 %d 字节（%s）" % (total, gib(total)))
        import shutil
        try:
            free = shutil.disk_usage(os.path.dirname(self.dest) or ".").free
            if free < total - have:
                raise RuntimeError("磁盘空间不足：需要 %s，可用 %s"
                                   % (gib(total - have), gib(free)))
        except RuntimeError:
            raise
        except Exception:
            pass
        with open(self.part, "wb") as f:
            f.truncate(total)
        self.st = {"total": total, "segments": self._build_segments(total),
                   "api": self.api_url,
                   "part_ctime": os.stat(self.part).st_ctime}
        self._save_state()

    def _resegment(self):
        """
        改分片数时重新切分。

        ── 关键 ──
        并行下载时已完成的数据 **不是连续前缀**，而是每段各自的 [start, next)；
        各段的「尾巴」还没下。早期版本把 done_bytes 当成前缀 [0, done) 重切，
        等于宣称没下的尾巴已完成 → 文件里出现整段全零的「空洞」，
        表现为下载完成、大小对、但 SHA256 校验失败。

        正确做法：收集真实的已完成区间，求它们在 [0, total) 里的补集（缺口），
        然后只把缺口切给新的连接。
        """
        st = self.st
        total = st["total"]
        n = self.segments_n

        # 1) 真实已完成区间（半开区间）
        done_ranges = []
        for s in st["segments"]:
            a = max(0, s["start"])
            b = min(s["next"], s["end"] + 1)
            if b > a:
                done_ranges.append((a, b))
        done_ranges.sort()

        merged = []
        for a, b in done_ranges:
            if merged and a <= merged[-1][1]:
                merged[-1] = (merged[-1][0], max(merged[-1][1], b))
            else:
                merged.append((a, b))

        # 2) 缺口 = 补集
        gaps = []
        pos = 0
        for a, b in merged:
            if a > pos:
                gaps.append((pos, a))
            pos = max(pos, b)
        if pos < total:
            gaps.append((pos, total))

        # 3) 缺口切成约 n 段：数量不够就把最长的对半切，绝不再跨过已下载区
        if not gaps:
            # 理论上已由 finalize 处理；保底不动分片，避免把整份文件重下
            self.log("重切分片：没有缺口（已下载完整），保持原样")
            return
        chunks = [g for g in gaps if g[1] > g[0]]
        while len(chunks) < n:
            chunks.sort(key=lambda r: r[1] - r[0], reverse=True)
            a, b = chunks[0]
            if b - a < 2:
                break
            mid = a + (b - a) // 2
            chunks[0] = (a, mid)
            chunks.append((mid, b))

        segs = [{"start": a, "end": b - 1, "next": a} for a, b in sorted(chunks)]
        st["segments"] = segs
        # 已落盘的字节记到基数里，新分片只覆盖缺口 → 进度不会归零
        st["base_done"] = sum(b - a for a, b in merged)
        self._save_state()
        keep = self._done_bytes() * 100.0 / total if total else 0
        self.log("分片数调整为 %d（按真实已完成区间重切，保留进度 %.2f%%，缺口 %d 处）"
                 % (len(segs), keep, len(gaps)))

    # ---------- 工作线程 ----------
    def _worker(self, idx):
        seg = self.st["segments"][idx]
        backoff = 1.0
        fh = open(self.part, "r+b")
        try:
            while seg["next"] <= seg["end"] and not self._cancel.is_set():
                while self._pause.is_set() and not self._cancel.is_set():
                    time.sleep(0.15)
                if self._cancel.is_set():
                    break
                pos = seg["next"]
                want_end = min(seg["end"], pos + SPAN - 1)
                advanced = False
                try:
                    # 注意：换票也要放在 try 里。以前它在外层，
                    # 一次异常就会让这个线程静默死掉，留下永远下不完的分片。
                    req = urllib.request.Request(
                        self._url_get(),
                        headers={"User-Agent": UA, "Range": "bytes=%d-%d" % (pos, want_end)})
                    with urllib.request.urlopen(req, timeout=SOCK_TIMEOUT) as r:
                        if r.status != 206:
                            raise RuntimeError("服务端未按 Range 返回 (status=%s)" % r.status)
                        while seg["next"] <= want_end and not self._cancel.is_set():
                            if self._pause.is_set():
                                break
                            # 限速时把读取块调细，让令牌发放更平滑、少浪费带宽
                            tile_size = READ_TILE
                            if self.limiter is not None and self.limiter.rate > 0:
                                tile_size = int(max(64 * 1024,
                                                    min(READ_TILE, self.limiter.rate / 16)))
                            tile = r.read(tile_size)
                            if not tile:
                                break
                            room = want_end - seg["next"] + 1
                            if len(tile) > room:
                                tile = tile[:room]
                            fh.seek(seg["next"])
                            fh.write(tile)
                            fh.flush()
                            if self.limiter is not None:
                                self.limiter.consume(len(tile))   # 全局限速
                            seg["next"] += len(tile)
                            advanced = True
                            with self._mu:
                                if time.time() - self._last_emit > 5:
                                    self._last_emit = time.time()
                                    self._save_state()
                    if advanced:
                        backoff = 1.0
                    else:
                        time.sleep(backoff)
                        backoff = min(backoff * 2, 30)
                except urllib.error.HTTPError as e:
                    if e.code == 402:
                        # 收费/余额不足：立刻停，绝不重试（重试过不去，还可能反复触发扣费）
                        self.error = ("HTTP 402：该下载需要付费或 Buzz 余额不足。已立即停止，不会重试。"
                                      "请先到 CivitAI 页面完成购买/充值。")
                        self.log(self.error)
                        self._cancel.set()
                        break
                    if self.hf is not None and e.code in (401, 403):
                        # HF 没有"换票"这回事。偶发 401/403 也报过（限流），
                        # 所以先等一会儿重试一次，连续两次才认定是真的没权限。
                        with self._mu:
                            self.auth_hits += 1
                            hits = self.auth_hits
                        if hits < 2:
                            self.log("HTTP %s（可能是限流），2 秒后重试一次…" % e.code)
                            time.sleep(2.0)
                            continue
                        self.needs_auth = True
                        self.error = ("HTTP %s：HuggingFace 要求登录授权。"
                                      "请在弹窗里填入 HF Token，并先到仓库页面点同意条款。"
                                      % e.code)
                        self.log(self.error)
                        self._cancel.set()
                        break
                    if e.code in (400, 401, 403):
                        # 签名过期 → 换新地址
                        try:
                            self._url_get(force=True)
                        except Exception as ee:
                            self.log("换票失败: %s" % ee)
                        time.sleep(1)
                    elif e.code == 416:
                        seg["next"] = seg["end"] + 1
                        self._save_state()
                        break
                    else:
                        with self._mu:
                            self.retries += 1
                        time.sleep(backoff)
                        backoff = min(backoff * 2, 30)
                except Exception:
                    # 连接被重置 / 读超时：已写入的字节保留，从断点继续
                    if not self._cancel.is_set():
                        with self._mu:
                            self.retries += 1
                        time.sleep(backoff)
                        backoff = min(backoff * 1.8, 30)
        finally:
            try:
                fh.close()
            except Exception:
                pass
            with self._mu:
                self._save_state()

    def _run(self):
        try:
            self.status = "preparing"
            self._emit("state", self.status)
            if self._hf_parse_error:
                raise RuntimeError(self._hf_parse_error)
            # ===== 来源预检 =====
            if self.hf is not None:
                # HuggingFace：没有付费模型，也没有 gated 的兜底拦截
                # （预检只做提示；真没权限时下载线程会立刻报错并停止）
                info = self.hf.precheck()
                self.paid_info = info
                if info.get("probe") == "auth":
                    self.log("提示：预检被拒绝（401/403），多半是 gated 仓库。"
                             "若接下来失败，请填 HF Token 并在网页同意条款。")
                self.log("来源：%s%s" % (self.hf.describe(),
                                        "（镜像 %s）" % self.hf.base_origin()
                                        if self.hf.mirror else ""))
            else:
                # ===== 付费护栏：只用元数据判定，发生在任何下载请求之前 =====
                info = probe_paid(self.api_url, self.token)
                self.paid_info = info
                if info.get("paid") and not self.allow_paid:
                    raise PaidModelError(info)
                if info.get("paid"):
                    self.log("注意：这是付费模型（%s），已按你的显式确认放行；"
                             "费用按 CivitAI 规则结算。" % paid_reason(info))
                elif info.get("unknown"):
                    self.log("提示：没能确认该版本是否付费（%s），继续下载。"
                             % (info.get("error") or "无元数据"))
            self._acquire_lock()
            self._load_or_init()
            self.status = "downloading"
            self._emit("state", self.status)
            n = len(self.st["segments"])
            self._emit("segments", n)
            self.log("开始下载，%d 连接并行 → %s" % (n, self.dest))
            threads = [threading.Thread(target=self._worker, args=(i,), daemon=True)
                       for i in range(n)]
            t0 = time.time()
            for t in threads:
                t.start()
            while any(t.is_alive() for t in threads):
                time.sleep(0.5)
                cur = self._done_bytes()
                now = time.time()
                self._samples.append((now, cur))
                self._emit("tick", None)
                if self._cancel.is_set():
                    break
            for t in threads:
                t.join(timeout=10)
            if self._cancel.is_set():
                self._save_state()
                self.status = "cancelled"
                self.log("已停止（进度已保存，可续传）")
                self._emit("state", self.status)
                return

            # ===== 收尾守门：所有分片都必须真正下到 end 之后 =====
            # 万一某个工作线程异常退出，这里补跑它负责的区间，
            # 绝不能在"没下满"的情况下去算 hash（那会把好文件误判成坏文件）。
            def unfinished():
                return [i for i, s in enumerate(self.st["segments"]) if s["next"] <= s["end"]]

            left = unfinished()
            rounds = 0
            while left and rounds < 5 and not self._cancel.is_set():
                rounds += 1
                self.log("⚠ 有 %d 个分片未完成（连接异常退出），第 %d 次补齐…" % (len(left), rounds))
                pool = [threading.Thread(target=self._worker, args=(i,), daemon=True) for i in left]
                for t in pool:
                    t.start()
                while any(t.is_alive() for t in pool):
                    time.sleep(0.5)
                    self._emit("tick", None)
                    if self._cancel.is_set():
                        break
                for t in pool:
                    t.join(timeout=10)
                left = unfinished()
            if left:
                self._save_state()
                self.status = "error"
                self.error = ("还有 %d 个分片没下完，已保留 .part 可续传；"
                              "为避免误判，本次不做 SHA256 校验。" % len(left))
                self.log(self.error)
                self._emit("state", self.status)
                return

            elapsed = time.time() - t0
            ok = self._finalize()
            if ok:
                self.status = "done"
                self.log("完成！用时 %.1f 分钟，换票 %d 次，重连 %d 次"
                         % (elapsed / 60, self.resolves, self.retries))
            else:
                self.status = "error"
                # 不要覆盖 _finalize 里给出的具体原因（SHA256 不符 / 大小不符…）
                if not self.error:
                    self.error = "校验未通过"
            self._emit("state", self.status)
        except Exception as e:
            self.status = "error"
            self.error = str(e)
            self.log("出错: %s" % e)
            self._emit("state", self.status)
        finally:
            self._release_lock()

    def _verify_hash(self):
        """用来源标注的 SHA256 校验成品；拿不到标注就跳过（返回 None）。"""
        if self.hf is not None:
            expected = self.hf.expected_sha256()
            if not expected:
                self.log("跳过 SHA256 校验（HF 该文件没有 LFS 摘要）")
                return None
            self.log("开始校验 SHA256（大文件约十几秒）…")
            h = hashlib.sha256()
            with open(self.part, "rb") as fh:
                while True:
                    b = fh.read(8 << 20)
                    if not b:
                        break
                    h.update(b)
            actual = h.hexdigest().upper()
            ok = actual == expected.upper()
            self.log("SHA256 %s：%s" % ("通过" if ok else "不匹配", actual))
            if not ok:
                self.log("HF LFS 标注为：%s" % expected.upper())
            return ok

        vid, fid = parse_civitai_url(self.api_url)
        if not vid:
            return None
        try:
            d = _http_json("https://civitai.red/api/v1/model-versions/%d" % vid, self.token)
        except Exception as e:
            self.log("跳过 SHA256 校验（取不到元数据：%s）" % e)
            return None
        files = d.get("files") or []
        expected = None
        if fid:
            for f in files:
                if f.get("id") == fid:
                    expected = (f.get("hashes") or {}).get("SHA256")
                    break
        if not expected and len(files) == 1:
            expected = (files[0].get("hashes") or {}).get("SHA256")
        if not expected:
            self.log("跳过 SHA256 校验（该文件没有标注 hash）")
            return None
        self.log("开始校验 SHA256（大文件约十几秒）…")
        h = hashlib.sha256()
        with open(self.part, "rb") as fh:
            while True:
                b = fh.read(8 << 20)
                if not b:
                    break
                h.update(b)
        actual = h.hexdigest().upper()
        ok = actual == expected.upper()
        self.log("SHA256 %s：%s" % ("通过" if ok else "不匹配", actual))
        if not ok:
            self.log("CivitAI 标注为：%s" % expected.upper())
        return ok

    def _finalize(self):
        # 预分配的文件长度一开始就等于总大小，所以"大小"证明不了下完了；
        # 必须先确认每个分片都真的写到了 end 之后，再谈校验。
        done = self._done_bytes()
        if done != self.st["total"]:
            self.error = ("下载不完整（%d/%d 字节，差 %.1f MB），已保留 .part 可续传；"
                          "为避免误判，本次不做 SHA256 校验。"
                          % (done, self.st["total"], (self.st["total"] - done) / 1048576))
            self.log(self.error)
            return False
        size = os.path.getsize(self.part)
        if size != self.st["total"]:
            self.error = ("文件大小不符 %d != %d，保留 .part 以便续传" % (size, self.st["total"]))
            self.log(self.error)
            return False
        try:
            with open(self.part, "rb") as f:
                head = f.read(8)
            n = struct.unpack("<Q", head)[0]
            if not (0 < n < 200 * 1024 * 1024):
                self.log("警告：safetensors 头长度异常 (%d)，文件可能损坏" % n)
        except Exception:
            pass
        if self.verify_hash:
            ok = self._verify_hash()
            if ok is False:
                self.hash_mismatch = True
                bad = self.dest + ".bad"
                try:
                    if os.path.exists(bad):
                        os.remove(bad)
                    os.replace(self.part, bad)
                    if os.path.exists(self.state_path):
                        os.remove(self.state_path)
                except Exception:
                    pass
                self.error = ("SHA256 与 CivitAI 标注不一致，已改名 %s；"
                              "建议删掉重新下载，别直接拿去跑。" % os.path.basename(bad))
                self.log(self.error)
                return False
        if os.path.exists(self.dest):
            os.remove(self.dest)
        os.replace(self.part, self.dest)
        if os.path.exists(self.state_path):
            os.remove(self.state_path)
        self.log("已落盘 → %s（%s）" % (self.dest, gib(size)))
        return True

    # ---------- 对外控制 ----------
    def start(self):
        if self._thread and self._thread.is_alive():
            return False
        self._cancel.clear()
        self._pause.clear()
        self.retries = 0
        self.resolves = 0
        self._samples.clear()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()
        return True

    def pause(self):
        if self.status == "downloading":
            self._pause.set()
            self.status = "paused"
            self._emit("state", self.status)
            self.log("已暂停")

    def resume(self):
        if self.status == "paused":
            self._pause.clear()
            self.status = "downloading"
            self._emit("state", self.status)
            self.log("继续")

    def cancel(self):
        self._cancel.set()
        self._pause.clear()

    @property
    def alive(self):
        return bool(self._thread and self._thread.is_alive())

    def snapshot(self) -> dict:
        total = (self.st or {}).get("total") or 0
        done = self._done_bytes()
        speed = 0.0
        if len(self._samples) >= 2:
            (t1, b1), (t2, b2) = self._samples[0], self._samples[-1]
            if t2 > t1:
                speed = max(0.0, (b2 - b1) / (t2 - t1))
        segs = []
        for s in (self.st or {}).get("segments", []):
            span = s["end"] - s["start"] + 1
            got = max(0, min(span, s["next"] - s["start"]))
            segs.append(got * 100.0 / span if span else 100.0)
        return {
            "status": self.status,
            "error": self.error,
            "total": total,
            "done": done,
            "pct": (done * 100.0 / total) if total else 0.0,
            "speed": speed,
            "eta": ((total - done) / speed) if speed > 2048 else 0,
            "retries": self.retries,
            "resolves": self.resolves,
            "segments": segs,
            "dest": self.dest,
            "needs_auth": bool(self.needs_auth),
            "hash_mismatch": bool(self.hash_mismatch),
        }

    def wait(self, timeout=None):
        if self._thread:
            self._thread.join(timeout)
