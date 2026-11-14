"""
HuggingFace 下载源（支持 hf-mirror 等镜像）

用法约定
--------
直接给"文件直链"，即带 /resolve/ 的地址：
    https://huggingface.co/<repo>/resolve/main/<path/to/file>
    https://huggingface.co/<repo>/blob/main/<path>      （blob 也认，自动换成 resolve）
镜像只需换域名，路径不变：
    https://hf-mirror.com/<repo>/resolve/main/<path>
本模块统一把它拆成 (repo, rev, path)，再按当前镜像重新拼出真正要下载的 URL。

元数据（大小 / SHA256）从 HF 的 tree API 取：
    /api/models/<repo>/tree/<rev>[/<dir>]
其中 LFS 文件的 lfs.oid 就是 sha256，可以直接当权威校验值用。
"""

import json
import time
import urllib.error
import urllib.parse
import urllib.request

UA = "Mozilla/5.0"
HF_ORIGIN = "https://huggingface.co"
DEFAULT_MIRROR = "https://hf-mirror.com"

# 已知的 HF 及其镜像域名（用于自动识别）
HF_HOSTS = (
    "huggingface.co", "hf.co", "hf-mirror.com", "hf-mirror.net",
    "hf-mirror.org", "huggingface.sukaka.top", "hf-api.gitee.com",
)


class HfError(RuntimeError):
    pass


def is_hf_url(url: str) -> bool:
    """判据：路径里有 /resolve/ 或 /blob/，或域名属于 HF/镜像。"""
    try:
        p = urllib.parse.urlparse((url or "").strip())
    except Exception:
        return False
    if not p.netloc:
        return False
    path = p.path or ""
    if "/resolve/" in path or "/blob/" in path:
        return True
    host = p.netloc.lower().split(":")[0]
    return any(host == h or host.endswith("." + h) for h in HF_HOSTS)


def _quote_path(path: str) -> str:
    return "/".join(urllib.parse.quote(seg) for seg in path.split("/") if seg != "")


class HfSource:
    """把一个 HF 文件直链变成可下载 + 可校验的源。"""

    kind = "hf"

    def __init__(self, url: str, token: str = "", mirror: str = ""):
        self.raw = (url or "").strip()
        self.token = (token or "").strip()
        self.mirror = (mirror or "").strip().rstrip("/")
        if self.mirror and not self.mirror.startswith("http"):
            self.mirror = "https://" + self.mirror
        self.repo, self.rev, self.path = self._parse(self.raw)
        self.name = self.path.rsplit("/", 1)[-1]
        self._meta = None
        self._meta_err = ""

    # ---------------- 解析 / 拼 URL ----------------
    @staticmethod
    def _parse(url: str):
        p = urllib.parse.urlparse(url)
        parts = [x for x in (p.path or "").split("/") if x]
        for key in ("resolve", "blob"):
            if key in parts:
                i = parts.index(key)
                repo = "/".join(parts[:i])
                rest = parts[i + 1:]
                if not repo or len(rest) < 2:
                    break
                rev = rest[0]
                path = "/".join(rest[1:])
                return repo, rev, urllib.parse.unquote(path)
        raise HfError(
            "看不懂这个 HuggingFace 链接。请用文件直链（带 /resolve/）：\n"
            "  https://huggingface.co/<仓库>/resolve/main/<文件路径>\n"
            "在 HF 文件页点「下载」复制到的就是这种地址。")

    def base_origin(self) -> str:
        return self.mirror or HF_ORIGIN

    @property
    def download_url(self) -> str:
        """当前镜像下的实际下载地址（每次取都重新拼，天然支持换镜像）。"""
        return "%s/%s/resolve/%s/%s" % (
            self.base_origin(), self.repo,
            urllib.parse.quote(self.rev), _quote_path(self.path))

    @property
    def origin_url(self) -> str:
        return "%s/%s/resolve/%s/%s" % (
            HF_ORIGIN, self.repo, urllib.parse.quote(self.rev), _quote_path(self.path))

    def headers(self) -> dict:
        h = {"User-Agent": UA}
        if self.token:
            h["Authorization"] = "Bearer " + self.token
        return h

    # ---------------- 元数据 ----------------
    def _tree_urls(self):
        dirname = self.path.rsplit("/", 1)[0] if "/" in self.path else ""
        tail = ("/" + _quote_path(dirname)) if dirname else ""
        seen = []
        for origin in (HF_ORIGIN, self.base_origin()):
            u = "%s/api/models/%s/tree/%s%s" % (
                origin, self.repo, urllib.parse.quote(self.rev), tail)
            if u not in seen:
                seen.append(u)
        return seen

    def fetch_meta(self, tries: int = 3) -> dict:
        """返回 {size, sha256}；取不到就返回 {}（不抛、不当成要授权）。

        注意：这里**不再**把 401/403 当成"仓库需要授权"——HF 的 API 也会因为
        限流/风控回 403，而公开文件照样能下。到底要不要授权，交给 probe_access()
        用真实的下载请求去判断。
        """
        if self._meta is not None:
            return self._meta
        last = ""
        for u in self._tree_urls():
            for i in range(tries):
                try:
                    req = urllib.request.Request(u, headers=self.headers())
                    with urllib.request.urlopen(req, timeout=45) as r:
                        data = json.loads(r.read().decode("utf-8", "replace"))
                    for it in data if isinstance(data, list) else []:
                        if it.get("path") == self.path:
                            lfs = it.get("lfs") or {}
                            self._meta = {
                                "size": lfs.get("size") or it.get("size"),
                                "sha256": (lfs.get("oid") or "").upper() or None,
                            }
                            return self._meta
                    last = "tree API 里没有该文件"
                    break
                except urllib.error.HTTPError as e:
                    last = "HTTP %s" % e.code
                    if e.code in (401, 403, 404):
                        break
                    time.sleep(1.0 * (i + 1))
                except Exception as e:
                    last = type(e).__name__
                    time.sleep(1.0 * (i + 1))
        self._meta_err = last
        self._meta = {}
        return self._meta

    # ---------------- 引擎接缝 ----------------
    def probe_access(self, tries: int = 2) -> str:
        """用 Range 0-0 真打一次下载地址，判断能不能下。

        返回 "ok" / "auth" / "unknown"。
        只有**连续几次**都被 401/403 拒绝才判 "auth"——HF 对同一 IP 的突发请求
        也会回 403，单次就下结论会把公开文件误判成 gated。
        """
        auth = 0
        for i in range(max(1, tries)):
            req = urllib.request.Request(
                self.download_url, headers=dict(self.headers(), Range="bytes=0-0"))
            try:
                with urllib.request.urlopen(req, timeout=45) as r:
                    r.read(1)
                    return "ok"
            except urllib.error.HTTPError as e:
                if e.code in (401, 403):
                    auth += 1
                    if auth >= max(1, tries):
                        return "auth"
                    time.sleep(1.5)
                    continue
                return "unknown"
            except Exception:
                if i == 0:
                    time.sleep(1.5)
                    continue
                return "unknown"
        return "auth" if auth else "unknown"

    def precheck(self) -> dict:
        """HF 没有付费模型；预检只给提示，不拦下载。

        真正的权限错误由下载线程处理（那里对 HF 的 401/403 是立刻停、不重试）。
        """
        self.fetch_meta()                      # 顺手拿 size/sha（拿不到不影响）
        acc = self.probe_access()
        return {"paid": False, "gated": False, "probe": acc}

    def expected_sha256(self):
        meta = self.fetch_meta()
        return meta.get("sha256")

    def describe(self) -> str:
        return "HuggingFace %s / %s" % (self.repo, self.path)
