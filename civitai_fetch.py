#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""civitai_fetch —— 命令行入口（下载逻辑全部复用 civitai_engine，与 GUI 同一套）"""
import argparse
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import civitai_engine as E  # noqa: E402

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass


def main():
    ap = argparse.ArgumentParser(description="CivitAI 断点续传下载器（命令行）")
    ap.add_argument("url", nargs="?", default="https://civitai.red/api/download/models/3404343?fileId=3295529")
    ap.add_argument("-o", "--out", help="目标文件完整路径")
    ap.add_argument("-d", "--dir", default=E.DEFAULT_MODELS_ROOT, help="模型根目录")
    ap.add_argument("-s", "--sub", default="checkpoints", help="子目录")
    ap.add_argument("-n", "--name", help="文件名（默认取自 CivitAI）")
    ap.add_argument("-c", "--connections", type=int, default=8, help="并行连接数")
    ap.add_argument("--allow-paid", action="store_true",
                    help="允许下载付费模型（默认拦截；确认会产生费用后再加这个参数）")
    a = ap.parse_args()

    token = E.read_token()
    if not token:
        sys.exit("缺少 API Key：%s" % E.TOKEN_FILE)

    dest = a.out
    if not dest:
        name = a.name
        if not name:
            ref = E.resolve_version_id(a.url, token)      # 支持页面地址与 API 地址
            if not ref.get("version_id"):
                sys.exit("链接里没有可识别的版本 ID：请用 /api/download/models/<版本ID> 或带上 ?modelVersionId=")
            meta = E.fetch_model_meta(ref["version_id"], token)
            name = meta["files"][0]["name"]
            if not a.sub:
                a.sub = E.resolve_target_folder(a.dir, meta.get("type") or "",
                                                meta.get("base_model") or "",
                                                meta.get("subfolder") or "")
        dest = os.path.join(a.dir, a.sub, name)

    os.makedirs(os.path.dirname(dest), exist_ok=True)
    print("目标: %s" % dest, flush=True)

    # 付费预检：只查元数据，不碰下载端点，不会产生费用
    info = E.probe_paid(a.url, token)
    if info.get("paid"):
        print("付费模型：%s" % E.paid_reason(info), flush=True)
        print("CivitAI 的 API 不返回金额，价格请到页面确认：%s" % info.get("page_url", ""), flush=True)
        if not a.allow_paid:
            sys.exit("已拦截：这是付费模型，未发起下载。确认要付费请加 --allow-paid 再运行。")

    def emit(kind, payload):
        if kind == "log":
            print(time.strftime("[%H:%M:%S] ") + payload, flush=True)

    eng = E.DownloadEngine(a.url, dest, a.connections, token, emit=emit, allow_paid=a.allow_paid)
    eng.start()
    while eng.alive:
        time.sleep(5)
        s = eng.snapshot()
        print("[%s] %6.2f%%  %s / %s  %s/s  剩余 %s  重连 %d"
              % (time.strftime("%H:%M:%S"), s["pct"], E.gib(s["done"]), E.gib(s["total"]),
                 E.human(s["speed"]), E.hms(s["eta"]), s["retries"]), flush=True)
    s = eng.snapshot()
    print("状态: %s%s" % (s["status"], (" - " + s["error"]) if s["error"] else ""))
    sys.exit(0 if s["status"] == "done" else 1)


if __name__ == "__main__":
    main()
