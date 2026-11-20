#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
civitai_manager —— 多任务下载管理

职责：
  * 任务队列：排队 / 运行 / 暂停 / 停止 / 重试 / 移除
  * 并发上限：最多同时跑 N 个任务，其余排队（一个结束自动补位）
  * 全局限速：所有任务的所有连接共享一个令牌桶
  * 每条任务独立的分片数（连接数）
  * 每任务独立进度快照，供界面轮询

不含任何 GUI 依赖，可单独测试或被 CLI 复用。
"""
from __future__ import annotations

import os
import threading
import time

import civitai_engine as E

STATE_TEXT = {
    "queued": "排队中",
    "starting": "准备中",
    "running": "下载中",
    "paused": "已暂停",
    "done": "已完成",
    "error": "出错",
    "stopped": "已停止",
}


class Task:
    def __init__(self, tid: str, api_url: str, dest: str, segments: int,
                 token: str = "", meta: dict = None, allow_paid: bool = False):
        self.tid = tid
        self.api_url = api_url
        self.dest = dest
        self.segments = max(1, min(32, int(segments or 8)))
        self.token = token
        self.meta = meta or {}
        self.allow_paid = allow_paid
        self.engine = None
        self.status = "queued"
        self.error = ""
        self.added = time.time()
        self.finished_at = None

    # ---------- 展示信息 ----------
    @property
    def name(self) -> str:
        name = os.path.basename(self.dest)
        return name[:-len(".safetensors")] if name.endswith(".safetensors") else name

    @property
    def folder(self) -> str:
        return os.path.dirname(self.dest)

    @property
    def paid(self) -> bool:
        return bool((self.meta.get("paid") or {}).get("paid"))

    def snapshot(self) -> dict:
        snap = self.engine.snapshot() if self.engine else {}
        total = snap.get("total") or 0
        done = snap.get("done") or 0
        if not self.engine:
            # 没有引擎（比如界面重启后恢复的历史任务）：从 .part.json 读进度，
            # 否则这一行会显示成 0%，看着像"任务丢了"。
            if os.path.exists(self.dest):
                try:
                    total = done = os.path.getsize(self.dest)
                except OSError:
                    pass
            else:
                hint = E.read_partial_info(self.dest)
                if hint:
                    total, done = hint["total"], hint["done"]
        pct = (done * 100.0 / total) if total else 0.0
        speed = snap.get("speed") or 0.0
        eta = snap.get("eta") or 0.0
        if self.status == "done":
            pct = 100.0
            speed = 0.0
            eta = 0.0
        elif self.status in ("paused", "stopped", "queued", "error"):
            speed = 0.0
        return {
            "tid": self.tid,
            "api_url": self.api_url,
            "state": self.status,
            "state_text": STATE_TEXT.get(self.status, self.status),
            "name": self.name,
            "dest": self.dest,
            "folder": self.folder,
            "size_total": total,
            "size_done": done,
            "pct": pct,
            "speed": speed,
            "eta": eta,
            "retries": snap.get("retries") or 0,
            "segments": self.segments,
            "seg_pct": snap.get("segments") or [],
            "paid": self.paid,
            "hash_mismatch": bool(self.engine.hash_mismatch) if self.engine else False,
            "needs_auth": bool(self.engine.needs_auth) if self.engine else False,
            "error": self.error,
        }


class TaskManager:
    """
    由界面（或测试）定期调用 tick() 驱动：
      - 回收已结束任务的状态
      - 在并发未满时启动排队任务
    """

    def __init__(self, emit=None, max_parallel: int = 2):
        self.tasks = []                      # 按加入顺序
        self._byid = {}
        self.max_parallel = max(1, int(max_parallel or 1))
        self.limiter = E.RateLimiter(0)      # 0 = 不限速
        self.hf_token = ""                   # HuggingFace Token（gated 仓库才需要）
        self.hf_mirror = ""                  # 例 https://hf-mirror.com；空 = 官方站
        self._lock = threading.RLock()
        self._emit_fn = emit
        self._seq = 0

    # ---------- 事件 ----------
    def _emit(self, kind, tid, payload=None):
        if self._emit_fn:
            try:
                self._emit_fn(kind, tid, payload)
            except Exception:
                pass

    # ---------- 增删 ----------
    def add(self, api_url: str, dest: str, segments: int, token: str = "",
            meta: dict = None, allow_paid: bool = False) -> Task:
        with self._lock:
            self._seq += 1
            tid = "t%d" % self._seq
            t = Task(tid, api_url, dest, segments, token, meta, allow_paid)
            self.tasks.append(t)
            self._byid[tid] = t
        self._emit("log", tid, "已加入队列：%s（分片 %d）" % (t.name, t.segments))
        return t

    def get(self, tid: str):
        return self._byid.get(tid)

    def remove(self, tid: str):
        t = self.get(tid)
        if not t:
            return
        if t.engine:
            t.engine.cancel()
            t.engine.wait(5)
        with self._lock:
            if t in self.tasks:
                self.tasks.remove(t)
            self._byid.pop(tid, None)
        self._emit("state", tid, "removed")
        self.pump()

    def clear_finished(self, targets=("done", "stopped")):
        """把已完成 / 已停止的任务从队列里一次性移除。

        只动队列，不动磁盘：模型文件、``.part`` 残留一律保留。
        返回 (移除数量, 被移除的任务名列表)。
        """
        with self._lock:
            keep, removed = [], []
            for t in self.tasks:
                if t.status in targets:
                    if t.engine and t.engine.alive:
                        t.engine.cancel()
                    self._byid.pop(t.tid, None)
                    removed.append(t.name)
                else:
                    keep.append(t)
            self.tasks = keep
        for t in removed:
            self._emit("log", "", "已从队列移除：%s" % t)
        self.pump()
        return len(removed), removed

    # ---------- 并发 ----------
    def _active(self):
        """真正在占用带宽的任务。暂停中的任务连接是空闲的，不占并发名额。"""
        return [t for t in self.tasks if t.status in ("starting", "running")]

    def pump(self):
        """在并发上限内把排队任务拉起来。暂停过的不再占名额，所以暂停两个不会堵住队列。"""
        with self._lock:
            active = len(self._active())
            for t in self.tasks:
                if active >= self.max_parallel:
                    break
                if t.status == "queued":
                    self._launch(t)
                    active += 1

    def _launch(self, t: Task):
        t.status = "starting"
        t.error = ""
        t.engine = E.DownloadEngine(
            api_url=t.api_url, dest_path=t.dest, segments=t.segments,
            token=t.token, allow_paid=t.allow_paid, limiter=self.limiter,
            hf_token=self.hf_token, hf_mirror=self.hf_mirror,
            emit=lambda kind, payload, tid=t.tid: self._on_engine(kind, payload, tid))
        t.engine.start()
        self._emit("state", t.tid, "starting")
        self._emit("log", t.tid, "开始下载（分片 %d，并发上限 %d）" % (t.segments, self.max_parallel))

    def _on_engine(self, kind, payload, tid):
        if kind == "log":
            self._emit("log", tid, payload)
        elif kind == "state":
            t = self.get(tid)
            if not t:
                return
            m = {"preparing": "starting", "downloading": "running",
                 "paused": "paused", "done": "done", "error": "error",
                 "cancelled": "stopped"}.get(payload, t.status)
            # 用户主动停止时保留 stopped，不要被 cancelled 覆盖成别的
            t.status = m
            if m in ("done", "error", "stopped"):
                t.finished_at = time.time()
                t.error = (t.engine.error if t.engine else "") or t.error
            self._emit("state", tid, t.status)

    # ---------- 单任务控制 ----------
    def start(self, tid: str) -> bool:
        """开始/继续。已结束的任务会重新排队（保留 .part 续传）。"""
        t = self.get(tid)
        if not t:
            return False
        if t.status in ("running", "starting"):
            return False
        if t.status == "paused" and t.engine and t.engine.alive:
            running = len(self._active())
            if running >= self.max_parallel:
                self._emit("log", tid, "并发上限 %d 已满（运行 %d），继续后可能短暂超限"
                           % (self.max_parallel, running))
            t.engine.resume()
            t.status = "running"
            self._emit("state", tid, "running")
            return True
        # queued / stopped / error / done → 重新入队
        if t.engine and t.engine.alive:
            t.engine.cancel()
            t.engine.wait(5)
        t.engine = None
        t.finished_at = None
        t.status = "queued"
        self._emit("state", tid, "queued")
        self.pump()
        return True

    def pause(self, tid: str) -> bool:
        t = self.get(tid)
        if not t or not t.engine:
            return False
        # starting（准备/探测）也要能暂停——链路差时这一步可能拖很久
        if t.status in ("starting", "running"):
            t.engine.pause()
            t.status = "paused"
            self._emit("state", tid, "paused")
            self._emit("log", tid, "已暂停（不占并发名额，队列里的任务会补位）")
            self.pump()          # 让出名额，排队任务立刻顶上
            return True
        return False

    def stop(self, tid: str) -> bool:
        t = self.get(tid)
        if not t:
            return False
        if t.engine and t.engine.alive:
            t.engine.cancel()
            t.status = "stopped"
            self._emit("state", tid, "stopped")
            self._emit("log", tid, "已停止（进度已保存，可续传）")
            self.pump()
            return True
        if t.status == "queued":
            t.status = "stopped"
            self._emit("state", tid, "stopped")
            return True
        return False

    def toggle(self, tid: str):
        t = self.get(tid)
        if not t:
            return
        if t.status == "running":
            self.pause(tid)
        else:
            self.start(tid)

    def set_segments(self, tid: str, n: int) -> bool:
        """只对未开始/已停止的任务改分片数；运行中不允许。"""
        t = self.get(tid)
        if not t or t.status in ("running", "starting", "paused"):
            return False
        t.segments = max(1, min(32, int(n)))
        self._emit("log", tid, "分片数设为 %d" % t.segments)
        return True

    # ---------- 清空重下 ----------
    def restart_clean(self, tid: str, delete_bad: bool = True) -> dict:
        """
        删除本地残留（.part / .part.json / 可选 .bad）后重新排队，从 0 开始。
        校验失败后用它：坏文件已经没救了，续传只会继续错下去。
        """
        t = self.get(tid)
        if not t:
            return {"ok": False, "removed": []}
        if t.engine and t.engine.alive:
            t.engine.cancel()
            t.engine.wait(5)
        t.engine = None
        removed = []
        targets = [t.dest + ".part", t.dest + ".part.json", t.dest + ".part.json.tmp"]
        if delete_bad:
            targets.append(t.dest + ".bad")
        for p in targets:
            try:
                if os.path.exists(p):
                    os.remove(p)
                    removed.append(os.path.basename(p))
            except OSError as e:
                self._emit("log", tid, "删除失败 %s：%s" % (os.path.basename(p), e))
        t.status = "queued"
        t.error = ""
        t.finished_at = None
        self._emit("state", tid, "queued")
        self._emit("log", tid, "已清空本地残留（%s），重新排队从头下载"
                   % (", ".join(removed) if removed else "无残留"))
        self.pump()
        return {"ok": True, "removed": removed}

    # ---------- 全局设置 ----------
    def set_max_parallel(self, n: int):
        self.max_parallel = max(1, int(n or 1))
        self.pump()

    def set_speed_limit(self, mbps: float):
        """0 或负数 = 不限速。"""
        bps = 0 if not mbps or mbps <= 0 else float(mbps) * 1024 * 1024
        self.limiter.set_rate(bps)

    # ---------- 驱动 ----------
    def tick(self):
        """界面的定时器调用：回收结束状态 + 补位启动排队任务。"""
        changed = False
        for t in list(self.tasks):
            eng = t.engine
            if not eng:
                continue
            if eng.alive:
                continue
            # 引擎线程已结束但状态还没落定（比如启动即失败）
            if t.status in ("starting", "running"):
                m = {"done": "done", "error": "error", "cancelled": "stopped"}.get(
                    eng.status, "error")
                t.status = m
                t.finished_at = time.time()
                t.error = eng.error or t.error
                self._emit("state", t.tid, t.status)
                changed = True
                if m == "done":
                    self._emit("log", t.tid, "任务完成")
        if changed:
            self.pump()
        else:
            self.pump()

    def snapshot(self) -> list:
        return [t.snapshot() for t in self.tasks]

    def totals(self) -> dict:
        total = done = 0
        speed = 0.0
        running = 0
        for t in self.tasks:
            s = t.snapshot()
            total += s["size_total"]
            done += s["size_done"]
            if s["state"] in ("running", "starting"):
                speed += s["speed"]
                running += 1
        return {"total": total, "done": done,
                "pct": (done * 100.0 / total) if total else 0.0,
                "speed": speed, "running": running, "count": len(self.tasks)}

    def stop_all(self):
        for t in list(self.tasks):
            if t.engine and t.engine.alive:
                t.engine.cancel()
                t.status = "stopped"
                self._emit("state", t.tid, "stopped")
