#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
CivitAI 模型下载器（图形界面 · 多任务）

① 添加任务：解析链接 → 选精度文件 → 自动按基模分目录 → 加入队列
② 任务队列：多任务并行、并发上限、全局限速、每条任务分片数、暂停/停止/重试/移除
③ 运行日志
"""
from __future__ import annotations

import os
import queue
import threading
import time
import tkinter as tk
import webbrowser
from tkinter import filedialog, messagebox, simpledialog, ttk

import civitai_engine as E
import civitai_manager as M

FONT = ("Microsoft YaHei UI", 10)
FONT_S = ("Microsoft YaHei UI", 9)
FONT_B = ("Microsoft YaHei UI", 10, "bold")
MONO = ("Consolas", 9)

STATE_TAG = {"done": "done", "error": "error", "running": "running",
             "starting": "running", "paused": "paused", "queued": "queued",
             "stopped": "stopped"}


class App(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("CivitAI 模型下载器 · 多任务")

        self.cfg = E.load_config()
        self.meta = None
        self.paid_info = {}
        self.q = queue.Queue()
        self.rows = {}          # tid -> tree item id
        self.parsing = False
        self._mismatch_prompted = set()
        self._auth_prompted = set()     # HF 401/403 已经弹过窗的任务

        self.manager = M.TaskManager(emit=self._on_manager_event,
                                     max_parallel=int(self.cfg.get("max_parallel", 2) or 2))

        self._build_style()
        self._build_ui()
        self._fit_to_screen()
        self._load_cfg_into_ui()
        self._restore_tasks()
        self.protocol("WM_DELETE_WINDOW", self._on_close)
        self.after(600, self._auto_parse)
        self.after(150, self._bring_front)
        self.after(500, self._poll)

    # ------------------------------------------------------------ 样式
    def _build_style(self):
        st = ttk.Style(self)
        try:
            st.theme_use("vista")
        except Exception:
            pass
        st.configure(".", font=FONT)
        st.configure("Card.TLabelframe", padding=10)
        st.configure("Card.TLabelframe.Label", font=FONT_B)
        st.configure("Big.TButton", font=FONT_B, padding=(12, 6))
        st.configure("Small.TLabel", font=FONT_S)
        st.configure("Hint.TLabel", font=FONT_S, foreground="#666")
        st.configure("Good.TLabel", font=FONT_S, foreground="#137333")
        st.configure("Bad.TLabel", font=FONT_S, foreground="#b3261e")
        st.configure("Treeview", font=FONT_S, rowheight=24)
        st.configure("Treeview.Heading", font=FONT_S)

    def _fit_to_screen(self):
        sw, sh = self.winfo_screenwidth(), self.winfo_screenheight()
        w = min(1240, sw - 60)
        h = min(880, sh - 70)
        self.minsize(min(960, w), min(620, h))
        self.geometry("%dx%d+%d+%d" % (w, h, max(0, (sw - w) // 2), max(0, (sh - h) // 2)))

    def _bring_front(self):
        try:
            self.deiconify()
            self.lift()
            self.focus_force()
            self.attributes("-topmost", True)
            self.after(800, lambda: self.attributes("-topmost", False))
        except Exception:
            pass

    # ------------------------------------------------------------ 界面
    def _build_ui(self):
        pad = {"padx": 8, "pady": 4}
        root = ttk.Frame(self, padding=10)
        root.pack(fill="both", expand=True)

        # ================= ① 添加任务 =================
        src = ttk.LabelFrame(root, text="① 添加下载任务", style="Card.TLabelframe")
        src.pack(fill="x")
        src.columnconfigure(1, weight=1)

        ttk.Label(src, text="下载地址").grid(row=0, column=0, sticky="w", **pad)
        self.var_url = tk.StringVar()
        ttk.Entry(src, textvariable=self.var_url, font=("Consolas", 9)).grid(
            row=0, column=1, sticky="ew", **pad)
        self.btn_parse = ttk.Button(src, text="解析", width=8, command=self.on_parse)
        self.btn_parse.grid(row=0, column=2, **pad)
        ttk.Label(src, text="粘贴 civitai 的 /api/download/models/… 链接，解析后可切换精度档位",
                  style="Hint.TLabel").grid(row=1, column=1, columnspan=2, sticky="w", padx=8)

        self.lbl_meta = ttk.Label(src, text="（尚未解析）", style="Hint.TLabel")
        self.lbl_meta.grid(row=2, column=0, columnspan=3, sticky="w", **pad)

        ttk.Label(src, text="选择文件").grid(row=3, column=0, sticky="w", **pad)
        self.var_file = tk.StringVar()
        self.cb_file = ttk.Combobox(src, textvariable=self.var_file, state="disabled", font=FONT_S)
        self.cb_file.grid(row=3, column=1, columnspan=2, sticky="ew", **pad)
        self.cb_file.bind("<<ComboboxSelected>>", self.on_pick_file)

        ttk.Label(src, text="模型根目录").grid(row=4, column=0, sticky="w", **pad)
        self.var_root = tk.StringVar(value=E.DEFAULT_MODELS_ROOT)
        ttk.Entry(src, textvariable=self.var_root).grid(row=4, column=1, sticky="ew", **pad)
        ttk.Button(src, text="浏览…", width=8, command=self.on_browse).grid(row=4, column=2, **pad)

        ttk.Label(src, text="保存位置").grid(row=5, column=0, sticky="w", **pad)
        row5 = ttk.Frame(src)
        row5.grid(row=5, column=1, columnspan=2, sticky="ew", **pad)
        self.var_sub = tk.StringVar(value="checkpoints")
        self.cb_sub = ttk.Combobox(row5, textvariable=self.var_sub, values=E.SUBFOLDERS, width=24)
        self.cb_sub.grid(row=0, column=0, sticky="w")
        self.var_bybase = tk.BooleanVar(value=True)
        ttk.Checkbutton(row5, text="按基模分目录", variable=self.var_bybase,
                        command=self._refresh_sub_choices).grid(row=0, column=1, padx=(10, 12))
        ttk.Label(row5, text="文件名").grid(row=0, column=2, sticky="e", padx=(0, 6))
        self.var_name = tk.StringVar()
        ttk.Entry(row5, textvariable=self.var_name).grid(row=0, column=3, sticky="ew")
        row5.columnconfigure(3, weight=1)

        ttk.Label(src, text="CivitAI API Key").grid(row=6, column=0, sticky="w", **pad)
        row6 = ttk.Frame(src)
        row6.grid(row=6, column=1, columnspan=2, sticky="ew", **pad)
        row6.columnconfigure(0, weight=1)
        self.var_token = tk.StringVar(value=E.read_token())
        ttk.Entry(row6, textvariable=self.var_token, show="•").grid(row=0, column=0, sticky="ew")
        ttk.Button(row6, text="保存", width=6, command=self.on_save_token).grid(row=0, column=1, padx=6)
        ttk.Label(row6, text="本任务分片数").grid(row=0, column=2, padx=(8, 4))
        self.var_seg = tk.IntVar(value=int(self.cfg.get("segments", 8) or 8))
        ttk.Spinbox(row6, from_=1, to=32, width=4, textvariable=self.var_seg).grid(row=0, column=3)
        self.var_autostart = tk.BooleanVar(value=bool(self.cfg.get("autostart", True)))
        ttk.Checkbutton(row6, text="加入后立即开始", variable=self.var_autostart).grid(
            row=0, column=4, padx=(12, 0))

        # 付费提示条
        self.paid_bar = ttk.Frame(src)
        self.lbl_paid = tk.Label(self.paid_bar, text="", bg="#fdecea", fg="#b3261e",
                                 font=FONT_B, anchor="w", justify="left",
                                 padx=10, pady=7, wraplength=980)
        self.lbl_paid.pack(side="left", fill="x", expand=True)
        ttk.Button(self.paid_bar, text="到 CivitAI 核对价格",
                   command=self.on_open_page).pack(side="right", padx=(8, 0))

        addbar = ttk.Frame(src)
        addbar.grid(row=8, column=0, columnspan=3, sticky="ew", pady=(8, 0))
        ttk.Button(addbar, text="＋ 加入队列", style="Big.TButton",
                   command=self.on_add_task).pack(side="left")
        ttk.Button(addbar, text="按基模重新归类", command=self.on_reapply_folder).pack(
            side="left", padx=8)

        # ================= ② 任务队列 =================
        qf = ttk.LabelFrame(root, text="② 任务队列", style="Card.TLabelframe")
        qf.pack(fill="both", expand=True, pady=(10, 0))

        top = ttk.Frame(qf)
        top.pack(fill="x")
        self.lbl_total = ttk.Label(top, text="共 0 个任务 · 运行 0 · 总进度 0.0% · 总速度 0 B/s",
                                   style="Small.TLabel")
        self.lbl_total.pack(side="left")

        twrap = ttk.Frame(qf)
        twrap.pack(fill="both", expand=True, pady=(6, 0))

        cols = ("state", "name", "size", "pct", "speed", "eta", "seg", "retry", "path")
        heads = {"state": "状态", "name": "文件名", "size": "大小", "pct": "进度",
                 "speed": "速度", "eta": "剩余", "seg": "分片", "retry": "重连", "path": "保存位置"}
        widths = {"state": 68, "name": 300, "size": 82, "pct": 66, "speed": 92,
                  "eta": 84, "seg": 46, "retry": 52, "path": 250}
        self.tree = ttk.Treeview(twrap, columns=cols, show="headings", selectmode="extended")
        for c in cols:
            self.tree.heading(c, text=heads[c])
            self.tree.column(c, width=widths[c], anchor="w" if c in ("name", "path") else "center",
                             stretch=(c == "name"))
        vs = ttk.Scrollbar(twrap, orient="vertical", command=self.tree.yview)
        self.tree.configure(yscrollcommand=vs.set)
        self.tree.grid(row=0, column=0, sticky="nsew")
        vs.grid(row=0, column=1, sticky="ns")
        twrap.columnconfigure(0, weight=1)
        twrap.rowconfigure(0, weight=1)
        self.tree.bind("<Double-1>", lambda e: self.on_toggle_sel())
        self.tree.tag_configure("done", foreground="#137333")
        self.tree.tag_configure("error", foreground="#b3261e")
        self.tree.tag_configure("running", foreground="#1a56db")
        self.tree.tag_configure("paused", foreground="#a15c00")
        self.tree.tag_configure("stopped", foreground="#666666")
        self.tree.tag_configure("queued", foreground="#444444")

        ctl = ttk.Frame(qf)
        ctl.pack(fill="x", pady=(8, 0))
        for text, cmd in (("▶ 开始/继续", self.on_start_sel), ("⏸ 暂停", self.on_pause_sel),
                          ("⏹ 停止", self.on_stop_sel), ("精确重试", self.on_retry_sel),
                          ("清空重下", self.on_clean_restart_sel),
                          ("移除", self.on_remove_sel), ("打开文件夹", self.on_open_dir_sel)):
            ttk.Button(ctl, text=text, command=cmd).pack(side="left", padx=(0, 6))
        ttk.Button(ctl, text="全部停止", command=self.on_stop_all).pack(side="right")

        netbar = ttk.Frame(qf)
        netbar.pack(fill="x", pady=(8, 0))
        ttk.Label(netbar, text="HF 镜像", style="Small.TLabel").pack(side="left")
        self.var_mirror = tk.StringVar(value=self.cfg.get("hf_mirror") or "（官方）")
        cb_mir = ttk.Combobox(netbar, textvariable=self.var_mirror, width=26,
                              values=["（官方）", "https://hf-mirror.com"])
        cb_mir.pack(side="left", padx=(4, 2))
        cb_mir.bind("<<ComboboxSelected>>", lambda e: self.on_mirror_change())
        cb_mir.bind("<FocusOut>", lambda e: self.on_mirror_change())
        ttk.Label(netbar, text="（HuggingFace 链接走这里，留空=官方站）",
                  style="Small.TLabel").pack(side="left")
        ttk.Button(netbar, text="HF Token", width=10,
                   command=self.on_set_hf_token).pack(side="right")
        self.lbl_hf_tok = ttk.Label(netbar, text=self._hf_token_hint(), style="Small.TLabel")
        self.lbl_hf_tok.pack(side="right", padx=(0, 6))

        setbar = ttk.Frame(qf)
        setbar.pack(fill="x", pady=(8, 0))
        ttk.Label(setbar, text="同时下载", style="Small.TLabel").pack(side="left")
        self.var_parallel = tk.IntVar(value=int(self.cfg.get("max_parallel", 2) or 2))
        sp = ttk.Spinbox(setbar, from_=1, to=8, width=4, textvariable=self.var_parallel,
                         command=self.on_parallel_change)
        sp.pack(side="left", padx=(4, 2))
        sp.bind("<FocusOut>", lambda e: self.on_parallel_change())
        ttk.Label(setbar, text="个任务", style="Small.TLabel").pack(side="left")

        ttk.Separator(setbar, orient="vertical").pack(side="left", fill="y", padx=12)
        ttk.Label(setbar, text="全局限速", style="Small.TLabel").pack(side="left")
        self.var_speed = tk.DoubleVar(value=float(self.cfg.get("speed_limit", 0) or 0))
        spd = ttk.Spinbox(setbar, from_=0, to=9999, increment=0.5, width=6,
                          textvariable=self.var_speed, command=self.on_speed_change)
        spd.pack(side="left", padx=(4, 2))
        spd.bind("<FocusOut>", lambda e: self.on_speed_change())
        ttk.Label(setbar, text="MB/s（0 = 不限速）", style="Small.TLabel").pack(side="left")

        ttk.Separator(setbar, orient="vertical").pack(side="left", fill="y", padx=12)
        ttk.Label(setbar, text="改所选任务的分片数", style="Small.TLabel").pack(side="left")
        self.var_seg_sel = tk.IntVar(value=8)
        ttk.Spinbox(setbar, from_=1, to=32, width=4, textvariable=self.var_seg_sel).pack(
            side="left", padx=(4, 4))
        ttk.Button(setbar, text="应用", width=6, command=self.on_apply_seg).pack(side="left")

        # ---- 队列区内部布局：按钮/设置先占位，表格吃剩余空间，避免被挤掉 ----
        for w in (top, twrap, ctl, setbar, netbar):
            w.pack_forget()
        top.pack(side="top", fill="x")
        netbar.pack(side="bottom", fill="x", pady=(8, 0))
        setbar.pack(side="bottom", fill="x", pady=(8, 0))
        ctl.pack(side="bottom", fill="x", pady=(8, 0))
        twrap.pack(side="top", fill="both", expand=True, pady=(6, 0))

        # ================= ③ 日志 =================
        logf = ttk.LabelFrame(root, text="③ 运行日志", style="Card.TLabelframe")
        logf.pack(fill="both", expand=False, pady=(10, 0))
        self.txt = tk.Text(logf, height=7, wrap="word", font=MONO,
                           bg="#1e1e1e", fg="#d4d4d4", insertbackground="#d4d4d4",
                           relief="flat", padx=8, pady=6)
        sb = ttk.Scrollbar(logf, orient="vertical", command=self.txt.yview)
        self.txt.configure(yscrollcommand=sb.set)
        self.txt.pack(side="left", fill="both", expand=True)
        sb.pack(side="right", fill="y")

        self.status = ttk.Label(root, text="就绪", style="Hint.TLabel", anchor="w")
        self.status.pack(fill="x", padx=12, pady=(4, 6))

        # ---- 按优先级重排：底部栏先占位，队列表最后吸收剩余空间（空间不足时它先缩）----
        for w in (src, qf, logf, self.status):
            w.pack_forget()
        self.status.pack(side="bottom", fill="x", padx=4, pady=(4, 0))
        logf.pack(side="bottom", fill="x", pady=(8, 0))
        src.pack(side="top", fill="x")
        qf.pack(side="top", fill="both", expand=True, pady=(8, 0))

        self.log_line("界面已就绪：解析 → 加入队列 → 队列里控制每个任务。")
        self.log_line("支持多任务并行；「同时下载」限制并发任务数，其余自动排队。")

    # ------------------------------------------------------------ 配置
    def _load_cfg_into_ui(self):
        c = self.cfg
        self.var_url.set(c.get("api_url") or
                         "https://civitai.red/api/download/models/3404343?fileId=3295529")
        self.var_root.set(c.get("models_root") or E.DEFAULT_MODELS_ROOT)
        self.var_sub.set(c.get("subfolder") or "checkpoints")
        self.var_bybase.set(bool(c.get("by_base_model", True)))
        self.var_name.set(c.get("filename") or "")
        self._refresh_sub_choices()

        self.manager.set_max_parallel(int(self.var_parallel.get() or 2))
        self.manager.set_speed_limit(float(self.var_speed.get() or 0))
        # HuggingFace 来源设置
        self.manager.hf_mirror = c.get("hf_mirror") or ""
        self.manager.hf_token = c.get("hf_token") or ""
        try:
            self.var_mirror.set(self.manager.hf_mirror or "（官方）")
            self.lbl_hf_tok.configure(text=self._hf_token_hint())
        except Exception:
            pass

        g = c.get("geometry") or ""
        if g:
            try:
                size = g.split("+")[0]
                gw, gh = [int(x) for x in size.split("x")]
                sw, sh = self.winfo_screenwidth(), self.winfo_screenheight()
                if gw <= sw - 20 and gh <= sh - 40:
                    self.geometry("%dx%d+%d+%d" % (gw, gh, max(0, (sw - gw) // 2),
                                                  max(0, (sh - gh) // 2)))
            except Exception:
                pass

        meta = c.get("last_meta")
        if meta and meta.get("files"):
            self.meta = meta
            self._on_meta(meta, quiet=True)

    def _save_cfg(self):
        E.save_config({
            "api_url": self.var_url.get().strip(),
            "models_root": self.var_root.get().strip(),
            "subfolder": self.var_sub.get().strip(),
            "by_base_model": bool(self.var_bybase.get()),
            "filename": self.var_name.get().strip(),
            "segments": int(self.var_seg.get() or 8),
            "max_parallel": int(self.var_parallel.get() or 2),
            "speed_limit": float(self.var_speed.get() or 0),
            "autostart": bool(self.var_autostart.get()),
            "geometry": self.geometry(),
            "last_meta": self.meta,
            "hf_mirror": self.manager.hf_mirror or "",
            "hf_token": self.manager.hf_token or "",
            # 任务队列也存下来，否则重启后界面是空的、看不到历史任务
            "tasks": [{
                "api_url": t.api_url,
                "dest": t.dest,
                "segments": t.segments,
                "allow_paid": bool(t.allow_paid),
                "status": t.status,
            } for t in self.manager.tasks],
        })

    # ------------------------------------------------------------ HF 来源设置
    def _hf_token_hint(self):
        t = (self.cfg.get("hf_token") or "").strip()
        return ("HF Token 已设置（%s…）" % t[:4]) if t else "HF Token 未设置（gated 仓库才需要）"

    def on_mirror_change(self):
        raw = (self.var_mirror.get() or "").strip()
        mirror = "" if raw.startswith("（") or raw.startswith("(") else raw
        self.cfg["hf_mirror"] = mirror
        self.manager.hf_mirror = mirror
        self._save_cfg()
        self.log_line("HF 镜像：%s" % (mirror or "官方 huggingface.co"))

    def on_set_hf_token(self):
        cur = self.cfg.get("hf_token") or ""
        val = simpledialog.askstring(
            "HuggingFace Token",
            "粘贴 HF Token（以 hf_ 开头，只需读权限）。\n留空并确定 = 清除。",
            show="*", initialvalue="", parent=self)
        if val is None:
            return
        val = val.strip()
        self.cfg["hf_token"] = val
        self.manager.hf_token = val
        self._save_cfg()
        self.lbl_hf_tok.configure(text=self._hf_token_hint())
        self.log_line("HF Token %s" % ("已保存（%s…）" % val[:4] if val else "已清除"))

    def _fp_of(self, name):
        low = (name or "").lower()
        for k in ("int8_convrot", "int8", "nvfp4", "fp8", "bf16", "fp16", "gguf"):
            if k in low:
                return k
        return ""

    def _parse_hf(self, url):
        """HuggingFace 直链：不需要版本/付费解析，只取大小和 LFS 摘要。"""
        self.btn_parse.configure(state="disabled", text="解析中")
        self.set_status("正在解析 HuggingFace 文件信息…")
        self.parsing = True
        mirror = self.cfg.get("hf_mirror") or ""
        tok = self.cfg.get("hf_token") or ""

        def work():
            try:
                src = E.HfSource(url, token=tok, mirror=mirror)
                mi = src.fetch_meta()
                if src.probe_access() == "auth":
                    self.q.put(("hf_auth", url))
                    return
                size = int(mi.get("size") or 0)
                meta = {
                    "model_name": src.repo,
                    "version_name": src.rev,
                    "base_model": "HuggingFace",
                    "type": "HF 文件",
                    "nsfw": False,
                    "files": [{"id": None, "name": src.name, "bytes": size,
                               "fp": self._fp_of(src.name), "type": "Model"}],
                    "paid": {"paid": False},
                    "_ref": {"file_id": None},
                    "_hf": {"mirror": src.base_origin(), "sha256": mi.get("sha256"),
                            "size": size},
                }
                self.q.put(("meta", meta))
            except Exception as ex:
                self.q.put(("meta_err", str(ex)))

        threading.Thread(target=work, daemon=True).start()

    def _restore_tasks(self):
        """恢复上次的任务列表：显示为「已停止」，不自动开跑。"""
        saved = self.cfg.get("tasks") or []
        if not saved:
            return
        n = 0
        for item in saved:
            try:
                dest = item.get("dest") or ""
                if not dest:
                    continue
                t = self.manager.add(
                    api_url=item.get("api_url") or "",
                    dest=dest,
                    segments=int(item.get("segments") or 8),
                    token=self.var_token.get().strip(),
                    meta={},
                    allow_paid=bool(item.get("allow_paid")))
                # 已完成 / 有残留 / 什么都没有 → 分别给出合适状态
                if os.path.exists(dest):
                    t.status = "done"
                else:
                    t.status = "stopped"
                n += 1
            except Exception as ex:
                self.log_line("恢复任务失败：%s" % ex)
        if n:
            self.log_line("已恢复 %d 个历史任务（状态为已停止，点「开始/继续」即可续传）" % n)

    # ------------------------------------------------------------ 工具
    def log_line(self, msg, tag_name=""):
        prefix = "[%s] " % time.strftime("%H:%M:%S") + (("[%s] " % tag_name) if tag_name else "")
        self.txt.insert("end", prefix + str(msg) + "\n")
        self.txt.see("end")
        n = int(self.txt.index("end-1c").split(".")[0])
        if n > 3000:
            self.txt.delete("1.0", "%d.0" % (n - 2500))

    def set_status(self, text, kind="hint"):
        try:
            self.status.configure(text=text, style={"hint": "Hint.TLabel",
                                                    "good": "Good.TLabel",
                                                    "bad": "Bad.TLabel"}[kind])
        except Exception:
            pass

    def _dest_path(self):
        root = self.var_root.get().strip()
        sub = self.var_sub.get().strip()
        name = self.var_name.get().strip()
        if not (root and sub and name):
            return None
        return os.path.join(root, sub, name)

    def _refresh_sub_choices(self):
        cur = self.var_sub.get().strip()
        vals = E.list_existing_subfolders(self.var_root.get().strip())
        for extra in ("checkpoints", "diffusion_models", "loras", "vae", "controlnet",
                      "upscale_models", "embeddings", cur):
            if extra and extra not in vals:
                vals.append(extra)
        self.cb_sub.configure(values=vals)
        if cur:
            self.var_sub.set(cur)

    # ------------------------------------------------------------ 解析
    def _auto_parse(self):
        if self.meta and self.meta.get("files"):
            return
        if self.var_url.get().strip() and self.var_token.get().strip():
            self.on_parse()

    def on_parse(self):
        url = self.var_url.get().strip()
        if not url:
            messagebox.showwarning("缺少地址", "请先粘贴 civitai 或 HuggingFace 链接。")
            return
        if E.is_hf_url(url):
            self._parse_hf(url)
            return
        self.btn_parse.configure(state="disabled", text="解析中")
        self.set_status("正在解析模型信息…")
        self.parsing = True
        token = self.var_token.get().strip()

        def work():
            try:
                # 页面地址 (/models/<模型ID>?modelVersionId=<版本ID>) 与 API 地址都支持
                ref = E.resolve_version_id(url, token)
                vid = ref.get("version_id")
                if not vid:
                    self.q.put(("meta_err",
                                "链接里没有可识别的版本 ID。若粘贴的是模型页地址，"
                                "请带上 ?modelVersionId=…（模型页右侧 Details 里能看到版本号）"))
                    return
                meta = E.fetch_model_meta(vid, token)
                meta["_ref"] = {k: ref.get(k) for k in
                                ("model_id", "version_id", "file_id", "assumed_latest")}
                self.q.put(("meta", meta))
            except Exception as ex:
                self.q.put(("meta_err", str(ex)))

        threading.Thread(target=work, daemon=True).start()

    def _on_meta(self, meta, quiet=False):
        self.meta = meta
        try:
            self.btn_parse.configure(state="normal", text="解析")
        except Exception:
            pass
        hf = meta.get("_hf")
        self.lbl_meta.configure(text="%s  ·  底模 %s  ·  类型 %s%s" % (
            meta.get("model_name") or "(无名称)", meta.get("base_model") or "?",
            meta.get("type") or "?", "  ·  NSFW" if meta.get("nsfw") else ""))
        if not hf:
            self._apply_folder(meta)
        self._apply_paid(meta.get("paid") or {})
        self._fill_files(meta)
        if not quiet:
            if hf:
                self.log_line("HuggingFace 文件：%s（%s）"
                              % (meta.get("model_name"), meta.get("version_name")))
                self.log_line("大小 %s%s" % (
                    E.human(hf.get("size") or 0),
                    "  镜像 " + hf["mirror"] if hf.get("mirror") else "  官方站"))
                self.log_line("SHA256：%s" % ("有 LFS 摘要，下完会校验"
                                             if hf.get("sha256") else "无摘要，将跳过校验"))
                self.log_line("保存位置请自行确认（HF 文件不带 CivitAI 的底模/类型信息）")
                self.set_status("解析完成", "good")
                return
            self.log_line("解析成功：%s（%d 个文件）"
                          % (meta.get("model_name"), len(meta.get("files") or [])))
            ref = meta.get("_ref") or {}
            self.log_line("版本 %s（模型 %s）%s" % (
                ref.get("version_id"), ref.get("model_id"),
                "  ← 链接未指定版本，已取最新版本" if ref.get("assumed_latest") else ""))
            self.set_status("解析完成", "good")

    def _apply_folder(self, meta):
        if self.var_bybase.get():
            sub = E.resolve_target_folder(self.var_root.get().strip(),
                                          meta.get("type") or "",
                                          meta.get("base_model") or "",
                                          meta.get("subfolder") or "")
            if sub:
                self.var_sub.set(sub)
                self.log_line("按基模归类 → %s（底模：%s）" % (sub, meta.get("base_model") or "?"))
        elif meta.get("subfolder"):
            self.var_sub.set(meta["subfolder"])
        self._refresh_sub_choices()

    def on_reapply_folder(self):
        if self.meta:
            self._apply_folder(self.meta)

    # ------------------------------------------------------------ 付费护栏
    def _apply_paid(self, info):
        self.paid_info = info or {}
        if not info or not info.get("paid"):
            self.paid_bar.grid_remove()
            return
        price = ("标价 %s Buzz。" % info["download_price"]) if info.get("download_price") else \
            "CivitAI 的 API 不返回金额，价格只在站内页面显示，请点右侧按钮核对。"
        self.lbl_paid.configure(text="⚠ 这是付费模型：%s。%s\n加入队列前会要求你二次确认。"
                                     % (E.paid_reason(info), price))
        self.paid_bar.grid(row=7, column=0, columnspan=3, sticky="ew", pady=(8, 0))
        self.log_line("付费检测：%s（需显式确认）" % E.paid_reason(info))

    def on_open_page(self):
        url = (self.paid_info or {}).get("page_url")
        if not url:
            vid, _ = E.parse_civitai_url(self.var_url.get())
            url = "https://civitai.com/models/%s?modelVersionId=%s" % (
                (self.meta or {}).get("model_id", ""), vid or "")
        try:
            import webbrowser
            webbrowser.open(url)
            self.log_line("已在浏览器打开：%s" % url)
        except Exception as ex:
            messagebox.showinfo("页面地址", "%s\n\n(%s)" % (url, ex))

    def _confirm_paid(self, info) -> bool:
        dlg = tk.Toplevel(self)
        dlg.title("确认付费下载")
        dlg.transient(self)
        dlg.resizable(False, False)
        frm = ttk.Frame(dlg, padding=18)
        frm.pack(fill="both", expand=True)
        ttk.Label(frm, text="这是一个付费模型", foreground="#b3261e",
                  font=("Microsoft YaHei UI", 15, "bold")).pack(anchor="w")
        ttk.Label(frm, text="模型：%s\n版本：%s\n文件：%s\n付费类型：%s" % (
            (self.meta or {}).get("model_name") or "?",
            (self.meta or {}).get("version_name") or "?",
            self.var_name.get() or "?", E.paid_reason(info)),
            style="Small.TLabel", justify="left").pack(anchor="w", pady=(10, 6))
        ttk.Label(frm, text="CivitAI 的 API 不返回金额，实际价格请到页面确认。",
                  style="Small.TLabel", foreground="#b3261e").pack(anchor="w")

        var = tk.BooleanVar(value=False)
        btns = ttk.Frame(frm)
        btns.pack(fill="x", pady=(16, 0))
        res = {"ok": False}

        def ok():
            res["ok"] = True
            dlg.destroy()

        ok_btn = ttk.Button(btns, text="仍要下载", command=ok, state="disabled")
        ok_btn.pack(side="right")
        ttk.Button(btns, text="取消（推荐）", command=dlg.destroy).pack(side="right", padx=8)
        ttk.Button(btns, text="查看价格", command=self.on_open_page).pack(side="left")
        ttk.Checkbutton(frm, text="我确认：该模型需要付费，下载会产生费用，费用由我自己承担",
                        variable=var,
                        command=lambda: ok_btn.configure(state="normal" if var.get() else "disabled")
                        ).pack(anchor="w", pady=(12, 0))
        dlg.update_idletasks()
        w, h = dlg.winfo_reqwidth(), dlg.winfo_reqheight()
        dlg.geometry("+%d+%d" % (max(0, self.winfo_rootx() + (self.winfo_width() - w) // 2),
                                 max(0, self.winfo_rooty() + (self.winfo_height() - h) // 3)))
        dlg.grab_set()
        self.wait_window(dlg)
        return res["ok"]

    # ------------------------------------------------------------ 文件列表
    def _file_label(self, f):
        """精度标签：优先 fp，没有就退回格式名（小写），都没有就留空。"""
        fp = (f.get("fp") or "").strip()
        if not fp:
            fp = (f.get("format") or "").strip().lower()
        return fp

    def _fill_files(self, meta):
        files = meta.get("files") or []
        tag = "【付费】" if (meta.get("paid") or {}).get("paid") else "【免费】"
        vals = []
        for f in files:
            fp = self._file_label(f)
            head = tag + (fp + " · " if fp else "")
            vals.append("%s%s · fileId %s" % (head, E.human(f.get("bytes") or 0), f.get("id")))
        self.cb_file.configure(values=vals, state="readonly" if vals else "disabled")
        if not vals:
            return
        fid = (meta.get("_ref") or {}).get("file_id")
        if not fid:
            _vid, fid = E.parse_civitai_url(self.var_url.get())
        idx = 0
        for i, f in enumerate(files):
            if fid and f.get("id") == fid:
                idx = i
                break
        self.cb_file.current(idx)
        self.on_pick_file()

    def on_pick_file(self, _evt=None):
        if not self.meta:
            return
        i = self.cb_file.current()
        files = self.meta.get("files") or []
        if not (0 <= i < len(files)):
            return
        f = files[i]
        self.var_name.set(f.get("name") or "model.safetensors")
        self.var_url.set(E.build_api_url(self.meta.get("version_id"), f.get("id")))
        self.log_line("已选：%s（%s，%s）" % (f.get("name"), self._file_label(f) or "默认精度",
                                              E.human(f.get("bytes") or 0)))

    def on_browse(self):
        d = filedialog.askdirectory(initialdir=self.var_root.get() or E.DEFAULT_MODELS_ROOT)
        if d:
            self.var_root.set(d)
            self._refresh_sub_choices()

    def on_save_token(self):
        t = self.var_token.get().strip()
        if not t:
            messagebox.showwarning("提示", "API Key 为空。")
            return
        E.save_token(t)
        self.log_line("API Key 已保存到 %s" % E.TOKEN_FILE)
        self.set_status("API Key 已保存", "good")

    # ------------------------------------------------------------ 任务操作
    def on_add_task(self):
        url = self.var_url.get().strip()
        dest = self._dest_path()
        token = self.var_token.get().strip()
        if not url:
            messagebox.showwarning("缺少地址", "请先填写或解析下载地址。")
            return
        if not dest:
            messagebox.showwarning("缺少目标", "请填写模型根目录、保存位置和文件名。")
            return
        if not token and not E.is_hf_url(url):
            messagebox.showwarning("缺少 API Key", "CivitAI 现在必须带 API Key 才能下载。")
            return
        for t in self.manager.tasks:
            if os.path.normcase(t.dest) == os.path.normcase(dest) and \
                    t.status not in ("done", "error", "stopped"):
                messagebox.showinfo("已在队列", "同一个目标文件已在队列中：\n%s" % dest)
                return

        paid = (self.meta or {}).get("paid") or {}
        allow_paid = False
        if paid.get("paid"):
            if not self._confirm_paid(paid):
                self.log_line("已取消：付费模型必须显式确认。")
                self.set_status("已取消（付费模型未确认）")
                return
            allow_paid = True

        try:
            os.makedirs(os.path.dirname(dest), exist_ok=True)
        except Exception as ex:
            messagebox.showerror("目录不可用", str(ex))
            return

        t = self.manager.add(api_url=url, dest=dest,
                             segments=int(self.var_seg.get() or 8), token=token,
                             meta=self.meta or {}, allow_paid=allow_paid)
        self._save_cfg()
        self.set_status("已加入队列：%s" % t.name, "good")
        if self.var_autostart.get():
            self.manager.start(t.tid)
        self.manager.pump()
        self._refresh_tree()

    def _sel_tids(self):
        return list(self.tree.selection())

    def on_start_sel(self):
        tids = self._sel_tids()
        if not tids:
            return
        for tid in tids:
            self.manager.start(tid)
        self.manager.pump()

    def on_pause_sel(self):
        for tid in self._sel_tids():
            self.manager.pause(tid)

    def on_stop_sel(self):
        tids = self._sel_tids()
        for tid in tids:
            self.manager.stop(tid)
        self.manager.pump()

    def on_retry_sel(self):
        for tid in self._sel_tids():
            self.manager.start(tid)
        self.manager.pump()

    def on_clean_restart_sel(self):
        """删除本地残留（.part/.part.json/.bad）后从 0 重新下载。"""
        tids = self._sel_tids()
        if not tids:
            messagebox.showinfo("提示", "先在队列里选中要重下的任务。")
            return
        names = []
        for tid in tids:
            t = self.manager.get(tid)
            if t:
                names.append("  · " + t.name)
        if not messagebox.askyesno(
                "清空重下",
                "将删除下列任务的本地残留并从 0 重新下载\n"
                "（.part 进度 / .part.json 状态 / .bad 坏文件）：\n\n%s\n\n确定？"
                % "\n".join(names[:10])):
            return
        for tid in tids:
            r = self.manager.restart_clean(tid)
            if r.get("removed"):
                self.log_line("已删除: %s" % ", ".join(r["removed"]))
        self._save_cfg()
        self._refresh_tree()

    def _check_hash_failures(self):
        """校验失败时主动提示是否清空重下（每个任务只提示一次）。"""
        for s in self.manager.snapshot():
            tid = s["tid"]
            if s.get("hash_mismatch") and s["state"] == "error" \
                    and tid not in self._mismatch_prompted:
                self._mismatch_prompted.add(tid)
                if messagebox.askyesno(
                        "文件校验失败",
                        "%s\n\nSHA256 与 CivitAI 标注不一致，坏文件已改名为 .bad。\n\n"
                        "要删除本地残留并从头重新下载吗？\n"
                        "（选「否」则保持现状，你可以之后手动点「清空重下」）" % s["name"]):
                    self.manager.restart_clean(tid)

    def _check_hf_auth(self):
        """任务因 HuggingFace 401/403 失败时，弹窗引导填 Token 并重试。"""
        for s in self.manager.snapshot():
            tid = s["tid"]
            if s.get("needs_auth") and tid not in self._auth_prompted:
                self._auth_prompted.add(tid)
                self._prompt_hf_auth(s.get("api_url") or "", retry_tid=tid)

    def _repo_of(self, url):
        """从 HF 直链里取出 (域名, owner/name)。"""
        try:
            import urllib.parse as _up
            p = _up.urlparse((url or "").strip())
            parts = [x for x in (p.path or "").split("/") if x]
            if "resolve" in parts:
                i = parts.index("resolve")
                return p.netloc or "huggingface.co", "/".join(parts[:i])
        except Exception:
            pass
        return "huggingface.co", ""

    def _prompt_hf_auth(self, url, retry_tid=None):
        """需要 HF 授权时的引导弹窗：打开仓库页 → 填 Token → 重试。"""
        host, repo = self._repo_of(url)
        page = "https://%s/%s" % ("huggingface.co", repo) if repo else "https://huggingface.co"
        msg = (
            "这个文件需要 HuggingFace 登录授权（gated 仓库）。\n\n"
            "仓库：%s\n\n"
            "步骤：\n"
            "  1. 打开仓库页面，登录并点 Agree / 同意条款\n"
            "  2. 到 https://huggingface.co/settings/tokens 建一个 Read 权限的 Token\n"
            "  3. 回到这里填入 Token，会自动重试下载\n\n"
            "现在打开仓库页面吗？" % (repo or url)
        )
        if not messagebox.askyesno("需要 HuggingFace 授权", msg):
            return False
        try:
            webbrowser.open(page)
        except Exception:
            pass
        val = simpledialog.askstring(
            "填入 HuggingFace Token",
            "粘贴 HF Token（以 hf_ 开头，Read 权限即可）：",
            show="*", parent=self)
        if not val or not val.strip():
            self.log_line("未填入 HF Token，已跳过（可稍后点「HF Token」按钮再设置）")
            return False
        val = val.strip()
        self.cfg["hf_token"] = val
        self.manager.hf_token = val
        self._save_cfg()
        self.lbl_hf_tok.configure(text=self._hf_token_hint())
        self.log_line("HF Token 已保存（%s…），重新开始该任务" % val[:4])
        if retry_tid:
            self._auth_prompted.discard(retry_tid)
            self.manager.start(retry_tid)
        self.set_status("已保存 Token，正在重试", "good")
        return True

    def on_remove_sel(self):
        tids = self._sel_tids()
        if not tids:
            return
        if not messagebox.askyesno("移除任务", "从队列移除选中的 %d 个任务？\n"
                                              "已下载的部分（.part）会保留，重新加入可续传。" % len(tids)):
            return
        for tid in tids:
            self.manager.remove(tid)
        self._save_cfg()
        self._refresh_tree()

    def on_toggle_sel(self):
        tids = self._sel_tids()
        if len(tids) == 1:
            self.manager.toggle(tids[0])

    def on_open_dir_sel(self):
        tids = self._sel_tids()
        if not tids:
            return
        t = self.manager.get(tids[0])
        if not t:
            return
        d = t.folder
        try:
            os.makedirs(d, exist_ok=True)
            os.startfile(d)
        except Exception as ex:
            messagebox.showerror("打开失败", str(ex))

    def on_stop_all(self):
        if self.manager.totals()["running"] and \
                not messagebox.askyesno("全部停止", "停止所有正在下载的任务？进度会保留。"):
            return
        self.manager.stop_all()

    def on_apply_seg(self):
        tids = self._sel_tids()
        if not tids:
            messagebox.showinfo("提示", "先在队列里选中任务，再应用分片数。")
            return
        n = int(self.var_seg_sel.get() or 8)
        ok = 0
        for tid in tids:
            if self.manager.set_segments(tid, n):
                ok += 1
        self.log_line("已把分片数设为 %d（%d 个任务；运行中的任务需先停止）" % (n, ok))
        self._refresh_tree()

    def on_parallel_change(self):
        try:
            n = int(self.var_parallel.get())
        except Exception:
            return
        self.manager.set_max_parallel(n)
        self.log_line("并发任务数上限 = %d" % n)

    def on_speed_change(self):
        try:
            mb = float(self.var_speed.get())
        except Exception:
            return
        self.manager.set_speed_limit(mb)
        self.log_line("全局限速 = %s" % ("不限" if mb <= 0 else "%.2f MB/s" % mb))

    # ------------------------------------------------------------ 事件循环
    def _on_manager_event(self, kind, tid, payload):
        self.q.put(("mgr", (kind, tid, payload)))

    def _poll(self):
        try:
            while True:
                kind, payload = self.q.get_nowait()
                if kind == "meta":
                    self.parsing = False
                    self._on_meta(payload)
                elif kind == "hf_auth":
                    self.parsing = False
                    self.btn_parse.configure(state="normal", text="解析")
                    self.log_line("该 HF 仓库需要登录授权，已弹出引导窗口")
                    if self._prompt_hf_auth(payload):
                        self.on_parse()          # 拿到 Token 后自动重新解析
                    else:
                        self.set_status("需要 HuggingFace 授权")
                elif kind == "meta_err":
                    self.parsing = False
                    self.btn_parse.configure(state="normal", text="解析")
                    self.log_line("解析失败：%s" % payload)
                    self.set_status("解析失败：" + payload, "bad")
                elif kind == "mgr":
                    ev, tid, val = payload
                    t = self.manager.get(tid)
                    label = t.name if t else tid
                    if ev == "log":
                        self.log_line(val, label)
                    elif ev == "state" and val == "done":
                        self.set_status("%s 已完成" % label, "good")
        except queue.Empty:
            pass

        self.manager.tick()
        self._refresh_tree()
        self._check_hash_failures()
        self._check_hf_auth()
        self.after(500, self._poll)

    def _refresh_tree(self):
        snaps = self.manager.snapshot()
        alive = set()
        for s in snaps:
            tid = s["tid"]
            alive.add(tid)
            vals = (s["state_text"], s["name"],
                    E.human(s["size_total"]) if s["size_total"] else "—",
                    "%.1f%%" % s["pct"], ("%s/s" % E.human(s["speed"])) if s["speed"] > 0 else "—",
                    E.hms(s["eta"]) if s["eta"] else "—", str(s["segments"]),
                    str(s["retries"]), s["folder"])
            tag = STATE_TAG.get(s["state"], "")
            if s["paid"]:
                vals = (vals[0], "【付费】" + vals[1]) + vals[2:]
            if tid in self.rows and self.tree.exists(self.rows[tid]):
                self.tree.item(self.rows[tid], values=vals, tags=(tag,))
            else:
                self.rows[tid] = self.tree.insert("", "end", iid=tid, values=vals, tags=(tag,))
        for tid in list(self.rows.keys()):
            if tid not in alive:
                try:
                    self.tree.delete(self.rows[tid])
                except Exception:
                    pass
                self.rows.pop(tid, None)

        tot = self.manager.totals()
        self.lbl_total.configure(
            text="共 %d 个任务 · 运行 %d · 总进度 %.1f%% · 总速度 %s/s"
                 % (tot["count"], tot["running"], tot["pct"], E.human(tot["speed"])))

    # ------------------------------------------------------------ 关闭
    def _on_close(self):
        if self.manager.totals()["running"]:
            if not messagebox.askyesno("退出", "还有任务在下载，退出会停止它们（进度已保存，下次可续传）。确定退出？"):
                return
            self.manager.stop_all()
            time.sleep(0.3)
        try:
            self._save_cfg()
        except Exception:
            pass
        self.destroy()


def main():
    try:
        import ctypes
        ctypes.windll.shcore.SetProcessDpiAwareness(1)
    except Exception:
        pass
    App().mainloop()


if __name__ == "__main__":
    main()
