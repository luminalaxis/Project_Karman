"""Project_Karman 런처 — 레포 루트의 Karman.bat 으로 실행합니다.

도구 목록은 같은 폴더의 tools.json 에서 읽습니다. 표준 라이브러리(tkinter)만 사용합니다.
"""

from __future__ import annotations

import importlib.util
import json
import os
import subprocess
import sys
import tkinter as tk
from pathlib import Path
from tkinter import messagebox

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
IS_WIN = os.name == "nt"

# ── 브랜드 색 ─────────────────────────────────────────────
BG = "#0E0F12"
CARD = "#17181D"
CARD_HOVER = "#1E2027"
LINE = "#2A2C34"
INK = "#F2F2EF"
MUTED = "#9A9CA5"
DIM = "#6B6E78"
RED = "#FF3B2F"
RED_HOVER = "#FF5A4F"
AMBER = "#F0B43C"
OK = "#4CC38A"

FONT = "Malgun Gothic" if IS_WIN else "Noto Sans CJK KR"


def console_python() -> str:
    """콘솔 창에서 쓸 python.exe (런처는 pythonw.exe 로 돌기 때문)."""
    exe = Path(sys.executable)
    if exe.name.lower() == "pythonw.exe":
        cand = exe.with_name("python.exe")
        if cand.exists():
            return str(cand)
    return str(exe)


def window_python() -> str:
    """콘솔 창 없이 GUI 를 띄울 pythonw.exe."""
    exe = Path(sys.executable)
    if IS_WIN and exe.name.lower() == "python.exe":
        cand = exe.with_name("pythonw.exe")
        if cand.exists():
            return str(cand)
    return str(exe)


def load_tools() -> list[dict]:
    with open(HERE / "tools.json", encoding="utf-8") as f:
        return json.load(f)["tools"]


def missing_modules(tool: dict) -> list[str]:
    importlib.invalidate_caches()
    pkg_dir = str(ROOT / tool["path"])
    added = pkg_dir not in sys.path
    if added:
        sys.path.insert(0, pkg_dir)  # 폴더에서 바로 실행하는 도구(lapsim 등)도 확인되도록
    try:
        out = []
        for mod in tool.get("requires", []):
            try:
                if importlib.util.find_spec(mod) is None:
                    out.append(mod)
            except (ImportError, ValueError):
                out.append(mod)
        return out
    finally:
        if added:
            sys.path.remove(pkg_dir)


def open_console(title: str, command: str, cwd: Path, pause: bool = False) -> None:
    if not IS_WIN:
        print(f"[console] ({cwd}) {command}")
        return
    tail = " & echo. & pause" if pause else ""
    line = f'cmd /s /k "title {title} & {command}{tail}"'
    subprocess.Popen(line, cwd=cwd, creationflags=subprocess.CREATE_NEW_CONSOLE)


def open_path(p: Path) -> None:
    if IS_WIN:
        os.startfile(p)  # noqa: S606
    else:
        print(f"[open] {p}")


class Button(tk.Label):
    """Windows 에서도 색이 그대로 나오도록 Label 로 만든 버튼."""

    def __init__(self, master, text, command, primary=False, **kw):
        self.bg = RED if primary else CARD
        self.hover = RED_HOVER if primary else LINE
        fg = "#FFFFFF" if primary else INK
        super().__init__(master, text=text, bg=self.bg, fg=fg, cursor="hand2",
                         font=(FONT, 10, "bold" if primary else "normal"),
                         padx=16, pady=7, highlightthickness=0 if primary else 1,
                         highlightbackground=LINE, **kw)
        self.command = command
        self.bind("<Button-1>", lambda e: self.command())
        self.bind("<Enter>", lambda e: self.config(bg=self.hover))
        self.bind("<Leave>", lambda e: self.config(bg=self.bg))

    def set_base(self, bg):
        self.bg = bg
        self.config(bg=bg)


class Launcher(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Project Karman")
        self.configure(bg=BG)
        self.resizable(False, False)
        ico = ROOT / "assets" / "karman.ico"
        if IS_WIN and ico.exists():
            try:
                self.iconbitmap(default=str(ico))
            except tk.TclError:
                pass
        self.close_after = tk.BooleanVar(value=False)
        self.tools = load_tools()
        self.cards: list[tuple[dict, dict]] = []
        self._build()
        self.bind("<Escape>", lambda e: self.destroy())
        for i in range(len(self.tools)):
            self.bind(str(i + 1), lambda e, i=i: self.primary_action(self.tools[i]))
        self.bind("<F5>", lambda e: self.refresh())
        self.update_idletasks()
        w, h = self.winfo_width(), self.winfo_height()
        x = (self.winfo_screenwidth() - w) // 2
        y = (self.winfo_screenheight() - h) // 3
        self.geometry(f"+{x}+{y}")

    # ── 화면 구성 ────────────────────────────────────────
    def _build(self):
        head = tk.Frame(self, bg=BG)
        head.pack(fill="x", padx=28, pady=(26, 6))
        logo = HERE / "logo.png"
        if logo.exists():
            self.logo_img = tk.PhotoImage(file=str(logo))
            tk.Label(head, image=self.logo_img, bg=BG).pack(side="left")
        else:
            tk.Label(head, text="PROJECT KARMAN", bg=BG, fg=INK, font=(FONT, 18, "bold")).pack(side="left")
        right = tk.Frame(head, bg=BG)
        right.pack(side="right", anchor="s")
        tk.Label(right, text="AERO TOOLKIT", bg=BG, fg=RED, font=(FONT, 9, "bold")).pack(anchor="e")
        tk.Label(right, text="e_formula 공력팀 도구 런처", bg=BG, fg=MUTED, font=(FONT, 10)).pack(anchor="e")

        tk.Frame(self, bg=LINE, height=1).pack(fill="x", padx=28, pady=(14, 14))

        body = tk.Frame(self, bg=BG)
        body.pack(fill="both", padx=28)
        for i, tool in enumerate(self.tools):
            self.cards.append((tool, self._card(body, i, tool)))

        foot = tk.Frame(self, bg=BG)
        foot.pack(fill="x", padx=28, pady=(10, 22))
        tk.Label(foot, text="숫자키 1–{}로 바로 실행 · F5 다시 확인 · Esc 닫기".format(len(self.tools)),
                 bg=BG, fg=DIM, font=(FONT, 9)).pack(side="left")
        cb = tk.Checkbutton(foot, text="실행하면 런처 닫기", variable=self.close_after, bg=BG, fg=MUTED,
                            selectcolor=CARD, activebackground=BG, activeforeground=INK,
                            font=(FONT, 9), highlightthickness=0, bd=0, cursor="hand2")
        cb.pack(side="right")
        tk.Label(foot, text=f"Python {sys.version_info.major}.{sys.version_info.minor}", bg=BG, fg=DIM,
                 font=(FONT, 9)).pack(side="right", padx=14)

    def _card(self, parent, idx, tool):
        card = tk.Frame(parent, bg=CARD, highlightthickness=1, highlightbackground=LINE)
        card.pack(fill="x", pady=5)
        inner = tk.Frame(card, bg=CARD)
        inner.pack(fill="x", padx=16, pady=12)

        num = tk.Label(inner, text=str(idx + 1), bg=CARD, fg=DIM, font=(FONT, 16, "bold"), width=2)
        num.grid(row=0, column=0, rowspan=3, sticky="n", padx=(0, 10))

        title_row = tk.Frame(inner, bg=CARD)
        title_row.grid(row=0, column=1, sticky="w")
        name = tk.Label(title_row, text=tool["name"], bg=CARD, fg=INK, font=(FONT, 13, "bold"))
        name.pack(side="left")
        chip = tk.Label(title_row, text="", bg=CARD, font=(FONT, 8, "bold"), padx=7, pady=1)
        chip.pack(side="left", padx=(10, 0))

        tag = tk.Label(inner, text=tool["tagline"], bg=CARD, fg=INK, font=(FONT, 10), anchor="w")
        tag.grid(row=1, column=1, sticky="w", pady=(2, 0))
        det = tk.Label(inner, text=tool.get("detail", ""), bg=CARD, fg=MUTED, font=(FONT, 9), anchor="w")
        det.grid(row=2, column=1, sticky="w")
        inner.grid_columnconfigure(1, weight=1, minsize=430)

        btns = tk.Frame(inner, bg=CARD)
        btns.grid(row=0, column=2, rowspan=3, sticky="e")
        primary = Button(btns, "", lambda t=tool: self.primary_action(t), primary=True)
        primary.pack(side="right")
        folder = Button(btns, "폴더", lambda t=tool: open_path(ROOT / t["path"]))
        folder.pack(side="right", padx=(0, 8))

        widgets = dict(card=card, inner=inner, num=num, title_row=title_row, name=name, tag=tag, det=det,
                       btns=btns, chip=chip, primary=primary, folder=folder)

        hover_targets = [card, inner, num, title_row, name, tag, det, btns]

        def paint(color):
            for w in hover_targets:
                w.config(bg=color)
            folder.set_base(color)
            if chip.cget("bg") in (CARD, CARD_HOVER):
                chip.config(bg=color)

        for w in hover_targets:
            w.bind("<Enter>", lambda e: paint(CARD_HOVER))
            w.bind("<Leave>", lambda e: paint(CARD))
            w.bind("<Double-Button-1>", lambda e, t=tool: self.primary_action(t))
        self._update_card(tool, widgets)
        return widgets

    def _update_card(self, tool, w):
        miss = missing_modules(tool) if tool["kind"] != "docs" else []
        tool["_missing"] = miss
        if tool["kind"] == "docs":
            w["chip"].config(text=tool.get("status", ""), fg=AMBER, bg=CARD)
            w["primary"].config(text="문서 열기")
        elif miss:
            w["chip"].config(text="설치 필요", fg="#FFFFFF", bg="#7A2A24")
            w["primary"].config(text="설치")
        else:
            w["chip"].config(text=tool.get("status", "준비됨"), fg=OK, bg=w["card"].cget("bg"))
            w["primary"].config(text="실행")

    def refresh(self):
        for tool, w in self.cards:
            self._update_card(tool, w)

    # ── 동작 ────────────────────────────────────────────
    def primary_action(self, tool):
        cwd = ROOT / tool["path"]
        if not cwd.exists():
            messagebox.showerror("Project Karman", f"폴더가 없습니다:\n{cwd}")
            return
        if tool["kind"] == "docs":
            readme = cwd / "README.md"
            open_path(readme if readme.exists() else cwd)
            return
        if tool.get("_missing"):
            ok = messagebox.askyesno(
                "설치 필요",
                f"{tool['name']} 실행에 필요한 모듈이 없습니다:\n  {', '.join(tool['_missing'])}\n\n"
                "새 터미널에서 설치할까요? 설치가 끝나면 런처에서 F5를 눌러 다시 확인하세요.")
            if ok:
                cmd = tool["install"].format(python=console_python())
                open_console(f"{tool['name']} 설치", cmd, cwd, pause=True)
            return
        self.launch(tool, cwd)
        if self.close_after.get():
            self.after(400, self.destroy)

    def launch(self, tool, cwd):
        if tool["kind"] == "gui":
            cmd = tool["run"].format(python=window_python())
            logs = HERE / "logs"
            logs.mkdir(exist_ok=True)
            log_path = logs / f"{tool['id']}.log"
            log = open(log_path, "w", encoding="utf-8")
            flags = subprocess.CREATE_NO_WINDOW if IS_WIN else 0
            proc = subprocess.Popen(cmd if IS_WIN else cmd.replace('"', "").split(), cwd=cwd,
                                    stdout=log, stderr=subprocess.STDOUT, creationflags=flags)
            self.after(4000, lambda: self._check_gui(tool, proc, log, log_path))
        else:
            cmd = tool["run"].format(python=console_python())
            open_console(tool["name"], cmd, cwd)

    def _check_gui(self, tool, proc, log, log_path):
        code = proc.poll()
        if code is None:  # 아직 실행 중이면 정상
            return
        log.close()
        if code != 0:
            text = log_path.read_text(encoding="utf-8", errors="replace").strip().splitlines()[-15:]
            messagebox.showerror(f"{tool['name']} 실행 실패",
                                 f"종료 코드 {code}\n\n" + "\n".join(text) + f"\n\n전체 로그: {log_path}")


if __name__ == "__main__":
    Launcher().mainloop()
