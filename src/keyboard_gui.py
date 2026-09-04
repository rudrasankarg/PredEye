"""
keyboard_gui.py — Gaze-controlled virtual keyboard with LSTM prediction slots

Layout (Tkinter):
    ┌─────────────────────────────────┐
    │     [PRED1]  [PRED2]  [PRED3]   │  ← prediction row (our novel addition)
    ├──────────────────────────────────┤
    │  NW    │   N    │   NE          │
    │  W     │   C    │   E           │  ← 3×3 command grid with dwell bars
    │  SW    │   S    │   SE          │
    ├──────────────────────────────────┤
    │        Typed text display        │
    └─────────────────────────────────┘

Dwell time: each cell shows a filling progress bar at the bottom.
The command fires only after the gaze stays in the zone for the full dwell period.
"""

import json
import sys
import threading
import time
import tkinter as tk
from pathlib import Path
from tkinter import font as tkfont
from typing import Callable, Optional

# Path to the editable phrase library
_PHRASES_JSON = Path(__file__).parent.parent / "phrases.json"


def _load_phrases() -> dict:
    """Load phrases from phrases.json; return empty dict on failure."""
    try:
        with open(_PHRASES_JSON, encoding="utf-8") as f:
            return json.load(f).get("categories", {})
    except Exception:
        return {
            "Greetings": ["Hello", "Thank you", "Please", "Goodbye",
                          "Good morning", "Good night", "Yes", "No", "How are you?"],
            "Needs":     ["I need help", "I am hungry", "I am thirsty",
                          "I am tired", "I am in pain", "Call a doctor",
                          "I need the bathroom", "I am cold", "I am hot"],
        }

try:
    from PIL import Image, ImageTk
    _PIL_AVAILABLE = True
except ImportError:
    _PIL_AVAILABLE = False

# ── Keyboard layout ────────────────────────────────────────────────────────────

LETTER_GROUPS: dict[int, str] = {
    1: "ABCD",
    2: "EFGH",
    3: "IJKL",
    4: "MNO",
    5: "SPACE",     # Centre = SPACE at top level; GO BACK at level 2
    6: "PQR",
    7: "STUV",
    8: "WXYZ",
    9: "DELETE",
}

DIR_POSITIONS: dict[int, tuple] = {
    1: (0, 0), 2: (0, 1), 3: (0, 2),
    4: (1, 0), 5: (1, 1), 6: (1, 2),
    7: (2, 0), 8: (2, 1), 9: (2, 2),
}

# ── Colour palette ─────────────────────────────────────────────────────────────

PALETTE = {
    "bg":           "#1a1a2e",
    "cell_idle":    "#16213e",
    "cell_hover":   "#0f3460",
    "cell_active":  "#e94560",
    "pred_bg":      "#0d2137",
    "pred_hover":   "#533483",
    "pred_active":  "#e94560",
    "text_primary": "#eaeaea",
    "text_dim":     "#7f8c8d",
    "accent":       "#00d2ff",
    "typed_bg":     "#0a0a1a",
    "dwell_bar":    "#00d2ff",
    "dwell_track":  "#0a0a1a",
}


class KeyboardGUI:
    """
    Main gaze keyboard window.

    External interface:
        gui.set_gaze_direction(d)         -> highlight cell d
        gui.set_dwell_progress(d, frac)   -> update dwell bar 0.0–1.0 for cell d
        gui.fire_command(d)               -> execute command at direction d
        gui.update_predictions(words)     -> update 3 prediction buttons
        gui.set_text(text)                -> set typed text directly
    """

    def __init__(
        self,
        on_text_change: Optional[Callable] = None,
        prediction_enabled: bool = True,
        tts_enabled: bool = True,
    ):
        self.on_text_change    = on_text_change
        self.prediction_enabled = prediction_enabled
        self.tts_enabled       = tts_enabled

        self._typed_text  = ""
        self._level       = 1
        self._selected_group = 0
        self._current_cells: dict[int, str] = {}
        self._gaze_dir    = 5
        self._pred_labels = ["", "", ""]
        self._last_fire_time = 0.0   # cooldown: prevents double-fires
        self._fire_cooldown  = 1.5   # seconds between allowed fires (hard backstop)
        self._is_done        = False # locks input when done typing
        self._session_start  = time.time()

        # Survey state
        self._is_in_survey   = False
        self._survey_q_idx   = 0
        self._survey_answers: dict = {}
        self._survey_user_id = "default"
        self._survey_mode    = "unknown"
        self._survey_callback = None
        self._survey_frame   = None

        # Phrase bank state
        self._phrase_panel       = None   # overlay Frame when open
        self._phrase_categories  = _load_phrases()
        self._phrase_zone_map: dict[int, str] = {}   # zone -> phrase/category
        self._phrase_mode        = "category"         # "category" or "phrases"
        self._phrase_active_cat  = ""                 # currently shown category

        # TTS
        self._tts = None
        if tts_enabled:
            try:
                # pyrefly: ignore [missing-import]
                import pyttsx3
                self._tts = pyttsx3.init()
                self._tts.setProperty("rate", 160)
            except Exception:
                pass

        self._build_window()
        self._set_level1()

    # ── Window construction ────────────────────────────────────────────────────

    def _build_window(self) -> None:
        self.root = tk.Tk()
        self.root.title("Gaze Keyboard — Mouse Mode (move mouse to zone, hold to select)")
        self.root.configure(bg=PALETTE["bg"])
        self.root.resizable(True, True)

        # Start maximized on Windows
        try:
            self.root.state("zoomed")
        except Exception:
            self.root.geometry("1100x700")

        # Fonts
        try:
            mono      = tkfont.Font(family="Consolas",  size=13, weight="bold")
            hdr       = tkfont.Font(family="Segoe UI",  size=10, weight="bold")
            big       = tkfont.Font(family="Consolas",  size=18, weight="bold")
            pred_font = tkfont.Font(family="Segoe UI",  size=12)
            small     = tkfont.Font(family="Segoe UI",  size=9)
        except Exception:
            mono = hdr = big = pred_font = small = tkfont.Font(size=11)

        self._font_mono = mono
        self._font_hdr  = hdr
        self._font_big  = big
        self._font_pred = pred_font
        self._font_small = small

        # ── Prediction row ─────────────────────────────────────────────────────
        pred_outer = tk.Frame(self.root, bg=PALETTE["pred_bg"])
        pred_outer.pack(fill="x", padx=12, pady=(10, 0))

        tk.Label(pred_outer, text="PREDICTIONS",
                 fg=PALETTE["text_dim"], bg=PALETTE["pred_bg"], font=hdr
                 ).pack(side="left", padx=(10, 0))

        self._pred_buttons: list[tk.Button] = []
        for i in range(3):
            btn = tk.Button(
                pred_outer,
                text="[PRED]",
                fg=PALETTE["text_primary"],
                bg=PALETTE["pred_bg"],
                activebackground=PALETTE["pred_active"],
                activeforeground="#fff",
                relief="flat",
                font=pred_font,
                width=16,
                pady=6,
                cursor="hand2",
                command=lambda idx=i: self._on_prediction_click(idx),
            )
            btn.pack(side="left", padx=8, pady=4)
            self._pred_buttons.append(btn)

        self._toggle_pred_btn = tk.Button(
            pred_outer,
            text="PRED: ON" if self.prediction_enabled else "PRED: OFF",
            fg=PALETTE["accent"],
            bg=PALETTE["pred_bg"],
            activebackground=PALETTE["cell_hover"],
            relief="flat",
            font=hdr,
            command=self._toggle_predictions,
            cursor="hand2",
        )
        self._toggle_pred_btn.pack(side="right", padx=10)

        # ── Phrase Bank button (zone 14) ───────────────────────────────────────
        self._phrase_btn = tk.Button(
            pred_outer,
            text="📋 PHRASES",
            fg="#f0e68c",
            bg=PALETTE["pred_bg"],
            activebackground="#7a5c00",
            relief="flat",
            font=hdr,
            command=self._open_phrase_bank,
            cursor="hand2",
            padx=8,
        )
        self._phrase_btn.pack(side="right", padx=(0, 6))

        self._done_btn = tk.Button(
            pred_outer,
            text="DONE TYPING",
            fg="#ffffff",
            bg="#27ae60",
            activebackground="#2ecc71",
            relief="flat",
            font=hdr,
            command=self._finish_typing,
            cursor="hand2",
            padx=10
        )
        self._done_btn.pack(side="right", padx=(0, 10))

        # Dwell bars for prediction buttons (zones 10-12)
        self._pred_dwell_canvases: list = []
        self._pred_dwell_bars:     list = []
        for btn in self._pred_buttons:
            # We'll draw dwell bars lazily using canvas overlaid on pred_outer
            # For now just stash references; bars drawn in set_dwell_progress
            self._pred_dwell_canvases.append(None)
            self._pred_dwell_bars.append(None)

        # Done button dwell state (zone 13)
        self._done_dwell_frac = 0.0

        # ── 3×3 grid ───────────────────────────────────────────────────────────
        grid_frame = tk.Frame(self.root, bg=PALETTE["bg"])
        grid_frame.pack(fill="both", expand=True, padx=12, pady=10)

        for col in range(3):
            grid_frame.columnconfigure(col, weight=1)
        for row in range(3):
            grid_frame.rowconfigure(row, weight=1)

        self._cell_frames:    dict[int, tk.Frame]  = {}
        self._cell_labels:    dict[int, tk.Label]  = {}
        self._cell_sublabels: dict[int, tk.Label]  = {}
        self._dwell_canvases: dict[int, tk.Canvas] = {}
        self._dwell_bars:     dict[int, int]       = {}  # canvas item ids

        for direction, (row, col) in DIR_POSITIONS.items():
            outer = tk.Frame(grid_frame, bg=PALETTE["cell_idle"], bd=2, relief="flat")
            outer.grid(row=row, column=col, padx=5, pady=5, sticky="nsew")

            # Direction number
            dir_lbl = tk.Label(outer, text=str(direction),
                                fg=PALETTE["text_dim"], bg=PALETTE["cell_idle"],
                                font=hdr, anchor="nw")
            dir_lbl.place(x=5, y=3)

            # Main content label
            content_lbl = tk.Label(outer, text="",
                                   fg=PALETTE["text_primary"], bg=PALETTE["cell_idle"],
                                   font=mono, wraplength=200, justify="center")
            content_lbl.place(relx=0.5, rely=0.42, anchor="center")

            # Sub-label
            sub_lbl = tk.Label(outer, text="",
                                fg=PALETTE["text_dim"], bg=PALETTE["cell_idle"],
                                font=small)
            sub_lbl.place(relx=0.5, rely=0.72, anchor="center")

            # Dwell progress bar (Canvas at bottom of each cell)
            dwell_canvas = tk.Canvas(outer, height=8, bg=PALETTE["dwell_track"],
                                      highlightthickness=0)
            dwell_canvas.place(relx=0, rely=1.0, anchor="sw", relwidth=1.0)
            bar_item = dwell_canvas.create_rectangle(0, 0, 0, 8,
                                                      fill=PALETTE["dwell_bar"],
                                                      outline="")

            self._cell_frames[direction]    = outer
            self._cell_labels[direction]    = content_lbl
            self._cell_sublabels[direction] = sub_lbl
            self._dwell_canvases[direction] = dwell_canvas
            self._dwell_bars[direction]     = bar_item

        # ── Typed text area ────────────────────────────────────────────────────
        text_frame = tk.Frame(self.root, bg=PALETTE["typed_bg"])
        text_frame.pack(fill="x", padx=12, pady=(0, 10))

        tk.Label(text_frame, text="Typed:", fg=PALETTE["text_dim"],
                 bg=PALETTE["typed_bg"], font=hdr).pack(side="left", padx=(8, 0))

        self._typed_display = tk.Label(
            text_frame, text="▌",
            fg=PALETTE["accent"], bg=PALETTE["typed_bg"],
            font=big, anchor="w",
        )
        self._typed_display.pack(side="left", fill="x", expand=True, padx=(8, 0))

        # Status bar
        self._status_bar = tk.Label(
            self.root,
            text="Move mouse to a zone and hold still to select",
            fg=PALETTE["text_dim"], bg=PALETTE["bg"],
            font=small, anchor="w",
        )
        self._status_bar.pack(fill="x", padx=12, pady=(0, 4))

        # ── Camera preview panel ────────────────────────────────────────────────
        cam_outer = tk.Frame(self.root, bg="#000", bd=1, relief="solid")
        cam_outer.place(relx=1.0, rely=1.0, anchor="se", x=-12, y=-30,
                        width=220, height=165)
        tk.Label(cam_outer, text="CAMERA PREVIEW",
                 fg=PALETTE["text_dim"], bg="#000",
                 font=small).pack()
        self._cam_canvas = tk.Canvas(cam_outer, width=220, height=150,
                                     bg="#111", highlightthickness=0)
        self._cam_canvas.pack()
        self._cam_photo = None   # keep reference to prevent GC
        self._cam_status_text = self._cam_canvas.create_text(
            110, 75, text="Press Enter/Space\nto start camera",
            fill=PALETTE["text_dim"], font=small, justify="center"
        )

    # ── Level management ───────────────────────────────────────────────────────

    def _set_level1(self) -> None:
        self._level = 1
        self._current_cells = {d: LETTER_GROUPS[d] for d in range(1, 10)}
        self._refresh_cells()

    def _set_level2(self, group_dir: int) -> None:
        self._level = 2
        self._selected_group = group_dir
        letters = list(LETTER_GROUPS[group_dir])
        positions = [d for d in range(1, 10) if d != 5]
        self._current_cells = {}
        for i, d in enumerate(positions):
            self._current_cells[d] = letters[i] if i < len(letters) else ""
        self._current_cells[5] = "<< BACK"   # GO BACK only at level 2
        self._refresh_cells()

    def _refresh_cells(self) -> None:
        for direction in range(1, 10):
            label = self._current_cells.get(direction, "")
            self._cell_labels[direction].config(text=label)
            if self._level == 1 and direction not in (5, 9):
                self._cell_sublabels[direction].config(text=f"({len(label)} letters)")
            else:
                self._cell_sublabels[direction].config(text="")
        # Reset all dwell bars when level changes
        self.root.after(0, lambda: [self.set_dwell_progress(d, 0.0) for d in range(1, 10)])

    # ── Gaze / dwell interface ─────────────────────────────────────────────────

    def set_gaze_direction(self, direction: int) -> None:
        """Highlight the focused cell. Handles zones 1-13. Thread-safe."""
        if direction == self._gaze_dir:
            return
        self._gaze_dir = direction
        if direction in range(10, 14):
            # Top-bar zones: highlight pred buttons or done button
            self.root.after(0, lambda d=direction: self._update_topbar_highlight(d))
        else:
            self.root.after(0, self._update_cell_highlight)

    def _update_cell_highlight(self) -> None:
        for d, frame in self._cell_frames.items():
            is_focused = (d == self._gaze_dir)
            color = PALETTE["cell_hover"] if is_focused else PALETTE["cell_idle"]
            frame.config(bg=color)
            self._cell_labels[d].config(bg=color)
            self._cell_sublabels[d].config(bg=color)
        # Unhighlight top-bar when grid zone is focused
        for btn in self._pred_buttons:
            btn.config(bg=PALETTE["pred_bg"])
        self._done_btn.config(bg="#27ae60")

    def _update_topbar_highlight(self, direction: int) -> None:
        # Unhighlight all grid cells
        for d, frame in self._cell_frames.items():
            frame.config(bg=PALETTE["cell_idle"])
            self._cell_labels[d].config(bg=PALETTE["cell_idle"])
            self._cell_sublabels[d].config(bg=PALETTE["cell_idle"])
        # Highlight correct top-bar element
        for i, btn in enumerate(self._pred_buttons):
            btn.config(bg=PALETTE["pred_hover"] if direction == 10 + i else PALETTE["pred_bg"])
        self._done_btn.config(bg="#2ecc71" if direction == 13 else "#27ae60")

    def set_dwell_progress(self, direction: int, fraction: float) -> None:
        """
        Update dwell progress bar for a cell (zones 1-9) or top-bar button (zones 10-13).
        fraction: 0.0 = empty, 1.0 = full.
        Thread-safe.
        """
        if direction in range(10, 14):
            # Top-bar button dwell — visualised as button background brightness change
            self.root.after(0, lambda d=direction, f=fraction: self._draw_topbar_dwell(d, f))
        else:
            self.root.after(0, lambda: self._draw_dwell(direction, fraction))

    def _draw_dwell(self, direction: int, fraction: float) -> None:
        canvas = self._dwell_canvases.get(direction)
        bar    = self._dwell_bars.get(direction)
        if canvas is None or bar is None:
            return
        width = canvas.winfo_width()
        if width <= 1:
            width = 200  # fallback before layout is done
        fill_w = max(0, min(width, int(width * fraction)))
        canvas.coords(bar, 0, 0, fill_w, 8)
        # Colour: cyan -> red as it fills
        if fraction < 0.6:
            color = PALETTE["dwell_bar"]
        elif fraction < 0.9:
            color = "#f39c12"  # orange
        else:
            color = PALETTE["cell_active"]  # red — about to fire
        canvas.itemconfig(bar, fill=color)

    # ── Command execution ──────────────────────────────────────────────────────

    def fire_command(self, direction: int) -> None:
        """Execute the command. Handles zones 1-13 + survey mode. Thread-safe."""
        import time as _time
        if getattr(self, "_is_done", False) and not self._is_in_survey:
            return
        now = _time.time()
        if now - self._last_fire_time < self._fire_cooldown:
            return   # too soon — debounce
        self._last_fire_time = now

        if self._is_in_survey:
            self.root.after(0, lambda: self._survey_fire(direction))
        elif direction in (10, 11, 12):     # prediction slots
            slot = direction - 10
            self.root.after(0, lambda s=slot: self._on_prediction_click(s))
        elif direction == 13:               # Done Typing
            self.root.after(0, self._finish_typing)
        elif direction == 14:              # Phrase Bank toggle
            self.root.after(0, self._open_phrase_bank)
        else:                               # normal grid zone 1-9
            self.root.after(0, lambda: self._execute_command(direction))

    def _execute_command(self, direction: int) -> None:
        # If phrase bank is open, route zones 1-9 into it
        if self._phrase_panel is not None:
            self._phrase_bank_fire(direction)
            return

        # Flash cell
        frame = self._cell_frames.get(direction)
        if frame:
            frame.config(bg=PALETTE["cell_active"])
            self._cell_labels[direction].config(bg=PALETTE["cell_active"])
            self.root.after(400, self._update_cell_highlight)

        # Reset dwell bar
        self.set_dwell_progress(direction, 0.0)

        label = self._current_cells.get(direction, "")

        if self._level == 1:
            if direction == 9:             # DELETE
                self._typed_text = self._typed_text[:-1]
                self._flush_display()
                self._set_status("Deleted last character")
                if self.on_text_change:
                    self.on_text_change(self._typed_text)
            elif direction == 5:           # SPACE
                self._typed_text += " "
                self._flush_display()
                self._speak("Space")
                self._set_status("Space added")
                if self.on_text_change:
                    self.on_text_change(self._typed_text)
            else:                          # letter group
                self._set_level2(direction)
                self._speak(f"Group {label}")
                self._set_status(f"Selected group: {label}  — now pick a letter")
        else:  # level 2
            if direction == 5 or label in ("GO BACK", "<< BACK", "BACK"):
                self._set_level1()
                self._speak("Back")
                self._set_status("Back to main menu")
            elif label:   # valid letter
                self._type_character(label)
                self._set_level1()
            # else: empty cell — do nothing, don't change level

    def _type_character(self, char: str) -> None:
        self._typed_text += char
        self._flush_display()
        self._speak(char)
        self._set_status(f"Typed: {char}")
        if self.on_text_change:
            self.on_text_change(self._typed_text)

    def _flush_display(self) -> None:
        self._typed_display.config(text=self._typed_text + "|")

    def _set_status(self, msg: str) -> None:
        self._status_bar.config(text=msg)

    # ── Prediction slots ───────────────────────────────────────────────────────

    def update_predictions(self, completions: list) -> None:
        self._pred_labels = (completions + ["", "", ""])[:3]
        self.root.after(0, self._refresh_predictions)

    def _refresh_predictions(self) -> None:
        for i, btn in enumerate(self._pred_buttons):
            word = self._pred_labels[i]
            btn.config(
                text=word,
                state="normal" if (word and self.prediction_enabled) else "disabled",
                bg=PALETTE["pred_bg"],
            )

    def _on_prediction_click(self, idx: int) -> None:
        word = self._pred_labels[idx]
        if not word or not self.prediction_enabled:
            return
        parts = self._typed_text.split()
        if self._typed_text and not self._typed_text.endswith(" "):
            parts = parts[:-1]
        parts.append(word)
        self._typed_text = " ".join(parts) + " "
        self._flush_display()
        self._speak(word)
        self._set_status(f"Prediction selected: {word}")
        if self.on_text_change:
            self.on_text_change(self._typed_text)

    def set_prediction_hover(self, slot: int) -> None:
        self.root.after(0, lambda: self._highlight_pred(slot))

    def _highlight_pred(self, slot: int) -> None:
        for i, btn in enumerate(self._pred_buttons):
            btn.config(bg=PALETTE["pred_hover"] if i == slot else PALETTE["pred_bg"])

    def fire_prediction(self, slot: int) -> None:
        self.root.after(0, lambda: self._on_prediction_click(slot))

    def _toggle_predictions(self) -> None:
        self.prediction_enabled = not self.prediction_enabled
        self._toggle_pred_btn.config(
            text="PRED: ON" if self.prediction_enabled else "PRED: OFF"
        )
        self._refresh_predictions()

    # ── Phrase Bank ───────────────────────────────────────────────────────────

    def _open_phrase_bank(self) -> None:
        """Open or close the phrase bank overlay panel."""
        if self._phrase_panel is not None:
            self._close_phrase_bank()
            return
        self._phrase_mode = "category"
        self._build_category_panel()

    def _close_phrase_bank(self) -> None:
        if self._phrase_panel is not None:
            self._phrase_panel.destroy()
            self._phrase_panel = None
        self._phrase_zone_map = {}
        self._phrase_btn.config(text="📋 PHRASES", fg="#f0e68c")

    def _build_category_panel(self) -> None:
        """Show a 3×3 grid of phrase categories to choose from."""
        if self._phrase_panel is not None:
            self._phrase_panel.destroy()

        self._phrase_btn.config(text="✖ CLOSE", fg="#e94560")
        overlay = tk.Frame(self.root, bg="#0a1628", bd=2, relief="solid")
        overlay.place(relx=0, rely=0.08, relwidth=1, relheight=0.87)
        self._phrase_panel = overlay
        self._phrase_zone_map = {}

        tk.Label(overlay, text="📋  PHRASE BANK — Select a category",
                 fg="#f0e68c", bg="#0a1628",
                 font=self._font_hdr).pack(pady=(10, 6))

        grid = tk.Frame(overlay, bg="#0a1628")
        grid.pack(fill="both", expand=True, padx=10, pady=(0, 10))
        cats = list(self._phrase_categories.keys())

        # Fill up to 9 slots (3×3)
        COLS = 3
        for col in range(COLS):
            grid.columnconfigure(col, weight=1)
        rows_needed = max(3, -(-len(cats) // COLS))  # ceiling division
        for r in range(rows_needed):
            grid.rowconfigure(r, weight=1)

        zone_order = [d for d in range(1, 10)]
        for i, zone in enumerate(zone_order):
            row, col = divmod(i, COLS)
            if i < len(cats):
                cat_name = cats[i]
                self._phrase_zone_map[zone] = ("category", cat_name)
                cell_bg = PALETTE["cell_idle"]
                cell_txt = cat_name
                sub_txt  = f"{len(self._phrase_categories[cat_name])} phrases"
            else:
                cell_bg = "#0d0d0d"
                cell_txt = ""
                sub_txt  = ""

            cell = tk.Frame(grid, bg=cell_bg, bd=1, relief="flat")
            cell.grid(row=row, column=col, padx=6, pady=6, sticky="nsew")
            tk.Label(cell, text=str(zone), fg=PALETTE["text_dim"],
                     bg=cell_bg, font=self._font_small).place(x=4, y=2)
            tk.Label(cell, text=cell_txt,
                     fg="#f0e68c", bg=cell_bg,
                     font=self._font_mono).place(relx=0.5, rely=0.42, anchor="center")
            tk.Label(cell, text=sub_txt,
                     fg=PALETTE["text_dim"], bg=cell_bg,
                     font=self._font_small).place(relx=0.5, rely=0.72, anchor="center")

            if i < len(cats):
                # Click support (mouse mode)
                cat_name_cap = cats[i]
                cell.bind("<Button-1>", lambda e, c=cat_name_cap: self._open_category(c))
                for w in cell.winfo_children():
                    w.bind("<Button-1>", lambda e, c=cat_name_cap: self._open_category(c))

    def _open_category(self, category: str) -> None:
        """Show the 9 phrases inside a category."""
        if self._phrase_panel is not None:
            self._phrase_panel.destroy()

        self._phrase_mode = "phrases"
        self._phrase_active_cat = category
        phrases = self._phrase_categories.get(category, [])

        overlay = tk.Frame(self.root, bg="#0a1628", bd=2, relief="solid")
        overlay.place(relx=0, rely=0.08, relwidth=1, relheight=0.87)
        self._phrase_panel = overlay
        self._phrase_zone_map = {}

        tk.Label(overlay, text=f"📋  {category}  — Gaze to select a phrase",
                 fg="#f0e68c", bg="#0a1628",
                 font=self._font_hdr).pack(pady=(10, 6))

        grid = tk.Frame(overlay, bg="#0a1628")
        grid.pack(fill="both", expand=True, padx=10, pady=0)
        COLS = 3
        for col in range(COLS):
            grid.columnconfigure(col, weight=1)
        for r in range(3):
            grid.rowconfigure(r, weight=1)

        zone_order = [d for d in range(1, 10)]
        for i, zone in enumerate(zone_order):
            row, col = divmod(i, COLS)

            if zone == 5:
                # Centre = GO BACK to categories
                cell_bg  = PALETTE["cell_hover"]
                cell_txt = "⬅  BACK"
                sub_txt  = "categories"
                self._phrase_zone_map[zone] = ("back", "")
            elif i < len(phrases) + (1 if i >= 4 else 0):
                # Adjust index to skip zone 5 slot
                pidx = i if i < 4 else i - 1
                if pidx < len(phrases):
                    phrase = phrases[pidx]
                    self._phrase_zone_map[zone] = ("phrase", phrase)
                    cell_bg  = PALETTE["cell_idle"]
                    cell_txt = phrase
                    sub_txt  = "👁 gaze to select"
                else:
                    cell_bg  = "#0d0d0d"
                    cell_txt = ""
                    sub_txt  = ""
            else:
                cell_bg  = "#0d0d0d"
                cell_txt = ""
                sub_txt  = ""

            cell = tk.Frame(grid, bg=cell_bg, bd=1, relief="flat")
            cell.grid(row=row, column=col, padx=6, pady=6, sticky="nsew")
            tk.Label(cell, text=str(zone), fg=PALETTE["text_dim"],
                     bg=cell_bg, font=self._font_small).place(x=4, y=2)
            lbl = tk.Label(cell, text=cell_txt,
                           fg="#ffffff", bg=cell_bg,
                           font=self._font_hdr,
                           wraplength=200, justify="center")
            lbl.place(relx=0.5, rely=0.42, anchor="center")
            tk.Label(cell, text=sub_txt,
                     fg=PALETTE["text_dim"], bg=cell_bg,
                     font=self._font_small).place(relx=0.5, rely=0.78, anchor="center")

            # Click support (mouse mode)
            kind, val = self._phrase_zone_map.get(zone, (None, None))
            if kind == "phrase":
                cell.bind("<Button-1>", lambda e, p=val: self._select_phrase(p))
                for w in cell.winfo_children():
                    w.bind("<Button-1>", lambda e, p=val: self._select_phrase(p))
            elif kind == "back":
                cell.bind("<Button-1>", lambda e: self._build_category_panel())
                for w in cell.winfo_children():
                    w.bind("<Button-1>", lambda e: self._build_category_panel())

        tk.Button(overlay, text="✖ Close Phrase Bank",
                  fg="#e94560", bg="#0a1628",
                  relief="flat", font=self._font_small,
                  command=self._close_phrase_bank).pack(side="bottom", pady=6)

    def _phrase_bank_fire(self, zone: int) -> None:
        """Called by fire_command when phrase panel is open."""
        kind_val = self._phrase_zone_map.get(zone)
        if kind_val is None:
            return
        kind, val = kind_val
        if kind == "phrase":
            self.root.after(0, lambda: self._select_phrase(val))
        elif kind == "category":
            self.root.after(0, lambda: self._open_category(val))
        elif kind == "back":
            self.root.after(0, self._build_category_panel)

    def _select_phrase(self, phrase: str) -> None:
        """Insert a full phrase into the typed text and close the bank."""
        if self._typed_text and not self._typed_text.endswith(" "):
            self._typed_text += " "
        self._typed_text += phrase + " "
        self._flush_display()
        self._speak(phrase)
        self._set_status(f"Phrase: {phrase}")
        if self.on_text_change:
            self.on_text_change(self._typed_text)
        self._close_phrase_bank()

    # ── TTS ───────────────────────────────────────────────────────────────────

    def _speak(self, text: str) -> None:
        if self._tts and self.tts_enabled:
            try:
                self._tts.say(text)
                threading.Thread(target=self._tts.runAndWait, daemon=True).start()
            except Exception:
                pass

    def _finish_typing(self) -> None:
        if self._is_done:
            return
        self._is_done = True
        self._speak("Typing finished. Please answer a short survey.")
        # Launch the survey — final screen shown after it completes
        self.show_survey(
            user_id   = self._survey_user_id,
            typed_text= self._typed_text,
            mode      = self._survey_mode,
            on_complete = self._show_final_screen,
        )

    def _show_final_screen(self) -> None:
        """Show the typed text after the survey is done."""
        for widget in self.root.winfo_children():
            widget.pack_forget()
        final_frame = tk.Frame(self.root, bg=PALETTE["bg"])
        final_frame.pack(fill="both", expand=True, padx=20, pady=20)
        tk.Label(final_frame, text="\u2705 FINAL TEXT",
                 fg="#27ae60", bg=PALETTE["bg"],
                 font=self._font_big).pack(pady=(50, 20))
        tk.Label(final_frame, text=self._typed_text,
                 fg="#ffffff", bg=PALETTE["bg"],
                 font=self._font_big,
                 wraplength=900, justify="center").pack(pady=40, expand=True)
        tk.Button(final_frame, text="EXIT APP",
                  font=self._font_hdr, bg="#e94560", fg="#fff",
                  command=self.root.quit, padx=20, pady=10).pack(pady=40)

    # ── Survey overlay ─────────────────────────────────────────────────────────

    # Survey questions definition
    _SURVEY_QUESTIONS = [
        {
            "key":      "eye_strain",
            "question": "How much eye strain did you feel?",
            "zones":    {
                1: ("None",     1),
                3: ("Mild",     2),
                7: ("High",     3),
                9: ("Severe",   4),
            },
        },
        {
            "key":      "typed_intended",
            "question": "Could you type what you intended?",
            "zones":    {
                1: ("Yes",       "Yes"),
                5: ("Partially", "Partially"),
                9: ("No",        "No"),
            },
        },
        {
            "key":      "overall_experience",
            "question": "How would you rate the overall experience?",
            "zones":    {
                1: ("Great \U0001f604",  "Great"),
                5: ("OK \U0001f610",     "OK"),
                9: ("Poor \U0001f615",   "Poor"),
            },
        },
    ]

    def show_survey(
        self,
        user_id: str,
        typed_text: str,
        mode: str,
        on_complete,
    ) -> None:
        """Show the gaze-navigable post-typing survey overlay."""
        self._survey_user_id  = user_id
        self._survey_typed    = typed_text
        self._survey_mode     = mode
        self._survey_callback = on_complete
        self._survey_q_idx    = 0
        self._survey_answers  = {}
        self._is_in_survey    = True
        self._session_end_time = time.time()
        self.root.after(0, self._build_survey_overlay)

    def _build_survey_overlay(self) -> None:
        """Build or rebuild the survey overlay for the current question."""
        if self._survey_frame is not None:
            self._survey_frame.destroy()

        q_data = self._SURVEY_QUESTIONS[self._survey_q_idx]
        n_q    = len(self._SURVEY_QUESTIONS)
        q_num  = self._survey_q_idx + 1

        overlay = tk.Frame(self.root, bg="#0d0d1a")
        overlay.place(relx=0, rely=0, relwidth=1, relheight=1)
        self._survey_frame   = overlay
        self._survey_zone_map = {}  # zone -> value

        # Header
        tk.Label(overlay,
                 text=f"Quick Survey  ({q_num}/{n_q})",
                 fg=PALETTE["text_dim"], bg="#0d0d1a",
                 font=self._font_hdr).pack(pady=(20, 4))

        tk.Label(overlay,
                 text=q_data["question"],
                 fg="#ffffff", bg="#0d0d1a",
                 font=self._font_big,
                 wraplength=900).pack(pady=(0, 30))

        # Grid of answer buttons
        btn_frame = tk.Frame(overlay, bg="#0d0d1a")
        btn_frame.pack(fill="both", expand=True, padx=30, pady=0)
        for col in range(3):
            btn_frame.columnconfigure(col, weight=1)
        btn_frame.rowconfigure(0, weight=1)

        active_zones = sorted(q_data["zones"].keys())
        for idx, zone in enumerate(active_zones):
            label_txt, value = q_data["zones"][zone]
            self._survey_zone_map[zone] = value
            col = idx % 3

            cell = tk.Frame(btn_frame, bg=PALETTE["cell_idle"], bd=2, relief="flat")
            cell.grid(row=0, column=col, padx=10, pady=10, sticky="nsew")

            # Zone number hint
            tk.Label(cell, text=f"zone {zone}",
                     fg=PALETTE["text_dim"], bg=PALETTE["cell_idle"],
                     font=self._font_small).place(x=5, y=3)

            # Answer label
            tk.Label(cell, text=label_txt,
                     fg=PALETTE["text_primary"], bg=PALETTE["cell_idle"],
                     font=self._font_big).place(relx=0.5, rely=0.45, anchor="center")

            # Dwell progress bar at bottom of each survey cell
            dwell_cv = tk.Canvas(cell, height=10, bg=PALETTE["dwell_track"],
                                  highlightthickness=0)
            dwell_cv.place(relx=0, rely=1.0, anchor="sw", relwidth=1.0)
            bar = dwell_cv.create_rectangle(0, 0, 0, 10,
                                             fill=PALETTE["dwell_bar"],
                                             outline="")
            # Store so set_dwell_progress can update them
            self._cell_frames[zone]    = cell
            self._dwell_canvases[zone] = dwell_cv
            self._dwell_bars[zone]     = bar

        # Instruction footer
        tk.Label(overlay,
                 text="Dwell on a zone (or blink in scan mode) to answer",
                 fg=PALETTE["text_dim"], bg="#0d0d1a",
                 font=self._font_small).pack(side="bottom", pady=10)

    def _survey_fire(self, zone: int) -> None:
        """Handle a zone selection during the survey."""
        q_data = self._SURVEY_QUESTIONS[self._survey_q_idx]
        if zone not in self._survey_zone_map:
            return  # zone not active for this question — ignore

        value = self._survey_zone_map[zone]
        self._survey_answers[q_data["key"]] = value
        self._survey_q_idx += 1

        if self._survey_q_idx < len(self._SURVEY_QUESTIONS):
            # Next question
            self.root.after(200, self._build_survey_overlay)
        else:
            # All questions answered — save and finish
            self._is_in_survey = False
            if self._survey_frame:
                self._survey_frame.destroy()
                self._survey_frame = None
            self._save_survey()
            if self._survey_callback:
                self._survey_callback()

    def _save_survey(self) -> None:
        try:
            import sys
            from pathlib import Path
            sys.path.insert(0, str(Path(__file__).parent))
            from survey import save_response
            session_sec = time.time() - self._session_start
            save_response(
                user_id    = self._survey_user_id,
                typed_text = getattr(self, "_survey_typed", self._typed_text),
                session_sec= session_sec,
                mode       = self._survey_mode,
                responses  = self._survey_answers,
            )
        except Exception as e:
            print(f"[Survey] Failed to save: {e}")

    # ── Public helpers ────────────────────────────────────────────────────────

    def set_text(self, text: str) -> None:
        self._typed_text = text
        self.root.after(0, self._flush_display)

    def get_text(self) -> str:
        return self._typed_text

    # ── Camera preview (called from gaze thread, renders on main thread) ──────

    def update_camera_preview(self, bgr_frame, eyes_detected: bool) -> None:
        """Thread-safe. Pass the annotated BGR numpy frame every N frames."""
        self.root.after(0, lambda: self._draw_camera_frame(bgr_frame, eyes_detected))

    def _draw_camera_frame(self, bgr_frame, eyes_detected: bool) -> None:
        import numpy as _np
        try:
            h, w = bgr_frame.shape[:2]
            scale = min(220 / w, 150 / h)
            nw, nh = int(w * scale), int(h * scale)
            rgb = bgr_frame[:, :, ::-1]   # BGR -> RGB

            if _PIL_AVAILABLE:
                from PIL import Image as _Img, ImageTk as _ITk
                img   = _Img.fromarray(rgb).resize((nw, nh), _Img.BILINEAR)
                photo = _ITk.PhotoImage(img)
            else:
                # pyrefly: ignore [missing-import]
                import cv2 as _cv2
                resized = _cv2.resize(rgb, (nw, nh))
                header  = f"P6\n{nw} {nh}\n255\n".encode()
                photo   = tk.PhotoImage(data=header + resized.tobytes())

            self._cam_photo = photo   # prevent GC
            self._cam_canvas.delete("all")
            self._cam_canvas.create_image(110, 75, image=photo, anchor="center")

            border = "#27ae60" if eyes_detected else "#e74c3c"
            self._cam_canvas.create_rectangle(0, 0, 219, 149,
                                              outline=border, width=3)
            label  = "Eyes OK" if eyes_detected else "NO EYES - move closer"
            lcolor = "#2ecc71" if eyes_detected else "#e74c3c"
            self._cam_canvas.create_text(110, 142, text=label,
                                         fill=lcolor, font=self._font_small)
        except Exception:
            pass

    def run(self) -> None:
        self.root.mainloop()



# ── Standalone demo ────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("Keyboard GUI demo — click prediction buttons to test")

    def on_change(text: str):
        print(f"Text: {text!r}")

    gui = KeyboardGUI(on_text_change=on_change, tts_enabled=False)
    gui.update_predictions(["hello", "world", "python"])
    gui.run()
