import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
from matplotlib.widgets import RectangleSelector
from scipy.optimize import curve_fit
import os
import platform
import threading
import math

# 自作モジュールのインポート
import config as cfg
from src import preprocessing as prep
from src import calculation as calc
from src import model

class RICSApp:
    def __init__(self, root):
        self.root = root
        self.root.title("RICS Analysis App v17.1 (Fix: Auto-Detect on Refresh)")
        self.root.geometry("1400x1000")

        # データ保持用
        self.raw_stack = None       
        self.processed_full = None
        self.roi_data = None
        self.acf_data = None
        
        # 表示用設定
        self.current_frame_idx = -1
        self.total_frames = 0
        self.fixed_ylim_x = None
        self.fixed_ylim_y = None
        
        # ヒートマップ結果保持用
        self.heatmap_d_map = None
        self.hm_window = None 
        
        # スレッド制御用
        self.heatmap_thread = None
        self.stop_event = threading.Event()
        self.progress_val = tk.DoubleVar(value=0.0)
        
        # リアルタイムプロット用
        self.live_fit_data = None 
        self.live_fit_lock = threading.Lock()

        # --- GUI変数 ---
        # Scan Params
        self.pixel_size_var = tk.DoubleVar(value=getattr(cfg, 'PIXEL_SIZE', 0.05) * 1000.0) 
        self.pixel_dwell_var = tk.DoubleVar(value=getattr(cfg, 'PIXEL_DWELL_TIME', 10e-6) * 1e6)
        self.line_time_var = tk.DoubleVar(value=getattr(cfg, 'LINE_TIME', 2e-3) * 1000.0) 

        self.ma_window_var = tk.IntVar(value=cfg.MOVING_AVG_WINDOW)
        self.roi_w_var = tk.IntVar(value=cfg.ROI_SIZE)
        self.roi_h_var = tk.IntVar(value=cfg.ROI_SIZE)
        self.roi_cx_var = tk.IntVar(value=128)
        self.roi_cy_var = tk.IntVar(value=128)
        
        self.omit_radius_var = tk.DoubleVar(value=0.0)
        self.fit_range_x_var = tk.IntVar(value=cfg.ROI_SIZE // 2)
        self.fit_range_y_var = tk.IntVar(value=cfg.ROI_SIZE // 2)
        self.auto_range_var = tk.BooleanVar(value=False)

        self.hm_window_var = tk.IntVar(value=32)
        self.hm_step_var = tk.IntVar(value=4)
        self.hm_autoscale_var = tk.BooleanVar(value=True)
        self.hm_percentile_var = tk.DoubleVar(value=95.0)
        self.hm_max_val_var = tk.DoubleVar(value=100.0)
        self.hm_interp_var = tk.StringVar(value="nearest")

        self.n_var = tk.StringVar(value="---")
        self.result_text = tk.StringVar(value="Ready...")
        self.heatmap_status = tk.StringVar(value="Idle")
        
        self.frame_info_var = tk.StringVar(value="No Data")
        self.drag_lines = {}
        self.dragging_item = None
        self.selector = None
        self.press_xy = None

        self.create_layout()
        self.setup_plots()

    def create_layout(self):
        main_frame = ttk.Frame(self.root)
        main_frame.pack(fill=tk.BOTH, expand=True)

        self.canvas = tk.Canvas(main_frame, width=400)
        scrollbar = ttk.Scrollbar(main_frame, orient="vertical", command=self.canvas.yview)
        
        self.scroll_inner = ttk.Frame(self.canvas, padding="10")
        self.canvas_window = self.canvas.create_window((0, 0), window=self.scroll_inner, anchor="nw")
        
        self.scroll_inner.bind("<Configure>", self._on_frame_configure)
        self.canvas.configure(yscrollcommand=scrollbar.set)
        
        scrollbar.pack(side=tk.LEFT, fill=tk.Y)
        self.canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=False)
        
        self.canvas.bind("<Enter>", self._bind_mouse_scroll)
        self.canvas.bind("<Leave>", self._unbind_mouse_scroll)

        self.graph_frame = ttk.Frame(main_frame)
        self.graph_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)

        self.create_widgets(self.scroll_inner)

    def _on_frame_configure(self, event):
        self.canvas.configure(scrollregion=self.canvas.bbox("all"))

    def _bind_mouse_scroll(self, event):
        system = platform.system()
        if system == "Darwin":
            self.canvas.bind_all("<MouseWheel>", self._on_mousewheel_mac)
        else:
            self.canvas.bind_all("<MouseWheel>", self._on_mousewheel_win)

    def _unbind_mouse_scroll(self, event):
        self.canvas.unbind_all("<MouseWheel>")

    def _on_mousewheel_mac(self, event):
        delta = event.delta
        if delta == 0: return
        self.canvas.yview_scroll(int(-1 * delta), "units")

    def _on_mousewheel_win(self, event):
        self.canvas.yview_scroll(int(-1*(event.delta/120)), "units")

    def create_widgets(self, parent):
        # 1. Data Loading
        ttk.Label(parent, text="1. Data Loading", font=("Arial", 12, "bold")).pack(pady=5, anchor="w")
        ttk.Button(parent, text="Load TIF Data", command=self.load_data).pack(fill=tk.X, pady=5)
        self.file_label = ttk.Label(parent, text="No file loaded", wraplength=350)
        self.file_label.pack()
        
        f_info = ttk.LabelFrame(parent, text="Frame Viewer")
        f_info.pack(fill=tk.X, pady=5)
        ttk.Label(f_info, textvariable=self.frame_info_var, foreground="blue").pack(anchor="w")
        
        self.frame_slider = tk.Scale(f_info, from_=0, to=1, orient=tk.HORIZONTAL, label="Frame Index", command=self.on_slider_change)
        self.frame_slider.pack(fill=tk.X, padx=5)
        self.frame_slider.configure(state="disabled")
        
        ttk.Button(f_info, text="Show Average Image", command=self.reset_to_average).pack(fill=tk.X, pady=2)

        ttk.Separator(parent, orient="horizontal").pack(fill=tk.X, pady=10)

        # 2. Scan Parameters Input
        ttk.Label(parent, text="2. Scan Parameters", font=("Arial", 12, "bold")).pack(pady=5, anchor="w")
        scan_grp = ttk.LabelFrame(parent, text="Microscope Settings")
        scan_grp.pack(fill=tk.X, pady=5)
        
        row_ps = ttk.Frame(scan_grp); row_ps.pack(fill=tk.X, pady=2)
        ttk.Label(row_ps, text="Pixel Size:").pack(side=tk.LEFT)
        ttk.Entry(row_ps, textvariable=self.pixel_size_var, width=8).pack(side=tk.LEFT, padx=5)
        ttk.Label(row_ps, text="nm").pack(side=tk.LEFT)
        
        row_pd = ttk.Frame(scan_grp); row_pd.pack(fill=tk.X, pady=2)
        ttk.Label(row_pd, text="Pixel Dwell:").pack(side=tk.LEFT)
        ttk.Entry(row_pd, textvariable=self.pixel_dwell_var, width=8).pack(side=tk.LEFT, padx=5)
        ttk.Label(row_pd, text="us").pack(side=tk.LEFT)
        
        row_lt = ttk.Frame(scan_grp); row_lt.pack(fill=tk.X, pady=2)
        ttk.Label(row_lt, text="Line Time:").pack(side=tk.LEFT)
        ttk.Entry(row_lt, textvariable=self.line_time_var, width=8).pack(side=tk.LEFT, padx=5)
        ttk.Label(row_lt, text="ms").pack(side=tk.LEFT)

        ttk.Separator(parent, orient="horizontal").pack(fill=tk.X, pady=10)

        # 3. ROI & Preprocessing
        ttk.Label(parent, text="3. ROI & Preprocessing", font=("Arial", 12, "bold")).pack(pady=5, anchor="w")
        
        bg_frame = ttk.Frame(parent); bg_frame.pack(fill=tk.X, pady=2)
        ttk.Label(bg_frame, text="Mov.Avg:").pack(side=tk.LEFT)
        ttk.Entry(bg_frame, textvariable=self.ma_window_var, width=5).pack(side=tk.LEFT, padx=5)

        roi_grp = ttk.LabelFrame(parent, text="ROI Config")
        roi_grp.pack(fill=tk.X, pady=5)
        
        ttk.Button(roi_grp, text="Select Full Image", command=self.set_full_roi).pack(fill=tk.X, pady=2)
        
        f_sz = ttk.Frame(roi_grp); f_sz.pack(fill=tk.X, pady=2)
        ttk.Label(f_sz, text="Size W:").pack(side=tk.LEFT)
        ttk.Entry(f_sz, textvariable=self.roi_w_var, width=5).pack(side=tk.LEFT)
        ttk.Label(f_sz, text=" H:").pack(side=tk.LEFT)
        ttk.Entry(f_sz, textvariable=self.roi_h_var, width=5).pack(side=tk.LEFT)

        f_pos = ttk.Frame(roi_grp); f_pos.pack(fill=tk.X, pady=2)
        ttk.Label(f_pos, text="Center X:").pack(side=tk.LEFT)
        ttk.Entry(f_pos, textvariable=self.roi_cx_var, width=5).pack(side=tk.LEFT)
        ttk.Label(f_pos, text=" Y:").pack(side=tk.LEFT)
        ttk.Entry(f_pos, textvariable=self.roi_cy_var, width=5).pack(side=tk.LEFT)

        ttk.Button(parent, text="Update Image & ACF", command=self.update_processing_and_acf).pack(fill=tk.X, pady=5)

        ttk.Separator(parent, orient="horizontal").pack(fill=tk.X, pady=10)

        # 4. Fitting Range & Omit
        ttk.Label(parent, text="4. Range & Omit", font=("Arial", 12, "bold")).pack(pady=5, anchor="w")

        omit_frame = ttk.Frame(parent); omit_frame.pack(fill=tk.X, pady=2)
        ttk.Label(omit_frame, text="Omit Radius:", foreground="red").pack(side=tk.LEFT)
        ttk.Entry(omit_frame, textvariable=self.omit_radius_var, width=5).pack(side=tk.LEFT, padx=5)
        ttk.Label(omit_frame, text="px").pack(side=tk.LEFT)

        range_frame = ttk.LabelFrame(parent, text="Fitting Range")
        range_frame.pack(fill=tk.X, pady=5)
        
        ttk.Checkbutton(range_frame, text="Auto-Detect Fit Range (Monotonic)", variable=self.auto_range_var).pack(anchor="w", padx=5, pady=2)
        
        fr_x = ttk.Frame(range_frame); fr_x.pack(fill=tk.X)
        ttk.Label(fr_x, text="Manual X (+/-):").pack(side=tk.LEFT)
        self.fit_range_x_var.trace_add("write", lambda *args: self.update_lines_from_entry())
        ttk.Entry(fr_x, textvariable=self.fit_range_x_var, width=5).pack(side=tk.LEFT, padx=5)
        
        fr_y = ttk.Frame(range_frame); fr_y.pack(fill=tk.X)
        ttk.Label(fr_y, text="Manual Y (+/-):").pack(side=tk.LEFT)
        self.fit_range_y_var.trace_add("write", lambda *args: self.update_lines_from_entry())
        ttk.Entry(fr_y, textvariable=self.fit_range_y_var, width=5).pack(side=tk.LEFT, padx=5)

        ttk.Button(parent, text="Refresh Plots", command=lambda: self.plot_results(None)).pack(fill=tk.X, pady=2)

        # Parameters
        self.params = {
            "D":  {"label": "D (um²/s)",   "val": 10.0},
            "G0": {"label": "G0 (Amp)",    "val": 0.01},
            "w0": {"label": "w0 (um)",     "val": cfg.W0},
            "wz": {"label": "wz (um)",     "val": cfg.WZ},
        }
        self.entries = {}
        self.checkvars = {}

        for key, info in self.params.items():
            row = ttk.Frame(parent); row.pack(fill=tk.X, pady=2)
            ttk.Label(row, text=info["label"], width=12).pack(side=tk.LEFT)
            ev = tk.DoubleVar(value=info["val"])
            ttk.Entry(row, textvariable=ev, width=8).pack(side=tk.LEFT, padx=5)
            self.entries[key] = ev
            cv = tk.BooleanVar(value=True if key in ["w0", "wz"] else False)
            ttk.Checkbutton(row, text="Fix", variable=cv).pack(side=tk.LEFT)
            self.checkvars[key] = cv

        n_frame = ttk.Frame(parent); n_frame.pack(fill=tk.X, pady=5)
        ttk.Label(n_frame, text="Calc. N:", foreground="blue").pack(side=tk.LEFT)
        ttk.Label(n_frame, textvariable=self.n_var, foreground="blue", font=("bold")).pack(side=tk.LEFT, padx=5)

        ttk.Separator(parent, orient="horizontal").pack(fill=tk.X, pady=10)

        # 5. Fitting
        ttk.Label(parent, text="5. Single Point Fitting", font=("Arial", 12, "bold")).pack(pady=5, anchor="w")
        ttk.Button(parent, text="Run Fitting (Selected ROI)", command=self.run_fitting).pack(fill=tk.X, pady=5)
        ttk.Label(parent, textvariable=self.result_text, relief="sunken", padding=5).pack(fill=tk.X)

        ttk.Separator(parent, orient="horizontal").pack(fill=tk.X, pady=10)

        # 6. Output
        ttk.Label(parent, text="6. Output", font=("Arial", 12, "bold")).pack(pady=5, anchor="w")
        ttk.Button(parent, text="Save Graphs as JPEG", command=self.save_graphs).pack(fill=tk.X, pady=5)

        ttk.Separator(parent, orient="horizontal").pack(fill=tk.X, pady=10)

        # 7. Heatmap
        ttk.Label(parent, text="7. Heatmap Analysis", font=("Arial", 12, "bold")).pack(pady=5, anchor="w")
        
        hm_conf = ttk.Frame(parent)
        hm_conf.pack(fill=tk.X, pady=2)
        ttk.Label(hm_conf, text="Win Size:").pack(side=tk.LEFT)
        ttk.Entry(hm_conf, textvariable=self.hm_window_var, width=5).pack(side=tk.LEFT, padx=5)
        ttk.Label(hm_conf, text="Step (px):").pack(side=tk.LEFT)
        ttk.Entry(hm_conf, textvariable=self.hm_step_var, width=5).pack(side=tk.LEFT, padx=5)
        
        vis_grp = ttk.LabelFrame(parent, text="Display Settings")
        vis_grp.pack(fill=tk.X, pady=5)
        
        v_row1 = ttk.Frame(vis_grp); v_row1.pack(fill=tk.X, pady=2)
        ttk.Checkbutton(v_row1, text="Scale by Percentile", variable=self.hm_autoscale_var).pack(side=tk.LEFT)
        ttk.Entry(v_row1, textvariable=self.hm_percentile_var, width=5).pack(side=tk.LEFT, padx=5)
        ttk.Label(v_row1, text="%").pack(side=tk.LEFT)
        
        v_row2 = ttk.Frame(vis_grp); v_row2.pack(fill=tk.X, pady=2)
        ttk.Label(v_row2, text="or Manual Max D (um^2/s):").pack(side=tk.LEFT)
        ttk.Entry(v_row2, textvariable=self.hm_max_val_var, width=6).pack(side=tk.LEFT, padx=5)
        
        v_row3 = ttk.Frame(vis_grp); v_row3.pack(fill=tk.X, pady=2)
        ttk.Label(v_row3, text="Interpolation:").pack(side=tk.LEFT)
        ttk.Combobox(v_row3, textvariable=self.hm_interp_var, values=["nearest", "bicubic", "bilinear"], width=10, state="readonly").pack(side=tk.LEFT, padx=5)
        
        ttk.Button(vis_grp, text="Re-draw Map Only", command=self.plot_heatmap_result).pack(fill=tk.X, pady=2)

        self.progress_bar = ttk.Progressbar(parent, variable=self.progress_val, maximum=100)
        self.progress_bar.pack(fill=tk.X, pady=5)
        ttk.Label(parent, textvariable=self.heatmap_status, font=("Arial", 9)).pack(anchor="w")

        hm_btns = ttk.Frame(parent)
        hm_btns.pack(fill=tk.X, pady=5)
        ttk.Button(hm_btns, text="Generate Heatmap (Live Plot)", command=self.start_heatmap_thread).pack(side=tk.LEFT, fill=tk.X, expand=True, padx=2)
        ttk.Button(hm_btns, text="Stop", command=self.stop_heatmap).pack(side=tk.LEFT, padx=2)
        ttk.Button(parent, text="Save Heatmap Image", command=self.save_heatmap_image).pack(fill=tk.X, pady=5)


    def setup_plots(self):
        self.fig = plt.Figure(figsize=(10, 8), dpi=100)
        gs = self.fig.add_gridspec(2, 2)
        self.ax_img = self.fig.add_subplot(gs[0, 0]) 
        self.ax_3d  = self.fig.add_subplot(gs[0, 1], projection='3d')
        self.ax_x   = self.fig.add_subplot(gs[1, 0])
        self.ax_y   = self.fig.add_subplot(gs[1, 1])
        
        self.fig.tight_layout(pad=3.0)
        self.canvas_fig = FigureCanvasTkAgg(self.fig, master=self.graph_frame)
        self.canvas_fig.get_tk_widget().pack(fill=tk.BOTH, expand=True)

        self.canvas_fig.mpl_connect('button_press_event', self.on_click)
        self.canvas_fig.mpl_connect('motion_notify_event', self.on_motion)
        self.canvas_fig.mpl_connect('button_release_event', self.on_release)
        
        self.selector = RectangleSelector(
            self.ax_img, self.on_select_roi, useblit=True, button=[1], 
            minspanx=5, minspany=5, spancoords='pixels', interactive=True,
            props=dict(facecolor='lime', edgecolor='lime', alpha=0.2, fill=True)
        )
        self.selector.set_active(False)

    # --- Mouse Events (省略なし) ---
    def on_click(self, event):
        if event.inaxes in [self.ax_x, self.ax_y]:
            if self.drag_lines:
                click_x = event.xdata
                if click_x is not None:
                    min_dist = float('inf')
                    target = None
                    active_lines = []
                    if event.inaxes == self.ax_x:
                        active_lines = [('x_min', self.drag_lines.get('x_min')), ('x_max', self.drag_lines.get('x_max'))]
                    elif event.inaxes == self.ax_y:
                        active_lines = [('y_min', self.drag_lines.get('y_min')), ('y_max', self.drag_lines.get('y_max'))]
                    for name, line in active_lines:
                        if line is None: continue
                        line_x = line.get_xdata()[0]
                        dist = abs(click_x - line_x)
                        if dist < min_dist:
                            min_dist = dist
                            target = name
                    if min_dist < (self.fit_range_x_var.get() * 0.2 + 1.0):
                        self.dragging_item = target
            return
        if event.inaxes == self.ax_img and event.xdata is not None:
            self.press_xy = (event.xdata, event.ydata)

    def on_motion(self, event):
        if self.dragging_item is None: return
        if event.xdata is None: return
        new_x = int(round(event.xdata))
        if 'x_' in self.dragging_item: self.fit_range_x_var.set(abs(new_x))
        elif 'y_' in self.dragging_item: self.fit_range_y_var.set(abs(new_x))

    def on_release(self, event):
        if self.dragging_item:
            self.dragging_item = None
            return
        if event.inaxes == self.ax_img and self.press_xy:
            release_xy = (event.xdata, event.ydata)
            if release_xy[0] is None: return
            dist = math.sqrt((release_xy[0] - self.press_xy[0])**2 + (release_xy[1] - self.press_xy[1])**2)
            self.press_xy = None
            if dist < 5: 
                new_cx = int(release_xy[0])
                new_cy = int(release_xy[1])
                self.roi_cx_var.set(new_cx)
                self.roi_cy_var.set(new_cy)
                self.update_processing_and_acf()

    def on_select_roi(self, eclick, erelease):
        x1, y1 = int(eclick.xdata), int(eclick.ydata)
        x2, y2 = int(erelease.xdata), int(erelease.ydata)
        xmin, xmax = sorted([x1, x2])
        ymin, ymax = sorted([y1, y2])
        w = xmax - xmin
        h = ymax - ymin
        cx = xmin + w // 2
        cy = ymin + h // 2
        if w < 2 or h < 2: return
        self.roi_w_var.set(w)
        self.roi_h_var.set(h)
        self.roi_cx_var.set(cx)
        self.roi_cy_var.set(cy)
        self.update_processing_and_acf()

    def update_lines_from_entry(self):
        if not self.drag_lines: return
        try:
            rx = self.fit_range_x_var.get()
            ry = self.fit_range_y_var.get()
            if 'x_min' in self.drag_lines: self.drag_lines['x_min'].set_xdata([-rx, -rx])
            if 'x_max' in self.drag_lines: self.drag_lines['x_max'].set_xdata([rx, rx])
            if 'y_min' in self.drag_lines: self.drag_lines['y_min'].set_xdata([-ry, -ry])
            if 'y_max' in self.drag_lines: self.drag_lines['y_max'].set_xdata([ry, ry])
            self.canvas_fig.draw_idle()
        except: pass

    # =======================================================
    # Helper: Auto Range Detection
    # =======================================================
    def detect_monotonic_decay_range(self, data_1d, min_len=3):
        n = len(data_1d)
        if n < min_len + 2: return n
        smooth = np.convolve(data_1d, np.ones(3)/3, mode='valid')
        diff = np.diff(smooth)
        idx = np.where(diff > 0)[0]
        if len(idx) > 0:
            return idx[0] + 1
        return n

    # =======================================================
    # Full ROI & Frame Logic
    # =======================================================
    def set_full_roi(self):
        if self.raw_stack is None: return
        _, H, W = self.raw_stack.shape
        self.roi_w_var.set(W)
        self.roi_h_var.set(H)
        self.roi_cx_var.set(W // 2)
        self.roi_cy_var.set(H // 2)
        self.update_processing_and_acf()

    def on_slider_change(self, val):
        self.current_frame_idx = int(val)
        self.update_processing_and_acf()

    def reset_to_average(self):
        self.current_frame_idx = -1
        self.update_processing_and_acf()

    # =======================================================
    # Heatmap Logic (Live Plot Added)
    # =======================================================
    def start_heatmap_thread(self):
        if self.processed_full is None:
            messagebox.showwarning("Warning", "Please load data first.")
            return
        if self.heatmap_thread is not None and self.heatmap_thread.is_alive():
            return 

        win_size = self.hm_window_var.get()
        step = self.hm_step_var.get()
        if step < 1: step = 1
        
        roi_coords = self.roi_coords
        
        self.stop_event.clear()
        self.progress_val.set(0)
        self.heatmap_status.set("Initializing...")
        
        # Init Live Data
        with self.live_fit_lock:
            self.live_fit_data = None
        
        fit_params = {k: v.get() for k, v in self.entries.items()}
        fixed_params = {k: v.get() for k, v in self.checkvars.items()}
        
        omit_r = self.omit_radius_var.get()
        range_x = self.fit_range_x_var.get()
        range_y = self.fit_range_y_var.get()
        auto_range = self.auto_range_var.get()
        
        # ★ Get Scan Params from GUI & Convert Units
        scan_params = {
            'pixel_size': self.pixel_size_var.get() * 1e-3,   # nm -> um
            'pixel_dwell': self.pixel_dwell_var.get() * 1e-6, # us -> s
            'line_time': self.line_time_var.get() * 1e-3,     # ms -> s
        }

        self.heatmap_thread = threading.Thread(
            target=self.run_heatmap_loop,
            args=(self.processed_full, roi_coords, win_size, step, fit_params, fixed_params, omit_r, range_x, range_y, auto_range, scan_params)
        )
        self.heatmap_thread.daemon = True
        self.heatmap_thread.start()
        self.root.after(100, self.monitor_heatmap_thread)

    def run_heatmap_loop(self, data, roi_coords, win_size, step, fit_params, fixed_params, omit_r, range_x, range_y, auto_range, scan_params):
        T, H, W = data.shape
        half_w = win_size // 2
        
        d_map = np.full((H, W), np.nan)
        roi_x, roi_y, roi_w, roi_h = roi_coords
        
        start_y, end_y = roi_y, roi_y + roi_h
        start_x, end_x = roi_x, roi_x + roi_w
        
        frees = [k for k, v in fixed_params.items() if not v]
        p0 = [fit_params[k] for k in frees]
        
        for y in range(start_y, end_y, step):
            if self.stop_event.is_set(): break
            
            prog = ((y - start_y) / (end_y - start_y)) * 100
            self.progress_val.set(prog)
            self.heatmap_status.set(f"Processing Row {y}/{end_y} (ROI)")

            for x in range(start_x, end_x, step):
                y1 = max(0, y - half_w)
                y2 = min(H, y + half_w)
                x1 = max(0, x - half_w)
                x2 = min(W, x + half_w)
                
                if (y2 - y1) < 4 or (x2 - x1) < 4: continue

                roi_img = data[:, y1:y2, x1:x2]
                try: acf = calc.calculate_2d_acf(roi_img)
                except: continue

                sub_h, sub_w = acf.shape
                sub_cy, sub_cx = sub_h // 2, sub_w // 2
                sx_axis = np.arange(-sub_cx, sub_cx + (1 if sub_w % 2 else 0))[:sub_w]
                sy_axis = np.arange(-sub_cy, sub_cy + (1 if sub_h % 2 else 0))[:sub_h]
                SX, SY = np.meshgrid(sx_axis, sy_axis)
                
                xdata = np.vstack((SX.ravel(), SY.ravel()))
                ydata = acf.ravel()

                dist_sq = SX.ravel()**2 + SY.ravel()**2
                mask_omit = dist_sq <= (omit_r**2) if omit_r > 0 else np.zeros_like(dist_sq, dtype=bool)
                
                curr_range_x = range_x
                curr_range_y = range_y
                if auto_range:
                    x_profile = acf[sub_cy, sub_cx:]
                    rx_detected = self.detect_monotonic_decay_range(x_profile)
                    y_profile = acf[sub_cy:, sub_cx]
                    ry_detected = self.detect_monotonic_decay_range(y_profile)
                    curr_range_x = rx_detected
                    curr_range_y = ry_detected

                mask_range = (np.abs(SX.ravel()) > curr_range_x) | (np.abs(SY.ravel()) > curr_range_y)
                mask_valid = ~(mask_omit | mask_range)
                
                x_fit = xdata[:, mask_valid]
                y_fit = ydata[mask_valid]
                
                if len(y_fit) < 5: continue

                def local_wrapper(xy, *args):
                    p = fit_params.copy()
                    for i, name in enumerate(frees): p[name] = args[i]
                    return model.rics_3d_equation(
                        xy, D=p["D"], G0=p["G0"], w0=p["w0"], wz=p["wz"],
                        pixel_size=scan_params['pixel_size'], 
                        pixel_dwell=scan_params['pixel_dwell'], 
                        line_time=scan_params['line_time']
                    ).ravel()

                try:
                    popt = None
                    if frees:
                        popt, _ = curve_fit(local_wrapper, x_fit, y_fit, p0=p0, bounds=([0]*len(frees), [np.inf]*len(frees)), maxfev=600)
                        d_idx = frees.index("D") if "D" in frees else -1
                        if d_idx >= 0:
                            d_val = popt[d_idx]
                            d_map[y:min(H, y+step), x:min(W, x+step)] = d_val
                        elif "D" in fit_params:
                            d_map[y:min(H, y+step), x:min(W, x+step)] = fit_params["D"]
                    else:
                        if "D" in fit_params:
                             d_map[y:min(H, y+step), x:min(W, x+step)] = fit_params["D"]

                    # Store Data for Live Plot
                    final_p = fit_params.copy()
                    if popt is not None:
                        for i, name in enumerate(frees): final_p[name] = popt[i]
                    
                    x_slice_vals = acf[sub_cy, :]
                    y_slice_vals = acf[:, sub_cx]
                    
                    x_axis_points = np.vstack((sx_axis, np.zeros_like(sx_axis)))
                    x_curve = model.rics_3d_equation(x_axis_points, **final_p, 
                                                     pixel_size=scan_params['pixel_size'], 
                                                     pixel_dwell=scan_params['pixel_dwell'], 
                                                     line_time=scan_params['line_time'])
                    
                    y_axis_points = np.vstack((np.zeros_like(sy_axis), sy_axis))
                    y_curve = model.rics_3d_equation(y_axis_points, **final_p, 
                                                     pixel_size=scan_params['pixel_size'], 
                                                     pixel_dwell=scan_params['pixel_dwell'], 
                                                     line_time=scan_params['line_time'])
                    
                    with self.live_fit_lock:
                        self.live_fit_data = {
                            "sx_axis": sx_axis,
                            "x_slice_vals": x_slice_vals,
                            "x_curve": x_curve,
                            "range_x": curr_range_x,
                            "sy_axis": sy_axis,
                            "y_slice_vals": y_slice_vals,
                            "y_curve": y_curve,
                            "range_y": curr_range_y,
                            "omit_r": omit_r
                        }

                except: pass

        self.heatmap_d_map = d_map

    def monitor_heatmap_thread(self):
        data_snapshot = None
        with self.live_fit_lock:
            if self.live_fit_data:
                data_snapshot = self.live_fit_data.copy()
                self.live_fit_data = None 

        if data_snapshot:
            self.update_live_plot(data_snapshot)

        if self.heatmap_thread.is_alive():
            self.root.after(100, self.monitor_heatmap_thread)
        else:
            self.progress_val.set(100)
            status = "Stopped." if self.stop_event.is_set() else "Completed."
            self.heatmap_status.set(status)
            self.plot_heatmap_result()

    def update_live_plot(self, d):
        self.ax_x.cla()
        self.ax_y.cla()
        
        self.ax_x.plot(d["sx_axis"], d["x_slice_vals"], 'b.', ms=4, alpha=0.5)
        self.ax_x.plot(d["sx_axis"], d["x_curve"], 'r-', lw=2)
        self.ax_x.axvline(-d["range_x"], color='orange', ls='--')
        self.ax_x.axvline(d["range_x"], color='orange', ls='--')
        self.ax_x.set_title("Live X Fit")
        self.ax_x.grid(True)
        
        self.ax_y.plot(d["sy_axis"], d["y_slice_vals"], 'b.', ms=4, alpha=0.5)
        self.ax_y.plot(d["sy_axis"], d["y_curve"], 'r-', lw=2)
        self.ax_y.axvline(-d["range_y"], color='orange', ls='--')
        self.ax_y.axvline(d["range_y"], color='orange', ls='--')
        self.ax_y.set_title("Live Y Fit")
        self.ax_y.grid(True)
        
        if self.fixed_ylim_x is not None: self.ax_x.set_ylim(self.fixed_ylim_x)
        if self.fixed_ylim_y is not None: self.ax_y.set_ylim(self.fixed_ylim_y)
        
        self.canvas_fig.draw()

    def stop_heatmap(self):
        self.stop_event.set()

    def plot_heatmap_result(self):
        if self.heatmap_d_map is None: return

        if self.hm_window is None or not tk.Toplevel.winfo_exists(self.hm_window):
            self.hm_window = tk.Toplevel(self.root)
            self.hm_window.title("Heatmap Result")
            self.hm_window.geometry("800x600")
            
            self.hm_fig = plt.Figure(figsize=(8, 6), dpi=100)
            self.hm_canvas = FigureCanvasTkAgg(self.hm_fig, master=self.hm_window)
            self.hm_canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
            toolbar = NavigationToolbar2Tk(self.hm_canvas, self.hm_window)
            toolbar.update()
            self.hm_canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        else:
            self.hm_window.lift()

        self.hm_fig.clf()
        self.hm_ax = self.hm_fig.add_subplot(111)

        display_map = self.heatmap_d_map.copy()
        valid_mask = ~np.isnan(display_map)
        if not np.any(valid_mask):
            self.hm_ax.text(0.5, 0.5, "No valid fitting results.", ha='center', va='center')
            self.hm_canvas.draw()
            return

        vmin, vmax = None, None
        
        if self.hm_autoscale_var.get():
            valid_data = display_map[valid_mask]
            if len(valid_data) > 0:
                vmin = np.nanmin(valid_data)
                try: perc_val = self.hm_percentile_var.get()
                except: perc_val = 95.0
                vmax = np.nanpercentile(valid_data, perc_val)
        else:
            valid_data = display_map[valid_mask]
            if len(valid_data) > 0:
                vmin = np.nanmin(valid_data)
                vmax = self.hm_max_val_var.get()
        
        if vmin is not None and vmax is not None:
            if vmin > vmax: vmin, vmax = vmax, vmin
            if vmin == vmax: vmax = vmin + 1e-9

        interp = self.hm_interp_var.get()
        im = self.hm_ax.imshow(display_map, cmap='jet', interpolation=interp, vmin=vmin, vmax=vmax)
        self.hm_ax.set_title(f"Diffusion Map (Interp: {interp})")
        self.hm_cbar = self.hm_fig.colorbar(im, ax=self.hm_ax, label="D (um^2/s)")
        self.hm_canvas.draw()

    def save_heatmap_image(self):
        if self.heatmap_d_map is None: return
        filepath = filedialog.asksaveasfilename(defaultextension=".png", filetypes=[("PNG", "*.png"), ("All", "*.*")])
        if filepath:
            try:
                fig_temp = plt.figure(figsize=(6, 5))
                ax_temp = fig_temp.add_subplot(111)
                display_map = self.heatmap_d_map.copy()
                vmin, vmax = None, None
                if self.hm_autoscale_var.get():
                    valid = display_map[~np.isnan(display_map)]
                    if len(valid) > 0:
                        vmin = np.nanmin(valid)
                        try: perc_val = self.hm_percentile_var.get()
                        except: perc_val = 95.0
                        vmax = np.nanpercentile(valid, perc_val)
                else:
                    valid = display_map[~np.isnan(display_map)]
                    if len(valid) > 0:
                        vmin = np.nanmin(valid)
                        vmax = self.hm_max_val_var.get()
                interp = self.hm_interp_var.get()
                im = ax_temp.imshow(display_map, cmap='jet', interpolation=interp, vmin=vmin, vmax=vmax)
                ax_temp.set_title("Diffusion Map")
                fig_temp.colorbar(im, ax=ax_temp, label="D (um^2/s)")
                fig_temp.savefig(filepath, dpi=300)
                plt.close(fig_temp)
                messagebox.showinfo("Success", f"Saved: {filepath}")
            except Exception as e: messagebox.showerror("Error", str(e))

    # --- Basic Data & Single Fit ---
    def load_data(self):
        filepath = filedialog.askopenfilename(filetypes=[("TIFF files", "*.tif"), ("All files", "*.*")])
        if not filepath: return
        try:
            self.file_label.config(text=os.path.basename(filepath))
            self.raw_stack = prep.load_tiff(filepath)
            
            # Frame Info
            self.total_frames, H, W = self.raw_stack.shape
            self.frame_info_var.set(f"Total Frames: {self.total_frames} ({W}x{H})")
            
            # Enable Slider
            self.frame_slider.configure(state="normal", to=self.total_frames-1)
            self.current_frame_idx = -1
            
            self.roi_cx_var.set(W // 2); self.roi_cy_var.set(H // 2)
            self.selector.set_active(True)
            self.update_processing_and_acf()
        except Exception as e: messagebox.showerror("Error", str(e))

    def update_processing_and_acf(self):
        if self.raw_stack is None: return
        try:
            win = max(1, self.ma_window_var.get())
            self.processed_full = prep.subtract_moving_average(self.raw_stack, win)
            
            roi_w = self.roi_w_var.get(); roi_h = self.roi_h_var.get()
            cx = self.roi_cx_var.get(); cy = self.roi_cy_var.get()
            _, H, W = self.processed_full.shape
            
            x_start = max(0, cx - roi_w // 2); x_end = min(W, cx + roi_w // 2)
            y_start = max(0, cy - roi_h // 2); y_end = min(H, cy + roi_h // 2)
            
            if (x_end-x_start)<2 or (y_end-y_start)<2: return

            self.roi_data = self.processed_full[:, y_start:y_end, x_start:x_end]
            self.roi_coords = (x_start, y_start, x_end - x_start, y_end - y_start)
            
            self.acf_data = calc.calculate_2d_acf(self.roi_data)
            self.plot_results(fit_data=None)
            
            g0 = self.entries["G0"].get()
            self.n_var.set(f"{1/g0:.2f}" if g0 > 0 else "Inf")
            self.result_text.set("ACF Updated.")
        except Exception as e: messagebox.showerror("Processing Error", str(e))

    def run_fitting(self):
        if self.acf_data is None: return
        vals = {k: v.get() for k, v in self.entries.items()}
        fixed = {k: v.get() for k, v in self.checkvars.items()}
        frees = [k for k, v in fixed.items() if not v]
        p0 = [vals[k] for k in frees]
        H, W = self.acf_data.shape
        cy, cx = H // 2, W // 2
        X_grid, Y_grid = np.meshgrid(np.arange(-cx, cx + (1 if W % 2 else 0))[:W], 
                                     np.arange(-cy, cy + (1 if H % 2 else 0))[:H])
        xdata_flat = np.vstack((X_grid.ravel(), Y_grid.ravel()))
        ydata_flat = self.acf_data.ravel()
        
        omit_r = self.omit_radius_var.get()
        range_x = self.fit_range_x_var.get(); range_y = self.fit_range_y_var.get()
        mask_omit = (X_grid.ravel()**2 + Y_grid.ravel()**2) <= (omit_r**2) if omit_r > 0 else np.zeros_like(ydata_flat, dtype=bool)
        mask_range = (np.abs(X_grid.ravel()) > range_x) | (np.abs(Y_grid.ravel()) > range_y)
        mask_valid = ~(mask_omit | mask_range)
        
        x_fit = xdata_flat[:, mask_valid]; y_fit = ydata_flat[mask_valid]
        if len(y_fit) == 0: return

        # ★ Use Params from GUI & Convert Units
        scan_params = {
            'pixel_size': self.pixel_size_var.get() * 1e-3,   # nm -> um
            'pixel_dwell': self.pixel_dwell_var.get() * 1e-6, # us -> s
            'line_time': self.line_time_var.get() * 1e-3,     # ms -> s
        }

        def fit_wrapper(xy, *args):
            p = vals.copy()
            for i, name in enumerate(frees): p[name] = args[i]
            return model.rics_3d_equation(xy, D=p["D"], G0=p["G0"], w0=p["w0"], wz=p["wz"],
                                          pixel_size=scan_params['pixel_size'], 
                                          pixel_dwell=scan_params['pixel_dwell'], 
                                          line_time=scan_params['line_time']).ravel()
        try:
            if frees:
                popt, _ = curve_fit(fit_wrapper, x_fit, y_fit, p0=p0, bounds=([0]*len(frees), [np.inf]*len(frees)))
                for i, name in enumerate(frees): self.entries[name].set(round(popt[i], 5))
                final_p = vals.copy()
                for i, name in enumerate(frees): final_p[name] = popt[i]
            else:
                final_p = vals
            
            fit_map = model.rics_3d_equation(xdata_flat, **final_p, 
                                             pixel_size=scan_params['pixel_size'], 
                                             pixel_dwell=scan_params['pixel_dwell'], 
                                             line_time=scan_params['line_time']).reshape(H, W)
            self.plot_results(fit_map)
            g0 = final_p["G0"]
            self.n_var.set(f"{1/g0:.2f}" if g0 > 1e-9 else "Inf")
            self.result_text.set(f"Fitting Done. ({np.sum(mask_valid)} pts)")
        except Exception as e: messagebox.showerror("Fitting Error", str(e))

    def save_graphs(self):
        if self.acf_data is None: return
        filepath = filedialog.asksaveasfilename(defaultextension=".jpg", filetypes=[("JPEG", "*.jpg"), ("All", "*.*")])
        if filepath:
            try: self.fig.savefig(filepath, dpi=300, format='jpg'); messagebox.showinfo("OK", f"Saved: {filepath}")
            except Exception as e: messagebox.showerror("Error", str(e))

    def plot_results(self, fit_data=None):
        self.ax_img.cla(); self.ax_x.cla(); self.ax_y.cla(); self.ax_3d.cla()
        self.drag_lines = {}

        if self.processed_full is None or self.acf_data is None:
            self.canvas_fig.draw(); return

        if self.current_frame_idx == -1:
            display_img = np.mean(self.processed_full, axis=0)
            title = "Average Image"
        else:
            idx = min(max(0, self.current_frame_idx), self.total_frames-1)
            display_img = self.processed_full[idx]
            title = f"Frame {idx}/{self.total_frames-1}"

        self.ax_img.imshow(display_img, cmap='gray')
        self.ax_img.set_title(f"{title} (Drag/Click to Move ROI)")
        self.ax_img.axis('off')
        
        x, y, w, h = self.roi_coords
        rect = patches.Rectangle((x, y), w, h, linewidth=2, edgecolor='lime', facecolor='none')
        self.ax_img.add_patch(rect)

        H, W = self.acf_data.shape
        cy, cx = H // 2, W // 2
        
        # ★ Fix: Apply Auto-Detect logic if checked
        if self.auto_range_var.get():
            x_profile = self.acf_data[cy, cx:]
            rx_new = self.detect_monotonic_decay_range(x_profile)
            y_profile = self.acf_data[cy:, cx]
            ry_new = self.detect_monotonic_decay_range(y_profile)
            self.fit_range_x_var.set(rx_new)
            self.fit_range_y_var.set(ry_new)

        x_axis = np.arange(-cx, cx + (1 if W % 2 else 0))[:W]
        y_axis = np.arange(-cy, cy + (1 if H % 2 else 0))[:H]
        X_grid, Y_grid = np.meshgrid(x_axis, y_axis)

        omit_r = self.omit_radius_var.get()
        range_x = self.fit_range_x_var.get(); range_y = self.fit_range_y_var.get()

        def plot_slice(ax, axis_vals, data_vals, fit_vals, title, fit_limit, line_prefix):
            dvals = np.abs(axis_vals)
            mask_active = (dvals <= fit_limit) & ~((dvals <= omit_r) if omit_r > 0 else False)
            ax.plot(axis_vals, data_vals, 'b-', alpha=0.2)
            ax.plot(axis_vals[mask_active], data_vals[mask_active], 'bo', markersize=4)
            if fit_vals is not None: ax.plot(axis_vals, fit_vals, 'r-', linewidth=2)
            if np.any(mask_active):
                v = data_vals[mask_active]
                mn, mx = np.min(v), np.max(v)
                m = (mx-mn)*0.1 if mx!=mn else 0.01
                ax.set_ylim(mn-m, mx+m)
            else: ax.autoscale(True, axis='y')
            self.drag_lines[f'{line_prefix}_min'] = ax.axvline(-fit_limit, color='r', linestyle='--', picker=5)
            self.drag_lines[f'{line_prefix}_max'] = ax.axvline(fit_limit, color='r', linestyle='--', picker=5)
            ax.set_title(title); ax.grid(True)

        plot_slice(self.ax_x, x_axis, self.acf_data[cy, :], fit_data[cy, :] if fit_data is not None else None, "Fast Scan (X)", range_x, "x")
        plot_slice(self.ax_y, y_axis, self.acf_data[:, cx], fit_data[:, cx] if fit_data is not None else None, "Slow Scan (Y)", range_y, "y")

        self.ax_3d.plot_surface(X_grid, Y_grid, self.acf_data, cmap='viridis', alpha=0.8)
        if fit_data is not None: self.ax_3d.plot_wireframe(X_grid, Y_grid, fit_data, color='red', alpha=0.5, rcount=10, ccount=10)
        
        mask_3d = (X_grid**2 + Y_grid**2) > (omit_r**2) if omit_r>0 else True
        vz = self.acf_data[mask_3d]
        if len(vz)>0: self.ax_3d.set_zlim(np.min(vz), np.max(vz))
        
        self.fixed_ylim_x = self.ax_x.get_ylim()
        self.fixed_ylim_y = self.ax_y.get_ylim()
        
        self.canvas_fig.draw()

if __name__ == "__main__":
    root = tk.Tk()
    app = RICSApp(root)
    root.mainloop()