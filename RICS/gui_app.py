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
        self.root.title("RICS Analysis App v10.0 (Separate Heatmap Window)")
        self.root.geometry("1400x1000")

        # データ保持用
        self.raw_stack = None       
        self.processed_full = None
        self.roi_data = None
        self.acf_data = None
        
        # ヒートマップ結果保持用
        self.heatmap_d_map = None
        self.hm_window = None # 別ウィンドウ保持用
        
        # スレッド制御用
        self.heatmap_thread = None
        self.stop_event = threading.Event()
        self.progress_val = tk.DoubleVar(value=0.0)

        # --- GUI変数 ---
        self.ma_window_var = tk.IntVar(value=cfg.MOVING_AVG_WINDOW)
        
        # ROI設定
        self.roi_w_var = tk.IntVar(value=cfg.ROI_SIZE)
        self.roi_h_var = tk.IntVar(value=cfg.ROI_SIZE)
        self.roi_cx_var = tk.IntVar(value=128)
        self.roi_cy_var = tk.IntVar(value=128)
        
        # Omit & Range
        self.omit_radius_var = tk.DoubleVar(value=0.0)
        self.fit_range_x_var = tk.IntVar(value=cfg.ROI_SIZE // 2)
        self.fit_range_y_var = tk.IntVar(value=cfg.ROI_SIZE // 2)

        # Heatmap設定
        self.hm_window_var = tk.IntVar(value=32)
        self.hm_step_var = tk.IntVar(value=4)
        
        # Heatmap Threshold (New)
        self.hm_use_threshold_var = tk.BooleanVar(value=False)
        self.hm_threshold_val_var = tk.DoubleVar(value=50.0)

        self.n_var = tk.StringVar(value="---")
        self.result_text = tk.StringVar(value="Ready...")
        self.heatmap_status = tk.StringVar(value="Idle")

        # マウス操作用
        self.drag_lines = {}
        self.dragging_item = None
        self.selector = None
        self.press_xy = None

        self.create_layout()
        self.setup_plots()

    def create_layout(self):
        main_frame = ttk.Frame(self.root)
        main_frame.pack(fill=tk.BOTH, expand=True)

        self.canvas = tk.Canvas(main_frame, width=380)
        scrollbar = ttk.Scrollbar(main_frame, orient="vertical", command=self.canvas.yview)
        
        self.scroll_inner = ttk.Frame(self.canvas, padding="10")
        self.canvas_window = self.canvas.create_window((0, 0), window=self.scroll_inner, anchor="nw")
        
        self.scroll_inner.bind("<Configure>", self._on_frame_configure)
        self.canvas.configure(yscrollcommand=scrollbar.set)
        
        scrollbar.pack(side=tk.LEFT, fill=tk.Y)
        self.canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=False)
        self.canvas.bind_all("<MouseWheel>", self._on_mousewheel)

        self.graph_frame = ttk.Frame(main_frame)
        self.graph_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)

        self.create_widgets(self.scroll_inner)

    def _on_frame_configure(self, event):
        self.canvas.configure(scrollregion=self.canvas.bbox("all"))

    def _on_mousewheel(self, event):
        system = platform.system()
        if system == "Darwin":
            delta = event.delta
            if abs(delta) >= 1:
                self.canvas.yview_scroll(int(-1 * delta), "units")
        else:
            self.canvas.yview_scroll(int(-1*(event.delta/120)), "units")

    def create_widgets(self, parent):
        # 1. Data Loading
        ttk.Label(parent, text="1. Data Loading", font=("Arial", 12, "bold")).pack(pady=5, anchor="w")
        ttk.Button(parent, text="Load TIF Data", command=self.load_data).pack(fill=tk.X, pady=5)
        self.file_label = ttk.Label(parent, text="No file loaded", wraplength=250)
        self.file_label.pack()

        ttk.Separator(parent, orient="horizontal").pack(fill=tk.X, pady=10)

        # 2. ROI & Preprocessing
        ttk.Label(parent, text="2. ROI & Preprocessing", font=("Arial", 12, "bold")).pack(pady=5, anchor="w")
        
        bg_frame = ttk.Frame(parent); bg_frame.pack(fill=tk.X, pady=2)
        ttk.Label(bg_frame, text="Mov.Avg:").pack(side=tk.LEFT)
        ttk.Entry(bg_frame, textvariable=self.ma_window_var, width=5).pack(side=tk.LEFT, padx=5)

        roi_grp = ttk.LabelFrame(parent, text="ROI Config (Draw/Click on Image)")
        roi_grp.pack(fill=tk.X, pady=5)
        
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

        ttk.Button(parent, text="Update Image & ACF (Manual)", command=self.update_processing_and_acf).pack(fill=tk.X, pady=5)

        ttk.Separator(parent, orient="horizontal").pack(fill=tk.X, pady=10)

        # 3. Fitting Range & Omit
        ttk.Label(parent, text="3. Range & Omit", font=("Arial", 12, "bold")).pack(pady=5, anchor="w")

        omit_frame = ttk.Frame(parent); omit_frame.pack(fill=tk.X, pady=2)
        ttk.Label(omit_frame, text="Omit Radius:", foreground="red").pack(side=tk.LEFT)
        ttk.Entry(omit_frame, textvariable=self.omit_radius_var, width=5).pack(side=tk.LEFT, padx=5)
        ttk.Label(omit_frame, text="px").pack(side=tk.LEFT)

        range_frame = ttk.LabelFrame(parent, text="Fitting Range (+/- px)")
        range_frame.pack(fill=tk.X, pady=5)
        fr_x = ttk.Frame(range_frame); fr_x.pack(fill=tk.X)
        ttk.Label(fr_x, text="X Range:").pack(side=tk.LEFT)
        self.fit_range_x_var.trace_add("write", lambda *args: self.update_lines_from_entry())
        ttk.Entry(fr_x, textvariable=self.fit_range_x_var, width=5).pack(side=tk.LEFT, padx=5)
        fr_y = ttk.Frame(range_frame); fr_y.pack(fill=tk.X)
        ttk.Label(fr_y, text="Y Range:").pack(side=tk.LEFT)
        self.fit_range_y_var.trace_add("write", lambda *args: self.update_lines_from_entry())
        ttk.Entry(fr_y, textvariable=self.fit_range_y_var, width=5).pack(side=tk.LEFT, padx=5)

        ttk.Button(parent, text="Refresh Plots (Apply Scale)", command=lambda: self.plot_results(None)).pack(fill=tk.X, pady=2)

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

        # 4. Fitting
        ttk.Label(parent, text="4. Single Point Fitting", font=("Arial", 12, "bold")).pack(pady=5, anchor="w")
        ttk.Button(parent, text="Run Fitting (Selected ROI)", command=self.run_fitting).pack(fill=tk.X, pady=5)
        ttk.Label(parent, textvariable=self.result_text, relief="sunken", padding=5).pack(fill=tk.X)

        ttk.Separator(parent, orient="horizontal").pack(fill=tk.X, pady=10)

        # 5. Output
        ttk.Label(parent, text="5. Output", font=("Arial", 12, "bold")).pack(pady=5, anchor="w")
        ttk.Button(parent, text="Save Graphs as JPEG", command=self.save_graphs).pack(fill=tk.X, pady=5)

        ttk.Separator(parent, orient="horizontal").pack(fill=tk.X, pady=10)

        # 6. Heatmap Analysis
        ttk.Label(parent, text="6. Heatmap Analysis (ROI Only)", font=("Arial", 12, "bold")).pack(pady=5, anchor="w")
        
        hm_conf = ttk.Frame(parent)
        hm_conf.pack(fill=tk.X, pady=2)
        ttk.Label(hm_conf, text="Win Size:").pack(side=tk.LEFT)
        ttk.Entry(hm_conf, textvariable=self.hm_window_var, width=5).pack(side=tk.LEFT, padx=5)
        ttk.Label(hm_conf, text="Step:").pack(side=tk.LEFT)
        ttk.Entry(hm_conf, textvariable=self.hm_step_var, width=5).pack(side=tk.LEFT, padx=5)
        
        # --- Threshold Config ---
        hm_thresh_f = ttk.Frame(parent)
        hm_thresh_f.pack(fill=tk.X, pady=2)
        ttk.Checkbutton(hm_thresh_f, text="Clamp High D (Use Threshold)", variable=self.hm_use_threshold_var).pack(side=tk.LEFT)
        
        hm_thresh_v = ttk.Frame(parent)
        hm_thresh_v.pack(fill=tk.X, pady=2)
        ttk.Label(hm_thresh_v, text="Threshold:").pack(side=tk.LEFT, padx=5)
        ttk.Entry(hm_thresh_v, textvariable=self.hm_threshold_val_var, width=8).pack(side=tk.LEFT)
        ttk.Button(hm_thresh_v, text="Re-draw Map Only", command=self.plot_heatmap_result).pack(side=tk.LEFT, padx=10)

        self.progress_bar = ttk.Progressbar(parent, variable=self.progress_val, maximum=100)
        self.progress_bar.pack(fill=tk.X, pady=5)
        ttk.Label(parent, textvariable=self.heatmap_status, font=("Arial", 9)).pack(anchor="w")

        hm_btns = ttk.Frame(parent)
        hm_btns.pack(fill=tk.X, pady=5)
        ttk.Button(hm_btns, text="Generate Heatmap", command=self.start_heatmap_thread).pack(side=tk.LEFT, fill=tk.X, expand=True, padx=2)
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

    # --- Mouse Events ---
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
    # Heatmap Logic
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
        
        fit_params = {k: v.get() for k, v in self.entries.items()}
        fixed_params = {k: v.get() for k, v in self.checkvars.items()}
        
        self.heatmap_thread = threading.Thread(
            target=self.run_heatmap_loop,
            args=(self.processed_full, roi_coords, win_size, step, fit_params, fixed_params)
        )
        self.heatmap_thread.daemon = True
        self.heatmap_thread.start()
        self.root.after(100, self.monitor_heatmap_thread)

    def run_heatmap_loop(self, data, roi_coords, win_size, step, fit_params, fixed_params):
        T, H, W = data.shape
        half_w = win_size // 2
        
        d_map = np.full((H, W), np.nan)
        roi_x, roi_y, roi_w, roi_h = roi_coords
        
        start_y, end_y = roi_y, roi_y + roi_h
        start_x, end_x = roi_x, roi_x + roi_w
        
        frees = [k for k, v in fixed_params.items() if not v]
        p0 = [fit_params[k] for k in frees]
        
        omit_r = self.omit_radius_var.get()
        range_x = self.fit_range_x_var.get()
        range_y = self.fit_range_y_var.get()
        
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
                mask_range = (np.abs(SX.ravel()) > range_x) | (np.abs(SY.ravel()) > range_y)
                mask_valid = ~(mask_omit | mask_range)
                
                x_fit = xdata[:, mask_valid]
                y_fit = ydata[mask_valid]
                
                if len(y_fit) < 5: continue

                def local_wrapper(xy, *args):
                    p = fit_params.copy()
                    for i, name in enumerate(frees): p[name] = args[i]
                    return model.rics_3d_equation(
                        xy, D=p["D"], G0=p["G0"], w0=p["w0"], wz=p["wz"],
                        pixel_size=cfg.PIXEL_SIZE, pixel_dwell=cfg.PIXEL_DWELL_TIME, line_time=cfg.LINE_TIME
                    ).ravel()

                try:
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
                except: pass

        self.heatmap_d_map = d_map

    def monitor_heatmap_thread(self):
        if self.heatmap_thread.is_alive():
            self.root.after(200, self.monitor_heatmap_thread)
        else:
            self.progress_val.set(100)
            status = "Stopped." if self.stop_event.is_set() else "Completed."
            self.heatmap_status.set(status)
            self.plot_heatmap_result()

    def stop_heatmap(self):
        self.stop_event.set()

    def plot_heatmap_result(self):
        """別ウィンドウにヒートマップを表示 (閾値処理あり)"""
        if self.heatmap_d_map is None: return

        # 別ウィンドウ作成 または 既存ウィンドウの取得
        if self.hm_window is None or not tk.Toplevel.winfo_exists(self.hm_window):
            self.hm_window = tk.Toplevel(self.root)
            self.hm_window.title("Heatmap Result")
            self.hm_window.geometry("800x600")
            
            # Matplotlib Figure
            self.hm_fig = plt.Figure(figsize=(8, 6), dpi=100)
            self.hm_ax = self.hm_fig.add_subplot(111)
            
            # Canvas & Toolbar
            self.hm_canvas = FigureCanvasTkAgg(self.hm_fig, master=self.hm_window)
            self.hm_canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
            toolbar = NavigationToolbar2Tk(self.hm_canvas, self.hm_window)
            toolbar.update()
            self.hm_canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        else:
            # 既存ウィンドウがあれば前面に持ってくる
            self.hm_window.lift()
            self.hm_ax.clear()

        # === 閾値処理 ===
        display_map = self.heatmap_d_map.copy()
        
        if self.hm_use_threshold_var.get():
            thresh = self.hm_threshold_val_var.get()
            # 計算された値全体の平均 (NaN除く)
            avg_d = np.nanmean(self.heatmap_d_map)
            
            # 閾値を超える場所を平均値で置換
            # (NaNと比較するとRuntimeWarningが出るので whereを使う)
            mask_high = np.where(display_map > thresh, True, False)
            display_map[mask_high] = avg_d

        im = self.hm_ax.imshow(display_map, cmap='jet', interpolation='nearest')
        self.hm_ax.set_title("Diffusion Map (ROI)")
        self.hm_fig.colorbar(im, ax=self.hm_ax, label="D (um^2/s)")
        
        self.hm_canvas.draw()


    def save_heatmap_image(self):
        # 閾値処理後の画像を保存したい場合は、現在のウィンドウから保存するか、
        # plot_heatmap_result と同じロジックを通す必要がある。
        if self.heatmap_d_map is None: return
        filepath = filedialog.asksaveasfilename(defaultextension=".png", filetypes=[("PNG", "*.png"), ("All", "*.*")])
        if filepath:
            try:
                # 一時的なFigureで作る (閾値処理も反映)
                fig_temp = plt.figure(figsize=(6, 5))
                ax_temp = fig_temp.add_subplot(111)
                
                display_map = self.heatmap_d_map.copy()
                if self.hm_use_threshold_var.get():
                    thresh = self.hm_threshold_val_var.get()
                    avg_d = np.nanmean(self.heatmap_d_map)
                    display_map[np.where(display_map > thresh, True, False)] = avg_d

                im = ax_temp.imshow(display_map, cmap='jet')
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
            _, H, W = self.raw_stack.shape
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

        def fit_wrapper(xy, *args):
            p = vals.copy()
            for i, name in enumerate(frees): p[name] = args[i]
            return model.rics_3d_equation(xy, D=p["D"], G0=p["G0"], w0=p["w0"], wz=p["wz"],
                                          pixel_size=cfg.PIXEL_SIZE, pixel_dwell=cfg.PIXEL_DWELL_TIME, line_time=cfg.LINE_TIME).ravel()
        try:
            if frees:
                popt, _ = curve_fit(fit_wrapper, x_fit, y_fit, p0=p0, bounds=([0]*len(frees), [np.inf]*len(frees)))
                for i, name in enumerate(frees): self.entries[name].set(round(popt[i], 5))
                final_p = vals.copy()
                for i, name in enumerate(frees): final_p[name] = popt[i]
            else:
                final_p = vals
            
            fit_map = model.rics_3d_equation(xdata_flat, **final_p, pixel_size=cfg.PIXEL_SIZE, 
                                             pixel_dwell=cfg.PIXEL_DWELL_TIME, line_time=cfg.LINE_TIME).reshape(H, W)
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

        avg_img = np.mean(self.processed_full, axis=0)
        self.ax_img.imshow(avg_img, cmap='gray')
        self.ax_img.set_title("Draw(Drag) / Move(Click) ROI")
        self.ax_img.axis('off')
        x, y, w, h = self.roi_coords
        rect = patches.Rectangle((x, y), w, h, linewidth=2, edgecolor='lime', facecolor='none')
        self.ax_img.add_patch(rect)

        H, W = self.acf_data.shape
        cy, cx = H // 2, W // 2
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
        
        self.canvas_fig.draw()

if __name__ == "__main__":
    root = tk.Tk()
    app = RICSApp(root)
    root.mainloop()