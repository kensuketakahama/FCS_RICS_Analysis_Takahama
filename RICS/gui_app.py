import json
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
from matplotlib.widgets import RectangleSelector, PolygonSelector
from matplotlib.path import Path
from scipy.optimize import curve_fit
import os
import threading
import math
import glob
import platform
import subprocess

# 自作モジュールのインポート
import config as cfg
from src import preprocessing as prep
from src import calculation as calc
from src import model

# =============================================================================
# ★ Mac Utilities
# =============================================================================
def disable_app_nap():
    try:
        import appnope
        appnope.nope()
    except ImportError:
        pass

def setup_dock_icon(root):
    if platform.system() == "Darwin":
        try:
            def on_dock_click(*args):
                root.deiconify()
                root.lift()
            root.createcommand("::tk::mac::ReopenApplication", on_dock_click)
        except Exception:
            pass

disable_app_nap()


# =============================================================================
# ★ Batch Configuration Window Class
# =============================================================================
class BatchConfigWindow(tk.Toplevel):
    def __init__(self, master, file_list, root_dir, execute_callback):
        super().__init__(master)
        self.title(f"Batch Processing Config - {len(file_list)} files found")
        self.geometry("1200x850")
        self.file_list = file_list
        self.root_dir = root_dir
        self.execute_callback = execute_callback
        
        self.current_preview_data = None
        self.roi_coords = (0, 0, 0, 0)
        
        # UI Variables
        self.ma_win = tk.IntVar(value=cfg.MOVING_AVG_WINDOW)
        self.px_size = tk.DoubleVar(value=getattr(cfg, 'PIXEL_SIZE', 0.05) * 1000.0)
        self.px_dwell = tk.DoubleVar(value=getattr(cfg, 'PIXEL_DWELL_TIME', 10e-6) * 1e6)
        self.line_time = tk.DoubleVar(value=getattr(cfg, 'LINE_TIME', 2e-3) * 1000.0)
        
        self.fit_D = tk.DoubleVar(value=10.0)
        self.fit_G0 = tk.DoubleVar(value=0.01)
        self.fit_w0 = tk.DoubleVar(value=cfg.W0)
        self.fit_wz = tk.DoubleVar(value=cfg.WZ)
        self.fix_w0 = tk.BooleanVar(value=True)
        self.fix_wz = tk.BooleanVar(value=True)
        
        self.hm_win = tk.IntVar(value=32)
        self.hm_step = tk.IntVar(value=4)
        self.omit_r = tk.DoubleVar(value=0.0)
        self.range_x = tk.IntVar(value=16)
        self.range_y = tk.IntVar(value=16)
        self.auto_range = tk.BooleanVar(value=False)
        self.mask_outside = tk.BooleanVar(value=False)
        
        self.hm_autoscale = tk.BooleanVar(value=True)
        self.hm_percentile = tk.DoubleVar(value=95.0)
        self.hm_max = tk.DoubleVar(value=100.0)
        self.hm_interp = tk.StringVar(value="nearest")

        self.roi_mode_var = tk.StringVar(value="common")  # "common" or "individual"
        self.roi_map = {}  # {filepath: {'type': 'rect', 'data': [x, y, w, h]}}
        self.common_roi = None  # 共通ROIデータ
        self.current_poly_patch = None # ポリゴン表示用

        self._create_layout()
        
        if self.file_list:
            self.file_lb.selection_set(0)
            self._on_file_select(None)

    def _create_layout(self):
        left_frame = ttk.Frame(self, width=300, padding=5)
        left_frame.pack(side=tk.LEFT, fill=tk.Y)
        ttk.Label(left_frame, text="Files Found (Relative Path):", font=("bold")).pack(anchor="w")
        
        sb = ttk.Scrollbar(left_frame)
        sb.pack(side=tk.RIGHT, fill=tk.Y)
        sb_x = ttk.Scrollbar(left_frame, orient=tk.HORIZONTAL)
        sb_x.pack(side=tk.BOTTOM, fill=tk.X)

        self.file_lb = tk.Listbox(left_frame, yscrollcommand=sb.set, xscrollcommand=sb_x.set, selectmode=tk.SINGLE, width=40)
        self.file_lb.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        sb.config(command=self.file_lb.yview)
        sb_x.config(command=self.file_lb.xview)
        
        for f in self.file_list:
            try: rel_path = os.path.relpath(f, self.root_dir)
            except ValueError: rel_path = os.path.basename(f)
            self.file_lb.insert(tk.END, rel_path)
            
        self.file_lb.bind('<<ListboxSelect>>', self._on_file_select)

        center_frame = ttk.Frame(self, padding=5)
        center_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        roi_ctrl_frame = ttk.LabelFrame(center_frame, text="ROI Settings")
        roi_ctrl_frame.pack(fill=tk.X, pady=5)
        
        ttk.Radiobutton(roi_ctrl_frame, text="Apply Common ROI to All", variable=self.roi_mode_var, value="common", command=self._refresh_roi_display).pack(anchor="w")
        ttk.Radiobutton(roi_ctrl_frame, text="Set Individual ROI for Each", variable=self.roi_mode_var, value="individual", command=self._refresh_roi_display).pack(anchor="w")
        
        btn_frame = ttk.Frame(roi_ctrl_frame)
        btn_frame.pack(fill=tk.X, pady=2)
        ttk.Button(btn_frame, text="Save ROI", command=self.save_current_roi).pack(side=tk.LEFT, padx=5)
        ttk.Button(btn_frame, text="Load ROI", command=self.load_current_roi).pack(side=tk.LEFT, padx=5)

        ttk.Label(center_frame, text="Preview & ROI Selection", font=("bold")).pack(anchor="w")
        
        self.fig = plt.Figure(figsize=(5, 5), dpi=100)
        self.ax = self.fig.add_subplot(111)
        self.canvas = FigureCanvasTkAgg(self.fig, master=center_frame)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        
        self.selector = RectangleSelector(self.ax, self._on_select_roi, useblit=True, button=[1], minspanx=5, minspany=5, spancoords='pixels', interactive=True, props=dict(facecolor='lime', edgecolor='lime', alpha=0.2, fill=True))

        right_frame = ttk.Frame(self, width=300, padding=10)
        right_frame.pack(side=tk.RIGHT, fill=tk.Y)
        
        p_grp = ttk.LabelFrame(right_frame, text="1. Pre-process & Scan")
        p_grp.pack(fill=tk.X, pady=5)
        self._add_entry(p_grp, "Mov.Avg:", self.ma_win)
        self._add_entry(p_grp, "Pixel Size (nm):", self.px_size)
        self._add_entry(p_grp, "Pixel Dwell (us):", self.px_dwell)
        self._add_entry(p_grp, "Line Time (ms):", self.line_time)
        
        f_grp = ttk.LabelFrame(right_frame, text="2. Fitting Initials")
        f_grp.pack(fill=tk.X, pady=5)
        self._add_entry(f_grp, "D (um2/s):", self.fit_D)
        self._add_entry(f_grp, "G0:", self.fit_G0)
        self._add_entry(f_grp, "w0 (um):", self.fit_w0, self.fix_w0)
        self._add_entry(f_grp, "wz (um):", self.fit_wz, self.fix_wz)
        
        h_grp = ttk.LabelFrame(right_frame, text="3. Heatmap Settings")
        h_grp.pack(fill=tk.X, pady=5)
        self._add_entry(h_grp, "Win Size:", self.hm_win)
        self._add_entry(h_grp, "Step:", self.hm_step)
        self._add_entry(h_grp, "Omit R (px):", self.omit_r)
        self._add_entry(h_grp, "Range X (px):", self.range_x)
        self._add_entry(h_grp, "Range Y (px):", self.range_y)
        ttk.Checkbutton(h_grp, text="Auto-Detect Range", variable=self.auto_range).pack(anchor="w")
        ttk.Checkbutton(h_grp, text="Mask Outside ROI (Mean Fill)", variable=self.mask_outside).pack(anchor="w")
        
        o_grp = ttk.LabelFrame(right_frame, text="4. Output Settings")
        o_grp.pack(fill=tk.X, pady=5)
        ttk.Checkbutton(o_grp, text="Auto Scale (%)", variable=self.hm_autoscale).pack(anchor="w")
        self._add_entry(o_grp, "Percentile:", self.hm_percentile)
        self._add_entry(o_grp, "Max D:", self.hm_max)
        ttk.Label(o_grp, text="Interpolation:").pack(anchor="w")
        ttk.Combobox(o_grp, textvariable=self.hm_interp, values=["nearest", "bicubic", "bilinear"]).pack(fill=tk.X)

        ttk.Button(right_frame, text="START BATCH ANALYSIS", command=self._on_run).pack(fill=tk.X, pady=20)

    def _add_entry(self, parent, label, var, check_var=None):
        f = ttk.Frame(parent); f.pack(fill=tk.X, pady=1)
        ttk.Label(f, text=label, width=15).pack(side=tk.LEFT)
        ttk.Entry(f, textvariable=var, width=8).pack(side=tk.LEFT)
        if check_var: ttk.Checkbutton(f, text="Fix", variable=check_var).pack(side=tk.LEFT, padx=5)

    def _on_file_select(self, event):
        idx = self.file_lb.curselection()
        if not idx: return
        fpath = self.file_list[idx[0]]
        self.current_file_path = fpath # パスを保持
        try:
            raw = prep.load_tiff(fpath)
            if raw.ndim >= 4: raw = raw[:, 0, ...] 
            if raw.ndim == 3: self.current_preview_data = np.mean(raw, axis=0)
            elif raw.ndim == 2: self.current_preview_data = raw
            else: self.current_preview_data = raw[0]
            H, W = self.current_preview_data.shape

            # 初期化: まだ共通ROIがない場合は全画面
            if self.common_roi is None:
                self.common_roi = {'type': 'rect', 'data': [0, 0, W, H]}
                self.roi_coords = (0, 0, W, H)

            self._refresh_roi_display() # 画像更新とROI描画
        except Exception as e: print(f"Preview Error: {e}")

    def _refresh_roi_display(self):
        """現在のモードとファイルに基づいてROIを表示"""
        if self.current_preview_data is None: return
        
        self.ax.cla()
        self.ax.imshow(self.current_preview_data, cmap='gray')
        self.ax.set_title(f"ROI: {self.roi_mode_var.get()}")
        
        # 表示すべきROIデータを取得
        target_roi = None
        if self.roi_mode_var.get() == "common":
            target_roi = self.common_roi
        else:
            # 個別モード
            if self.current_file_path in self.roi_map:
                target_roi = self.roi_map[self.current_file_path]
            else:
                # 未設定の場合はデフォルト(全画面)または空
                H, W = self.current_preview_data.shape
                target_roi = {'type': 'rect', 'data': [0, 0, W, H]}
                # マップに登録しておく
                self.roi_map[self.current_file_path] = target_roi

        # 描画とセレクターの制御
        if target_roi:
            if target_roi['type'] == 'rect':
                x, y, w, h = target_roi['data']
                self.roi_coords = (x, y, w, h)
                
                # RectangleSelectorを有効化・同期
                self.selector.set_active(True)
                self.selector.set_visible(True)
                # extentsを設定して表示を同期 (xmin, xmax, ymin, ymax)
                self.selector.extents = (x, x+w, y, y+h)
                
                # パッチ描画 (Selectorが表示するので不要だが、明示的に描くなら)
                rect = patches.Rectangle((x, y), w, h, linewidth=2, edgecolor='lime', facecolor='none')
                self.ax.add_patch(rect)
                
            elif target_roi['type'] == 'poly':
                # ポリゴンの場合は矩形セレクターを無効化してパッチだけ表示
                self.selector.set_active(False)
                self.selector.set_visible(False)
                verts = target_roi['data']
                poly = patches.Polygon(verts, closed=True, linewidth=2, edgecolor='cyan', facecolor='none')
                self.ax.add_patch(poly)
                self.roi_coords = (0,0,0,0) # ダミー

        self.canvas.draw()

    def _on_select_roi(self, eclick, erelease):
        """矩形選択時のコールバック"""
        x1, y1 = int(eclick.xdata), int(eclick.ydata)
        x2, y2 = int(erelease.xdata), int(erelease.ydata)
        xmin, xmax = sorted([x1, x2])
        ymin, ymax = sorted([y1, y2])
        w, h = xmax - xmin, ymax - ymin
        
        new_roi = {'type': 'rect', 'data': [xmin, ymin, w, h]}
        self.roi_coords = (xmin, ymin, w, h)
        
        # モードに応じてデータを更新
        if self.roi_mode_var.get() == "common":
            self.common_roi = new_roi
        else:
            if self.current_file_path:
                self.roi_map[self.current_file_path] = new_roi

    def save_current_roi(self):
        """現在のROIをJSON保存"""
        # 現在表示中のROIデータを特定
        target_roi = self.common_roi if self.roi_mode_var.get() == "common" else self.roi_map.get(self.current_file_path)
        
        if not target_roi: return
        
        fpath = filedialog.asksaveasfilename(defaultextension=".json", filetypes=[("JSON", "*.json")])
        if fpath:
            try:
                with open(fpath, 'w') as f:
                    json.dump(target_roi, f)
                messagebox.showinfo("Saved", f"ROI saved to {os.path.basename(fpath)}")
            except Exception as e:
                messagebox.showerror("Error", str(e))

    def load_current_roi(self):
        """JSONからROIを読込"""
        fpath = filedialog.askopenfilename(filetypes=[("JSON", "*.json")])
        if not fpath: return
        
        try:
            with open(fpath, 'r') as f:
                loaded_roi = json.load(f)
            
            # 形式チェック
            if 'type' not in loaded_roi or 'data' not in loaded_roi:
                raise ValueError("Invalid ROI file format")

            # データ更新
            if self.roi_mode_var.get() == "common":
                self.common_roi = loaded_roi
            else:
                if self.current_file_path:
                    self.roi_map[self.current_file_path] = loaded_roi
            
            # 再描画
            self._refresh_roi_display()
            
        except Exception as e:
            messagebox.showerror("Error", f"Failed to load ROI: {e}")

    def _on_run(self):
        # ★ バリデーション: 個別モードの場合、全ファイルにROIが設定されているか確認
        if self.roi_mode_var.get() == "individual":
            missing = [f for f in self.file_list if f not in self.roi_map]
            if missing:
                msg = f"ROI is not set for {len(missing)} files.\nFirst missing: {os.path.basename(missing[0])}"
                messagebox.showwarning("ROI Missing", msg)
                return
            
            # パラメータにROIマップを渡す
            roi_param = self.roi_map
        else:
            # 共通モード
            roi_param = self.common_roi

        params = {
            'mov_avg': self.ma_win.get(),
            'scan_params': {'pixel_size': self.px_size.get() * 1e-3, 'pixel_dwell': self.px_dwell.get() * 1e-6, 'line_time': self.line_time.get() * 1e-3},
            'fit_params': {"D": self.fit_D.get(), "G0": self.fit_G0.get(), "w0": self.fit_w0.get(), "wz": self.fit_wz.get()},
            'fixed_params': {"w0": self.fix_w0.get(), "wz": self.fix_wz.get(), "D": False, "G0": False},
            'heatmap': {
                'win': self.hm_win.get(), 'step': self.hm_step.get(),
                'omit': self.omit_r.get(), 'rx': self.range_x.get(), 'ry': self.range_y.get(),
                'auto_range': self.auto_range.get(),
                'mask': self.mask_outside.get()
            },
            'roi': roi_param, # ★ 変更: 辞書または単一ROIデータを渡す
            'output': {'auto': self.hm_autoscale.get(), 'perc': self.hm_percentile.get(), 'max': self.hm_max.get(), 'interp': self.hm_interp.get()}
        }
        self.execute_callback(self.file_list, params)
        self.destroy()

    def _on_select_roi(self, eclick, erelease):
        x1, y1 = int(eclick.xdata), int(eclick.ydata)
        x2, y2 = int(erelease.xdata), int(erelease.ydata)
        xmin, xmax = sorted([x1, x2])
        ymin, ymax = sorted([y1, y2])
        self.roi_coords = (xmin, ymin, xmax-xmin, ymax-ymin)
        self._redraw_preview()

    def _redraw_preview(self):
        if self.current_preview_data is None: return
        self.ax.cla()
        self.ax.imshow(self.current_preview_data, cmap='gray')
        self.ax.set_title("Draw ROI (Applied to ALL files)")
        x, y, w, h = self.roi_coords
        rect = patches.Rectangle((x, y), w, h, linewidth=2, edgecolor='lime', facecolor='none')
        self.ax.add_patch(rect)
        self.canvas.draw()

    def _on_run(self):
        params = {
            'mov_avg': self.ma_win.get(),
            'scan_params': {'pixel_size': self.px_size.get() * 1e-3, 'pixel_dwell': self.px_dwell.get() * 1e-6, 'line_time': self.line_time.get() * 1e-3},
            'fit_params': {"D": self.fit_D.get(), "G0": self.fit_G0.get(), "w0": self.fit_w0.get(), "wz": self.fit_wz.get()},
            'fixed_params': {"w0": self.fix_w0.get(), "wz": self.fix_wz.get(), "D": False, "G0": False},
            'heatmap': {
                'win': self.hm_win.get(), 'step': self.hm_step.get(),
                'omit': self.omit_r.get(), 'rx': self.range_x.get(), 'ry': self.range_y.get(),
                'auto_range': self.auto_range.get(),
                'mask': self.mask_outside.get()
            },
            'roi': self.roi_coords,
            'output': {'auto': self.hm_autoscale.get(), 'perc': self.hm_percentile.get(), 'max': self.hm_max.get(), 'interp': self.hm_interp.get()}
        }
        self.execute_callback(self.file_list, params)
        self.destroy()

# =============================================================================
# ★ ROI Analysis Window Class
# =============================================================================
class ROIAnalysisWindow(tk.Toplevel):
    def __init__(self, master):
        super().__init__(master)
        self.title("Freehand ROI Analysis Tool")
        self.geometry("1100x850")
        
        self.ref_image = None
        self.diff_map = None
        self.poly_selector = None
        self.roi_verts = None
        
        self.disp_auto = tk.BooleanVar(value=True)
        self.disp_perc = tk.DoubleVar(value=95.0)
        self.disp_max = tk.DoubleVar(value=50.0)
        self.view_mode = tk.StringVar(value="heatmap")
        
        self._create_layout()

    def _create_layout(self):
        ctrl_frame = ttk.Frame(self, padding=10)
        ctrl_frame.pack(side=tk.TOP, fill=tk.X)
        
        ttk.Label(ctrl_frame, text="1. Load Files", font=("bold")).pack(anchor="w")
        btn_frame = ttk.Frame(ctrl_frame); btn_frame.pack(fill=tk.X, pady=5)
        ttk.Button(btn_frame, text="Load Reference Image", command=self.load_ref_image).pack(side=tk.LEFT, padx=5)
        ttk.Button(btn_frame, text="Load Diffusion CSV", command=self.load_diff_data).pack(side=tk.LEFT, padx=5)
        ttk.Button(btn_frame, text="Reset All", command=self.reset_all).pack(side=tk.RIGHT, padx=5)
        
        ttk.Separator(ctrl_frame, orient="horizontal").pack(fill=tk.X, pady=10)
        ttk.Label(ctrl_frame, text="2. Display Settings (Does not overlay)", font=("bold")).pack(anchor="w")
        view_frame = ttk.Frame(ctrl_frame); view_frame.pack(fill=tk.X, pady=5)
        ttk.Label(view_frame, text="Show: ").pack(side=tk.LEFT)
        ttk.Radiobutton(view_frame, text="Heatmap", variable=self.view_mode, value="heatmap", command=self.refresh_plot).pack(side=tk.LEFT, padx=5)
        ttk.Radiobutton(view_frame, text="Reference Image", variable=self.view_mode, value="reference", command=self.refresh_plot).pack(side=tk.LEFT, padx=5)
        
        scale_frame = ttk.Frame(ctrl_frame); scale_frame.pack(fill=tk.X, pady=2)
        ttk.Label(scale_frame, text="Heatmap Scale: ").pack(side=tk.LEFT)
        ttk.Checkbutton(scale_frame, text="Auto (%)", variable=self.disp_auto).pack(side=tk.LEFT, padx=5)
        ttk.Entry(scale_frame, textvariable=self.disp_perc, width=5).pack(side=tk.LEFT)
        ttk.Label(scale_frame, text=" or Max D:").pack(side=tk.LEFT, padx=5)
        ttk.Entry(scale_frame, textvariable=self.disp_max, width=6).pack(side=tk.LEFT)
        ttk.Button(scale_frame, text="Update Heatmap", command=self.refresh_plot).pack(side=tk.LEFT, padx=20)
        
        ttk.Separator(ctrl_frame, orient="horizontal").pack(fill=tk.X, pady=10)
        ttk.Label(ctrl_frame, text="3. Draw ROI & Analyze", font=("bold")).pack(anchor="w")
        op_frame = ttk.Frame(ctrl_frame); op_frame.pack(fill=tk.X, pady=5)
        self.btn_draw = ttk.Button(op_frame, text="Start Drawing ROI", command=self.start_drawing, state="disabled")
        self.btn_draw.pack(side=tk.LEFT, padx=5)
        self.btn_calc = ttk.Button(op_frame, text="Calculate Mean D", command=self.calculate_roi_stats, state="disabled")
        self.btn_calc.pack(side=tk.LEFT, padx=5)
        self.lbl_result = ttk.Label(op_frame, text="Result: ---", foreground="blue", font=("Arial", 12, "bold"))
        self.lbl_result.pack(side=tk.LEFT, padx=20)

        self.fig = plt.Figure(figsize=(8, 6), dpi=100)
        self.ax = self.fig.add_subplot(111)
        self.canvas = FigureCanvasTkAgg(self.fig, master=self)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        toolbar = NavigationToolbar2Tk(self.canvas, self)
        toolbar.update()

    def reset_all(self):
        self.ref_image = None; self.diff_map = None; self.roi_verts = None
        if self.poly_selector:
            self.poly_selector.set_active(False)
            self.poly_selector = None
        self.lbl_result.config(text="Result: ---")
        self.btn_draw.config(state="disabled"); self.btn_calc.config(state="disabled")
        self.ax.cla(); self.canvas.draw()

    def load_ref_image(self):
        path = filedialog.askopenfilename(title="Select Reference Image", filetypes=[("Images", "*.tif *.tiff *.png *.jpg"), ("All", "*.*")])
        if not path: return
        try:
            if path.lower().endswith(('.tif', '.tiff')):
                raw = prep.load_tiff(path)
                if raw.ndim >= 4: raw = raw[:, 0, ...]
                if raw.ndim == 3: self.ref_image = np.mean(raw, axis=0)
                elif raw.ndim == 2: self.ref_image = raw
                else: self.ref_image = raw[0]
            else:
                self.ref_image = plt.imread(path)
                if self.ref_image.ndim == 3: self.ref_image = np.mean(self.ref_image, axis=2)
            self.view_mode.set("reference"); self.refresh_plot(); self.check_ready()
        except Exception as e: messagebox.showerror("Error", str(e))

    def load_diff_data(self):
        path = filedialog.askopenfilename(title="Select Diffusion Map CSV", filetypes=[("CSV files", "*.csv"), ("All", "*.*")])
        if not path: return
        try:
            self.diff_map = np.loadtxt(path, delimiter=',')
            self.view_mode.set("heatmap"); self.refresh_plot(); self.check_ready()
        except Exception as e: messagebox.showerror("Error", str(e))

    def check_ready(self):
        if self.ref_image is not None or self.diff_map is not None: self.btn_draw.config(state="normal")
        if self.diff_map is not None: self.btn_calc.config(state="normal")

    def refresh_plot(self):
        self.ax.cla()
        mode = self.view_mode.get()
        if mode == "heatmap" and self.diff_map is not None:
            masked_map = np.ma.masked_invalid(self.diff_map)
            valid = self.diff_map[np.isfinite(self.diff_map)]
            vmin, vmax = 0, 1
            if len(valid) > 0:
                vmin = np.min(valid)
                if self.disp_auto.get(): vmax = np.percentile(valid, self.disp_perc.get())
                else: vmax = self.disp_max.get()
            self.ax.imshow(masked_map, cmap='jet', vmin=vmin, vmax=vmax)
            self.ax.set_title("Diffusion Map View")
        elif mode == "reference" and self.ref_image is not None:
            self.ax.imshow(self.ref_image, cmap='gray')
            self.ax.set_title("Reference Image View")
        elif self.diff_map is not None: self.ax.imshow(self.diff_map, cmap='jet')
        elif self.ref_image is not None: self.ax.imshow(self.ref_image, cmap='gray')
            
        if self.roi_verts is not None:
            poly = patches.Polygon(self.roi_verts, closed=True, linewidth=2, edgecolor='red', facecolor='none')
            self.ax.add_patch(poly)
        self.canvas.draw()

    def start_drawing(self):
        if self.poly_selector: self.poly_selector.set_active(False)
        self.lbl_result.config(text="Drawing... Click points, click start to finish.")
        def on_select(verts):
            self.roi_verts = verts; self.refresh_plot()
            self.lbl_result.config(text="Result: ROI Defined. Press Calculate.")
        self.poly_selector = PolygonSelector(self.ax, on_select, props=dict(color='r', linewidth=2, alpha=0.8))

    def calculate_roi_stats(self):
        if self.diff_map is None or self.roi_verts is None: messagebox.showwarning("Warning", "Load Diffusion Data and Draw ROI."); return
        try:
            H, W = self.diff_map.shape
            y, x = np.mgrid[:H, :W]
            points = np.vstack((x.ravel(), y.ravel())).T
            path = Path(self.roi_verts)
            mask = path.contains_points(points).reshape(H, W)
            roi_values = self.diff_map[mask]
            valid_values = roi_values[np.isfinite(roi_values)]
            if len(valid_values) == 0: res_str = "No valid data in ROI."
            else:
                mean_val = np.mean(valid_values); std_val = np.std(valid_values)
                res_str = f"Mean D: {mean_val:.3f} ± {std_val:.3f} (n={len(valid_values)})"
                self.refresh_plot()
                cx = np.mean([v[0] for v in self.roi_verts]); cy = np.mean([v[1] for v in self.roi_verts])
                self.ax.text(cx, cy, f"{mean_val:.2f}", color='white', fontweight='bold', ha='center', bbox=dict(facecolor='black', alpha=0.5, edgecolor='none'))
                self.canvas.draw()
            self.lbl_result.config(text=res_str); messagebox.showinfo("ROI Result", res_str)
        except Exception as e: messagebox.showerror("Error", str(e))


# =============================================================================
# ★ Main Application Class
# =============================================================================
class RICSApp:
    def __init__(self, root):
        self.root = root
        self.root.title("RICS Analysis App v23.7 (Fixed ROI Logic)")
        self.root.geometry("1400x1000")
        self.roi_verts = None

        setup_dock_icon(self.root)

        # Data holders
        self.raw_stack = None       
        self.processed_full = None
        self.roi_data = None
        self.acf_data = None
        self.current_file_path = None
        
        self.roi_mask = None 
        self.roi_mode = "rect" # "rect" or "poly"
        self.show_roi_rect = True # ★ Flag to control Rect display (False on Reset)
        
        # Display settings
        self.current_frame_idx = -1
        self.total_frames = 0
        self.fixed_ylim_x = None
        self.fixed_ylim_y = None
        
        # Heatmap results
        self.heatmap_d_map = None
        self.hm_window = None 
        
        # Threading
        self.heatmap_thread = None
        self.stop_event = threading.Event()
        self.progress_val = tk.DoubleVar(value=0.0)
        
        # Live Plot
        self.live_fit_data = None 
        self.live_fit_lock = threading.Lock()
        
        # Batch State
        self.is_batch_running = False
        self.batch_files = []
        self.batch_index = 0
        self.batch_params = None 
        self.batch_stop_req = False

        # GUI Variables
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
        self.mask_outside_var = tk.BooleanVar(value=False) # New
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
        self.batch_info_var = tk.StringVar(value="Batch: Idle")
        
        self.drag_lines = {}
        self.dragging_item = None
        self.selector = None
        self.poly_selector = None
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
        self.scroll_inner.bind("<Configure>", lambda e: self.canvas.configure(scrollregion=self.canvas.bbox("all")))
        self.canvas.configure(yscrollcommand=scrollbar.set)
        scrollbar.pack(side=tk.LEFT, fill=tk.Y)
        self.canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=False)
        self.graph_frame = ttk.Frame(main_frame)
        self.graph_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)
        self.canvas.bind("<Enter>", self._bind_mouse_scroll)
        self.canvas.bind("<Leave>", self._unbind_mouse_scroll)
        self.create_widgets(self.scroll_inner)

    def _bind_mouse_scroll(self, event):
        if platform.system() == "Darwin": self.canvas.bind_all("<MouseWheel>", self._on_mousewheel_mac)
        else: self.canvas.bind_all("<MouseWheel>", self._on_mousewheel_win)
    def _unbind_mouse_scroll(self, event): self.canvas.unbind_all("<MouseWheel>")
    def _on_mousewheel_mac(self, event): self.canvas.yview_scroll(int(-1 * event.delta), "units") if event.delta else None
    def _on_mousewheel_win(self, event): self.canvas.yview_scroll(int(-1*(event.delta/120)), "units")

    def create_widgets(self, parent):
        ttk.Label(parent, text="1. Data Loading", font=("Arial", 12, "bold")).pack(pady=5, anchor="w")
        ttk.Button(parent, text="Load Single TIF Data", command=self.load_data).pack(fill=tk.X, pady=2)
        self.file_label = ttk.Label(parent, text="No file loaded", wraplength=350)
        self.file_label.pack()
        
        batch_frame = ttk.LabelFrame(parent, text="Batch Processing"); batch_frame.pack(fill=tk.X, pady=5)
        ttk.Button(batch_frame, text="Open Batch Config Window", command=self.open_batch_window).pack(fill=tk.X, pady=2)
        ttk.Button(batch_frame, text="Open ROI Analysis Tool (Offline)", command=self.open_roi_tool).pack(fill=tk.X, pady=2)
        ttk.Label(batch_frame, textvariable=self.batch_info_var, foreground="red", font=("Arial", 10, "bold"), wraplength=350).pack(anchor="w")
        
        f_info = ttk.LabelFrame(parent, text="Frame Viewer"); f_info.pack(fill=tk.X, pady=5)
        ttk.Label(f_info, textvariable=self.frame_info_var, foreground="blue").pack(anchor="w")
        self.frame_slider = tk.Scale(f_info, from_=0, to=1, orient=tk.HORIZONTAL, label="Frame Index", command=self.on_slider_change)
        self.frame_slider.pack(fill=tk.X, padx=5); self.frame_slider.configure(state="disabled")
        ttk.Button(f_info, text="Show Average Image", command=self.reset_to_average).pack(fill=tk.X, pady=2)

        ttk.Separator(parent, orient="horizontal").pack(fill=tk.X, pady=10)
        ttk.Label(parent, text="2. Scan Parameters", font=("Arial", 12, "bold")).pack(pady=5, anchor="w")
        scan_grp = ttk.LabelFrame(parent, text="Microscope Settings"); scan_grp.pack(fill=tk.X, pady=5)
        self._add_entry(scan_grp, "Pixel Size (nm):", self.pixel_size_var)
        self._add_entry(scan_grp, "Pixel Dwell (us):", self.pixel_dwell_var)
        self._add_entry(scan_grp, "Line Time (ms):", self.line_time_var)

        ttk.Separator(parent, orient="horizontal").pack(fill=tk.X, pady=10)
        ttk.Label(parent, text="3. ROI & Preprocessing", font=("Arial", 12, "bold")).pack(pady=5, anchor="w")
        bg_frame = ttk.Frame(parent); bg_frame.pack(fill=tk.X, pady=2)
        ttk.Label(bg_frame, text="Mov.Avg:").pack(side=tk.LEFT); ttk.Entry(bg_frame, textvariable=self.ma_window_var, width=5).pack(side=tk.LEFT)
        roi_grp = ttk.LabelFrame(parent, text="ROI Config"); roi_grp.pack(fill=tk.X, pady=5)
        ttk.Button(roi_grp, text="Select Full Image", command=lambda: self.set_full_roi(show_visuals=True)).pack(fill=tk.X, pady=2)
        roi_btns = ttk.Frame(roi_grp); roi_btns.pack(fill=tk.X, pady=2)
        ttk.Button(roi_btns, text="Rect ROI", command=self.use_rect_roi, width=8).pack(side=tk.LEFT, padx=2)
        ttk.Button(roi_btns, text="Draw Poly", command=self.use_poly_roi, width=8).pack(side=tk.LEFT, padx=2)
        ttk.Button(roi_btns, text="Reset", command=self.reset_roi, width=8).pack(side=tk.LEFT, padx=2)

        io_btns = ttk.Frame(roi_grp); io_btns.pack(fill=tk.X, pady=2)
        ttk.Button(io_btns, text="Save ROI", command=self.save_roi).pack(side=tk.LEFT, padx=5, expand=True, fill=tk.X)
        ttk.Button(io_btns, text="Load ROI", command=self.load_roi).pack(side=tk.LEFT, padx=5, expand=True, fill=tk.X)
        
        self._add_roi_entry(roi_grp, "Size W:", self.roi_w_var, " H:", self.roi_h_var)
        self._add_roi_entry(roi_grp, "Center X:", self.roi_cx_var, " Y:", self.roi_cy_var)
        ttk.Checkbutton(roi_grp, text="Mask Outside ROI (Mean Fill)", variable=self.mask_outside_var).pack(anchor="w", padx=5)
        ttk.Button(parent, text="Update Image & ACF", command=self.update_processing_and_acf).pack(fill=tk.X, pady=5)

        ttk.Separator(parent, orient="horizontal").pack(fill=tk.X, pady=10)
        ttk.Label(parent, text="4. Range & Omit", font=("Arial", 12, "bold")).pack(pady=5, anchor="w")
        omit_frame = ttk.Frame(parent); omit_frame.pack(fill=tk.X, pady=2)
        ttk.Label(omit_frame, text="Omit Radius:", foreground="red").pack(side=tk.LEFT)
        ttk.Entry(omit_frame, textvariable=self.omit_radius_var, width=5).pack(side=tk.LEFT, padx=5)
        range_frame = ttk.LabelFrame(parent, text="Fitting Range"); range_frame.pack(fill=tk.X, pady=5)
        ttk.Checkbutton(range_frame, text="Auto-Detect", variable=self.auto_range_var).pack(anchor="w")
        self._add_entry(range_frame, "Range X:", self.fit_range_x_var)
        self._add_entry(range_frame, "Range Y:", self.fit_range_y_var)
        self.fit_range_x_var.trace_add("write", lambda *args: self.update_lines_from_entry())
        self.fit_range_y_var.trace_add("write", lambda *args: self.update_lines_from_entry())
        ttk.Button(parent, text="Refresh Plots", command=lambda: self.plot_results(None)).pack(fill=tk.X, pady=2)

        self.params = { "D": {"val": 10.0, "label": "D (um²/s)"}, "G0": {"val": 0.01, "label": "G0"}, "w0": {"val": cfg.W0, "label": "w0"}, "wz": {"val": cfg.WZ, "label": "wz"} }
        self.entries = {}; self.checkvars = {}
        for key, info in self.params.items():
            row = ttk.Frame(parent); row.pack(fill=tk.X, pady=2)
            ttk.Label(row, text=info["label"], width=12).pack(side=tk.LEFT)
            ev = tk.DoubleVar(value=info["val"]); ttk.Entry(row, textvariable=ev, width=8).pack(side=tk.LEFT, padx=5)
            self.entries[key] = ev
            cv = tk.BooleanVar(value=True if key in ["w0", "wz"] else False)
            ttk.Checkbutton(row, text="Fix", variable=cv).pack(side=tk.LEFT); self.checkvars[key] = cv

        ttk.Separator(parent, orient="horizontal").pack(fill=tk.X, pady=10)
        ttk.Label(parent, text="5. Single Fit", font=("Arial", 12, "bold")).pack(pady=5, anchor="w")
        ttk.Button(parent, text="Run Fitting", command=self.run_fitting).pack(fill=tk.X, pady=5)
        ttk.Label(parent, textvariable=self.result_text, relief="sunken", padding=5).pack(fill=tk.X)

        ttk.Separator(parent, orient="horizontal").pack(fill=tk.X, pady=10)
        ttk.Label(parent, text="6. Output", font=("Arial", 12, "bold")).pack(pady=5, anchor="w")
        ttk.Button(parent, text="Save Graphs", command=self.save_graphs).pack(fill=tk.X, pady=5)

        ttk.Separator(parent, orient="horizontal").pack(fill=tk.X, pady=10)
        ttk.Label(parent, text="7. Heatmap", font=("Arial", 12, "bold")).pack(pady=5, anchor="w")
        hm_conf = ttk.Frame(parent); hm_conf.pack(fill=tk.X, pady=2)
        ttk.Label(hm_conf, text="Win:").pack(side=tk.LEFT); ttk.Entry(hm_conf, textvariable=self.hm_window_var, width=5).pack(side=tk.LEFT)
        ttk.Label(hm_conf, text="Step:").pack(side=tk.LEFT); ttk.Entry(hm_conf, textvariable=self.hm_step_var, width=5).pack(side=tk.LEFT)
        vis_grp = ttk.LabelFrame(parent, text="Display"); vis_grp.pack(fill=tk.X, pady=5)
        ttk.Checkbutton(vis_grp, text="Auto Scale", variable=self.hm_autoscale_var).pack(side=tk.LEFT)
        ttk.Entry(vis_grp, textvariable=self.hm_percentile_var, width=5).pack(side=tk.LEFT); ttk.Label(vis_grp, text="%").pack(side=tk.LEFT)
        
        ttk.Label(vis_grp, text=" | Max D:").pack(side=tk.LEFT, padx=5)
        ttk.Entry(vis_grp, textvariable=self.hm_max_val_var, width=5).pack(side=tk.LEFT)
        
        self.progress_bar = ttk.Progressbar(parent, variable=self.progress_val, maximum=100); self.progress_bar.pack(fill=tk.X, pady=5)
        ttk.Label(parent, textvariable=self.heatmap_status, font=("Arial", 9)).pack(anchor="w")
        
        hm_btns = ttk.Frame(parent); hm_btns.pack(fill=tk.X, pady=5)
        ttk.Button(hm_btns, text="Gen Heatmap", command=self.start_heatmap_thread).pack(side=tk.LEFT, fill=tk.X, expand=True)
        ttk.Button(hm_btns, text="Stop", command=self.stop_heatmap).pack(side=tk.LEFT)
        
        # ★ 追加: Regen View (計算済みデータの再描画)
        ttk.Button(hm_btns, text="Regen View", command=self.plot_heatmap_result).pack(side=tk.LEFT, padx=5)
        
        ttk.Button(parent, text="Save Heatmap", command=self.save_heatmap_image).pack(fill=tk.X, pady=5)

    def save_roi(self):
        """現在のROIをJSON形式で保存"""
        data = {}
        if self.roi_mode == "poly" and self.roi_verts is not None:
            # numpy配列などはリストに変換
            data = {'type': 'poly', 'data': np.array(self.roi_verts).tolist()}
        else:
            # Rectモード
            data = {'type': 'rect', 'data': list(self.roi_coords)}
            
        fpath = filedialog.asksaveasfilename(defaultextension=".json", filetypes=[("JSON", "*.json")])
        if fpath:
            try:
                with open(fpath, 'w') as f:
                    json.dump(data, f)
                messagebox.showinfo("Success", "ROI Saved.")
            except Exception as e:
                messagebox.showerror("Error", str(e))

    def load_roi(self):
        """JSONからROIを読み込んで適用"""
        fpath = filedialog.askopenfilename(filetypes=[("JSON", "*.json")])
        if not fpath: return
        
        try:
            with open(fpath, 'r') as f:
                data = json.load(f)
            
            roi_type = data.get('type')
            roi_data = data.get('data')
            
            if roi_type == 'poly':
                self.use_poly_roi() # モード切替
                self.roi_verts = roi_data
                # マスク生成と描画更新
                # on_poly相当の処理を実行
                if self.raw_stack is not None:
                    _, H, W = self.raw_stack.shape
                    y, x = np.mgrid[:H, :W]
                    points = np.vstack((x.ravel(), y.ravel())).T
                    path = Path(self.roi_verts)
                    mask = path.contains_points(points).reshape(H, W)
                    self.roi_mask = mask
                    
                    # BBox計算
                    xs = [v[0] for v in self.roi_verts]; ys = [v[1] for v in self.roi_verts]
                    x1, x2 = int(min(xs)), int(max(xs))
                    y1, y2 = int(min(ys)), int(max(ys))
                    self.roi_coords = (x1, y1, x2-x1, y2-y1)
                    
                    self.update_processing_and_acf()
                    
            elif roi_type == 'rect':
                self.use_rect_roi() # モード切替
                self.roi_coords = tuple(roi_data)
                self.update_processing_and_acf()
                
        except Exception as e:
            messagebox.showerror("Error", f"Failed to load ROI: {e}")

    def _add_entry(self, p, l, v):
        f = ttk.Frame(p); f.pack(fill=tk.X, pady=1)
        ttk.Label(f, text=l, width=15).pack(side=tk.LEFT); ttk.Entry(f, textvariable=v, width=8).pack(side=tk.LEFT, padx=5)
    def _add_roi_entry(self, p, l1, v1, l2, v2):
        f = ttk.Frame(p); f.pack(fill=tk.X, pady=1)
        ttk.Label(f, text=l1).pack(side=tk.LEFT); ttk.Entry(f, textvariable=v1, width=5).pack(side=tk.LEFT)
        ttk.Label(f, text=l2).pack(side=tk.LEFT); ttk.Entry(f, textvariable=v2, width=5).pack(side=tk.LEFT)

    def setup_plots(self):
        self.fig = plt.Figure(figsize=(10, 8), dpi=100); gs = self.fig.add_gridspec(2, 2)
        self.ax_img = self.fig.add_subplot(gs[0, 0]); self.ax_3d = self.fig.add_subplot(gs[0, 1], projection='3d')
        self.ax_x = self.fig.add_subplot(gs[1, 0]); self.ax_y = self.fig.add_subplot(gs[1, 1])
        self.fig.tight_layout(pad=3.0)
        self.canvas_fig = FigureCanvasTkAgg(self.fig, master=self.graph_frame); self.canvas_fig.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        self.canvas_fig.mpl_connect('button_press_event', self.on_click)
        self.canvas_fig.mpl_connect('motion_notify_event', self.on_motion)
        self.canvas_fig.mpl_connect('button_release_event', self.on_release)
        # Default Rect
        self.selector = RectangleSelector(self.ax_img, self.on_select_roi, useblit=True, button=[1], minspanx=5, minspany=5, spancoords='pixels', interactive=True, props=dict(facecolor='lime', edgecolor='lime', alpha=0.2, fill=True))
        self.selector.set_active(False)

    def reset_roi(self):
        """★ Reset all ROI selections and hide visual frame"""
        self.roi_mode = "rect"
        self.roi_mask = None
        
        # ポリゴンセレクターの無効化
        if self.poly_selector:
            self.poly_selector.set_active(False)
            self.poly_selector = None
        
        # 矩形セレクターの完全無効化と非表示
        if self.selector:
            self.selector.set_active(False)
            self.selector.set_visible(False) # ★重要: 画面から消す
            # 描画アーティファクトが残る場合があるので、念のためartistsも非表示に
            try:
                for artist in self.selector.artists:
                    artist.set_visible(False)
            except:
                pass

        self.result_text.set("ROI Selection Reset (Full Image)")
        # Full image, Visuals OFF
        self.set_full_roi(show_visuals=False)

    def use_rect_roi(self):
        if self.poly_selector:
            self.poly_selector.set_active(False)
            self.poly_selector = None
        self.roi_mode = "rect"
        self.roi_mask = None 
        self.show_roi_rect = True 
        
        if self.selector:
            self.selector.set_active(True)
            self.selector.set_visible(True) # ★重要: 再表示する
        
        self.result_text.set("Switched to Rectangular ROI.")
        self.plot_results()

    def use_poly_roi(self):
        # 1. 長方形セレクターを無効化
        if self.selector:
            self.selector.set_active(False)
        
        # 2. 変数設定
        self.roi_mode = "poly"
        self.roi_mask = None
        self.show_roi_rect = False # 枠線表示OFF
        
        # 3. 古いポリゴンセレクターがあれば破棄
        if self.poly_selector:
            self.poly_selector.set_active(False)
            self.poly_selector = None
        
        # 4. ★重要★ 先に画面をクリアして、古いROI表示（長方形など）を消す
        # これを後にすると、せっかく作ったPolygonSelectorがcla()で消されてしまう
        self.plot_results()
        
        # 5. コールバック定義
        def on_poly(verts):
            self.roi_verts = verts
            if self.raw_stack is None: return
            _, H, W = self.raw_stack.shape
            y, x = np.mgrid[:H, :W]
            points = np.vstack((x.ravel(), y.ravel())).T
            path = Path(verts)
            mask = path.contains_points(points).reshape(H, W)
            self.roi_mask = mask
            
            # Update BBox
            xs = [v[0] for v in verts]; ys = [v[1] for v in verts]
            x1, x2 = int(min(xs)), int(max(xs))
            y1, y2 = int(min(ys)), int(max(ys))
            w = x2 - x1; h = y2 - y1
            cx = x1 + w // 2; cy = y1 + h // 2
            self.roi_w_var.set(w); self.roi_h_var.set(h)
            self.roi_cx_var.set(cx); self.roi_cy_var.set(cy)
            
            # 計算更新（plot_resultsが呼ばれ、contourが描画される）
            self.update_processing_and_acf()
            
        # 6. 新しいポリゴンセレクターを作成（クリア済みの画面にアタッチ）
        self.poly_selector = PolygonSelector(self.ax_img, on_poly, props=dict(color='cyan', linewidth=2, alpha=0.5))
        self.result_text.set("Draw Polygon on Image...")

    def on_click(self, event):
        if event.inaxes in [self.ax_x, self.ax_y]:
            if self.drag_lines:
                click_x = event.xdata
                if click_x is not None:
                    min_dist = float('inf'); target = None
                    active_lines = []
                    if event.inaxes == self.ax_x: active_lines = [('x_min', self.drag_lines.get('x_min')), ('x_max', self.drag_lines.get('x_max'))]
                    elif event.inaxes == self.ax_y: active_lines = [('y_min', self.drag_lines.get('y_min')), ('y_max', self.drag_lines.get('y_max'))]
                    for name, line in active_lines:
                        if line:
                            dist = abs(click_x - line.get_xdata()[0])
                            if dist < min_dist: min_dist = dist; target = name
                    if min_dist < (self.fit_range_x_var.get() * 0.2 + 1.0): self.dragging_item = target
            return
        if event.inaxes == self.ax_img and event.xdata: self.press_xy = (event.xdata, event.ydata)

    def on_motion(self, event):
        if self.dragging_item and event.xdata:
            new_x = int(round(event.xdata))
            if 'x_' in self.dragging_item: self.fit_range_x_var.set(abs(new_x))
            elif 'y_' in self.dragging_item: self.fit_range_y_var.set(abs(new_x))

    def on_release(self, event):
        if self.dragging_item: self.dragging_item = None; return
        if event.inaxes == self.ax_img and self.press_xy:
            if self.roi_mode == "poly": return
            
            rx = (event.xdata, event.ydata)
            # Only update center if simple click (and Rect mode)
            if self.roi_mask is None and rx[0] and math.sqrt((rx[0]-self.press_xy[0])**2 + (rx[1]-self.press_xy[1])**2) < 5:
                self.roi_cx_var.set(int(rx[0])); self.roi_cy_var.set(int(rx[1]))
                self.update_processing_and_acf()

    def on_select_roi(self, eclick, erelease):
        self.roi_mask = None 
        x1, y1 = int(eclick.xdata), int(eclick.ydata); x2, y2 = int(erelease.xdata), int(erelease.ydata)
        w = abs(x2-x1); h = abs(y2-y1); cx = min(x1,x2)+w//2; cy = min(y1,y2)+h//2
        self.roi_w_var.set(w); self.roi_h_var.set(h); self.roi_cx_var.set(cx); self.roi_cy_var.set(cy)
        self.update_processing_and_acf()

    def update_lines_from_entry(self):
        if not self.drag_lines: return
        try:
            rx = self.fit_range_x_var.get(); ry = self.fit_range_y_var.get()
            if 'x_min' in self.drag_lines: self.drag_lines['x_min'].set_xdata([-rx, -rx])
            if 'x_max' in self.drag_lines: self.drag_lines['x_max'].set_xdata([rx, rx])
            if 'y_min' in self.drag_lines: self.drag_lines['y_min'].set_xdata([-ry, -ry])
            if 'y_max' in self.drag_lines: self.drag_lines['y_max'].set_xdata([ry, ry])
            self.canvas_fig.draw_idle()
        except: pass

    def set_full_roi(self, show_visuals=True):
        """Reset ROI to full image dimensions."""
        if self.raw_stack is not None:
            _, H, W = self.raw_stack.shape
            self.roi_w_var.set(W); self.roi_h_var.set(H); self.roi_cx_var.set(W//2); self.roi_cy_var.set(H//2)
            self.roi_mask = None 
            self.show_roi_rect = show_visuals # ★ Control visual flag
            self.update_processing_and_acf()

    def on_slider_change(self, val): self.current_frame_idx = int(val); self.update_processing_and_acf()
    def reset_to_average(self): self.current_frame_idx = -1; self.update_processing_and_acf()
    def detect_monotonic_decay_range(self, data, min_len=3):
        n=len(data); smooth=np.convolve(data,np.ones(3)/3,mode='valid'); diff=np.diff(smooth)
        idx=np.where(diff>0)[0]; return idx[0]+1 if len(idx)>0 else n

    # =======================================================
    # Batch & Core Logic
    # =======================================================
    def open_roi_tool(self):
        ROIAnalysisWindow(self.root)

    def open_batch_window(self):
        root_dir = filedialog.askdirectory(title="Select Root Directory for Batch Analysis")
        if not root_dir: return
        exts = ['*.tif', '*.tiff', '*.TIF', '*.TIFF']
        files = []
        for ext in exts: files.extend(glob.glob(os.path.join(root_dir, '**', ext), recursive=True))
        files = sorted(list(set(files)))
        if not files: messagebox.showinfo("Info", "No TIFF files found."); return
        BatchConfigWindow(self.root, files, root_dir, self.start_visual_batch)

    def start_visual_batch(self, file_list, params):
        self.batch_files = file_list; self.batch_params = params
        self.batch_index = 0; self.is_batch_running = True; self.batch_stop_req = False
        self.process_batch_next()

    def process_batch_next(self):
        if not self.is_batch_running or self.batch_stop_req:
            self.is_batch_running = False; self.batch_info_var.set("Batch: Stopped.")
            return
        if self.batch_index >= len(self.batch_files):
            self.is_batch_running = False; self.batch_info_var.set("Batch: Finished All.")
            messagebox.showinfo("Batch Done", "All files processed."); return

        fpath = self.batch_files[self.batch_index]
        fname = os.path.basename(fpath)
        self.batch_info_var.set(f"Batch: {self.batch_index+1}/{len(self.batch_files)} - {fname}")
        
        try:
            self.load_file_internal(fpath)
            p = self.batch_params
            self.ma_window_var.set(p['mov_avg'])
            _, H, W = self.raw_stack.shape
            roi_data_src = p['roi'] # 辞書(個別) または dict(共通データ)
            
            # 個別モード(辞書)の場合、当該ファイル用のROIを取得
            current_roi_data = None
            if isinstance(roi_data_src, dict) and 'type' not in roi_data_src:
                # {path: roi_data} の形式とみなす
                current_roi_data = roi_data_src.get(fpath)
            else:
                # 共通ROIデータそのもの
                current_roi_data = roi_data_src

            # ROI適用
            if current_roi_data and current_roi_data.get('type') == 'poly':
                # ポリゴンマスク生成
                verts = current_roi_data['data']
                y, x = np.mgrid[:H, :W]
                points = np.vstack((x.ravel(), y.ravel())).T
                path = Path(verts)
                self.roi_mask = path.contains_points(points).reshape(H, W)
                
                # BBox設定
                xs = [v[0] for v in verts]; ys = [v[1] for v in verts]
                x1, x2 = int(min(xs)), int(max(xs))
                y1, y2 = int(min(ys)), int(max(ys))
                self.roi_coords = (x1, y1, x2-x1, y2-y1)
                
            elif current_roi_data and current_roi_data.get('type') == 'rect':
                # 矩形
                self.roi_mask = None
                rx, ry, rw, rh = current_roi_data['data']
                if rx+rw <= W and ry+rh <= H:
                    self.roi_coords = (rx, ry, rw, rh)
                else:
                    self.set_full_roi(show_visuals=False)
            else:
                self.set_full_roi(show_visuals=False)
            
            self.update_processing_and_acf()
            self.start_heatmap_thread_batch(p)
        except Exception as e:
            print(f"Skipping {fname}: {e}")
            self.batch_index += 1; self.root.after(100, self.process_batch_next)


    def start_heatmap_thread_batch(self, p):
        self.stop_event.clear(); self.progress_val.set(0); self.heatmap_status.set("Batch Analyzing...")
        with self.live_fit_lock: self.live_fit_data = None
        roi = self.roi_coords; win = p['heatmap']['win']; step = p['heatmap']['step']
        self.heatmap_thread = threading.Thread(
            target=self.run_heatmap_loop,
            args=(self.processed_full, roi, win, step, 
                  p['fit_params'], p['fixed_params'], 
                  p['heatmap']['omit'], p['heatmap']['rx'], p['heatmap']['ry'], 
                  p['heatmap']['auto_range'], p['scan_params'], p['heatmap']['mask'], None) # Batch uses rect only
        )
        self.heatmap_thread.daemon = True; self.heatmap_thread.start()
        self.root.after(100, self.monitor_heatmap_thread)

    def save_batch_result(self):
        if self.heatmap_d_map is None: return
        p = self.batch_params['output']
        base_path = os.path.splitext(self.current_file_path)[0]
        
        # ★修正: 連番チェックを入れる
        save_path = self._get_unique_filepath(base_path + "_heatmap.png")
        # CSVはPNGと同じベース名に合わせるか、独自に連番を振る
        csv_base = os.path.splitext(save_path)[0] 
        csv_path = csv_base + ".csv"
        # 万が一CSVだけ残っているケースも考慮するなら以下を使用
        # csv_path = self._get_unique_filepath(base_path + "_heatmap_data.csv")
        
        ma_val = self.batch_params['mov_avg']
        win_val = self.batch_params['heatmap']['win']
        step_val = self.batch_params['heatmap']['step']
        info_text = f"Mov.Avg: {ma_val} | Win: {win_val} | Step: {step_val}"

        try:
            fig_temp = plt.figure(figsize=(6, 6))
            ax_temp = fig_temp.add_axes([0.1, 0.15, 0.8, 0.8])
            display_map = self.heatmap_d_map.copy()
            # (描画処理省略...既存コードと同じ)
            valid = display_map[~np.isnan(display_map)]
            vmin, vmax = None, None
            if len(valid) > 0:
                if p['auto']: vmin = np.nanmin(valid); vmax = np.nanpercentile(valid, p['perc'])
                else: vmin = np.nanmin(valid); vmax = p['max']
            im = ax_temp.imshow(display_map, cmap='jet', interpolation=p['interp'], vmin=vmin, vmax=vmax)
            ax_temp.set_title(f"D Map: {os.path.basename(self.current_file_path)}")
            fig_temp.colorbar(im, ax=ax_temp, label="D (um^2/s)")
            fig_temp.text(0.5, 0.05, info_text, ha='center', fontsize=10, bbox=dict(facecolor='white', alpha=0.8))
            
            fig_temp.savefig(save_path, dpi=300); plt.close(fig_temp)
            np.savetxt(csv_path, self.heatmap_d_map, delimiter=',')
            print(f"Saved: {save_path} & {csv_path}")
        except Exception as e: print(f"Save Error: {e}")

    # =======================================================
    # Main Logic
    # =======================================================
    def load_data(self):
        filepath = filedialog.askopenfilename(filetypes=[("TIFF files", "*.tif"), ("All files", "*.*")])
        if not filepath: return
        self.load_file_internal(filepath)

    def load_file_internal(self, filepath):
        try:
            self.file_label.config(text=os.path.basename(filepath))
            img = prep.load_tiff(filepath)
            img = np.squeeze(img)
            if img.ndim == 2: img = img[np.newaxis, :, :]
            elif img.ndim >= 4:
                while img.ndim > 3: img = img[:, 0, ...]
            self.raw_stack = img
            
            self.total_frames, H, W = self.raw_stack.shape
            self.frame_info_var.set(f"Total Frames: {self.total_frames} ({W}x{H})")
            self.frame_slider.configure(state="normal", to=self.total_frames-1)
            self.current_frame_idx = -1
            self.current_file_path = filepath
            if not self.is_batch_running:
                self.roi_cx_var.set(W // 2); self.roi_cy_var.set(H // 2)
            self.roi_mask = None # Reset
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
            
            x_start = cx - roi_w // 2; x_end = x_start + roi_w
            y_start = cy - roi_h // 2; y_end = y_start + roi_h
            
            # Prepare Mask Logic for Single Fit
            use_mask = self.mask_outside_var.get()
            
            # Clip bounds
            vx1 = max(0, x_start); vx2 = min(W, x_end)
            vy1 = max(0, y_start); vy2 = min(H, y_end)
            
            if vx2 > vx1 and vy2 > vy1:
                # Base valid extraction
                base_crop = self.processed_full[:, vy1:vy2, vx1:vx2]
                
                if use_mask:
                    # Create container filled with mean
                    mean_val = np.mean(base_crop)
                    self.roi_data = np.full((self.total_frames, roi_h, roi_w), mean_val, dtype=base_crop.dtype)
                    
                    # Place valid rect
                    oy = vy1 - y_start; ox = vx1 - x_start
                    
                    # If Polygon Mask exists, modify base_crop before placing
                    if self.roi_mask is not None:
                        # Extract mask crop
                        mask_crop = self.roi_mask[vy1:vy2, vx1:vx2]
                        # Apply to data: False -> mean_val
                        # Note: This affects base_crop reference, so copy if needed
                        masked_data = base_crop.copy()
                        masked_data[:, ~mask_crop] = mean_val
                        self.roi_data[:, oy:oy+(vy2-vy1), ox:ox+(vx2-vx1)] = masked_data
                    else:
                        self.roi_data[:, oy:oy+(vy2-vy1), ox:ox+(vx2-vx1)] = base_crop
                else:
                    # Normal: Just use valid intersection (size might differ from roi_w/h if at border)
                    self.roi_data = base_crop
            else:
                self.roi_data = np.zeros((self.total_frames, roi_h, roi_w))

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
        if self.auto_range_var.get():
            x_profile = self.acf_data[cy, cx:]; rx_new = self.detect_monotonic_decay_range(x_profile)
            y_profile = self.acf_data[cy:, cx]; ry_new = self.detect_monotonic_decay_range(y_profile)
            self.fit_range_x_var.set(rx_new); self.fit_range_y_var.set(ry_new)
        X_grid, Y_grid = np.meshgrid(np.arange(-cx, cx + (1 if W % 2 else 0))[:W], np.arange(-cy, cy + (1 if H % 2 else 0))[:H])
        xdata_flat = np.vstack((X_grid.ravel(), Y_grid.ravel())); ydata_flat = self.acf_data.ravel()
        omit_r = self.omit_radius_var.get(); range_x = self.fit_range_x_var.get(); range_y = self.fit_range_y_var.get()
        mask_omit_arr = (X_grid.ravel()**2 + Y_grid.ravel()**2) <= (omit_r**2) if omit_r > 0 else np.zeros_like(ydata_flat, dtype=bool)
        mask_range = (np.abs(X_grid.ravel()) > range_x) | (np.abs(Y_grid.ravel()) > range_y)
        mask_valid = ~(mask_omit_arr | mask_range)
        x_fit = xdata_flat[:, mask_valid]; y_fit = ydata_flat[mask_valid]
        if len(y_fit) == 0: return
        scan_params = {'pixel_size': self.pixel_size_var.get()*1e-3, 'pixel_dwell': self.pixel_dwell_var.get()*1e-6, 'line_time': self.line_time_var.get()*1e-3}
        def fit_wrapper(xy, *args):
            p = vals.copy()
            for i, name in enumerate(frees): p[name] = args[i]
            return model.rics_3d_equation(xy, D=p["D"], G0=p["G0"], w0=p["w0"], wz=p["wz"], **scan_params).ravel()
        try:
            if frees:
                popt, _ = curve_fit(fit_wrapper, x_fit, y_fit, p0=p0, bounds=([0]*len(frees), [np.inf]*len(frees)))
                for i, name in enumerate(frees): self.entries[name].set(round(popt[i], 5))
                final_p = vals.copy()
                for i, name in enumerate(frees): final_p[name] = popt[i]
            else: final_p = vals
            fit_map = model.rics_3d_equation(xdata_flat, **final_p, **scan_params).reshape(H, W)
            self.plot_results(fit_map)
            g0 = final_p["G0"]
            self.n_var.set(f"{1/g0:.2f}" if g0 > 1e-9 else "Inf")
            self.result_text.set(f"Fitting Done. ({np.sum(mask_valid)} pts)")
        except Exception as e: messagebox.showerror("Fitting Error", str(e))

    def start_heatmap_thread(self):
        if self.is_batch_running: return
        if self.processed_full is None: messagebox.showwarning("Warning", "Please load data first."); return
        fp = {k: v.get() for k, v in self.entries.items()}
        fx = {k: v.get() for k, v in self.checkvars.items()}
        sp = {'pixel_size': self.pixel_size_var.get()*1e-3, 'pixel_dwell': self.pixel_dwell_var.get()*1e-6, 'line_time': self.line_time_var.get()*1e-3}
        self.stop_event.clear(); self.progress_val.set(0)
        with self.live_fit_lock: self.live_fit_data = None
        
        # Check mask setting
        use_mask = self.mask_outside_var.get()
        roi_mask_data = self.roi_mask if use_mask else None
        
        self.heatmap_thread = threading.Thread(
            target=self.run_heatmap_loop,
            args=(self.processed_full, self.roi_coords, self.hm_window_var.get(), self.hm_step_var.get(),
                  fp, fx, self.omit_radius_var.get(), self.fit_range_x_var.get(), self.fit_range_y_var.get(),
                  self.auto_range_var.get(), sp, use_mask, roi_mask_data)
        )
        self.heatmap_thread.daemon = True; self.heatmap_thread.start()
        self.root.after(100, self.monitor_heatmap_thread)

    def run_heatmap_loop(self, data, roi, win, step, fit_p, fixed_p, omit, rx, ry, auto, scan_p, use_mask, poly_mask):
        T, H, W = data.shape
        d_map = np.full((H, W), np.nan)
        roi_x, roi_y, roi_w, roi_h = roi
        frees = [k for k, v in fixed_p.items() if not v]
        p0 = [fit_p[k] for k in frees]
        half_w = win // 2 
        
        for y in range(roi_y, roi_y+roi_h, step):
            if self.stop_event.is_set(): break
            prog = ((y - roi_y) / roi_h) * 100
            self.progress_val.set(prog)
            for x in range(roi_x, roi_x+roi_w, step):
                
                # ★ Center Check: If poly_mask is used, skip if center is False
                if poly_mask is not None:
                    # Boundary check for center
                    if 0 <= y < H and 0 <= x < W:
                        if not poly_mask[y, x]:
                            continue
                
                # Window Coords
                y1 = y - half_w; y2 = y + half_w
                x1 = x - half_w; x2 = x + half_w
                
                # Check bounds
                if not use_mask:
                    # Clip if mask OFF
                    y1 = max(0, y1); y2 = min(H, y2)
                    x1 = max(0, x1); x2 = min(W, x2)
                    if (y2-y1)<4 or (x2-x1)<4: continue
                    roi_img = data[:, y1:y2, x1:x2]
                else:
                    # Mask ON (Mean Fill)
                    valid_y1 = max(roi_y, y1); valid_y2 = min(roi_y+roi_h, y2)
                    valid_x1 = max(roi_x, x1); valid_x2 = min(roi_x+roi_w, x2)
                    
                    valid_y1 = max(0, valid_y1); valid_y2 = min(H, valid_y2)
                    valid_x1 = max(0, valid_x1); valid_x2 = min(W, valid_x2)
                    
                    if valid_y2 <= valid_y1 or valid_x2 <= valid_x1: continue
                        
                    valid_data = data[:, valid_y1:valid_y2, valid_x1:valid_x2]
                    mean_val = np.mean(valid_data)
                    
                    # Create window filled with mean
                    win_h_sz = 2 * half_w; win_w_sz = 2 * half_w
                    roi_img = np.full((T, win_h_sz, win_w_sz), mean_val, dtype=data.dtype)
                    
                    off_y = valid_y1 - y1; off_x = valid_x1 - x1
                    cpy_h = valid_y2 - valid_y1; cpy_w = valid_x2 - valid_x1
                    
                    if poly_mask is not None:
                        # Extract mask for valid region
                        m_sub = poly_mask[valid_y1:valid_y2, valid_x1:valid_x2]
                        masked_vdata = valid_data.copy()
                        masked_vdata[:, ~m_sub] = mean_val
                        roi_img[:, off_y:off_y+cpy_h, off_x:off_x+cpy_w] = masked_vdata
                    else:
                        roi_img[:, off_y:off_y+cpy_h, off_x:off_x+cpy_w] = valid_data

                try:
                    acf = calc.calculate_2d_acf(roi_img)
                    sh, sw = acf.shape; scy, scx = sh//2, sw//2
                    sx = np.arange(-scx, scx+(1 if sw%2 else 0))[:sw]
                    sy = np.arange(-scy, scy+(1 if sh%2 else 0))[:sh]
                    SX, SY = np.meshgrid(sx, sy)
                    crx, cry = rx, ry
                    if auto:
                        crx = self.detect_monotonic_decay_range(acf[scy, scx:])
                        cry = self.detect_monotonic_decay_range(acf[scy:, scx])
                    mask_omit_arr = (SX**2 + SY**2 <= omit**2) if omit > 0 else np.zeros_like(SX, dtype=bool)
                    mask_range = (np.abs(SX)>crx) | (np.abs(SY)>cry)
                    mask = ~(mask_omit_arr | mask_range)
                    xf = np.vstack((SX.ravel(), SY.ravel()))[:, mask.ravel()]
                    yf = acf.ravel()[mask.ravel()]
                    if len(yf) < 5: continue
                    def wrap(xy, *a):
                        curr = fit_p.copy()
                        for i, n in enumerate(frees): curr[n] = a[i]
                        return model.rics_3d_equation(xy, D=curr["D"], G0=curr["G0"], w0=curr["w0"], wz=curr["wz"], **scan_p).ravel()
                    if frees:
                        popt, _ = curve_fit(wrap, xf, yf, p0=p0, bounds=([0]*len(frees), [np.inf]*len(frees)), maxfev=600)
                        if "D" in frees: d_map[y:min(H, y+step), x:min(W, x+step)] = popt[frees.index("D")]
                    else: d_map[y:min(H, y+step), x:min(W, x+step)] = fit_p["D"]
                    
                    with self.live_fit_lock:
                        curr = fit_p.copy()
                        if frees:
                            for i, n in enumerate(frees): curr[n] = popt[i]
                        xc = model.rics_3d_equation(np.vstack((sx, np.zeros_like(sx))), **curr, **scan_p)
                        yc = model.rics_3d_equation(np.vstack((np.zeros_like(sy), sy)), **curr, **scan_p)
                        self.live_fit_data = {
                            "sx_axis": sx, "x_slice_vals": acf[scy, :], "x_curve": xc, "range_x": crx,
                            "sy_axis": sy, "y_slice_vals": acf[:, scx], "y_curve": yc, "range_y": cry
                        }
                except Exception: pass
        self.heatmap_d_map = d_map

    def monitor_heatmap_thread(self):
        try:
            with self.live_fit_lock:
                if self.live_fit_data:
                    self.update_live_plot(self.live_fit_data); self.live_fit_data = None
        except Exception: pass
        if self.heatmap_thread.is_alive():
            self.root.after(100, self.monitor_heatmap_thread)
        else:
            self.progress_val.set(100)
            if self.is_batch_running:
                self.save_batch_result(); self.batch_index += 1; self.root.after(200, self.process_batch_next)
            else:
                self.heatmap_status.set("Completed."); self.plot_heatmap_result()

    def update_live_plot(self, d):
        try:
            self.ax_x.cla(); self.ax_y.cla()
            self.ax_x.plot(d["sx_axis"], d["x_slice_vals"], 'b.', ms=4, alpha=0.5); self.ax_x.plot(d["sx_axis"], d["x_curve"], 'r-')
            self.ax_x.axvline(-d["range_x"], color='orange', ls='--'); self.ax_x.axvline(d["range_x"], color='orange', ls='--')
            self.ax_x.set_title("Live X Fit")
            self.ax_y.plot(d["sy_axis"], d["y_slice_vals"], 'b.', ms=4, alpha=0.5); self.ax_y.plot(d["sy_axis"], d["y_curve"], 'r-')
            self.ax_y.axvline(-d["range_y"], color='orange', ls='--'); self.ax_y.axvline(d["range_y"], color='orange', ls='--')
            self.ax_y.set_title("Live Y Fit")
            
            def apply_zoom(ax, axis, vals, limit):
                ax.set_xlim(-limit*1.5, limit*1.5)
                mask = np.abs(axis) <= (limit * 1.5)
                if np.any(mask):
                    v = vals[mask]
                    if np.all(np.isfinite(v)):
                        mn, mx = np.min(v), np.max(v)
                        m = (mx-mn)*0.1 if mx!=mn else 0.01; ax.set_ylim(mn-m, mx+m)
            apply_zoom(self.ax_x, d["sx_axis"], d["x_slice_vals"], d["range_x"])
            apply_zoom(self.ax_y, d["sy_axis"], d["y_slice_vals"], d["range_y"])
            self.canvas_fig.draw()
        except Exception: pass

    def stop_heatmap(self): 
        self.stop_event.set()
        if self.is_batch_running: self.batch_stop_req = True

    def plot_heatmap_result(self):
        # (前半の描画処理は変更なし)
        if self.heatmap_d_map is None: return
        if self.hm_window is None or not tk.Toplevel.winfo_exists(self.hm_window):
            self.hm_window = tk.Toplevel(self.root)
            self.hm_window.title("Heatmap Result"); self.hm_window.geometry("800x600")
            self.hm_fig = plt.Figure(figsize=(8, 6), dpi=100)
            self.hm_canvas = FigureCanvasTkAgg(self.hm_fig, master=self.hm_window)
            self.hm_canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
            toolbar = NavigationToolbar2Tk(self.hm_canvas, self.hm_window); toolbar.update()
            self.hm_canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        else: self.hm_window.lift()
        
        self.hm_fig.clf(); self.hm_ax = self.hm_fig.add_subplot(111)
        # ... (中略: imshowなどの描画コード) ...
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
            if len(valid_data) > 0: vmin = np.nanmin(valid_data); vmax = self.hm_max_val_var.get()
            
        if vmin is not None and vmax is not None:
            if vmin > vmax: vmin, vmax = vmax, vmin
            if vmin == vmax: vmax = vmin + 1e-9
            
        interp = self.hm_interp_var.get()
        im = self.hm_ax.imshow(display_map, cmap='jet', interpolation=interp, vmin=vmin, vmax=vmax)
        self.hm_ax.set_title(f"Diffusion Map (Interp: {interp})")
        self.hm_cbar = self.hm_fig.colorbar(im, ax=self.hm_ax, label="D (um^2/s)")
        
        # ★修正: CSV自動保存時の上書き防止
        if self.current_file_path:
            base = os.path.splitext(self.current_file_path)[0]
            # ここでユニークパスを取得
            csv_path = self._get_unique_filepath(base + "_heatmap_data.csv")
            try: np.savetxt(csv_path, self.heatmap_d_map, delimiter=',')
            except: pass
            
        tk.Button(self.hm_window, text="Draw Freehand ROI & Calc Mean", command=self.open_roi_tool).pack(pady=5)
        self.hm_canvas.draw()

    def save_heatmap_image(self):
        if self.heatmap_d_map is None: return
        filepath = filedialog.asksaveasfilename(defaultextension=".png", filetypes=[("PNG", "*.png"), ("All", "*.*")])
        
        ma_val = self.ma_window_var.get()
        win_val = self.hm_window_var.get()
        step_val = self.hm_step_var.get()
        info_text = f"Mov.Avg: {ma_val} | Win: {win_val} | Step: {step_val}"

        if filepath:
            try:
                # (画像の保存処理はそのまま)
                fig_temp = plt.figure(figsize=(6, 6))
                ax_temp = fig_temp.add_axes([0.1, 0.15, 0.8, 0.8])
                display_map = self.heatmap_d_map.copy()
                # ... (中略: imshow等) ...
                vmin, vmax = None, None
                if self.hm_autoscale_var.get():
                    valid = display_map[~np.isnan(display_map)]
                    if len(valid) > 0:
                        vmin = np.nanmin(valid); vmax = np.nanpercentile(valid, self.hm_percentile_var.get())
                else:
                    valid = display_map[~np.isnan(display_map)]
                    if len(valid) > 0: vmin = np.nanmin(valid); vmax = self.hm_max_val_var.get()
                
                interp = self.hm_interp_var.get()
                im = ax_temp.imshow(display_map, cmap='jet', interpolation=interp, vmin=vmin, vmax=vmax)
                ax_temp.set_title("Diffusion Map")
                fig_temp.colorbar(im, ax=ax_temp, label="D (um^2/s)")
                fig_temp.text(0.5, 0.05, info_text, ha='center', fontsize=10, bbox=dict(facecolor='white', alpha=0.8))
                fig_temp.savefig(filepath, dpi=300); plt.close(fig_temp)
                
                # ★修正: CSV保存時の上書き防止
                # 画像のパスに基づいてCSV名を作るが、それが既存なら連番を振る
                csv_base_path = os.path.splitext(filepath)[0] + ".csv"
                csv_path = self._get_unique_filepath(csv_base_path)
                
                np.savetxt(csv_path, self.heatmap_d_map, delimiter=',')
                
                messagebox.showinfo("Success", f"Saved PNG:\n{filepath}\n\nSaved CSV:\n{csv_path}")
            except Exception as e: messagebox.showerror("Error", str(e))

    def save_graphs(self):
        if self.acf_data is None: return
        filepath = filedialog.asksaveasfilename(defaultextension=".jpg", filetypes=[("JPEG", "*.jpg"), ("All", "*.*")])
        if filepath:
            try: self.fig.savefig(filepath, dpi=300, format='jpg'); messagebox.showinfo("OK", f"Saved: {filepath}")
            except Exception as e: messagebox.showerror("Error", str(e))

    def _get_unique_filepath(self, filepath):
        """同名ファイルがある場合、末尾に連番を付与して重複しないパスを返す"""
        if not os.path.exists(filepath):
            return filepath
        base, ext = os.path.splitext(filepath)
        counter = 1
        while True:
            new_path = f"{base}_{counter}{ext}"
            if not os.path.exists(new_path):
                return new_path
            counter += 1

    def plot_results(self, fit_data=None):
        self.ax_img.cla(); self.ax_x.cla(); self.ax_y.cla(); self.ax_3d.cla()
        self.drag_lines = {}
        if self.processed_full is None or self.acf_data is None: self.canvas_fig.draw(); return
        if self.current_frame_idx == -1: display_img = np.mean(self.processed_full, axis=0); title = "Average Image"
        else: idx = min(max(0, self.current_frame_idx), self.total_frames-1); display_img = self.processed_full[idx]; title = f"Frame {idx}/{self.total_frames-1}"
        self.ax_img.imshow(display_img, cmap='gray')
        self.ax_img.set_title(f"{title} (Drag ROI)")
        self.ax_img.axis('off')
        
        # ★ Draw ROI based on mode
        if self.roi_mask is not None:
            # Poly mode: draw contour
            self.ax_img.contour(self.roi_mask, colors='r', linewidths=2)
        else:
            # Rect mode
            x, y, w, h = self.roi_coords
            # Check if visual flag is True
            if self.show_roi_rect:
                rect = patches.Rectangle((x, y), w, h, linewidth=2, edgecolor='lime', facecolor='none')
                self.ax_img.add_patch(rect)

        H, W = self.acf_data.shape; cy, cx = H // 2, W // 2
        x_axis = np.arange(-cx, cx + (1 if W % 2 else 0))[:W]; y_axis = np.arange(-cy, cy + (1 if H % 2 else 0))[:H]
        X_grid, Y_grid = np.meshgrid(x_axis, y_axis)
        omit_r = self.omit_radius_var.get(); range_x = self.fit_range_x_var.get(); range_y = self.fit_range_y_var.get()
        def plot_slice(ax, axis_vals, data_vals, fit_vals, title, fit_limit, line_prefix):
            dvals = np.abs(axis_vals)
            mask_omit_arr = (dvals <= omit_r) if omit_r > 0 else np.zeros_like(dvals, dtype=bool)
            mask_active = (dvals <= fit_limit) & (~mask_omit_arr)
            ax.plot(axis_vals, data_vals, 'b-', alpha=0.2)
            ax.plot(axis_vals[mask_active], data_vals[mask_active], 'bo', markersize=4)
            if fit_vals is not None: ax.plot(axis_vals, fit_vals, 'r-', linewidth=2)
            ax.set_xlim(-fit_limit*1.5, fit_limit*1.5)
            if np.any(mask_active): 
                v = data_vals[mask_active]; mn, mx = np.min(v), np.max(v)
                m = (mx-mn)*0.1 if mx!=mn else 0.01; ax.set_ylim(mn-m, mx+m)
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