import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.widgets import RectangleSelector
from scipy.optimize import curve_fit
import os
import platform # OS判定用に追加

# 自作モジュールのインポート
import config as cfg
from src import preprocessing as prep
from src import calculation as calc
from src import model

class RICSApp:
    def __init__(self, root):
        self.root = root
        self.root.title("RICS Analysis App v7.1 (Mac Scroll Fix)")
        self.root.geometry("1400x950")

        # データ保持用
        self.raw_stack = None       
        self.processed_full = None
        self.roi_data = None
        self.acf_data = None
        
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

        self.n_var = tk.StringVar(value="---")
        self.result_text = tk.StringVar(value="Ready...")

        # ドラッグ操作用
        self.drag_lines = {}
        self.dragging_item = None
        
        # ROI Selector保持用
        self.selector = None

        # レイアウト構築
        self.create_layout()
        self.setup_plots()

    def create_layout(self):
        main_frame = ttk.Frame(self.root)
        main_frame.pack(fill=tk.BOTH, expand=True)

        # --- 左側：スクロール可能なコントロールパネル ---
        self.canvas = tk.Canvas(main_frame, width=350)
        scrollbar = ttk.Scrollbar(main_frame, orient="vertical", command=self.canvas.yview)
        
        self.scroll_inner = ttk.Frame(self.canvas, padding="10")
        
        # Windowとして配置
        self.canvas_window = self.canvas.create_window((0, 0), window=self.scroll_inner, anchor="nw")
        
        # Scroll設定
        self.scroll_inner.bind("<Configure>", self._on_frame_configure)
        self.canvas.configure(yscrollcommand=scrollbar.set)
        
        # Pack
        scrollbar.pack(side=tk.LEFT, fill=tk.Y)
        self.canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=False)
        
        # マウスホイールバインディング (OSごとの挙動差を吸収)
        # bind_allだと全画面で反応するため、Canvas上でのみ反応するようにbindする方法もあるが
        # 利便性のためbind_allにし、ハンドラ側で制御する
        self.canvas.bind_all("<MouseWheel>", self._on_mousewheel)
        # Linuxなどは Button-4/5 の場合があるが、まずはMouseWheelで対応

        # --- 右側：グラフエリア ---
        self.graph_frame = ttk.Frame(main_frame)
        self.graph_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)

        self.create_widgets(self.scroll_inner)

    def _on_frame_configure(self, event):
        """内部フレームのサイズが変わったらスクロール領域を更新"""
        self.canvas.configure(scrollregion=self.canvas.bbox("all"))

    def _on_mousewheel(self, event):
        """マウスホイールイベントハンドラ (Mac対応版)"""
        # OS判定
        system = platform.system()
        
        if system == "Darwin": # macOS
            # Macのトラックパッドは event.delta が小さい整数 (1, 2...) や小数を返す
            # 120で割ると0になって動かなくなるため、そのまま使うか、適度な係数を掛ける
            # 方向は event.delta が正なら上スクロール(中身は下へ)、負なら下スクロール
            delta = event.delta
            # 感度調整: 必要に応じて係数を変えてください
            if abs(delta) >= 1:
                self.canvas.yview_scroll(int(-1 * delta), "units")
        else:
            # Windows / Linux (通常 120単位)
            self.canvas.yview_scroll(int(-1*(event.delta/120)), "units")

    def create_widgets(self, parent):
        # 1. Data Loading
        ttk.Label(parent, text="1. Data Loading", font=("Arial", 12, "bold")).pack(pady=5, anchor="w")
        ttk.Button(parent, text="Load TIF Data", command=self.load_data).pack(fill=tk.X, pady=5)
        self.file_label = ttk.Label(parent, text="No file loaded", wraplength=200)
        self.file_label.pack()

        ttk.Separator(parent, orient="horizontal").pack(fill=tk.X, pady=10)

        # 2. ROI & Preprocessing
        ttk.Label(parent, text="2. ROI & Preprocessing", font=("Arial", 12, "bold")).pack(pady=5, anchor="w")
        
        bg_frame = ttk.Frame(parent); bg_frame.pack(fill=tk.X, pady=2)
        ttk.Label(bg_frame, text="Mov.Avg:").pack(side=tk.LEFT)
        ttk.Entry(bg_frame, textvariable=self.ma_window_var, width=5).pack(side=tk.LEFT, padx=5)

        roi_grp = ttk.LabelFrame(parent, text="ROI Config (Draw on Image)")
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

        # 4. Execution
        ttk.Button(parent, text="Run Fitting", command=self.run_fitting).pack(fill=tk.X, pady=10)
        ttk.Label(parent, textvariable=self.result_text, relief="sunken", padding=5).pack(fill=tk.X)

        ttk.Separator(parent, orient="horizontal").pack(fill=tk.X, pady=10)

        # 5. Output
        ttk.Label(parent, text="5. Output", font=("Arial", 12, "bold")).pack(pady=5, anchor="w")
        ttk.Button(parent, text="Save Graphs as JPEG", command=self.save_graphs).pack(fill=tk.X, pady=5)

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
        
        # ROI Selector
        self.selector = RectangleSelector(
            self.ax_img, 
            self.on_select_roi,
            useblit=True,
            button=[1], 
            minspanx=5, minspany=5,
            spancoords='pixels',
            interactive=True,
            props=dict(facecolor='lime', edgecolor='lime', alpha=0.2, fill=True)
        )
        self.selector.set_active(False)

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

    def on_click(self, event):
        if event.inaxes not in [self.ax_x, self.ax_y]: return
        if not self.drag_lines: return
        
        click_x = event.xdata
        if click_x is None: return

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

    def on_motion(self, event):
        if self.dragging_item is None: return
        if event.xdata is None: return
        
        new_x = int(round(event.xdata))
        
        if 'x_' in self.dragging_item:
            self.fit_range_x_var.set(abs(new_x))
        elif 'y_' in self.dragging_item:
            self.fit_range_y_var.set(abs(new_x))

    def on_release(self, event):
        self.dragging_item = None

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
        except:
            pass

    def save_graphs(self):
        if self.acf_data is None:
            messagebox.showwarning("Warning", "No data to save.")
            return

        filepath = filedialog.asksaveasfilename(
            defaultextension=".jpg",
            filetypes=[("JPEG files", "*.jpg"), ("All files", "*.*")],
            title="Save Graphs as JPEG"
        )
        if not filepath: return

        try:
            self.fig.savefig(filepath, dpi=300, format='jpg')
            messagebox.showinfo("Success", f"Graphs saved to:\n{filepath}")
        except Exception as e:
            messagebox.showerror("Error", f"Failed to save file:\n{e}")

    def load_data(self):
        filepath = filedialog.askopenfilename(filetypes=[("TIFF files", "*.tif"), ("All files", "*.*")])
        if not filepath: return
        try:
            self.file_label.config(text=os.path.basename(filepath))
            self.raw_stack = prep.load_tiff(filepath)
            
            _, H, W = self.raw_stack.shape
            self.roi_cx_var.set(W // 2)
            self.roi_cy_var.set(H // 2)
            
            self.selector.set_active(True)
            self.update_processing_and_acf()
        except Exception as e:
            messagebox.showerror("Error", str(e))

    def update_processing_and_acf(self):
        if self.raw_stack is None: return
        try:
            win = max(1, self.ma_window_var.get())
            self.processed_full = prep.subtract_moving_average(self.raw_stack, win)
            
            roi_w = self.roi_w_var.get()
            roi_h = self.roi_h_var.get()
            cx = self.roi_cx_var.get()
            cy = self.roi_cy_var.get()
            _, H, W = self.processed_full.shape
            
            x_start = max(0, cx - roi_w // 2)
            x_end = min(W, cx + roi_w // 2)
            y_start = max(0, cy - roi_h // 2)
            y_end = min(H, cy + roi_h // 2)

            if (x_end - x_start) < 2 or (y_end - y_start) < 2:
                return

            self.roi_data = self.processed_full[:, y_start:y_end, x_start:x_end]
            self.roi_coords = (x_start, y_start, x_end - x_start, y_end - y_start)
            
            self.acf_data = calc.calculate_2d_acf(self.roi_data)
            
            self.plot_results(fit_data=None)
            
            g0 = self.entries["G0"].get()
            self.n_var.set(f"{1/g0:.2f}" if g0 > 0 else "Inf")
            self.result_text.set("ACF Updated.")

        except Exception as e:
            messagebox.showerror("Processing Error", str(e))

    def run_fitting(self):
        if self.acf_data is None: return

        vals = {k: v.get() for k, v in self.entries.items()}
        fixed = {k: v.get() for k, v in self.checkvars.items()}
        frees = [k for k, v in fixed.items() if not v]
        p0 = [vals[k] for k in frees]

        H, W = self.acf_data.shape
        cy, cx = H // 2, W // 2
        x_axis = np.arange(-cx, cx + (1 if W % 2 else 0))[:W]
        y_axis = np.arange(-cy, cy + (1 if H % 2 else 0))[:H]
        X_grid, Y_grid = np.meshgrid(x_axis, y_axis)
        
        xdata_flat = np.vstack((X_grid.ravel(), Y_grid.ravel()))
        ydata_flat = self.acf_data.ravel()

        omit_r = self.omit_radius_var.get()
        range_x = self.fit_range_x_var.get()
        range_y = self.fit_range_y_var.get()

        dist_sq = X_grid.ravel()**2 + Y_grid.ravel()**2
        mask_omit = dist_sq <= (omit_r**2) if omit_r > 0 else np.zeros_like(dist_sq, dtype=bool)
        mask_range_out = (np.abs(X_grid.ravel()) > range_x) | (np.abs(Y_grid.ravel()) > range_y)
        
        mask_invalid = mask_omit | mask_range_out
        mask_valid = ~mask_invalid

        xdata_fit = xdata_flat[:, mask_valid]
        ydata_fit = ydata_flat[mask_valid]

        if len(ydata_fit) == 0:
            messagebox.showerror("Error", "No data points left for fitting!")
            return

        def fit_wrapper(xy_mesh, *args):
            p = vals.copy()
            for i, name in enumerate(frees): p[name] = args[i]
            full_calc = model.rics_3d_equation(
                xy_mesh, D=p["D"], G0=p["G0"], w0=p["w0"], wz=p["wz"],
                pixel_size=cfg.PIXEL_SIZE, pixel_dwell=cfg.PIXEL_DWELL_TIME, line_time=cfg.LINE_TIME
            )
            return full_calc.ravel()

        try:
            if frees:
                bounds = ([0]*len(frees), [np.inf]*len(frees))
                popt, _ = curve_fit(fit_wrapper, xdata_fit, ydata_fit, p0=p0, bounds=bounds)
                for i, name in enumerate(frees): self.entries[name].set(round(popt[i], 5))
            else:
                popt = []

            final_p = vals.copy()
            for i, name in enumerate(frees): final_p[name] = popt[i]
            
            fit_map = model.rics_3d_equation(
                xdata_flat, **final_p, pixel_size=cfg.PIXEL_SIZE, pixel_dwell=cfg.PIXEL_DWELL_TIME, line_time=cfg.LINE_TIME
            ).reshape(H, W)
            
            self.plot_results(fit_map)
            
            g0 = final_p["G0"]
            self.n_var.set(f"{1/g0:.2f}" if g0 > 1e-9 else "Inf")
            self.result_text.set(f"Fitting Done. (Used {np.sum(mask_valid)} pts)")
            
        except Exception as e:
            messagebox.showerror("Fitting Error", str(e))

    def plot_results(self, fit_data=None):
        self.ax_img.cla(); self.ax_3d.cla(); self.ax_x.cla(); self.ax_y.cla()
        self.drag_lines = {}

        if self.processed_full is None or self.acf_data is None:
            self.canvas_fig.draw(); return

        # 1. Image Preview
        avg_img = np.mean(self.processed_full, axis=0)
        self.ax_img.imshow(avg_img, cmap='gray')
        self.ax_img.set_title("Draw ROI with Mouse")
        self.ax_img.axis('off')
        
        x, y, w, h = self.roi_coords
        rect = patches.Rectangle((x, y), w, h, linewidth=2, edgecolor='lime', facecolor='none')
        self.ax_img.add_patch(rect)
        
        # Prep Grid
        H, W = self.acf_data.shape
        cy, cx = H // 2, W // 2
        x_axis = np.arange(-cx, cx + (1 if W % 2 else 0))[:W]
        y_axis = np.arange(-cy, cy + (1 if H % 2 else 0))[:H]
        X_grid, Y_grid = np.meshgrid(x_axis, y_axis)

        omit_r = self.omit_radius_var.get()
        range_x = self.fit_range_x_var.get()
        range_y = self.fit_range_y_var.get()

        def plot_slice(ax, axis_vals, data_vals, fit_vals, title, fit_limit, line_prefix):
            dist_vals = np.abs(axis_vals)
            mask_omit = dist_vals <= omit_r if omit_r > 0 else np.zeros_like(dist_vals, dtype=bool)
            mask_range = dist_vals <= fit_limit
            mask_active = mask_range & (~mask_omit)
            
            ax.plot(axis_vals[mask_omit & mask_range], data_vals[mask_omit & mask_range], 'x', color='lightgray')
            ax.plot(axis_vals[~mask_range], data_vals[~mask_range], '.', color='lightgray', alpha=0.5)
            ax.plot(axis_vals[mask_active], data_vals[mask_active], 'bo', label='Valid Data', markersize=4)
            ax.plot(axis_vals, data_vals, 'b-', alpha=0.2)

            if fit_vals is not None:
                ax.plot(axis_vals, fit_vals, 'r-', label='Fit', linewidth=2)

            if np.any(mask_active):
                valid_y = data_vals[mask_active]
                ymin, ymax = np.min(valid_y), np.max(valid_y)
                margin = (ymax - ymin) * 0.1 if ymax != ymin else 0.01
                ax.set_ylim(ymin - margin, ymax + margin)
            else:
                ax.autoscale(True, axis='y')

            l1 = ax.axvline(x=-fit_limit, color='r', linestyle='--', picker=5)
            l2 = ax.axvline(x=fit_limit, color='r', linestyle='--', picker=5)
            self.drag_lines[f'{line_prefix}_min'] = l1
            self.drag_lines[f'{line_prefix}_max'] = l2

            ax.set_title(title)
            ax.set_xlabel("Lag (pixel)")
            ax.set_ylabel("ACF (G)")
            ax.grid(True)

        # 2. X Slice
        plot_slice(self.ax_x, x_axis, self.acf_data[cy, :], 
                   fit_data[cy, :] if fit_data is not None else None, 
                   "Fast Scan (X)", range_x, "x")

        # 3. Y Slice
        plot_slice(self.ax_y, y_axis, self.acf_data[:, cx], 
                   fit_data[:, cx] if fit_data is not None else None, 
                   "Slow Scan (Y)", range_y, "y")

        # 4. 3D Plot
        self.ax_3d.plot_surface(X_grid, Y_grid, self.acf_data, cmap='viridis', alpha=0.8)
        if fit_data is not None:
            self.ax_3d.plot_wireframe(X_grid, Y_grid, fit_data, color='red', alpha=0.5, rcount=15, ccount=15)
        self.ax_3d.set_title("3D ACF")
        self.ax_3d.set_xlabel("X Lag")
        self.ax_3d.set_ylabel("Y Lag")
        self.ax_3d.set_zlabel("ACF (G)")

        dist_sq = X_grid**2 + Y_grid**2
        mask_3d_omit = dist_sq <= (omit_r**2) if omit_r > 0 else np.zeros_like(dist_sq, dtype=bool)
        valid_z = self.acf_data[~mask_3d_omit]
        
        if len(valid_z) > 0:
            zmin, zmax = np.min(valid_z), np.max(valid_z)
            self.ax_3d.set_zlim(zmin, zmax)

        self.canvas_fig.draw()

if __name__ == "__main__":
    root = tk.Tk()
    app = RICSApp(root)
    root.mainloop()