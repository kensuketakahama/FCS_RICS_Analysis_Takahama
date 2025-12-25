import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from scipy.optimize import curve_fit
import os

# 自作モジュールのインポート
import config as cfg
from src import preprocessing as prep
from src import calculation as calc
from src import model

class FCSApp:
    def __init__(self, root):
        self.root = root
        self.root.title("FCS Analysis App v9.0 (Grid Layout & Auto-Clamp)")
        self.root.geometry("1400x1000")

        # Data
        self.raw_stack = None
        self.trace_raw = None
        self.trace_processed = None
        self.time_axis = None
        
        # ACF Results
        self.acf_lags = None
        self.acf_G = None
        self.acf_sigma = None
        
        # --- Config Parameters ---
        self.pixel_time_var = tk.DoubleVar(value=2.0)
        self.use_detrend_var = tk.BooleanVar(value=False)
        self.detrend_cutoff_var = tk.DoubleVar(value=5.0)
        self.use_log_bin_var = tk.BooleanVar(value=False)
        self.n_segments_var = tk.IntVar(value=1)

        # --- Fitting Parameters ---
        # 辞書で一元管理
        self.params = {
            'N':        {'val': 1.0,   'fix': False, 'b_on': False, 'min': 0.0, 'max': 'inf'},
            'D':        {'val': 414.0, 'fix': False, 'b_on': False, 'min': 0.0, 'max': 'inf'},
            'w0':       {'val': cfg.W0,'fix': True,  'b_on': False, 'min': 0.0, 'max': 'inf'},
            'wz':       {'val': cfg.WZ,'fix': True,  'b_on': False, 'min': 0.0, 'max': 'inf'},
            'T':        {'val': 0.1,   'fix': False, 'b_on': True,  'min': 0.0, 'max': 1.0},
            'tau_trip': {'val': 1e-6,  'fix': False, 'b_on': False, 'min': 0.0, 'max': 'inf'},
            'y0':       {'val': 0.0,   'fix': False, 'b_on': False, 'min': -1.0,'max': 1.0}
        }
        
        # Tkinter変数への変換
        self.p_vars = {}
        for k, v in self.params.items():
            self.p_vars[k] = {
                'val': tk.DoubleVar(value=v['val']),
                'fix': tk.BooleanVar(value=v['fix']),
                'b_on': tk.BooleanVar(value=v['b_on']),
                'min': tk.StringVar(value=str(v['min'])),
                'max': tk.StringVar(value=str(v['max']))
            }

        # Fitting Range
        self.range_min_var = tk.DoubleVar(value=1e-5)
        self.range_max_var = tk.DoubleVar(value=1.0)

        self.drag_lines = {}
        self.dragging_item = None

        self.create_layout()
        self.setup_plots()

    def create_layout(self):
        panel = ttk.Frame(self.root, padding="10")
        panel.pack(side=tk.LEFT, fill=tk.Y)

        # 1. Load
        ttk.Label(panel, text="1. Data Loading", font=("bold")).pack(anchor="w", pady=5)
        ttk.Button(panel, text="Load Data", command=self.load_data).pack(fill=tk.X)
        self.file_label = ttk.Label(panel, text="No file", wraplength=200)
        self.file_label.pack(pady=2)
        
        ttk.Separator(panel, orient="horizontal").pack(fill=tk.X, pady=10)

        # 2. Config
        ttk.Label(panel, text="2. Analysis Config", font=("bold")).pack(anchor="w", pady=5)
        f1 = ttk.Frame(panel); f1.pack(fill=tk.X, pady=2)
        ttk.Label(f1, text="Pixel Time (us):").pack(side=tk.LEFT)
        ttk.Entry(f1, textvariable=self.pixel_time_var, width=8).pack(side=tk.LEFT, padx=5)

        # Detrend
        d_frame = ttk.LabelFrame(panel, text="Preprocessing")
        d_frame.pack(fill=tk.X, pady=5)
        ttk.Checkbutton(d_frame, text="Bleach Correction (Detrend)", variable=self.use_detrend_var, 
                        command=self.update_analysis).pack(anchor="w")
        
        # ACF Mode
        b_frame = ttk.LabelFrame(panel, text="ACF Mode")
        b_frame.pack(fill=tk.X, pady=5)
        f_seg = ttk.Frame(b_frame); f_seg.pack(fill=tk.X, pady=2)
        ttk.Label(f_seg, text="Segments:").pack(side=tk.LEFT)
        ttk.Entry(f_seg, textvariable=self.n_segments_var, width=5).pack(side=tk.LEFT, padx=5)
        ttk.Checkbutton(b_frame, text="Log Binning", variable=self.use_log_bin_var,
                        command=self.update_analysis).pack(anchor="w")
        ttk.Button(b_frame, text="Update ACF", command=self.update_analysis).pack(fill=tk.X, pady=2)

        ttk.Separator(panel, orient="horizontal").pack(fill=tk.X, pady=10)

        # 3. Fitting Parameters (Grid Layout)
        ttk.Label(panel, text="3. Fitting Parameters", font=("bold")).pack(anchor="w", pady=5)
        
        # Grid Frame作成
        grid_frame = ttk.Frame(panel)
        grid_frame.pack(fill=tk.X, pady=2)

        # Headers
        headers = ["Param", "Value", "Fix", "Limit?", "Min", "Max"]
        for col, text in enumerate(headers):
            ttk.Label(grid_frame, text=text, font=("bold", 9)).grid(row=0, column=col, padx=3, pady=2)

        # Rows
        param_order = ['N', 'D', 'w0', 'wz', 'T', 'tau_trip', 'y0']
        for row_idx, key in enumerate(param_order, start=1):
            vars = self.p_vars[key]
            
            # Param Name
            ttk.Label(grid_frame, text=key).grid(row=row_idx, column=0, padx=3, pady=2, sticky="e")
            
            # Value
            ttk.Entry(grid_frame, textvariable=vars['val'], width=8).grid(row=row_idx, column=1, padx=3)
            
            # Fix Checkbox
            ttk.Checkbutton(grid_frame, variable=vars['fix']).grid(row=row_idx, column=2, padx=3)
            
            # Limit Checkbox
            ttk.Checkbutton(grid_frame, variable=vars['b_on']).grid(row=row_idx, column=3, padx=3)
            
            # Min / Max
            ttk.Entry(grid_frame, textvariable=vars['min'], width=6).grid(row=row_idx, column=4, padx=1)
            ttk.Entry(grid_frame, textvariable=vars['max'], width=6).grid(row=row_idx, column=5, padx=1)

        ttk.Button(panel, text="Run Fitting", command=self.run_fitting).pack(fill=tk.X, pady=15)

        # Results
        res_grp = ttk.LabelFrame(panel, text="Results")
        res_grp.pack(fill=tk.X, pady=5)
        self.res_txt = tk.StringVar(value="")
        ttk.Label(res_grp, textvariable=self.res_txt, justify="left", font=("Consolas", 10)).pack(padx=5, pady=5, anchor="w")
        
        ttk.Button(panel, text="Save Graph", command=self.save_graph).pack(fill=tk.X, pady=5)


    def setup_plots(self):
        self.fig = plt.Figure(figsize=(10, 8), dpi=100)
        gs = self.fig.add_gridspec(2, 1, height_ratios=[1, 1.5])
        
        self.ax_trace = self.fig.add_subplot(gs[0, 0])
        self.ax_acf = self.fig.add_subplot(gs[1, 0])
        self.fig.tight_layout(pad=3.0)
        
        right_frame = ttk.Frame(self.root)
        right_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)
        
        self.canvas = FigureCanvasTkAgg(self.fig, master=right_frame)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

        self.canvas.mpl_connect('button_press_event', self.on_click)
        self.canvas.mpl_connect('motion_notify_event', self.on_motion)
        self.canvas.mpl_connect('button_release_event', self.on_release)

    def load_data(self):
        path = filedialog.askopenfilename(filetypes=[("TIF", "*.tif"), ("All", "*.*")])
        if not path: return
        try:
            self.file_label.config(text=os.path.basename(path))
            raw = prep.load_tiff(path)
            self.trace_raw = raw.flatten().astype(np.float64)
            self.trace_processed = self.trace_raw.copy()
            self.update_analysis()
        except Exception as e: messagebox.showerror("Error", str(e))

    def update_analysis(self):
        if self.trace_raw is None: return
        
        pt_sec = self.pixel_time_var.get() * 1e-6
        n = len(self.trace_raw)
        
        # Detrend
        if self.use_detrend_var.get():
            cutoff_sec = self.detrend_cutoff_var.get() * 1e-3
            self.trace_processed = prep.detrend_1d_trace(self.trace_raw, pt_sec, cutoff_sec)
        else:
            self.trace_processed = self.trace_raw.copy()

        self.time_axis = np.arange(n) * pt_sec
        
        # ACF
        n_seg = max(1, self.n_segments_var.get())
        lags_raw, G_raw, sem_raw = calc.calculate_segmented_acf(self.trace_processed, n_segments=n_seg)
        time_lags = lags_raw * pt_sec
        
        if self.use_log_bin_var.get():
            self.acf_lags, self.acf_G, self.acf_sigma = calc.log_binning_weighted(time_lags, G_raw, sem_raw)
        else:
            self.acf_lags = time_lags
            self.acf_G = G_raw
            self.acf_sigma = sem_raw
        
        self.plot_graphs()

    def plot_graphs(self, fit_curve_lags=None, fit_curve_G=None):
        self.ax_trace.clear()
        step = max(1, len(self.trace_processed) // 10000)
        
        if self.use_detrend_var.get():
            self.ax_trace.plot(self.time_axis[::step], self.trace_raw[::step], 'k-', lw=0.5, alpha=0.3)
            self.ax_trace.plot(self.time_axis[::step], self.trace_processed[::step], 'g-', lw=0.5)
        else:
            self.ax_trace.plot(self.time_axis[::step], self.trace_raw[::step], 'g-', lw=0.5)
            
        self.ax_trace.set_title("Intensity Trace")
        self.ax_trace.set_xlabel("Time (s)")
        self.ax_trace.grid(True)

        self.ax_acf.clear()
        if self.acf_lags is not None and len(self.acf_lags) > 0:
            plot_step = 1
            if not self.use_log_bin_var.get() and len(self.acf_lags) > 5000:
                plot_step = len(self.acf_lags) // 2000
            
            if self.use_log_bin_var.get():
                self.ax_acf.errorbar(self.acf_lags, self.acf_G, yerr=self.acf_sigma, 
                                     fmt='bo', ms=3, capsize=2, alpha=0.5, label='Data')
            else:
                self.ax_acf.plot(self.acf_lags[::plot_step], self.acf_G[::plot_step], 
                                 'b.', ms=2, alpha=0.3, label='Data')
            self.ax_acf.set_xscale('log')

        if fit_curve_lags is not None:
            self.ax_acf.plot(fit_curve_lags, fit_curve_G, 'r-', lw=2, label='Fit')

        t_min = self.range_min_var.get()
        t_max = self.range_max_var.get()
        self.drag_lines['min'] = self.ax_acf.axvline(t_min, color='orange', ls='--', picker=5)
        self.drag_lines['max'] = self.ax_acf.axvline(t_max, color='orange', ls='--', picker=5)

        self.ax_acf.set_title("Autocorrelation G(tau)")
        self.ax_acf.set_xlabel("Lag Time (s)")
        self.ax_acf.grid(True, which="both", alpha=0.4)
        self.ax_acf.legend()
        self.canvas.draw()

    def run_fitting(self):
        if self.acf_lags is None: return
        
        t_min = self.range_min_var.get()
        t_max = self.range_max_var.get()
        mask = (self.acf_lags >= t_min) & (self.acf_lags <= t_max)
        
        x_fit = self.acf_lags[mask]
        y_fit = self.acf_G[mask]
        sigma_fit = self.acf_sigma[mask]
        
        if self.n_segments_var.get() == 1 and not self.use_log_bin_var.get():
            sigma_fit = None 
            absolute_sigma = False
        else:
            if len(sigma_fit) > 0:
                mean_s = np.mean(sigma_fit[sigma_fit > 0]) if np.any(sigma_fit > 0) else 1.0
                sigma_fit[sigma_fit == 0] = mean_s
            absolute_sigma = True
        
        if len(y_fit) < 5:
            messagebox.showerror("Error", "Not enough points")
            return

        # Prepare Params
        param_order = ['N', 'D', 'w0', 'wz', 'T', 'tau_trip', 'y0']
        
        vals = {}
        fixed = {}
        bounds_low = []
        bounds_high = []
        free_keys = []
        p0 = []

        for k in param_order:
            v = self.p_vars[k]
            val = v['val'].get() # 初期値
            is_fix = v['fix'].get()
            vals[k] = val
            fixed[k] = is_fix
            
            if not is_fix:
                free_keys.append(k)
                
                # --- Bounds Logic & Initial Value Clamping ---
                # CheckboxがONなら入力値を、OFFならデフォルト(0-inf)を使用
                if v['b_on'].get():
                    try: mn = float(v['min'].get())
                    except: mn = -np.inf
                    try:
                        s_max = v['max'].get()
                        mx = np.inf if (not s_max or s_max.lower() == 'inf') else float(s_max)
                    except: mx = np.inf
                else:
                    # Default Safe Bounds
                    if k == 'y0': mn = -np.inf
                    elif k == 'T': mn = 0.0; mx = 1.0 # Tは物理的に0-1
                    else: mn = 0.0
                    mx = np.inf
                    if k == 'T' and not v['b_on'].get(): mx = 1.0 # T Default

                # ★ Clamping: 初期値が範囲外なら強制的に内側へ入れる
                # これにより「範囲外のパラメータ」が入力されても計算がスタートできる
                epsilon = 1e-9
                if val < mn: val = mn + epsilon
                if val > mx: val = mx - epsilon
                
                p0.append(val)
                bounds_low.append(mn)
                bounds_high.append(mx)

        if len(free_keys) == 0:
            self.finalize_fit(vals)
            return

        def wrapper(t, *args):
            current = vals.copy()
            for i, key in enumerate(free_keys):
                current[key] = args[i]
            return model.fcs_standard_model(
                t, current['N'], current['D'], current['w0'], current['wz'],
                current['T'], current['tau_trip'], current['y0']
            )

        try:
            popt, _ = curve_fit(wrapper, x_fit, y_fit, p0=p0, sigma=sigma_fit, 
                                absolute_sigma=absolute_sigma, bounds=(bounds_low, bounds_high), maxfev=3000)
            
            for i, key in enumerate(free_keys):
                vals[key] = popt[i]
                self.p_vars[key]['val'].set(round(popt[i], 6))
            
            self.finalize_fit(vals)
            
        except Exception as e:
            messagebox.showerror("Fit Error", str(e))

    def finalize_fit(self, vals):
        T_val = vals['T']
        g0 = (1.0 / vals['N']) * (1.0 / (1.0 - T_val)) + vals['y0'] if T_val < 1.0 else 0
        
        res_str = (
            f"D = {vals['D']:.2f}\n"
            f"N = {vals['N']:.3f}\n"
            f"T = {vals['T']*100:.1f} %\n"
            f"tau_T = {vals['tau_trip']*1e6:.1f} us\n"
            f"G(0) = {g0:.4f}\n"
            f"w0 = {vals['w0']:.3f}"
        )
        self.res_txt.set(res_str)
        
        smooth = np.logspace(np.log10(min(self.acf_lags[self.acf_lags>0])), np.log10(max(self.acf_lags)), 200)
        smooth_G = model.fcs_standard_model(
            smooth, vals['N'], vals['D'], vals['w0'], vals['wz'],
            vals['T'], vals['tau_trip'], vals['y0']
        )
        self.plot_graphs(smooth, smooth_G)

    def on_click(self, event):
        if event.inaxes != self.ax_acf: return
        if not self.drag_lines: return
        if event.xdata is None: return
        min_dist = float('inf'); target = None
        for name, line in self.drag_lines.items():
            lx = line.get_xdata()[0]
            dist = abs(np.log10(event.xdata) - np.log10(lx))
            if dist < 0.2:
                if dist < min_dist: min_dist = dist; target = name
        if target: self.dragging_item = target

    def on_motion(self, event):
        if self.dragging_item is None: return
        if event.xdata is None: return
        new_val = event.xdata
        if self.dragging_item == 'min': self.range_min_var.set(new_val)
        else: self.range_max_var.set(new_val)
        line = self.drag_lines[self.dragging_item]
        line.set_xdata([new_val, new_val])
        self.canvas.draw_idle()

    def on_release(self, event): self.dragging_item = None
    def save_graph(self):
        path = filedialog.asksaveasfilename(defaultextension=".png", filetypes=[("PNG", "*.png")])
        if path:
            try: self.fig.savefig(path, dpi=300); messagebox.showinfo("Saved", path)
            except Exception as e: messagebox.showerror("Error", str(e))

if __name__ == "__main__":
    root = tk.Tk()
    app = FCSApp(root)
    root.mainloop()