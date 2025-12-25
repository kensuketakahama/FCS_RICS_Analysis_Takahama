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
        self.root.title("FCS Analysis App v7.0 (All Points / No Binning)")
        self.root.geometry("1300x950")

        # Data
        self.raw_stack = None
        self.trace_raw = None
        self.trace_processed = None
        self.time_axis = None
        
        # ACF Results
        self.acf_lags = None
        self.acf_G = None
        self.acf_sigma = None
        
        # --- Parameters ---
        self.pixel_time_var = tk.DoubleVar(value=2.0)
        
        # Analysis Settings
        self.use_detrend_var = tk.BooleanVar(value=False)
        self.detrend_cutoff_var = tk.DoubleVar(value=5.0)
        
        # ★ New Options ★
        self.use_log_bin_var = tk.BooleanVar(value=False) # デフォルトOFF (全点使用)
        self.n_segments_var = tk.IntVar(value=1)          # デフォルト1 (全データ一括)

        # Beam Parameters
        self.w0_var = tk.DoubleVar(value=cfg.W0)
        self.wz_var = tk.DoubleVar(value=cfg.WZ)
        self.fix_w0_var = tk.BooleanVar(value=True)
        self.fix_wz_var = tk.BooleanVar(value=True)

        # Fitting Initial Guesses
        self.fit_N_var = tk.DoubleVar(value=1.0)
        self.fit_D_var = tk.DoubleVar(value=414.0)
        self.fit_T_var = tk.DoubleVar(value=0.1)
        self.fit_trip_var = tk.DoubleVar(value=1e-6)
        self.fit_y0_var = tk.DoubleVar(value=0.0)

        # Fix Flags
        self.fix_N_var = tk.BooleanVar(value=False)
        self.fix_D_var = tk.BooleanVar(value=False)
        self.fix_T_var = tk.BooleanVar(value=False)
        self.fix_trip_var = tk.BooleanVar(value=False)
        self.fix_y0_var = tk.BooleanVar(value=False)

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
        
        # Binning & Segments
        b_frame = ttk.LabelFrame(panel, text="ACF Calculation Mode")
        b_frame.pack(fill=tk.X, pady=5)
        
        # Segments
        f_seg = ttk.Frame(b_frame); f_seg.pack(fill=tk.X, pady=2)
        ttk.Label(f_seg, text="Segments (Avg):").pack(side=tk.LEFT)
        ttk.Entry(f_seg, textvariable=self.n_segments_var, width=5).pack(side=tk.LEFT, padx=5)
        ttk.Label(f_seg, text="(1 = Full Trace)").pack(side=tk.LEFT)
        
        # Log Binning
        ttk.Checkbutton(b_frame, text="Use Log Binning (Reduce Pts)", variable=self.use_log_bin_var,
                        command=self.update_analysis).pack(anchor="w")
        
        ttk.Button(b_frame, text="Update ACF", command=self.update_analysis).pack(fill=tk.X, pady=5)

        # Beam
        ttk.Label(panel, text="Beam Params:", font=("small")).pack(anchor="w", pady=(5,0))
        f2 = ttk.Frame(panel); f2.pack(fill=tk.X, pady=2)
        ttk.Label(f2, text="w0 (um):").pack(side=tk.LEFT)
        ttk.Entry(f2, textvariable=self.w0_var, width=6).pack(side=tk.LEFT)
        ttk.Checkbutton(f2, text="Fix", variable=self.fix_w0_var).pack(side=tk.LEFT)
        
        f3 = ttk.Frame(panel); f3.pack(fill=tk.X, pady=2)
        ttk.Label(f3, text="wz (um):").pack(side=tk.LEFT)
        ttk.Entry(f3, textvariable=self.wz_var, width=6).pack(side=tk.LEFT)
        ttk.Checkbutton(f3, text="Fix", variable=self.fix_wz_var).pack(side=tk.LEFT)

        ttk.Separator(panel, orient="horizontal").pack(fill=tk.X, pady=10)

        # 3. Fit Params
        ttk.Label(panel, text="3. Fit Initial / Fix", font=("bold")).pack(anchor="w", pady=5)
        
        def make_row(lbl, var, fix_var):
            f = ttk.Frame(panel); f.pack(fill=tk.X, pady=1)
            ttk.Label(f, text=lbl, width=8).pack(side=tk.LEFT)
            ttk.Entry(f, textvariable=var, width=8).pack(side=tk.LEFT, padx=2)
            ttk.Checkbutton(f, text="Fix", variable=fix_var).pack(side=tk.LEFT)

        make_row("N:", self.fit_N_var, self.fix_N_var)
        make_row("D:", self.fit_D_var, self.fix_D_var)
        make_row("T:", self.fit_T_var, self.fix_T_var)
        make_row("tau_T:", self.fit_trip_var, self.fix_trip_var)
        make_row("y0:", self.fit_y0_var, self.fix_y0_var)

        ttk.Button(panel, text="Run Fitting (All Points)", command=self.run_fitting).pack(fill=tk.X, pady=10)

        # Results
        res_grp = ttk.LabelFrame(panel, text="Final Results")
        res_grp.pack(fill=tk.X, pady=5)
        self.res_txt = tk.StringVar(value="")
        ttk.Label(res_grp, textvariable=self.res_txt, justify="left", font=("Consolas", 10)).pack(padx=5, pady=5, anchor="w")
        
        ttk.Separator(panel, orient="horizontal").pack(fill=tk.X, pady=10)
        ttk.Button(panel, text="Save Graph Image", command=self.save_graph).pack(fill=tk.X, pady=5)

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
            cutoff_ms = self.detrend_cutoff_var.get()
            cutoff_sec = cutoff_ms * 1e-3
            self.trace_processed = prep.detrend_1d_trace(self.trace_raw, pt_sec, cutoff_sec)
        else:
            self.trace_processed = self.trace_raw.copy()

        self.time_axis = np.arange(n) * pt_sec
        
        # --- ACF Calculation ---
        # User specified segments (Default 1 = Full Trace)
        n_seg = max(1, self.n_segments_var.get())
        
        lags_raw, G_raw, sem_raw = calc.calculate_segmented_acf(self.trace_processed, n_segments=n_seg)
        time_lags = lags_raw * pt_sec
        
        if self.use_log_bin_var.get():
            # Binning Mode
            self.acf_lags, self.acf_G, self.acf_sigma = calc.log_binning_weighted(time_lags, G_raw, sem_raw)
        else:
            # All Points Mode (No Binning)
            # ラグ0はショットノイズなので除外するのが一般的だが、ここではすべて保持
            self.acf_lags = time_lags
            self.acf_G = G_raw
            # 標準誤差 (n_seg=1 の場合はダミー1e-5が入っている)
            self.acf_sigma = sem_raw
        
        self.plot_graphs()

    def plot_graphs(self, fit_curve_lags=None, fit_curve_G=None):
        self.ax_trace.clear()
        step = max(1, len(self.trace_processed) // 10000)
        
        if self.use_detrend_var.get():
            self.ax_trace.plot(self.time_axis[::step], self.trace_raw[::step], 'k-', lw=0.5, alpha=0.3, label='Raw')
            self.ax_trace.plot(self.time_axis[::step], self.trace_processed[::step], 'g-', lw=0.5, label='Detrended')
            self.ax_trace.legend(loc='upper right')
        else:
            self.ax_trace.plot(self.time_axis[::step], self.trace_raw[::step], 'g-', lw=0.5)
            
        self.ax_trace.set_title(f"Intensity Trace")
        self.ax_trace.set_xlabel("Time (s)")
        self.ax_trace.grid(True)

        self.ax_acf.clear()
        if self.acf_lags is not None and len(self.acf_lags) > 0:
            # データ点数が多すぎる場合、描画時は間引く (Fittingは全点使う)
            plot_step = 1
            if not self.use_log_bin_var.get() and len(self.acf_lags) > 5000:
                plot_step = len(self.acf_lags) // 2000 # 最大2000点くらいに抑える
            
            # 誤差バーは重くなるので、Binning OFF時は点で表示
            if self.use_log_bin_var.get():
                self.ax_acf.errorbar(self.acf_lags, self.acf_G, yerr=self.acf_sigma, 
                                     fmt='bo', ms=3, capsize=2, alpha=0.5, label='Data')
            else:
                self.ax_acf.plot(self.acf_lags[::plot_step], self.acf_G[::plot_step], 
                                 'b.', ms=2, alpha=0.3, label='Data (All Points)')
                
            self.ax_acf.set_xscale('log')

        if fit_curve_lags is not None:
            self.ax_acf.plot(fit_curve_lags, fit_curve_G, 'r-', lw=2, label='Fit')

        t_min = self.range_min_var.get()
        t_max = self.range_max_var.get()
        self.drag_lines['min'] = self.ax_acf.axvline(t_min, color='orange', ls='--', picker=5, label='Range')
        self.drag_lines['max'] = self.ax_acf.axvline(t_max, color='orange', ls='--', picker=5)

        self.ax_acf.set_title("Autocorrelation G(tau)")
        self.ax_acf.set_xlabel("Lag Time (s)")
        self.ax_acf.set_ylabel("G(tau)")
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
        
        # Sigma処理
        # Seg=1 (全点) の場合、sigmaはダミーなので重みなし(=1.0)にする
        if self.n_segments_var.get() == 1 and not self.use_log_bin_var.get():
            sigma_fit = None # 重みなし最小二乗法
            absolute_sigma = False
        else:
            if len(sigma_fit) > 0:
                mean_sigma = np.mean(sigma_fit[sigma_fit > 0]) if np.any(sigma_fit > 0) else 1.0
                sigma_fit[sigma_fit == 0] = mean_sigma
            absolute_sigma = True
        
        if len(y_fit) < 5:
            messagebox.showerror("Error", "Not enough points in range")
            return
            
        vals = {
            'N': self.fit_N_var.get(), 'D': self.fit_D_var.get(),
            'w0': self.w0_var.get(), 'wz': self.wz_var.get(),
            'T': self.fit_T_var.get(), 'tau_trip': self.fit_trip_var.get(),
            'y0': self.fit_y0_var.get()
        }
        
        fixed = {
            'N': self.fix_N_var.get(), 'D': self.fix_D_var.get(),
            'w0': self.fix_w0_var.get(), 'wz': self.fix_wz_var.get(),
            'T': self.fix_T_var.get(), 'tau_trip': self.fix_trip_var.get(),
            'y0': self.fix_y0_var.get()
        }
        
        param_order = ['N', 'D', 'w0', 'wz', 'T', 'tau_trip', 'y0']
        free_keys = [k for k in param_order if not fixed[k]]
        p0 = [vals[k] for k in free_keys]
        
        b_min = []; b_max = []
        for k in free_keys:
            if k == 'y0': b_min.append(-np.inf)
            elif k == 'T': b_min.append(0); b_max.append(1.0); continue
            else: b_min.append(0)
            b_max.append(np.inf)

        def wrapper(t, *args):
            current = vals.copy()
            for i, k in enumerate(free_keys): current[k] = args[i]
            return model.fcs_standard_model(
                t, current['N'], current['D'], current['w0'], current['wz'],
                current['T'], current['tau_trip'], current['y0']
            )

        try:
            if len(free_keys) > 0:
                # フィッティング実行
                # sigma_fit=None なら重みなし
                popt, _ = curve_fit(wrapper, x_fit, y_fit, p0=p0, sigma=sigma_fit, 
                                    absolute_sigma=absolute_sigma, bounds=(b_min, b_max), maxfev=3000)
                for i, k in enumerate(free_keys):
                    vals[k] = popt[i]
                    if k=='N': self.fit_N_var.set(round(vals[k], 4))
                    if k=='D': self.fit_D_var.set(round(vals[k], 2))
                    if k=='w0': self.w0_var.set(round(vals[k], 4))
                    if k=='wz': self.wz_var.set(round(vals[k], 4))
                    if k=='T': self.fit_T_var.set(round(vals[k], 4))
                    if k=='tau_trip': self.fit_trip_var.set(vals[k])
                    if k=='y0': self.fit_y0_var.set(round(vals[k], 6))
            
            T_val = vals['T']
            if T_val < 1.0: g0_theo = (1.0 / vals['N']) * (1.0 / (1.0 - T_val)) + vals['y0']
            else: g0_theo = 0
            
            res_str = (
                f"D = {vals['D']:.2f} um^2/s\n"
                f"N = {vals['N']:.3f}\n"
                f"T = {vals['T']*100:.1f} %\n"
                f"tau_T = {vals['tau_trip']*1e6:.1f} us\n"
                f"y0 = {vals['y0']:.2e}\n"
                f"G(0) = {g0_theo:.4f}"
            )
            self.res_txt.set(res_str)
            
            smooth_lags = np.logspace(np.log10(min(self.acf_lags[self.acf_lags>0])), np.log10(max(self.acf_lags)), 200)
            smooth_G = model.fcs_standard_model(
                smooth_lags, vals['N'], vals['D'], vals['w0'], vals['wz'],
                vals['T'], vals['tau_trip'], vals['y0']
            )
            self.plot_graphs(smooth_lags, smooth_G)
            
        except Exception as e: messagebox.showerror("Fit Error", str(e))

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