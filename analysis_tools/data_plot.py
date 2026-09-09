import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.ticker as mticker
import matplotlib.tri as mtri
from pathlib import Path
import matplotlib.tri as tri
import importlib
# importlib.reload(data_calculation)
from data_calculation import DataCal 
from modelData_processor import ModelDataProcessor
import joblib
import tensorflow as tf
from matplotlib.backends.backend_pdf import PdfPages
from tqdm import tqdm
import time
import pandas as pd
import cmocean

class DataPlot:
    # ---------- helpers ----------

    plt.rcParams.update({
        "font.size": 20,
        "axes.titlesize": 18,
        "axes.labelsize": 18,
        "xtick.labelsize": 16,
        "ytick.labelsize": 16,
        "legend.fontsize": 16,
        "figure.titlesize": 20,
    })

    @staticmethod
    def _resolve_labels_cmap(component: str, cmap_name: str):
        comp = str(component).lower()
        if comp == 'x':
            comp_label = 'X-Force'
            title_text = 'Gradient of Radiation X-Force'
            # cmap = plt.cm.get_cmap(cmap_name).copy()
            cmap = cmocean.cm.balance.copy()
        elif comp == 'y':
            comp_label = 'Y-Force'
            title_text = 'Gradient of Radiation Y-Force'
            # cmap = plt.cm.get_cmap(cmap_name).copy()
            cmap = cmocean.cm.balance.copy()
        elif comp == 'hsig':
            comp_label = 'Hsig'
            title_text = 'Significant Wave Height'
            # cmap = plt.cm.get_cmap('viridis' if cmap_name == 'coolwarm' else cmap_name).copy()
            # cmap = cmocean.cm.dense.copy()
            cmap = cmocean.cm.haline.copy()
            
        else:
            raise ValueError("component must be 'x', 'y', or 'hsig'.")
        return comp, comp_label, title_text, cmap

    @staticmethod
    def _resolve_norm(vals, cbar_range):
        if cbar_range is None:
            vmin = np.nanmin(vals)
            vmax = np.nanmax(vals)
            if (not np.isfinite(vmin)) or (not np.isfinite(vmax)) or vmin == vmax:
                vmin, vmax = 0.0, 1.0
        else:
            vmin, vmax = cbar_range
        return mcolors.Normalize(vmin=vmin, vmax=vmax)

    @staticmethod
    def _save_fig(fig, save_path, dpi):
        if save_path is not None:
            p = Path(save_path)
            p.parent.mkdir(parents=True, exist_ok=True)
            fig.savefig(p, dpi=dpi, bbox_inches='tight')

    # ---------- API 1: scatter ----------
    @staticmethod
    def plot_scatter(
        x, y, values,
        *,
        component='x',              # 'x' | 'y' | 'hsig'
        title_prefix='Radiation Force',
        cmap_name='coolwarm',
        point_size=12,
        equal_axes=True,
        cbar_range=None,
        show=True,
        save_path=None,
        dpi=200,
        tight_layout=True,
        return_fig_ax=True,
        ax=None
    ):
        x = np.asarray(x); y = np.asarray(y); vals = np.asarray(values, dtype=float)
        if x.shape != y.shape or x.shape != vals.shape:
            raise ValueError("x, y, and values must have the same shape.")

        comp, comp_label, title_text, cmap = DataPlot._resolve_labels_cmap(component, cmap_name)
        norm = DataPlot._resolve_norm(vals, cbar_range)
        cmap.set_bad('black')

        created_fig = False
        if ax is None:
            fig, ax = plt.subplots(figsize=(8, 7))
            created_fig = True
        else:
            fig = ax.figure

        sc = ax.scatter(x, y, c=vals, s=point_size, cmap=cmap, norm=norm)
        cbar = fig.colorbar(sc, ax=ax); cbar.set_label(comp_label)

        ax.set_title(f"{title_text if comp=='hsig' else title_prefix} ({comp_label}) — Scatter")
        ax.set_xlabel("x"); ax.set_ylabel("y")
        if equal_axes:
            ax.set_aspect('equal', adjustable='box')
            ax.autoscale(enable=True, tight=True)
        ax.grid(True, alpha=0.2)

        if tight_layout:
            plt.tight_layout()
        if save_path:
            DataPlot._save_fig(fig, save_path, dpi)

        if show:
            plt.show()
        else:
            plt.close(fig)

        return (fig, ax) if return_fig_ax else None

    # ---------- API 2: triangulation ----------
    @staticmethod
    def plot_triangulation(
        x, y, values,
        *,
        component='x',
        title_prefix='Radiation Force',
        cmap_name='coolwarm',
        shading='flat',
        equal_axes=True,
        cbar_range=None,
        show=True,
        save_path=None,
        dpi=200,
        tight_layout=True,
        return_fig_ax=True,
        ax=None,
        annotate_extreme=None,         
        annotate_k=1,                   
        annotate_text=True,             
        annotate_scatter_kwargs=None,   
        annotate_text_kwargs=None       
    ):

        x = np.asarray(x); y = np.asarray(y); vals = np.asarray(values, dtype=float)
        if x.shape != y.shape or x.shape != vals.shape:
            raise ValueError("x, y, and values must have the same shape.")

        comp, comp_label, title_text, cmap = DataPlot._resolve_labels_cmap(component, cmap_name)
        norm = DataPlot._resolve_norm(vals, cbar_range)
        cmap.set_bad('black')

        tri = mtri.Triangulation(x, y)
        vals_masked = np.ma.masked_invalid(vals)

        created_fig = False
        if ax is None:
            fig, ax = plt.subplots(figsize=(8, 7))
            created_fig = True
        else:
            fig = ax.figure


        tc = ax.tripcolor(tri, vals_masked, cmap=cmap, norm=norm, shading=shading)
        cbar = fig.colorbar(tc, ax=ax)
        cbar.set_label(comp_label)

        # --- FIX: ticks follow cbar_range (or norm), not raw data min/max ---
        finite_vals = vals[np.isfinite(vals)]

        if cbar_range is not None:
            vmin, vmax = float(cbar_range[0]), float(cbar_range[1])
        elif finite_vals.size > 0:
            vmin, vmax = float(finite_vals.min()), float(finite_vals.max())
        else:
            vmin, vmax = 0.0, 1.0

        cbar.set_ticks(np.linspace(vmin, vmax, 6))
        ax.set_title(f"{title_text if comp=='hsig' else title_prefix} ({comp_label}) — Triangulated", pad=25)
        ax.set_xlabel("x"); ax.set_ylabel("y")
        if equal_axes:
            ax.set_aspect('equal', adjustable='box')
            ax.autoscale(enable=True, tight=True)
        ax.grid(True, alpha=0.2)

        if annotate_extreme in ('max', 'min') and finite_vals.size > 0:
            vals_arr = vals.copy()
            finite_mask = np.isfinite(vals_arr)
            idx_all = np.where(finite_mask)[0]
            vals_finite = vals_arr[finite_mask]

            k = max(1, int(annotate_k))
            k = min(k, vals_finite.size)

            if annotate_extreme == 'max':
                part = np.argpartition(vals_finite, -k)[-k:]
                order = np.argsort(vals_finite[part])[::-1]
                chosen_rel = part[order]
            else: 
                part = np.argpartition(vals_finite, k-1)[:k]
                order = np.argsort(vals_finite[part]) 
                chosen_rel = part[order]


            chosen_abs = idx_all[chosen_rel]

            if annotate_scatter_kwargs is None:
                annotate_scatter_kwargs = dict(s=150, marker="*", edgecolor='white', linewidths=1.5, zorder=6)
            if 'color' not in annotate_scatter_kwargs:
                annotate_scatter_kwargs['color'] = 'blue' if annotate_extreme == 'max' else 'black'

            if annotate_text_kwargs is None:
                annotate_text_kwargs = dict(
                    xytext=(6, 6), textcoords='offset points',
                    fontsize=12, weight='bold', color=annotate_scatter_kwargs.get('color', 'blue'),
                    bbox=dict(boxstyle='round,pad=0.2', fc='white', alpha=0.7),
                    arrowprops=dict(arrowstyle='->', lw=1)
                )

            label_once = f"Top {k} {'Max' if annotate_extreme=='max' else 'Min'}"

            for j, idx in enumerate(chosen_abs, 1):
                xx, yy, vv = x[idx], y[idx], vals[idx]
                scatter_label = (label_once if (j == 1 and annotate_text) else None)
                ax.scatter(xx, yy, label=None, **annotate_scatter_kwargs)
                if annotate_text:
                    ax.annotate(f"{vv:.3g}", xy=(xx, yy), **annotate_text_kwargs)

            if annotate_text:
                handles, labels = ax.get_legend_handles_labels()
                if any(labels):
                    ax.legend(loc="best")


        if tight_layout:
            plt.tight_layout()
            
        if save_path:
            DataPlot._save_fig(fig, save_path, dpi)

        if show:
            plt.show()
        else:
            plt.close(fig)

        return (fig, ax) if return_fig_ax else None

    @staticmethod
    def select_row(test_df, wind_val, wind_dir, wave_h, wave_dir):
        mask = (
            (test_df["Wind Value"] == wind_val) &
            (test_df["Wind Direction"] == wind_dir) &
            (test_df["Wave Height"] == wave_h) &
            (test_df["Wave Direction"] == wave_dir)
        )
        matches = test_df[mask]
        if matches.empty:
            raise ValueError("No scenario found matching the given parameters.")
      
        return matches.iloc[0]

    @staticmethod
    def invert_with_legacy_scaler(o_res, target_scaler, n_feat=349):
        o = np.asarray(o_res)
        if o.ndim == 2 and o.shape[0] == 1:
            o = o.T                     # (N,1)
        elif o.ndim == 1:
            o = o[:, None]              # (N,1)
        elif o.ndim == 2 and o.shape[1] == 1:
            pass                        
        else:
            raise ValueError(f"{o.shape}")
        N = o.shape[0]
        o_expanded = np.repeat(o, n_feat, axis=1)   # (N,349)
        y_inv_full = target_scaler.inverse_transform(o_expanded)  # (N,349)
        pred_valid = y_inv_full[:, 0]   # (N,)
        return pred_valid
    
    @staticmethod
    def compute_vmin_vmax(arr, comp, cbar_range, outlier_percentile, c_min):
        if cbar_range is not None: 
            return float(cbar_range[0]), float(cbar_range[1])
        
        temp_arr = arr[np.isfinite(arr)]
        
        if c_min is not None and comp in ("x", "y") and len(temp_arr) > 0:
            temp_arr = temp_arr[temp_arr > c_min]
        
        if len(temp_arr) == 0: 
            return (-1.0, 1.0) if comp in ("x", "y") else (0.0, 1.0)
        
        # --- FIX: Handle None case for outlier_percentile ---
        if outlier_percentile is None:
            if comp in ("x", "y"):
                vmax = float(np.nanmax(np.abs(temp_arr)))
                vmin = -vmax
            else:
                vmin = float(np.nanmin(temp_arr))
                vmax = float(np.nanmax(temp_arr))
        else:
            # Standard percentile logic
            lo, hi = outlier_percentile
            q_low, q_high = np.nanpercentile(temp_arr, [lo, hi])
            
            if comp in ("x", "y"):
                vmax = float(max(abs(q_low), abs(q_high)))
                vmin = -vmax
            else:
                vmin, vmax = float(q_low), float(q_high)
                
        # Final guard: ensure vmax > vmin
        if vmin == vmax:
            vmin -= 0.1
            vmax += 0.1

        return vmin, vmax

    # ------------------------------------------------------------------ #
    #  Transect helpers                                                  #
    # ------------------------------------------------------------------ #
    @staticmethod
    def segment_indices(vector):
        """Split an ordered grid vector into transect (cross-grid) segments.

        The DUCK grid points are stored row-by-row: within one transect the
        coordinate increases monotonically, and it *drops* when the grid wraps
        to the next transect.  Each place where ``vector[i+1] < vector[i]``
        therefore marks the last node of a segment.

        Returns
        -------
        list[(start, end)]
            Inclusive index ranges, one per transect.  ``vector[start:end+1]``
            selects the nodes of that transect.
        """
        vector = np.asarray(vector)
        marked_indices = []
        for i in range(len(vector) - 1):
            if vector[i + 1] < vector[i]:
                marked_indices.append(i)
        marked_indices.append(len(vector) - 1)

        segments = []
        start_idx = 0
        for end_idx in marked_indices:
            segments.append((start_idx, end_idx))   # end inclusive
            start_idx = end_idx + 1
        return segments

    @staticmethod
    def _compute_scenario_error_fields(
        test_df, wind_val, wind_dir, wave_h, wave_dir,
        xp_vector, yp_vector, component, model_dir,
    ):
        """Run the DON for one scenario and return the truth / prediction /
        error fields on the full grid.  Shared by ``plot_duck_scenario`` and
        ``plot_transect_error`` so the prediction logic lives in one place.
        """
        row = DataPlot.select_row(test_df, wind_val, wind_dir, wave_h, wave_dir)

        hsig_data_coarse = np.array(row["Hsig Data"])
        forces_data_coarse = np.array(row["Forces Data"])

        comp = component.lower()
        if comp == "x":
            true_vals = forces_data_coarse[:, 0]
        elif comp == "y":
            true_vals = forces_data_coarse[:, 1]
        elif comp == "hsig":
            true_vals = hsig_data_coarse
        else:
            raise ValueError("component must be 'x', 'y', or 'hsig'")

        valid_mask = hsig_data_coarse != -9

        b_key = list(row["key"])
        x_wind, y_wind = DataCal.convert_wind_components(b_key[0], b_key[1])
        cos_wave_dir = np.cos(np.radians(b_key[3]))
        sin_wave_dir = np.sin(np.radians(b_key[3]))
        b_test_input = [x_wind, y_wind, b_key[2], cos_wave_dir, sin_wave_dir]

        valid_idx = np.where(valid_mask)[0]
        valid_coords = np.column_stack((xp_vector[valid_idx], yp_vector[valid_idx]))
        t_test = ModelDataProcessor.trunk_scale_and_reload(
            valid_coords,
            scaler_path_0=f"{model_dir}/scaler_col_0.pkl",
            scaler_path_1=f"{model_dir}/scaler_col_1.pkl",
        )

        b_batch = np.array([b_test_input] * len(t_test))
        b_test_scaled = ModelDataProcessor.load_and_scale_new_data(
            b_batch,
            scaler_path_0_1=f"{model_dir}/scaler_factor_0_1.pkl",
            scaler_path_2=f"{model_dir}/scaler_factor_2.pkl",
        )

        load_model = tf.keras.models.load_model(model_dir, compile=False)
        o_res = load_model([b_test_scaled, t_test])
        target_scaler = joblib.load(f"{model_dir}/target_scaler.pkl")
        pred_valid = target_scaler.inverse_transform(o_res).ravel()

        pred_full = np.full_like(true_vals, np.nan, dtype=float)
        pred_full[valid_idx] = pred_valid

        abs_error = np.abs(pred_full - true_vals)
        abs_error[~valid_mask] = np.nan

        l2_norm_true_scalar = np.linalg.norm(true_vals[valid_mask])
        relative_l2_error_field = abs_error / (l2_norm_true_scalar + 1e-10)

        return {
            "row": row,
            "comp": comp,
            "true_vals": true_vals,
            "pred_full": pred_full,
            "valid_mask": valid_mask,
            "abs_error": abs_error,
            "relative_l2_error_field": relative_l2_error_field,
        }

    @staticmethod
    def plot_transect_error(
        test_df,
        wind_val,
        wind_dir,
        wave_h,
        wave_dir,
        xp_vector,
        yp_vector,
        segment,                       # int or list[int]: which transect(s) to plot
        component="x",                 # "x" | "y" | "hsig"
        seg_vector="y",                # which coord defines transects: "x"->xp, "y"->yp
        error_type="relative_l2",      # "relative_l2" | "absolute"
        along="auto",                  # profile x-axis: "x" | "y" | "dist" | "auto"
        text_size=18,
        model_dir="/workspace/shukaic/wave_DON_WorkingFolder/model_DUCK_xforces",
        show_map=True,
        truth_color=None,              # fixed color for truth profile; None -> per-transect
        pred_color=None,               # fixed color for prediction profile; None -> per-transect
        err_cbar_max=99,               # percentile used for the error-field map colorbar max
        err_decimals=6,                # decimal places for error colorbar / profile tick labels
        save_path=None,
        plot=True,
    ):
        """Plot the error profile along one or more cross-grid transects.

        The grid is split into transects with :meth:`segment_indices` applied to
        ``seg_vector`` (``yp_vector`` by default).  For each selected ``segment``
        (an integer position into that list, or a list of them) the truth /
        prediction and the error are plotted as 1-D profiles along the transect.

        Parameters
        ----------
        segment : int | list[int]
            Index (or indices) of the transect(s) to plot.
        seg_vector : {"x", "y"}
            Coordinate vector used to detect transect boundaries.
        error_type : {"relative_l2", "absolute"}
            Which error field to show in the lower profile.
        along : {"x", "y", "dist", "auto"}
            What to use as the profile's horizontal axis.  ``"auto"`` picks the
            coordinate that varies most along the transect; ``"dist"`` uses the
            cumulative along-transect distance.
        """
        fields = DataPlot._compute_scenario_error_fields(
            test_df, wind_val, wind_dir, wave_h, wave_dir,
            xp_vector, yp_vector, component, model_dir,
        )
        comp = fields["comp"]
        true_vals = fields["true_vals"]
        pred_full = fields["pred_full"]
        valid_mask = fields["valid_mask"]

        if error_type == "absolute":
            err_field = fields["abs_error"]
            err_label = "Absolute Error"
        else:
            err_field = fields["relative_l2_error_field"]
            err_label = "Relative $L_2$ Error"

        comp_label = {
            "x": "X-Force (N/m$^2$)",
            "y": "Y-Force (N/m$^2$)",
            "hsig": "Hsig (m)",
        }[comp]

        base_vec = yp_vector if seg_vector.lower() == "y" else xp_vector
        segments = DataPlot.segment_indices(base_vec)

        seg_list = [segment] if np.isscalar(segment) else list(segment)
        for s in seg_list:
            if s < 0 or s >= len(segments):
                raise IndexError(
                    f"segment {s} out of range (found {len(segments)} transects)"
                )

        ncol = 2 if show_map else 1
        fig = plt.figure(figsize=(11 * ncol, 9))
        if show_map:
            ax_map = fig.add_subplot(1, ncol, 1)
            ax_top = fig.add_subplot(2, ncol, 2)
            ax_bot = fig.add_subplot(2, ncol, 4)
        else:
            ax_map = None
            ax_top = fig.add_subplot(2, 1, 1)
            ax_bot = fig.add_subplot(2, 1, 2)

        if show_map:
            triangulation = tri.Triangulation(xp_vector, yp_vector)
            err_masked = np.where(valid_mask, err_field, np.nan)
            finite = err_masked[np.isfinite(err_masked)]
            vmax_map = float(np.nanpercentile(finite, err_cbar_max)) if finite.size else 1.0
            vmax_map = max(vmax_map, 1e-9)
            cmap = cmocean.cm.amp.copy()
            cmap.set_bad("black")
            tcm = ax_map.tripcolor(
                triangulation, err_masked, cmap=cmap,
                norm=mcolors.Normalize(vmin=0, vmax=vmax_map), shading="flat",
            )
            cb = fig.colorbar(tcm, ax=ax_map)
            cb.set_label(err_label, fontsize=text_size)
            cb.ax.tick_params(labelsize=text_size - 4)
            # Explicit evenly-spaced ticks so the top tick equals vmax_map (largest shown value).
            cb.set_ticks(np.linspace(0, vmax_map, 6))
            cb.ax.yaxis.set_major_formatter(mticker.FormatStrFormatter(f"%.{err_decimals}f"))
            ax_map.set_title("Error field & selected transect(s)", fontsize=text_size)
            ax_map.set_xlabel("x", fontsize=text_size)
            ax_map.set_ylabel("y", fontsize=text_size)
            ax_map.tick_params(labelsize=text_size - 4)

        colors = plt.cm.tab10(np.linspace(0, 1, max(len(seg_list), 1)))
        for s, color in zip(seg_list, colors):
            start, end = segments[s]
            idx = np.arange(start, end + 1)
            idx = idx[valid_mask[idx]]          # keep only wet nodes
            if idx.size == 0:
                print(f"Segment {s}: no valid (wet) nodes, skipped.")
                continue

            xs, ys = xp_vector[idx], yp_vector[idx]

            # choose horizontal axis
            mode = along.lower()
            if mode == "auto":
                mode = "x" if xs.ptp() >= ys.ptp() else "y"
            if mode == "x":
                t = xs; t_label = "x"
            elif mode == "y":
                t = ys; t_label = "y"
            else:  # cumulative distance
                d = np.sqrt(np.diff(xs) ** 2 + np.diff(ys) ** 2)
                t = np.concatenate([[0.0], np.cumsum(d)])
                t_label = "Along-transect distance"
            order = np.argsort(t)
            t = t[order]; idx = idx[order]

            lbl = f"transect {s}"
            t_color = truth_color if truth_color is not None else color
            p_color = pred_color if pred_color is not None else color
            ax_top.plot(t, true_vals[idx], "-", color=t_color, ms=3,
                        label=f"{lbl} truth")
            ax_top.plot(t, pred_full[idx], "--", color=p_color, ms=3,
                        markerfacecolor="none", label=f"{lbl} pred")
            ax_bot.plot(t, err_field[idx], "-", color=color, ms=3, label=lbl)

            if show_map:
                ax_map.plot(xs, ys, "-", color=color, lw=2.5, label=lbl)

        # ----- cosmetics -----
        ax_top.set_ylabel(comp_label, fontsize=text_size)
        ax_top.set_title("Truth vs. Prediction along transect", fontsize=text_size)
        ax_top.legend(fontsize=text_size - 6, ncol=2)
        ax_top.grid(alpha=0.3)
        ax_top.tick_params(labelsize=text_size - 4)

        ax_bot.set_ylabel(err_label, fontsize=text_size)
        ax_bot.set_xlabel(t_label, fontsize=text_size)
        ax_bot.set_title(f"{err_label} along transect", fontsize=text_size)
        ax_bot.yaxis.set_major_formatter(mticker.FormatStrFormatter(f"%.{err_decimals}f"))
        if len(seg_list) > 1:
            ax_bot.legend(fontsize=text_size - 6, ncol=2)
        ax_bot.grid(alpha=0.3)
        ax_bot.tick_params(labelsize=text_size - 4)

        if show_map:
            ax_map.legend(fontsize=text_size - 6)

        fig.suptitle(
            f"Wind {wind_val} m/s at {wind_dir}°, Wave H {wave_h} m at {wave_dir}°"
            f"  —  {comp_label.split(' ')[0]} transect error",
            fontsize=text_size + 2,
        )
        fig.tight_layout(rect=[0, 0, 1, 0.96])

        if save_path:
            out = f"{save_path}/{comp}_transect{'_'.join(map(str, seg_list))}_error.pdf"
            fig.savefig(out, dpi=300, bbox_inches="tight")
            print("saved:", out)
        # if plot:
        #     plt.show()

        return fig

    @staticmethod
    def plot_duck_scenario(
        test_df,
        wind_val,
        wind_dir,
        wave_h,
        wave_dir,
        xp_vector, 
        yp_vector,
        component="x",                 # "x" | "y" | "hsig"
        text_size=25,    
        model_dir="/workspace/shukaic/wave_DON_WorkingFolder/model_DUCK_xforces", 
        cbar_range=None,
        # tensor_DON=True,
        # outlier_percentile=(1, 99),
        outlier_percentile=None,
        clip_min = None,
        err_cbar_max = None,
        plot_params=None,
        save_path = None,
        plot = True

    ):
        # ----- pick the row -----
        row = DataPlot.select_row(test_df, wind_val, wind_dir, wave_h, wave_dir)

        # ----- unpack inputs -----
        hsig_data_coarse = np.array(row["Hsig Data"])
        forces_data_coarse = np.array(row["Forces Data"])

        comp = component.lower()
        if comp == "x":
            true_vals = forces_data_coarse[:, 0]
            label = "X-Force (N/m$^2$)"
            title_gt = "Ground Truth X-Forces"
            title_pred = "DON Model Prediction X-Forces"
            title_err = "DeepONet Relative $L_2$ Error (X-Forces)"
            cmap = cmocean.cm.balance.copy()
            vmax_fixed = 5
        elif comp == "y":
            true_vals = forces_data_coarse[:, 1]
            label = "Y-Force (N/m$^2$)"
            title_gt = "Ground Truth Y-Forces"
            title_pred = "DON Model Prediction Y-Forces"
            title_err = "DeepONet Relative $L_2$ Error (Y-Forces)"
            cmap = cmocean.cm.balance.copy()
            vmax_fixed = 5
        elif comp == "hsig":
            true_vals = hsig_data_coarse
            label = "Hsig (m)"
            title_gt = "Ground Truth Hsig"
            title_pred = "DON Model Prediction Hsig"
            title_err = "DeepONet Relative $L_2$ Error (Hsig)"
            cmap = cmocean.cm.haline.copy()
            vmax_fixed = None 
        else:
            raise ValueError("component must be 'x', 'y', or 'hsig'")

        valid_mask = hsig_data_coarse != -9
        triangulation = tri.Triangulation(xp_vector, yp_vector)

        b_key = list(row["key"])
        x_wind, y_wind = DataCal.convert_wind_components(b_key[0], b_key[1])
        cos_wave_dir = np.cos(np.radians(b_key[3]))
        sin_wave_dir = np.sin(np.radians(b_key[3]))
        b_test_input = [x_wind, y_wind, b_key[2], cos_wave_dir, sin_wave_dir]

        # trunk coords
        valid_idx = np.where(valid_mask)[0]
        valid_coords = np.column_stack((xp_vector[valid_idx], yp_vector[valid_idx]))
        t_test = ModelDataProcessor.trunk_scale_and_reload(
            valid_coords,
            scaler_path_0=f"{model_dir}/scaler_col_0.pkl",
            scaler_path_1=f"{model_dir}/scaler_col_1.pkl",
        )

        b_batch = np.array([b_test_input] * len(t_test))

        b_test_scaled = ModelDataProcessor.load_and_scale_new_data(
            b_batch,
            scaler_path_0_1=f"{model_dir}/scaler_factor_0_1.pkl",
            scaler_path_2=f"{model_dir}/scaler_factor_2.pkl",
        )

        out_dir = model_dir
        load_model = tf.keras.models.load_model(out_dir, compile=False)
        load_model.summary()
        init_time = time.time()
        o_res = load_model([b_test_scaled, t_test])  # (len(valid_idx), 1)

        end_time = time.time()

        target_scaler = joblib.load(f"{out_dir}/target_scaler.pkl")
        pred_valid = target_scaler.inverse_transform(o_res).ravel()
        
        predict_time = end_time - init_time
        print(f"Prediction time: {predict_time :.6f} seconds")

    
        pred_full = np.full_like(true_vals, np.nan, dtype=float)
        pred_full[valid_idx] = pred_valid

        true_masked = np.where(valid_mask, true_vals, np.nan)
        pred_masked = np.where(valid_mask, pred_full, np.nan)
        abs_error = np.abs(pred_full - true_vals)
        abs_error[~valid_mask] = np.nan


        l2_norm_true_scalar = np.linalg.norm(true_vals[valid_mask])
        relative_l2_error_field = abs_error / (l2_norm_true_scalar + 1e-10)

        dry_mask = np.where(~valid_mask, -1.0, np.nan)

       
        fig, axs = plt.subplots(2, 2, figsize=(22, 22))
        plt.subplots_adjust(wspace=0.4, hspace=0.15)
        cmap.set_bad("black")
        vmin_gt, vmax_gt = DataPlot.compute_vmin_vmax(true_masked, comp, cbar_range, outlier_percentile, clip_min)
        norm_gt = mcolors.Normalize(vmin=vmin_gt, vmax=vmax_gt)
        tc0 = axs[0, 0].tripcolor(triangulation, true_masked, cmap=cmap, norm=norm_gt, shading="flat")
        axs[0, 0].set_title(title_gt, fontsize=text_size)

        # (0,1) Prediction
        vmin_pr, vmax_pr = DataPlot.compute_vmin_vmax(pred_masked, comp, cbar_range, outlier_percentile, clip_min)
        norm_pr = mcolors.Normalize(vmin=vmin_pr, vmax=vmax_pr)
        axs[0, 1].tripcolor(triangulation, dry_mask, cmap=mcolors.ListedColormap(["black"]), shading="flat")
        tc1 = axs[0, 1].tripcolor(triangulation, pred_masked, cmap=cmap, norm=norm_pr, shading="flat")
        axs[0, 1].set_title(title_pred, fontsize=text_size)


        # (1,0) L2 relative Error
        if err_cbar_max is not None:
            vmax_err = float(np.nanpercentile(relative_l2_error_field, err_cbar_max))
        else:
            max_val = np.nanmax(relative_l2_error_field)
            vmax_err = float(max_val)

        if not (np.isfinite(vmax_err) and vmax_err > 0):
            vmax_err = 0.1  # Default to 10% if data is missing or all zero

        vmax_err = min(vmax_err, 2.0)

        norm_err = mcolors.Normalize(vmin=0, vmax=vmax_err)
        axs[1, 0].tripcolor(triangulation, dry_mask, cmap=mcolors.ListedColormap(["black"]), shading="flat")
        tc2 = axs[1, 0].tripcolor(triangulation, relative_l2_error_field , cmap=cmocean.cm.amp, norm=norm_err, shading="flat")
        axs[1, 0].set_title(title_err, fontsize=text_size)

        axs[1, 1].axis("off")

        fig.suptitle(
            f"Wind {wind_val} m/s at {wind_dir}°, Wave H {wave_h} m at {wave_dir}°",
            fontsize=text_size + 2,
            y = 0.94
        )


        
        for ax in axs.flat:
            ax.tick_params(axis="x", labelsize=text_size)
            ax.tick_params(axis="y", labelsize=text_size)

        cbar0 = fig.colorbar(tc0, ax=axs[0, 0])
        cbar0.set_label(label, fontsize=text_size)
        cbar0.ax.tick_params(labelsize=text_size)
        gt_ticks  = np.linspace(vmin_gt,  vmax_gt, 6) 
        cbar0.set_ticks(gt_ticks)


        cbar1 = fig.colorbar(tc1, ax=axs[0, 1])
        cbar1.set_label(label, fontsize=text_size)
        cbar1.ax.tick_params(labelsize=text_size)
        pr_ticks  = np.linspace(vmin_pr,  vmax_pr, 6) 
        cbar1.set_ticks(pr_ticks)

        cbar2 = fig.colorbar(tc2, ax=axs[1, 0])
        # cbar2.set_label("Absolute Error", fontsize=text_size)
        cbar2.set_label("Relative $L_2$ Error", fontsize=text_size)
        cbar2.ax.tick_params(labelsize=text_size)
        err_ticks = np.linspace(0, vmax_err, 6) 
        cbar2.set_ticks(err_ticks)

        max_err_idx = np.nanargmax(abs_error)
        max_err_val = abs_error[max_err_idx]
        max_err_x = xp_vector[max_err_idx]
        max_err_y = yp_vector[max_err_idx]
        max_err_true = true_vals[max_err_idx]
        print(f"Max AE = {max_err_val:.4f} at (x={max_err_x:.2f}, y={max_err_y:.2f})")
        print(f"True value at this location = {max_err_true:.4f}")


        if save_path:
            fig.savefig(save_path+"/"+comp + "_scenario_result.pdf", dpi=300)
            print("it is saved in ",save_path+"/"+comp+ "_scenario_result.pdf")
        if plot:

            plt.show()

        return fig, axs


    @staticmethod
    def plot_duck_scenario_transect(
        test_df,
        wind_val,
        wind_dir,
        wave_h,
        wave_dir,
        xp_vector,
        yp_vector,
        component="x",                 # "x" | "y" | "hsig"
        text_size=25,
        model_dir="/workspace/shukaic/wave_DON_WorkingFolder/model_DUCK_xforces",
        cbar_range=None,
        outlier_percentile=None,
        clip_min=None,
        err_cbar_max=None,
        segment=None,                  # None -> transect of the max-error node; int -> that transect
        seg_vector="y",                # which coord defines transects: "x"->xp, "y"->yp
        error_type="relative_l2",      # "relative_l2" | "absolute"
        profile_show="all",            # "all" | "values" | "error"
        along="auto",                  # profile x-axis: "x" | "y" | "dist" | "auto"
        err_decimals=6,
        save_path=None,
        plot=True,
    ):
        """Combined scenario + transect view.

        Renders the three :meth:`plot_duck_scenario` maps (ground truth,
        prediction, error field) on the top two rows, picks a cross-grid
        transect (via :meth:`segment_indices` on ``seg_vector``), highlights it
        and the max-error point on the error map, and plots the error profile
        along that transect in a full-width panel underneath.

        Parameters
        ----------
        segment : int or None
            ``None`` (default) auto-selects the transect that contains the node
            with the largest absolute error.  Pass an integer to plot a
            specific transect instead (index into the segment list).
        profile_show : {"all", "values", "error"}
            What to draw in the bottom panel.  ``"values"`` shows the truth and
            prediction profiles, ``"error"`` shows only the error profile, and
            ``"all"`` (default) shows both with truth/prediction on the left
            y-axis and the error on a twin right y-axis.

        Returns
        -------
        (fig, dict)
            The figure and a small dict describing the auto-selected transect
            (``segment``, ``max_err_idx``, ``max_err_xy``, ``max_err_val``).
        """
        # ----- shared prediction / error fields (single model load) -----
        fields = DataPlot._compute_scenario_error_fields(
            test_df, wind_val, wind_dir, wave_h, wave_dir,
            xp_vector, yp_vector, component, model_dir,
        )
        comp = fields["comp"]
        true_vals = fields["true_vals"]
        pred_full = fields["pred_full"]
        valid_mask = fields["valid_mask"]
        abs_error = fields["abs_error"]

        # ----- labels / cmap (mirror plot_duck_scenario) -----
        if comp == "x":
            label = "X-Force (N/m$^2$)"
            title_gt = "Ground Truth X-Forces"
            title_pred = "DON Model Prediction X-Forces"
            title_err = "DeepONet Relative $L_2$ Error (X-Forces)"
            cmap = cmocean.cm.balance.copy()
        elif comp == "y":
            label = "Y-Force (N/m$^2$)"
            title_gt = "Ground Truth Y-Forces"
            title_pred = "DON Model Prediction Y-Forces"
            title_err = "DeepONet Relative $L_2$ Error (Y-Forces)"
            cmap = cmocean.cm.balance.copy()
        else:  # hsig
            label = "Hsig (m)"
            title_gt = "Ground Truth Hsig"
            title_pred = "DON Model Prediction Hsig"
            title_err = "DeepONet Relative $L_2$ Error (Hsig)"
            cmap = cmocean.cm.haline.copy()

        if error_type == "absolute":
            err_field = fields["abs_error"]
            err_label = "Absolute Error"
        else:
            err_field = fields["relative_l2_error_field"]
            err_label = "Relative $L_2$ Error"

        comp_label = label

        triangulation = tri.Triangulation(xp_vector, yp_vector)
        true_masked = np.where(valid_mask, true_vals, np.nan)
        pred_masked = np.where(valid_mask, pred_full, np.nan)
        dry_mask = np.where(~valid_mask, -1.0, np.nan)

        # ----- locate the max-error point and its transect -----
        max_err_idx = int(np.nanargmax(abs_error))
        max_err_val = float(abs_error[max_err_idx])
        max_err_x = float(xp_vector[max_err_idx])
        max_err_y = float(yp_vector[max_err_idx])
        print(f"Max AE = {max_err_val:.4f} at (x={max_err_x:.2f}, y={max_err_y:.2f})")
        print(f"True value at this location = {true_vals[max_err_idx]:.4f}")

        base_vec = yp_vector if seg_vector.lower() == "y" else xp_vector
        segments = DataPlot.segment_indices(base_vec)

        if segment is None:
            # auto: the transect that contains the max-error node
            seg_id = next(
                (i for i, (s, e) in enumerate(segments) if s <= max_err_idx <= e), None
            )
            if seg_id is None:
                raise RuntimeError("Could not locate the max-error node in any transect.")
            print(f"Max-error node falls in transect segment {seg_id}.")
        else:
            # user-defined transect
            seg_id = int(segment)
            if seg_id < 0 or seg_id >= len(segments):
                raise IndexError(
                    f"segment {seg_id} out of range (found {len(segments)} transects)"
                )
            print(f"Plotting user-defined transect segment {seg_id}.")

        # ----- figure layout: 2 map rows + 1 wide profile row -----
        fig = plt.figure(figsize=(22, 30))
        gs = fig.add_gridspec(3, 2, height_ratios=[1, 1, 0.7],
                              wspace=0.4, hspace=0.25)
        ax_gt = fig.add_subplot(gs[0, 0])
        ax_pred = fig.add_subplot(gs[0, 1])
        ax_err = fig.add_subplot(gs[1, 0])
        fig.add_subplot(gs[1, 1]).axis("off")
        ax_prof = fig.add_subplot(gs[2, :])

        cmap.set_bad("black")

        # (0,0) Ground truth
        vmin_gt, vmax_gt = DataPlot.compute_vmin_vmax(
            true_masked, comp, cbar_range, outlier_percentile, clip_min)
        norm_gt = mcolors.Normalize(vmin=vmin_gt, vmax=vmax_gt)
        tc0 = ax_gt.tripcolor(triangulation, true_masked, cmap=cmap,
                              norm=norm_gt, shading="flat")
        ax_gt.set_title(title_gt, fontsize=text_size)

        # (0,1) Prediction
        vmin_pr, vmax_pr = DataPlot.compute_vmin_vmax(
            pred_masked, comp, cbar_range, outlier_percentile, clip_min)
        norm_pr = mcolors.Normalize(vmin=vmin_pr, vmax=vmax_pr)
        ax_pred.tripcolor(triangulation, dry_mask,
                          cmap=mcolors.ListedColormap(["black"]), shading="flat")
        tc1 = ax_pred.tripcolor(triangulation, pred_masked, cmap=cmap,
                                norm=norm_pr, shading="flat")
        ax_pred.set_title(title_pred, fontsize=text_size)

        # (1,0) Error field
        if err_cbar_max is not None:
            vmax_err = float(np.nanpercentile(err_field, err_cbar_max))
        else:
            vmax_err = float(np.nanmax(err_field))
        if not (np.isfinite(vmax_err) and vmax_err > 0):
            vmax_err = 0.1
        if error_type != "absolute":
            vmax_err = min(vmax_err, 2.0)
        norm_err = mcolors.Normalize(vmin=0, vmax=vmax_err)
        ax_err.tripcolor(triangulation, dry_mask,
                         cmap=mcolors.ListedColormap(["black"]), shading="flat")
        amp = cmocean.cm.amp.copy()
        amp.set_bad("black")
        tc2 = ax_err.tripcolor(triangulation, np.where(valid_mask, err_field, np.nan),
                               cmap=amp, norm=norm_err, shading="flat")
        ax_err.set_title(title_err, fontsize=text_size)

        # ----- build the selected transect's profile (wet nodes only) -----
        start, end = segments[seg_id]
        idx = np.arange(start, end + 1)
        idx = idx[valid_mask[idx]]
        xs, ys = xp_vector[idx], yp_vector[idx]

        mode = along.lower()
        if mode == "auto":
            mode = "x" if xs.ptp() >= ys.ptp() else "y"
        if mode == "x":
            t = xs; t_label = "x"
        elif mode == "y":
            t = ys; t_label = "y"
        else:
            d = np.sqrt(np.diff(xs) ** 2 + np.diff(ys) ** 2)
            t = np.concatenate([[0.0], np.cumsum(d)])
            t_label = "Along-transect distance"
        order = np.argsort(t)
        t = t[order]; idx = idx[order]

        # highlight the transect and the max-error node on the error map
        ax_err.plot(xs, ys, "-", color="deepskyblue", lw=2.5,
                    label=f"transect {seg_id}")
        ax_err.scatter(max_err_x, max_err_y, color="lime", s=160, marker="o",
                       edgecolors="k", zorder=5, label="max error")
        ax_err.legend(fontsize=text_size - 8, loc="best")

        # ----- (2,:) profile along the transect -----
        mode_show = profile_show.lower()
        if mode_show not in ("all", "values", "error"):
            raise ValueError("profile_show must be 'all', 'values', or 'error'")
        show_values = mode_show in ("all", "values")
        show_error = mode_show in ("all", "error")
        t_max = t[idx == max_err_idx]

        handles, labels = [], []  # collect across both axes for one legend
        ax_right = None

        if show_values:
            # truth / prediction on the (left) value axis
            h_true, = ax_prof.plot(t, true_vals[idx], "-o", color="navy", lw=2,
                                   ms=5, label="truth")
            h_pred, = ax_prof.plot(t, pred_full[idx], "--s", color="darkorange",
                                   lw=2, ms=5, markerfacecolor="none",
                                   label="prediction")
            ax_prof.set_ylabel(comp_label, fontsize=text_size)
            handles += [h_true, h_pred]
            labels += ["truth", "prediction"]

        # the axis that the error profile is drawn on
        if show_error and show_values:
            ax_right = ax_prof.twinx()         # error on the right axis
            err_ax = ax_right
        elif show_error:
            err_ax = ax_prof                   # error only -> use the left axis
        else:
            err_ax = None

        if err_ax is not None:
            h_err, = err_ax.plot(t, err_field[idx], "-o", color="crimson", lw=2,
                                 ms=5, label=err_label)
            err_ax.set_ylabel(err_label, fontsize=text_size)
            err_ax.yaxis.set_major_formatter(
                mticker.FormatStrFormatter(f"%.{err_decimals}f"))
            err_ax.tick_params(labelsize=text_size - 4)
            handles.append(h_err)
            labels.append(err_label)
            # mark the max-error node on the error curve
            if t_max.size:
                err_ax.scatter(t_max, [err_field[max_err_idx]], color="lime",
                               s=160, marker="o", edgecolors="k", zorder=5,
                               label="max error")
                handles.append(err_ax.collections[-1])
                labels.append("max error")

        ax_prof.set_xlabel(t_label, fontsize=text_size)
        seg_note = "contains max error" if segment is None else "user-selected"
        ax_prof.set_title(
            f"Transect {seg_id} profile ({seg_note})",
            fontsize=text_size,
        )
        ax_prof.grid(alpha=0.3)
        ax_prof.legend(handles, labels, fontsize=text_size - 6, ncol=2,
                       loc="best")
        ax_prof.tick_params(labelsize=text_size - 4)

        # ----- colorbars / ticks -----
        for ax in (ax_gt, ax_pred, ax_err):
            ax.tick_params(axis="x", labelsize=text_size)
            ax.tick_params(axis="y", labelsize=text_size)

        cbar0 = fig.colorbar(tc0, ax=ax_gt)
        cbar0.set_label(label, fontsize=text_size)
        cbar0.ax.tick_params(labelsize=text_size)
        cbar0.set_ticks(np.linspace(vmin_gt, vmax_gt, 6))

        cbar1 = fig.colorbar(tc1, ax=ax_pred)
        cbar1.set_label(label, fontsize=text_size)
        cbar1.ax.tick_params(labelsize=text_size)
        cbar1.set_ticks(np.linspace(vmin_pr, vmax_pr, 6))

        cbar2 = fig.colorbar(tc2, ax=ax_err)
        cbar2.set_label(err_label, fontsize=text_size)
        cbar2.ax.tick_params(labelsize=text_size)
        cbar2.set_ticks(np.linspace(0, vmax_err, 6))
        cbar2.ax.yaxis.set_major_formatter(
            mticker.FormatStrFormatter(f"%.{err_decimals}f"))

        fig.suptitle(
            f"Wind {wind_val} m/s at {wind_dir}°, Wave H {wave_h} m at {wave_dir}°",
            fontsize=text_size + 2,
            y=0.95,
        )

        info = {
            "segment": seg_id,
            "max_err_idx": max_err_idx,
            "max_err_xy": (max_err_x, max_err_y),
            "max_err_val": max_err_val,
        }

        if save_path:
            out = f"{save_path}/{comp}_scenario_transect{seg_id}.pdf"
            fig.savefig(out, dpi=300, bbox_inches="tight")
            print("it is saved in", out)
        if plot:
            plt.show()

        return fig, info

    @staticmethod
    def plot_duck_scenario_pdf(
        row, xp_vector, yp_vector,
        loaded_model, target_scaler,
        trunk_scaler_0, trunk_scaler_1,
        branch_scaler_01, branch_scaler_2,
        component="x", text_size=25, clip_min=None, outlier_percentile=None,
        cbar_range=None
    ):
        # 1. Unpack Scenario Data
        hsig_data_coarse = np.array(row["Hsig Data"])
        forces_data_coarse = np.array(row["Forces Data"])
        comp = component.lower()
        
        # Identical logic to your previous "if comp == ..." block
        if comp == "x":
            true_vals = forces_data_coarse[:, 0]
            label = "X-Force (N/m$^2$)"
            title_gt, title_pred = "Ground Truth X-Forces", "DON Model Prediction X-Forces"
            title_err = "DeepONet Relative $L_2$ Error (X-Forces)"
            cmap = cmocean.cm.balance.copy()
        elif comp == "y":
            true_vals = forces_data_coarse[:, 1]
            label = "Y-Force (N/m$^2$)"
            title_gt, title_pred = "Ground Truth Y-Forces", "DON Model Prediction Y-Forces"
            title_err = "DeepONet Relative $L_2$ Error (Y-Forces)"
            cmap = cmocean.cm.balance.copy()
        elif comp == "hsig":
            true_vals = hsig_data_coarse
            label = "Hsig (m)"
            title_gt, title_pred = "Ground Truth Hsig", "DON Model Prediction Hsig"
            title_err = "DeepONet Relative $L_2$ Error (Hsig)"
            cmap = cmocean.cm.haline.copy()
        
        valid_mask = hsig_data_coarse != -9
        valid_idx = np.where(valid_mask)[0]
        triangulation = tri.Triangulation(xp_vector, yp_vector)

        # 2. Scaling & Prediction (The HPC-Safe Way)
        b_key = list(row["key"])
        x_wind, y_wind = DataCal.convert_wind_components(b_key[0], b_key[1])
        b_input = np.array([[x_wind, y_wind, b_key[2], np.cos(np.radians(b_key[3])), np.sin(np.radians(b_key[3]))]], dtype=np.float32)
        
        valid_coords = np.column_stack((xp_vector[valid_idx], yp_vector[valid_idx]))
        t_test = valid_coords.copy().astype(np.float32)
        t_test[:, 0] = trunk_scaler_0.transform(t_test[:, 0].reshape(-1,1)).ravel()
        t_test[:, 1] = trunk_scaler_1.transform(t_test[:, 1].reshape(-1,1)).ravel()

        b_test_scaled = b_input.copy()
        b_test_scaled[:, 0:2] = branch_scaler_01.transform(b_test_scaled[:, 0:2])
        b_test_scaled[:, 2:3] = branch_scaler_2.transform(b_test_scaled[:, 2:3])
        b_batch = np.repeat(b_test_scaled, len(t_test), axis=0)

        # Predict using the already loaded model
        o_res = loaded_model([b_batch, t_test], training=False)
        pred_valid = target_scaler.inverse_transform(o_res).ravel()

        # 3. Data Stitching & Masking (Restored to previous logic)
        pred_full = np.full_like(true_vals, np.nan, dtype=float)
        pred_full[valid_idx] = pred_valid
        
        true_masked = np.where(valid_mask, true_vals, np.nan)
        pred_masked = np.where(valid_mask, pred_full, np.nan)
        abs_error = np.abs(pred_full - true_vals)
        abs_error[~valid_mask] = np.nan

        rel_l2_error_field = abs_error / (np.linalg.norm(true_vals[valid_mask]) + 1e-10)
        dry_mask = np.where(~valid_mask, -1.0, np.nan)

        # 4. Figure Layout (Restored 2x2 with subplots_adjust)
        fig, axs = plt.subplots(2, 2, figsize=(22, 22))
        plt.subplots_adjust(wspace=0.4, hspace=0.15)
        cmap.set_bad("black")

        # Compute vmin/vmax exactly like previous way
        vmin_gt, vmax_gt = DataPlot.compute_vmin_vmax(true_masked, comp, cbar_range, outlier_percentile, clip_min)
        vmin_pr, vmax_pr = DataPlot.compute_vmin_vmax(pred_masked, comp, cbar_range, outlier_percentile, clip_min)
        
        # Calculate Error Max for relative L2
        max_val_err = np.nanmax(rel_l2_error_field)
        vmax_err = min(float(np.nanpercentile(rel_l2_error_field, 99)), 2.0) if np.isfinite(max_val_err) else 0.1

        # Plotting (0,0) GT
        norm_gt = mcolors.Normalize(vmin=vmin_gt, vmax=vmax_gt)
        tc0 = axs[0, 0].tripcolor(triangulation, true_masked, cmap=cmap, norm=norm_gt, shading="flat")
        axs[0, 0].set_title(title_gt, fontsize=text_size)

        # Plotting (0,1) Pred
        norm_pr = mcolors.Normalize(vmin=vmin_pr, vmax=vmax_pr)
        axs[0, 1].tripcolor(triangulation, dry_mask, cmap=mcolors.ListedColormap(["black"]), shading="flat")
        tc1 = axs[0, 1].tripcolor(triangulation, pred_masked, cmap=cmap, norm=norm_pr, shading="flat")
        axs[0, 1].set_title(title_pred, fontsize=text_size)

        # Plotting (1,0) Rel Error
        norm_err = mcolors.Normalize(vmin=0, vmax=vmax_err)
        axs[1, 0].tripcolor(triangulation, dry_mask, cmap=mcolors.ListedColormap(["black"]), shading="flat")
        tc2 = axs[1, 0].tripcolor(triangulation, rel_l2_error_field, cmap=cmocean.cm.amp, norm=norm_err, shading="flat")
        axs[1, 0].set_title(title_err, fontsize=text_size)

        axs[1, 1].axis("off")

        # Titles and Ticks (Restored exact formatting)
        fig.suptitle(f"Wind {row['Wind Value']} m/s at {row['Wind Direction']}°, Wave H {row['Wave Height']} m at {row['Wave Direction']}°",
                     fontsize=text_size + 2, y=0.94)

        for ax in axs.flat:
            ax.tick_params(axis="both", labelsize=text_size)

        # Colorbars (Restored exact tick and label logic)
        cbars = [fig.colorbar(tc0, ax=axs[0,0]), fig.colorbar(tc1, ax=axs[0,1]), fig.colorbar(tc2, ax=axs[1,0])]
        cbar_labels = [label, label, "Relative $L_2$ Error"]
        cbar_ticks = [np.linspace(vmin_gt, vmax_gt, 6), np.linspace(vmin_pr, vmax_pr, 6), np.linspace(0, vmax_err, 6)]

        for cbar, l, ticks in zip(cbars, cbar_labels, cbar_ticks):
            cbar.set_label(l, fontsize=text_size)
            cbar.ax.tick_params(labelsize=text_size)
            cbar.set_ticks(ticks)

        # Max Error Annotation (Restored)
        max_err_idx = np.nanargmax(abs_error)
        max_ex, max_ey, max_ev = xp_vector[max_err_idx], yp_vector[max_err_idx], abs_error[max_err_idx]
        axs[1, 0].scatter(max_ex, max_ey, color="blue", s=120, marker="*", edgecolors="k", zorder=5)
        axs[1, 0].annotate(f"Max AE\n{max_ev:.2f}", (max_ex, max_ey), xytext=(10, 10), 
                            textcoords="offset points", fontsize=text_size-4, color="blue", weight="bold")

        return fig

    @staticmethod
    def plotSWANResult_old(
        xp_vector, yp_vector,
        prediction_result,
        component='x',
        cmap_name='coolwarm',
        shading='flat',
        point_size=12,
        title_prefix='Radiation Force',
        equal_axes=True,
        cbar_range=None   
    ):
        xp_vector = np.asarray(xp_vector)
        yp_vector = np.asarray(yp_vector)
        if xp_vector.shape != yp_vector.shape:
            raise ValueError("xp_vector and yp_vector must have the same shape.")

        comp = component.lower()
        if comp == 'x':
            comp_label = 'X-Force'
            title_text = 'Gradient of Radiation X-Force'
        elif comp == 'y':
            comp_label = 'Y-Force'
            title_text = 'Gradient of Radiation Y-Force'
        elif comp == 'hsig':
            comp_label = 'Hsig'
            title_text = 'Significant Wave Height'
        else:
            raise ValueError("component must be 'x', 'y', or 'hsig'.")


        vals_full = np.asarray(prediction_result, dtype=float)
        if vals_full.shape != xp_vector.shape:
            raise ValueError("In full-size mode, prediction_result must match xp/yp shape.")
        scatter_x = xp_vector
        scatter_y = yp_vector
        scatter_vals = vals_full

        # --- Triangulation ---
        triangulation = tri.Triangulation(xp_vector, yp_vector)

        if cbar_range is not None:
            vmin, vmax = cbar_range
        else:
            vmin = np.nanmin(vals_full)
            vmax = np.nanmax(vals_full)
            if not np.isfinite(vmin) or not np.isfinite(vmax) or vmin == vmax:
                vmin, vmax = 0.0, 1.0

        norm = mcolors.Normalize(vmin=vmin, vmax=vmax)

        if comp in ('x', 'y'):
            cmap = plt.cm.get_cmap(cmap_name).copy()   
            cmap = cmocean.cm.balance.copy()         
        else:
            cmap = plt.cm.get_cmap('viridis' if cmap_name == 'coolwarm' else cmap_name).copy() 
            cmap = cmocean.cm.haline.copy() 

        cmap.set_bad('black')

        # --- Plot ---
        fig, axs = plt.subplots(1, 2, figsize=(16, 7))

        tc = axs[0].tripcolor(triangulation, vals_full, cmap=cmap, norm=norm, shading=shading)
        cbar1 = fig.colorbar(tc, ax=axs[0])
        cbar1.set_label(comp_label)
        axs[0].set_title(f"{title_prefix if comp!='hsig' else title_text} ({comp_label}) — Triangulated")

        sc = axs[1].scatter(scatter_x, scatter_y, c=scatter_vals, s=point_size, cmap=cmap, norm=norm)
        cbar2 = fig.colorbar(sc, ax=axs[1])
        cbar2.set_label(comp_label)
        axs[1].set_title(f"{title_prefix if comp!='hsig' else title_text} ({comp_label}) — Scatter (No Triangulation)")

        for ax in axs:
            ax.set_xlabel("x")
            ax.set_ylabel("y")
            if equal_axes:
                ax.set_aspect('equal', adjustable='box')
                ax.autoscale(enable=True, tight=True)
            ax.grid(True, alpha=0.2)

        plt.tight_layout()
        plt.show()


    @staticmethod
    def collect_max_error_points(
        test_df,
        xp_vector,
        yp_vector,
        *,
        component="x",  # "x" | "y" | "hsig"
        model_dir="/workspace/shukaic/wave_DON_WorkingFolder/model_DUCK_xforces",
        tensor_DON=True
    ):
        """
        For each scenario (row) in test_df, find the location with the maximum absolute error.
        Returns a pandas DataFrame: one row per scenario with x,y, abs_error, true, pred, and forcings.
        """
        # Load once
        model = tf.keras.models.load_model(model_dir, compile=False)
        target_scaler = joblib.load(f"{model_dir}/target_scaler.pkl")

        xp_vector = np.asarray(xp_vector, dtype=float)
        yp_vector = np.asarray(yp_vector, dtype=float)

        rows = []

        for i, row in test_df.iterrows():
            # --- Ground truth ---
            hsig = np.array(row["Hsig Data"])
            forces = np.array(row["Forces Data"])

            comp = component.lower()
            if comp == "x":
                true_vals = forces[:, 0]
            elif comp == "y":
                true_vals = forces[:, 1]
            elif comp == "hsig":
                true_vals = hsig
            else:
                raise ValueError("component must be 'x', 'y', or 'hsig'")

            # mask out dry points
            valid_mask = (hsig != -9)
            if not np.any(valid_mask):
                continue

            valid_idx = np.where(valid_mask)[0]
            valid_coords = np.column_stack([xp_vector[valid_idx], yp_vector[valid_idx]])

            # --- Trunk scaling ---
            t_test = ModelDataProcessor.trunk_scale_and_reload(
                valid_coords,
                scaler_path_0=f"{model_dir}/scaler_col_0.pkl",
                scaler_path_1=f"{model_dir}/scaler_col_1.pkl",
            )

            # --- Branch input ---
            if "key" in row and row["key"] is not None:
                b_key = list(row["key"])
            else:
                b_key = [
                    float(row["Wind Value"]),
                    float(row["Wind Direction"]),
                    float(row["Wave Height"]),
                    float(row["Wave Direction"]),
                ]

            x_wind, y_wind = DataCal.convert_wind_components(b_key[0], b_key[1])
            cos_wave_dir = np.cos(np.radians(b_key[3]))
            sin_wave_dir = np.sin(np.radians(b_key[3]))
            b_test_input = [x_wind, y_wind, b_key[2], cos_wave_dir, sin_wave_dir]

            if tensor_DON:
                b_batch = np.array([b_test_input])               # (1, 5)
            else:
                b_batch = np.array([b_test_input] * len(t_test)) # (N, 5)

            b_test_scaled = ModelDataProcessor.load_and_scale_new_data(
                b_batch,
                scaler_path_0_1=f"{model_dir}/scaler_factor_0_1.pkl",
                scaler_path_2=f"{model_dir}/scaler_factor_2.pkl",
            )

            # --- Predict ---
            o_res = model([b_test_scaled, t_test])

            # --- Inverse scale ---
            if tensor_DON:
                pred_valid = DataPlot.invert_with_legacy_scaler(
                    o_res, target_scaler, n_feat=len(valid_idx)
                )
            else:
                pred_valid = target_scaler.inverse_transform(o_res).ravel()

            # reconstruct full vector to align with true_vals indexing
            pred_full = np.full_like(true_vals, np.nan, dtype=float)
            pred_full[valid_idx] = pred_valid

            # --- Error & max index for this scenario ---
            abs_err = np.abs(pred_full - true_vals)
            abs_err[~valid_mask] = np.nan
            if np.all(np.isnan(abs_err)):
                continue

            max_local_idx = int(np.nanargmax(abs_err))
            rows.append({
                "scenario_idx": int(i),
                "x": float(xp_vector[max_local_idx]),
                "y": float(yp_vector[max_local_idx]),
                "abs_error": float(abs_err[max_local_idx]),
                "true_value": float(true_vals[max_local_idx]),
                "pred_value": float(pred_full[max_local_idx]),
                "wind": float(b_key[0]),
                "wind_dir": float(b_key[1]),
                "wave_h": float(b_key[2]),
                "wave_dir": float(b_key[3]),
                "component": component,
            })

        return pd.DataFrame(rows).sort_values("abs_error", ascending=False).reset_index(drop=True)
    

    @staticmethod
    def plot_max_errors_on_map(
            xp_vector,
            yp_vector,
            worst_points_df,
            *,
            show_background=True,
            background_alpha=0.08,
            size_by_error=True,
            annotate_top_k=5,
            fixed_vmin=None,
            fixed_vmax=None,
            cmap="viridis",
            title=None,

            # NEW: saving options
            save_dir=None,
            save_name="max_error_map",
            save_dpi=300,
        ):
        """
        Scatter-plot the worst-error location of each scenario on the (xp, yp) domain.
        """
        import os

        xp_vector = np.asarray(xp_vector, dtype=float)
        yp_vector = np.asarray(yp_vector, dtype=float)

        errs = worst_points_df["abs_error"].to_numpy()

        # handle color scale
        vmin = fixed_vmin if fixed_vmin is not None else np.nanmin(errs)
        vmax = fixed_vmax if fixed_vmax is not None else np.nanmax(errs)
        if vmax - vmin < 1e-12:   # avoid divide-by-zero issue
            vmax = vmin + 1e-12

        # marker size scaling
        if size_by_error:
            e = (errs - vmin) / (vmax - vmin)
            sizes = 60 + 240 * e
        else:
            sizes = 60 * np.ones_like(errs)

        plt.figure(figsize=(8, 7))

        # background points
        if show_background:
            plt.scatter(
                xp_vector, yp_vector,
                s=3, alpha=background_alpha, linewidths=0, color="gray"
            )

        # main scatter
        sc = plt.scatter(
            worst_points_df["x"].to_numpy(),
            worst_points_df["y"].to_numpy(),
            c=errs,
            s=sizes,
            cmap=cmap,
            vmin=vmin, vmax=vmax,
            edgecolor="k",
            linewidths=0.5,
        )

        # colorbar
        cbar = plt.colorbar(sc)
        cbar.set_label("Absolute Error at Scenario Worst Point")
        cbar.ax.set_title(f"{vmax:.4f}", fontsize=10, pad=6)

        cbar.ax.text(
            0.5, 1.02,
            f"max = {vmax:.4f}",
            ha="center", va="bottom", fontsize=9, transform=cbar.ax.transAxes
        )

        plt.gca().set_aspect("equal", adjustable="box")
        plt.xlabel("X coordinate")
        plt.ylabel("Y coordinate")
        plt.title(title or "Worst-case Error Locations Across Scenarios")

        # annotate top-k largest errors
        if annotate_top_k and annotate_top_k > 0:
            top = worst_points_df.nlargest(annotate_top_k, "abs_error")
            for _, r in top.iterrows():
                txt = (
                    f"Scenario {int(r['scenario_idx'])}\n"
                    f"Wind {r['wind']:.1f} m/s @ {r['wind_dir']:.0f}°\n"
                    f"Hs {r['wave_h']:.2f} m @ {r['wave_dir']:.0f}°"
                )
                plt.annotate(
                    txt,
                    (r["x"], r["y"]),
                    xytext=(8, 8),
                    textcoords="offset points",
                    fontsize=9,
                    bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="gray", alpha=0.75)
                )

        plt.tight_layout()

        if save_dir is not None:
            os.makedirs(save_dir, exist_ok=True)
            save_path = os.path.join(save_dir, f"{save_name}")
            plt.savefig(save_path, dpi=save_dpi, bbox_inches="tight")
            print(f"Figure saved to: {save_path}")

        plt.show()
