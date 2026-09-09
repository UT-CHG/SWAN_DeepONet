# model_Analysis.py
import numpy as np
import joblib
import tensorflow as tf

# make sure these are importable here:
from data_calculation import DataCal
from modelData_processor import ModelDataProcessor
import matplotlib.pyplot as plt
import pandas as pd
import os

class ModelAnalysis:
    @staticmethod
    def _true_vector_for_component(row, component: str):
        hsig_data_coarse   = np.array(row["Hsig Data"])
        forces_data_coarse = np.array(row["Forces Data"])
        comp = str(component).lower()
        if comp == "x":
            return forces_data_coarse[:, 0]
        elif comp == "y":
            return forces_data_coarse[:, 1]
        elif comp == "hsig":
            return hsig_data_coarse
        else:
            raise ValueError("component must be 'x', 'y', or 'hsig'.")

    @staticmethod
    def _predict_one_case(row, xp_vector, yp_vector, model, target_scaler, model_dir: str):
        # valid mask from Hsig
        hsig_data_coarse = np.array(row["Hsig Data"])
        valid_mask = (hsig_data_coarse != -9)
        valid_idx = np.where(valid_mask)[0]

        # branch features
        b_key = list(row["key"])
        x_wind, y_wind = DataCal.convert_wind_components(b_key[0], b_key[1])
        cos_wave_dir   = np.cos(np.radians(b_key[3]))
        sin_wave_dir   = np.sin(np.radians(b_key[3]))
        b_test_input   = [x_wind, y_wind, b_key[2], cos_wave_dir, sin_wave_dir]

        # trunk features at valid points
        xp_vector = np.asarray(xp_vector); yp_vector = np.asarray(yp_vector)
        valid_xy  = np.column_stack((xp_vector[valid_idx], yp_vector[valid_idx]))

        # scale trunk
        t_test = ModelDataProcessor.trunk_scale_and_reload(
            valid_xy,
            scaler_path_0=f"{model_dir}/scaler_col_0.pkl",
            scaler_path_1=f"{model_dir}/scaler_col_1.pkl",
        )

        # scale branch
        b_batch = np.array([b_test_input] * len(t_test))
        b_test_scaled = ModelDataProcessor.load_and_scale_new_data(
            b_batch,
            scaler_path_0_1=f"{model_dir}/scaler_factor_0_1.pkl",
            scaler_path_2  =f"{model_dir}/scaler_factor_2.pkl",
        )

        # predict & inverse-scale
        o_res = model([b_test_scaled, t_test])  # shape (n_valid, 1)
        pred_valid = target_scaler.inverse_transform(o_res).ravel()

        # stitch to full length
        pred_full = np.full(valid_mask.shape, np.nan, dtype=float)
        pred_full[valid_idx] = pred_valid
        return pred_full, valid_mask
    
    @staticmethod
    def compute_metrics_comprehensive(
        test_df,
        xp_vector,
        yp_vector,
        *,
        model_dir: str,
        component: str = "y",
        machine = None,
    ):

        if machine == 'Vista':
             model = tf.keras.models.load_model(os.path.join(model_dir, 'model'), compile=False)
        else:
            model = tf.keras.models.load_model(model_dir, compile=False)
        target_scaler = joblib.load(f"{model_dir}/target_scaler.pkl")

        N = len(xp_vector)
        sum_err_abs = np.zeros(N, dtype=float)
        sum_true = np.zeros(N, dtype=float)
        count_pointwise = np.zeros(N, dtype=np.int64)
        
        rel_l2_list = []
        relative_max_error = []
        all_true_vectors = []
        all_pred_vectors = []
        all_valid_masks = []
        scenario_keys = []

        for _, row in test_df.iterrows():
            true_vec = ModelAnalysis._true_vector_for_component(row, component)
            pred_full, valid_mask = ModelAnalysis._predict_one_case(
                row, xp_vector, yp_vector, model, target_scaler, model_dir
            )


            err_abs = np.abs(pred_full - true_vec)
            

            valid = valid_mask & np.isfinite(err_abs) & np.isfinite(true_vec)
            
            if np.any(valid):
                diff_vec = (pred_full[valid] - true_vec[valid])
    
                l2_diff = np.linalg.norm(diff_vec, ord=2)
                l2_true = np.linalg.norm(true_vec[valid], ord=2)
                
   
                if l2_true > 0:
                    rel_l2_list.append(l2_diff / l2_true)


                sum_err_abs[valid] += err_abs[valid]
                sum_true[valid] += true_vec[valid]
                count_pointwise[valid] += 1

                max_true = np.max(np.abs(true_vec[valid]))
                max_error = np.max(err_abs[valid])
                relative_MaxErro_scenario = max_error / max_true
            
            relative_max_error.append(relative_MaxErro_scenario)
            all_true_vectors.append(true_vec.copy())
            all_pred_vectors.append(pred_full.copy())
            all_valid_masks.append(valid_mask.copy())
            scenario_keys.append(row['key'])



        with np.errstate(invalid="ignore", divide="ignore"):
            mae_full = sum_err_abs / count_pointwise
            mean_true_full = sum_true / count_pointwise

        mae_full[count_pointwise == 0] = np.nan
        mean_true_full[count_pointwise == 0] = np.nan

        return {
            "pointwise_mae": mae_full,
            "pointwise_true_mean": mean_true_full,
            "rel_l2_per_case": np.array(rel_l2_list),
            "mean_rel_l2": np.mean(rel_l2_list) if rel_l2_list else np.nan,
            "median_rel_l2": np.median(rel_l2_list) if rel_l2_list else np.nan,
            "relative_max_error": relative_max_error,
            "all_true_vectors": np.array(all_true_vectors),
            "all_pred_vectors": np.array(all_pred_vectors),
            "all_valid_masks": np.array(all_valid_masks),
            "scenario_keys": scenario_keys,
        }
        
        
    @staticmethod
    def compute_per_case_metrics(
        test_df,
        xp_vector,
        yp_vector,
        *,
        model_dir: str,
        component: str = "y",
    ):
        """
        Compute per-scenario metrics:
        - worst point absolute error
        - MAE of the scenario
        - RMSE of the scenario
        - scenario conditions (wind, wave)
        Returns a DataFrame.
        """
        # preload model once
        model = tf.keras.models.load_model(model_dir, compile=False)
        target_scaler = joblib.load(f"{model_dir}/target_scaler.pkl")

        records = []  # list of dicts

        for _, row in test_df.iterrows():

            # --- true vector ---
            true_vec = ModelAnalysis._true_vector_for_component(row, component)

            # --- prediction ---
            pred_full, valid_mask = ModelAnalysis._predict_one_case(
                row, xp_vector, yp_vector, model, target_scaler, model_dir
            )

            # valid only
            valid_pred = pred_full[valid_mask]
            valid_true = true_vec[valid_mask]

            # --- errors ---
            abs_err = np.abs(valid_pred - valid_true)
            sq_err  = (valid_pred - valid_true) ** 2

            worst_point_err = np.max(abs_err)
            mae_case        = np.mean(abs_err)
            rmse_case       = np.sqrt(np.mean(sq_err))

            # scenario conditions (wind, wave)
            w_speed, w_dir, wave_h, wave_dir = row["key"]

            # store one record
            records.append({
                "wind_speed": w_speed,
                "wind_dir": w_dir,
                "wave_height": wave_h,
                "wave_dir": wave_dir,
                "worst_point_error": worst_point_err,
                "mae": mae_case,
                "rmse": rmse_case,
            })

        # convert to DataFrame
        return pd.DataFrame(records)

    
