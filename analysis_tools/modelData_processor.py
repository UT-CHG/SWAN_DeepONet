from sklearn.preprocessing import MinMaxScaler
import joblib
import numpy as np

class ModelDataProcessor:
    
    @staticmethod
    def load_and_scale_new_data(new_b_train, scaler_path_0_1, scaler_path_2):
        """
        Load the saved scalers and apply scaling to new data.
        
        Parameters:
        - new_b_train: np.array, new input data with 4 columns.
        - scaler_path_0_1: str, path to load MinMaxScaler for columns 0 and 1.
        - scaler_path_2: str, path to load MinMaxScaler for column 2.
        - scaler_path_3: str, path to load scaling factor for column 3.
        
        Returns:
        - new_b_train_scaled: np.array, scaled new data.
        """

        scaler_0_1 = joblib.load(scaler_path_0_1)
        

        scaler_2 = joblib.load(scaler_path_2)
        

        new_scaled_0_1 = scaler_0_1.transform(new_b_train[:, [0, 1]])
        

        new_scaled_2 = scaler_2.transform(new_b_train[:, [2]].reshape(-1, 1))
        
        new_b_train_scaled = np.hstack((new_scaled_0_1, new_scaled_2, new_b_train[:, [3, 4]]))
        
        return new_b_train_scaled

    @staticmethod
    def trunk_scale_and_reload(t_train, t_val=None, save_scalers=True, reload_scalers=False, scaler_path_0=None, scaler_path_1=None):
        """
        Scales t_train and optionally t_val columns independently to range [0, 1],
        with the ability to save and reload scalers.
        
        Args:
            t_train (numpy.ndarray): Training data with two columns to scale.
            t_val (numpy.ndarray, optional): Validation data with two columns to scale. Defaults to None.
            save_scalers (bool): Whether to save the scalers for later use. Defaults to True.
            reload_scalers (bool): Whether to reload scalers from the provided paths. Defaults to False.
            scaler_path_0 (str): Path to save or reload the scaler for column 0.
            scaler_path_1 (str): Path to save or reload the scaler for column 1.
        
        Returns:
            tuple: Scaled t_train and optionally t_val arrays.
        """
        # Initialize scalers
        if reload_scalers:
            # Reload saved scalers
            if scaler_path_0 and scaler_path_1:
                scaler_col_0 = joblib.load(scaler_path_0)
                scaler_col_1 = joblib.load(scaler_path_1)
            else:
                raise ValueError("Scaler paths must be provided for reloading scalers.")
        else:
            # Initialize new scalers for each column
            scaler_col_0 = MinMaxScaler(feature_range=(0, 1))  # For column 0
            scaler_col_1 = MinMaxScaler(feature_range=(0, 1))  # For column 1
            
            # Fit scalers on t_train
            scaler_col_0.fit(t_train[:, 0].reshape(-1, 1))
            scaler_col_1.fit(t_train[:, 1].reshape(-1, 1))
            
            # Save scalers if required
            if save_scalers:
                if scaler_path_0:
                    joblib.dump(scaler_col_0, scaler_path_0)
                if scaler_path_1:
                    joblib.dump(scaler_col_1, scaler_path_1)
        
        # Scale t_train using the scalers
        t_train_scaled_col_0 = scaler_col_0.transform(t_train[:, 0].reshape(-1, 1))
        t_train_scaled_col_1 = scaler_col_1.transform(t_train[:, 1].reshape(-1, 1))
        t_train_scaled = np.hstack([t_train_scaled_col_0, t_train_scaled_col_1])
        
        # Scale t_val if provided
        if t_val is not None:
            t_val_scaled_col_0 = scaler_col_0.transform(t_val[:, 0].reshape(-1, 1))
            t_val_scaled_col_1 = scaler_col_1.transform(t_val[:, 1].reshape(-1, 1))
            t_val_scaled = np.hstack([t_val_scaled_col_0, t_val_scaled_col_1])
            return t_train_scaled, t_val_scaled
        
        return t_train_scaled