import numpy as np

class DataCal:
    @staticmethod
    def convert_wind_components(speed, direction):
        # Convert wind direction to radians
        direction_rad = np.deg2rad(direction)
        # Calculate x and y components of the wind speed
        x_wind = speed * np.cos(direction_rad)
        y_wind = speed * np.sin(direction_rad)
        return x_wind, y_wind

    