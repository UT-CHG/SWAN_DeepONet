import re
import numpy as np
import pandas as pd
from collections import defaultdict


class DataExtraction1D:
    def __init__(self):
        self.data_dict = {
            'wind': [],
            'wind_direction': [],
            'initial_wave_h': [],
            'initial_wave_d': [],
            'dist': [],
            'depth': [],
            'Hsig': [],
            'RTpeak': [],
            'Tm01': [],
            'Tm02': [],
            'FSpr': [],
            'Dir': [],
            'x_wforce': [],
            'y_wforce': []
        }

    # Function to parse a section of the file
    def parse_section(self, section):
        lines = section.splitlines()
        # Extract parameters from the first line
        match = re.search(r'Wind: (\d+), Direction: (-?\d+); Initial_Wave_H: (\d+(\.\d+)?); Initial_Wave_D: (-?\d+)', lines[0])

        if match:
            wind = int(match.group(1))
            direction = int(match.group(2))
            initial_wave_h = float(match.group(3))
            initial_wave_d = int(match.group(5))

            # Extract data from the table
            for line in lines[5:]:
                if line.strip() == '':
                    continue
                parts = line.split()
                if len(parts) >= 12:
                    try:
                        dist = float(parts[0])
                        water_depth = float(parts[1])
                        wave_height = float(parts[2])
                        RTpeak = float(parts[3])
                        Tm_01 = float(parts[4])
                        Tm_02 = float(parts[5])
                        FSpr = float(parts[6])
                        dir = float(parts[7])
                        x_wforce = float(parts[10])
                        y_wforce = float(parts[11])

                        self.data_dict['wind'].append(wind)
                        self.data_dict['wind_direction'].append(direction)
                        self.data_dict['initial_wave_h'].append(initial_wave_h)
                        self.data_dict['initial_wave_d'].append(initial_wave_d)
                        self.data_dict['dist'].append(dist)
                        self.data_dict['depth'].append(water_depth)
                        self.data_dict['Hsig'].append(wave_height)
                        self.data_dict['RTpeak'].append(RTpeak)
                        self.data_dict['Tm01'].append(Tm_01)
                        self.data_dict['Tm02'].append(Tm_02)
                        self.data_dict['FSpr'].append(FSpr)
                        self.data_dict['Dir'].append(dir)
                        self.data_dict['x_wforce'].append(x_wforce)
                        self.data_dict['y_wforce'].append(y_wforce)
                    except ValueError:
                        continue
    import re



    def adjust_direction(self, direction):
        return direction + 360 if direction < 0 else direction

    def read_data_txt(self, file_path):
        with open(file_path, 'r') as file:
            content = file.read()

        # Split content by 'Wind: ' to separate sections
        sections = content.split('Wind: ')[1:]
        for section in sections:
            self.parse_section('Wind: ' + section)
        
        # Convert lists to numpy arrays for easy indexing
        self.data_dict = {key: np.array(value) for key, value in self.data_dict.items()}
        data_df = pd.DataFrame(self.data_dict)
        
        data_df['wind_direction'] = data_df['wind_direction'].apply(self.adjust_direction)
        data_df['initial_wave_d'] = data_df['initial_wave_d'].apply(self.adjust_direction)

        return data_df


class DataExtraction2D:
    def __init__(self):
        # Data structure to hold parsed data
        self.data_dict = defaultdict(dict)

    def get_specific_data(self, filename, variable_type, relevant_keys):
        """
        Parses a file (Hsig or forces) and retrieves data for the given keys.

        Args:
            filename (str): Path to the input file.
            variable_type (str): Type of variable (e.g., "hsig", "forces").
            relevant_keys (set): Set of keys to retrieve data for.

        Returns:
            dict: A dictionary with the relevant keys and their parsed data.
        """
        specific_data = {}

        with open(filename, 'r') as file:
            content = file.read()

        # Split the file content into sections
        sections = content.strip().split("\n\n")
        for section in sections:
            lines = section.splitlines()

            # Extract the parameters from the first line
            match = re.search(
                r'Wind: (\d+), Direction: (-?\d+); Initial_Wave_H: (\d+(\.\d+)?); Initial_Wave_D: (-?\d+)',
                lines[0]
            )
            if match:
                wind = int(match.group(1))
                direction = int(match.group(2))
                initial_wave_h = float(match.group(3))
                initial_wave_d = int(match.group(5))

                # Create a unique key for this combination of parameters
                key = (wind, direction, initial_wave_h, initial_wave_d)

                # Only process if the key is relevant
                if key in relevant_keys:
                    data = []
                    for line in lines[1:]:
                        if line.strip() == '':
                            continue
                        parts = line.split()
                        if variable_type == 'hsig':
                            try:
                                value = float(parts[0])
                                data.append(value)
                            except ValueError:
                                continue
                        elif variable_type == 'forces':
                            try:
                                x_force = float(parts[0])
                                y_force = float(parts[1])
                                data.append((x_force, y_force))
                            except ValueError:
                                continue
                    specific_data[key] = data

        return specific_data
    
    def GetScenariosToPD(self, filename):
        # File path
        file_path = filename

        # Initialize variables
        data = []
        current_config = None
        iteration_count = 0

        # Read the file
        with open(file_path, "r") as file:
            lines = file.readlines()

        # Process each line
        for line in lines:
            # Check for a new combination
            if line.startswith("Running SWAN with Wind:"):
                # Save the last combination's data
                if current_config and iteration_count > 0:
                    data.append((*current_config, iteration_count))
                
                # Parse the configuration
                current_config = re.match(
                    r"Running SWAN with Wind: (\d+) m/s, Direction: ([\-\d]+) degree, Wave Height: ([\d\.]+) m, Wave Direction: ([\-\d]+) degree",
                    line.strip()
                )
                if current_config:
                    current_config = current_config.groups()  # Extract values as a tuple
                iteration_count = 0  # Reset iteration count for the new configuration

            # Count iterations
            elif re.match(r"\s*iteration\s+\d+;", line):
                iteration_count += 1

        if current_config and iteration_count > 0:
            data.append((*current_config, iteration_count))

        columns = ["Wind Value", "Wind Direction", "Wave Height", "Wave Direction", "Iterations"]
        df = pd.DataFrame(data, columns=columns)

        print(df)
        return df
    
    def DataConcat(self, relevant_keys, hsig_data, forces_data):
        result = []
        for key in relevant_keys:
            wind, direction, wave_height, wave_direction = key
            result.append({
                "Wind Value": wind,
                "Wind Direction": direction,
                "Wave Height": wave_height,
                "Wave Direction": wave_direction,
                "Hsig Data": hsig_data.get(key, None),
                "Forces Data": forces_data.get(key, None),
            })

        result_df = pd.DataFrame(result)

        return result_df