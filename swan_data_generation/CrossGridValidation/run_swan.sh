#!/bin/bash



# --- Setup ---
TEST_FILE="test_list.txt"
TIME_LOG="DUCK_execution_times.txt"

# Clear or initialize the time log
echo "Scenario_Key | SWAN_Execution_Time_Seconds" > "$TIME_LOG"

# Check if the test list exists
if [ ! -f "$TEST_FILE" ]; then
    echo "Error: $TEST_FILE not found! Export it from Python first."
    exit 1
fi

while read -r wind_val wind_dir h wave_dir
do
    echo "--------------------------------------------------------"
    echo "Running SWAN with Wind: $wind_val m/s, Direction: $wind_dir degree, Wave Height: $h m, Wave Direction: $wave_dir degree"

    # 1. Create SWAN input file (Identical to your previous sed command)
    sed -e "s/{{WIND}}/$wind_val $wind_dir/g" \
        -e "s/{{BOUN_CON}}/BOUN SIDE NE CCW CON PAR $h 10.718 $wave_dir 2./g" \
        DUCK_DON_template.swn > DUCK_DON.swn

    # 2. Timing: Start
    start_time=$(date +%s.%N)

    # 3. Run SWAN
    ./swanrun -input DUCK_DON.swn

    # 4. Timing: End & Calculation
    end_time=$(date +%s.%N)
    duration=$(echo "$end_time - $start_time" | bc)
    
    # Save timing to separate file for comparison with DON
    echo "($wind_val, $wind_dir, $h, $wave_dir), $duration" >> "$TIME_LOG"

    # 5. Maintain original output format for post-processing
    # These lines are identical to your original loop to keep data structures consistent
    header="Wind: $wind_val, Direction: $wind_dir; Initial_Wave_H: $h; Initial_Wave_D: $wave_dir"
    
    echo "$header" >> DUCK_training_hsig.txt
    echo "$header" >> DUCK_training_forces.txt
    echo "$header" >> DUCK_training_dir.txt
    echo "$header" >> DUCK_training_PeakPeriod.txt

    cat 2d_duck_hsig.txt >> DUCK_training_hsig.txt
    cat 2d_duck_forces.txt >> DUCK_training_forces.txt
    cat 2d_duck_meandir.txt >> DUCK_training_dir.txt
    cat 2d_duck_RTP.txt >> DUCK_training_PeakPeriod.txt

    echo "" >> DUCK_training_hsig.txt
    echo "" >> DUCK_training_forces.txt
    echo "" >> DUCK_training_dir.txt
    echo "" >> DUCK_training_PeakPeriod.txt

done < "$TEST_FILE"

echo "Processing complete. Timing data saved in $TIME_LOG"
