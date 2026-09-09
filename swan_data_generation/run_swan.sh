#!/bin/bash



wind_values=(1 2 3 4 5 6 7 8)  
wind_direction=(0 45 90 135 180 225 270 315) 
wave_height=(1.6 1.8 2.0 2.2 2.4 2.6) 
wave_direction=(-20 0 20 40 60 80 100 120 140 160)



# Loop over each wind value
for wind_val in "${wind_values[@]}"
do
    for wind_dir in "${wind_direction[@]}"
    do
        for h in "${wave_height[@]}"
        do
            for wave_dir in "${wave_direction[@]}"
            do
            # Create a new SWAN input file with the current wind value
            echo "Running SWAN with Wind: $wind_val m/s, Direction: $wind_dir degree, Wave Height: $h m, Wave Direction: $wave_dir degree"
            #sed -e "s/{{WIND}}/$wind_val $wind_dir/g" -e "s/{{BOUN_CON}}/BOUN SIDE NE CCW CON PAR $h 3.5 $wave_dir 2./g" DUCK_DON_template.swn > DUCK_DON.swn
            sed -e "s/{{WIND}}/$wind_val $wind_dir/g" -e "s/{{BOUN_CON}}/BOUN SIDE NE CCW CON PAR $h 10.718 $wave_dir 2./g" DUCK_DON_template.swn > DUCK_DON.swn

            # Run SWAN with the new input file
            ./swanrun -input DUCK_DON.swn

            # Append the wind value to the output file
            echo "Wind: $wind_val, Direction: $wind_dir; Initial_Wave_H: $h; Initial_Wave_D: $wave_dir" >> DUCK_training_hsig.txt
            echo "Wind: $wind_val, Direction: $wind_dir; Initial_Wave_H: $h; Initial_Wave_D: $wave_dir" >> DUCK_training_forces.txt
            echo "Wind: $wind_val, Direction: $wind_dir; Initial_Wave_H: $h; Initial_Wave_D: $wave_dir" >> DUCK_training_dir.txt
            echo "Wind: $wind_val, Direction: $wind_dir; Initial_Wave_H: $h; Initial_Wave_D: $wave_dir" >> DUCK_training_PeakPeriod.txt

            # Append the relevant output to the results_summary.txt file
            cat 2d_duck_hsig.txt >>DUCK_training_hsig.txt
            cat 2d_duck_forces.txt >>DUCK_training_forces.txt
            cat 2d_duck_meandir.txt >>DUCK_training_dir.txt
            cat 2d_duck_RTP.txt >>DUCK_training_PeakPeriod.txt

            echo "" >> DUCK_training_hsig.txt  # Add a blank line for readability
            echo "" >> DUCK_training_forces.txt  # Add a blank line for readability
            echo "" >> DUCK_training_dir.txt  # Add a blank line for readability
            echo "" >> DUCK_training_PeakPeriod.txt # Add a blank line for readability
            done
        done
    done
done
