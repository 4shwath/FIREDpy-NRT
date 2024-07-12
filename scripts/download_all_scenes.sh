#!/bin/bash

# Define your list of fire IDs
#fire_ids=(9844 11862 12887 202 280 5072 5532 7123 25993 47422 2282 2614 4413 5587 5615 5950 2819 5447 7792 8676)
fire_ids=(2614, 7792, 2282, 5615)

# Iterate through the first five fire IDs
for ((i=0; i<20; i++))
do
    # Retry up to 3 times for sentinel download if the script fails
    retries=0
    while [ $retries -lt 3 ]
    do
        python3 download_scenes.py -id "${fire_ids[i]}" -s "sentinel"
        if [ $? -eq 0 ]; then
            break
        else
            retries=$((retries + 1))
            echo "Failed to execute, retrying (attempt $retries)..."
        fi
    done
    if [ $retries -eq 3 ]; then
        echo "Failed to execute after 3 attempts, moving to the next fire ID."
    fi

    # Retry up to 3 times for landsat download if the script fails
    retries=0
    while [ $retries -lt 3 ]
    do
        python3 download_scenes.py -id "${fire_ids[i]}" -s "landsat"
        if [ $? -eq 0 ]; then
            break
        else
            retries=$((retries + 1))
            echo "Failed to execute, retrying (attempt $retries)..."
        fi
    done
    if [ $retries -eq 3 ]; then
        echo "Failed to execute after 3 attempts, moving to the next fire ID."
    fi
done
