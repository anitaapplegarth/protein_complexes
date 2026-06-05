#!/bin/bash
# Run from your ml_pipeline directory
# Creates a folder called 'split_results_collected' with renamed files

mkdir -p split_results_collected

for model_dir in randomforest lightgbm xgboost; do
    # Short model names for filenames
    case $model_dir in
        randomforest) model="rf" ;;
        lightgbm) model="lgbm" ;;
        xgboost) model="xgb" ;;
    esac
    
    for experiment_dir in "$model_dir"/*/; do
        # Get the experiment name (e.g., cp_all_stoich_features)
        experiment=$(basename "$experiment_dir")
        
        # Find split_results.csv in any subfolder (*_family_splits)
        for splits_dir in "$experiment_dir"*_family_splits/; do
            if [ -f "${splits_dir}split_results.csv" ]; then
                # Remove '_features' suffix from experiment name
                clean_name=$(echo "$experiment" | sed 's/_features$//')
                cp "${splits_dir}split_results.csv" "split_results_collected/${clean_name}_${model}.csv"
                echo "Copied: ${clean_name}_${model}.csv"
            fi
        done
    done
done

echo ""
echo "Total files collected:"
ls split_results_collected/ | wc -l
echo ""
echo "Files:"
ls split_results_collected/
