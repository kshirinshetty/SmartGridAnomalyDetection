import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import VarianceThreshold
import os

# Chk/make directory
output_directory = 'processed'
if not os.path.exists(output_directory):
    os.makedirs(output_directory)

# Process F0L and F0M first
f0l_processed = None
f0m_processed = None

for filename_base in ['F0L', 'F0M']:
    input_filename = f'dataset/{filename_base}.csv'

    try:
        df = pd.read_csv(input_filename)
        print(f"\nProcessing {input_filename}")

        # Feature Selection

        # 1. Variance Thresholding
        threshold_variance = 0.1  # Adjust as needed
        selector_variance = VarianceThreshold(threshold=threshold_variance)
        df_high_variance = pd.DataFrame(selector_variance.fit_transform(df),
                                        columns=df.columns[selector_variance.get_support()])
        print(f"Removed {df.shape[1] - df_high_variance.shape[1]} low variance features.")

        # 2. Correlation Analysis
        def remove_highly_correlated(df_corr, threshold_corr):
            corr_matrix = df_corr.corr().abs()
            upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
            to_drop = [column for column in upper.columns if any(upper[column] > threshold_corr)]
            df_cleaned = df_corr.drop(columns=to_drop)
            return df_cleaned, to_drop

        correlation_threshold = 0.9  # Adjust as needed
        df_selected_features, dropped_corr = remove_highly_correlated(df_high_variance, correlation_threshold)
        print(f"Removed {len(dropped_corr)} highly correlated features: {dropped_corr}")

        # Z-score Standardization
        if not df_selected_features.empty and df_selected_features.shape[1] > 1:
            time_column = df_selected_features.iloc[:, 0]
            data_to_scale = df_selected_features.iloc[:, 1:]

            scaler = StandardScaler()
            scaled_data = scaler.fit_transform(data_to_scale)
            scaled_df = pd.DataFrame(scaled_data, columns=data_to_scale.columns)
            scaled_df = pd.concat([time_column.reset_index(drop=True), scaled_df], axis=1)

            if filename_base == 'F0L':
                f0l_processed = scaled_df
            else:
                f0m_processed = scaled_df

            print(f"Z-score scaling complete for {filename_base}.")

        elif df_selected_features.shape[1] <= 1:
            print(f"Warning: Only one or zero columns remaining after feature selection for {filename_base}. Skipping scaling.")
            if filename_base == 'F0L':
                f0l_processed = df_selected_features
            else:
                f0m_processed = df_selected_features
        else:
            print(f"Warning: DataFrame is empty after feature selection for {filename_base}.")
            if filename_base == 'F0L':
                f0l_processed = pd.DataFrame()
            else:
                f0m_processed = pd.DataFrame()

    except FileNotFoundError:
        print(f"Error: Could not find {input_filename}")
    except Exception as e:
        print(f"An error occurred while processing {input_filename}: {e}")

# Process the remaining datasets (F1L-F7L & F1M-F7M)

# Using the feature selection learned from F0L and F0M
f0l_selected_columns = f0l_processed.columns.tolist() if f0l_processed is not None else []
f0m_selected_columns = f0m_processed.columns.tolist() if f0m_processed is not None else []

if not f0l_selected_columns or not f0m_selected_columns:
    print("Error: Could not retrieve surviving columns from processed F0L or F0M.")
    exit()

processed_data = {}

for i in range(1, 8):
    for suffix in ['L', 'M']:
        input_filename = f'dataset/F{i}{suffix}.csv'
        selected_columns = f0l_selected_columns if suffix == 'L' else f0m_selected_columns

        try:
            df = pd.read_csv(input_filename)
            print(f"\nProcessing {input_filename}")

            # Keep only the columns that survived in F0L/F0M
            common_columns = [col for col in df.columns if col in selected_columns]
            df_selected = df[common_columns]
            print(f"Kept {len(common_columns)} columns based on F0{suffix}.")

            # Z-score Standardization
            if not df_selected.empty and df_selected.shape[1] > 1:
                time_column = df_selected.iloc[:, 0]
                data_to_scale = df_selected.iloc[:, 1:]

                scaler = StandardScaler()
                scaled_data = scaler.fit_transform(data_to_scale)
                scaled_df = pd.DataFrame(scaled_data, columns=data_to_scale.columns)
                scaled_df = pd.concat([time_column.reset_index(drop=True), scaled_df], axis=1)
                processed_data[f'F{i}{suffix}'] = scaled_df
                print(f"Z-score scaling complete for F{i}{suffix}.")

            elif df_selected.shape[1] <= 1:
                print(f"Warning: Only one or zero columns remaining for F{i}{suffix}. No scaling.")
                processed_data[f'F{i}{suffix}'] = df_selected
            else:
                print(f"Warning: DataFrame is empty for F{i}{suffix}.")
                processed_data[f'F{i}{suffix}'] = pd.DataFrame()

        except FileNotFoundError:
            print(f"Warning: Could not find {input_filename}")
        except Exception as e:
            print(f"An error occurred while processing {input_filename}: {e}")


training_data = []
testing_data = []

print("Loading data for concatenation.")

for i in range(1, 8):
    for suffix in ['L', 'M']:
        key = f'F{i}{suffix}'
        if key in processed_data and not processed_data[key].empty:
            df = processed_data[key].copy()

            # Split into TRAINING and TESTING (e.g., 80% training, 20% testing)
            train_split = int(0.5 * len(df))

            # Add 'source' column only to the training data
            train_df = df.iloc[:train_split].copy()
            train_df['source'] = key
            training_data.append(train_df)

            # Testing data without the 'source' column
            test_df = df.iloc[train_split:].copy()
            testing_data.append(test_df)
        else:
            print(f"Warning: No data available for {key} for concatenation.")

# Concatenate all training and testing data
training_df = pd.concat(training_data, ignore_index=True)
testing_df = pd.concat(testing_data, ignore_index=True)
'''
# Save the concatenated datasets
training_df.to_csv('processed/preTRAINING.csv', index=False)
testing_df.to_csv('processed/preTESTING.csv', index=False)


shuffled_training_df = pd.read_csv('processed/preTRAINING.csv')
shuffled_testing_df = pd.read_csv('processed/preTESTING.csv')

print("Shuffling now. ")
'''
# Shuffle the datasets
training_df.sample(frac=1, random_state=69).to_csv('processed/TRAINING.csv', index=False)
testing_df.sample(frac=1, random_state=69).to_csv('processed/TESTING.csv', index=False)

print("Done.")
