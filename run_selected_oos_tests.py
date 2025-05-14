# run_selected_oos_tests.py
import pandas as pd
import json
import subprocess
import os
import shutil 
import time # Added for potential sleep

# --- 配置 ---
# !! 请务必将下一行修改为你实际的 all_runs_summary.csv 文件路径 !!
SUMMARY_CSV_FILE = r"ga_multiple_runs_results\ga_runs_20250514_182400\all_runs_summary.csv" 

RUN_IDS_TO_TEST_OOS = [4, 8, 2, 1] 
MAIN_SCRIPT_FOR_OOS = "main.py"
BEST_GENES_FILE_FOR_MAIN = "best_genes.json" 
RESULTS_DIR = "oos_test_results_latest_ga" # 使用一个新的目录名以避免与旧结果混淆

# --- 脚本主逻辑 ---

def run_oos_for_run_id(df_summary: pd.DataFrame, run_id: int, base_results_dir: str):
    print(f"\n--- Starting OOS Test for Run ID: {run_id} ---")

    run_data = df_summary[df_summary['run_id'] == run_id]
    if run_data.empty:
        print(f"Error: Run ID {run_id} not found in {SUMMARY_CSV_FILE}.")
        return
    
    run_status = run_data.iloc[0]['status']
    # Corrected status check:
    # We consider it a success if the word "Success" is in the status string.
    # This allows for statuses like "Success (with_general_errors_in_log)" or "Success (no_final_summary_parsed)"
    if not isinstance(run_status, str) or "Success" not in run_status:
        print(f"Skipping Run ID {run_id} because its status does not indicate overall success (Status: {run_status}).")
        return

    genes_str = run_data.iloc[0]['best_genes_str']
    # Check if genes_str is NaN or actually "Not found..." string from the CSV
    if pd.isna(genes_str) or (isinstance(genes_str, str) and "Not found" in genes_str) :
        print(f"Skipping Run ID {run_id} because best_genes_str is missing or indicates no genes found.")
        return
        
    try:
        # It seems genes_str might sometimes be a stringified dict already, not a JSON string of a dict string
        # Let's try to handle both potential cases, though json.loads should handle dict-like strings
        if isinstance(genes_str, str):
            genes_dict = json.loads(genes_str)
        else: # Should not happen if CSV is consistent, but as a fallback
            print(f"Warning: genes_str for Run ID {run_id} is not a string. Attempting to use as is if dict.")
            if isinstance(genes_str, dict):
                genes_dict = genes_str
            else:
                raise TypeError("genes_str is neither a string nor a dictionary.")
                
    except (json.JSONDecodeError, TypeError) as e:
        print(f"Error processing genes_str for Run ID {run_id}: {e}")
        print(f"Problematic genes_str type: {type(genes_str)}, value: {genes_str}")
        return

    try:
        with open(BEST_GENES_FILE_FOR_MAIN, 'w') as f:
            json.dump(genes_dict, f, indent=4)
        print(f"Successfully wrote genes for Run ID {run_id} to {BEST_GENES_FILE_FOR_MAIN}")
    except Exception as e:
        print(f"Error writing {BEST_GENES_FILE_FOR_MAIN} for Run ID {run_id}: {e}")
        return

    current_oos_run_dir = os.path.join(base_results_dir, f"oos_run_{run_id}")
    os.makedirs(current_oos_run_dir, exist_ok=True)
    
    oos_log_file_path = os.path.join(current_oos_run_dir, f"oos_output_run_{run_id}.txt")
    print(f"Running OOS backtest for Run ID {run_id}. Output will be logged to: {oos_log_file_path}")

    command = ["python", MAIN_SCRIPT_FOR_OOS, "backtest_oos"]
    full_oos_log_content = ""
    try:
        # import locale # Add this if not already imported at the top
        # preferred_encoding = locale.getpreferredencoding() # Get system's preferred encoding

        with subprocess.Popen(
            command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
            # encoding=preferred_encoding if preferred_encoding else 'utf-8', # Use system preferred or fallback to utf-8
            encoding='utf-8', # 或者直接强制使用 utf-8
            errors='replace', bufsize=1, universal_newlines=True
        ) as process:
            log_lines = []
            if process.stdout:
                for line in process.stdout:
                    print(line, end='') # Print to console as it comes
                    log_lines.append(line)
            process.wait()
            full_oos_log_content = "".join(log_lines)

        with open(oos_log_file_path, 'w', encoding='utf-8', errors='replace') as f_log: # Write log file also with utf-8
            f_log.write(full_oos_log_content)

        if process.returncode == 0:
            print(f"\nOOS backtest for Run ID {run_id} completed. Log: {oos_log_file_path}")
            files_to_move = ["oos_kline_data.csv", "oos_trades_log.csv", BEST_GENES_FILE_FOR_MAIN]
            for file_name in files_to_move:
                src_path = file_name 
                # Add run_id to the destination filename to avoid overwriting from different OOS runs
                dst_file_name_base, dst_file_ext = os.path.splitext(file_name)
                dst_path = os.path.join(current_oos_run_dir, f"{dst_file_name_base}_run_{run_id}{dst_file_ext}")
                if os.path.exists(src_path):
                    try: shutil.move(src_path, dst_path); print(f"Moved {src_path} to {dst_path}")
                    except Exception as e_move: print(f"Error moving {src_path} to {dst_path}: {e_move}")
        else:
            print(f"Error: OOS backtest for Run ID {run_id} failed with return code {process.returncode}.")
    except FileNotFoundError: print(f"Error: '{MAIN_SCRIPT_FOR_OOS}' not found.")
    except Exception as e:
        print(f"An error occurred during OOS test for Run ID {run_id}: {e}"); import traceback; print(traceback.format_exc())
    print(f"--- Finished OOS Test for Run ID: {run_id} ---")

if __name__ == "__main__":
    if not os.path.exists(SUMMARY_CSV_FILE):
        print(f"Error: Summary CSV file not found at '{SUMMARY_CSV_FILE}'")
        print("Please update the SUMMARY_CSV_FILE variable in this script to the correct path.")
        exit(1)
        
    if not os.path.exists(RESULTS_DIR):
        os.makedirs(RESULTS_DIR); print(f"Created OOS results directory: {RESULTS_DIR}")
    try:
        df_summary_all_runs = pd.read_csv(SUMMARY_CSV_FILE)
    except Exception as e: print(f"Error reading summary CSV file '{SUMMARY_CSV_FILE}': {e}"); exit(1)

    for run_id_to_test in RUN_IDS_TO_TEST_OOS:
        run_oos_for_run_id(df_summary_all_runs, run_id_to_test, RESULTS_DIR)
        time.sleep(0.5) # Small delay between runs
    print("\nAll selected OOS tests finished.")
