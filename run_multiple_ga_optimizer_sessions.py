# trading_bot/run_multiple_ga_optimizer_sessions.py
import subprocess
import os
import json
import shutil
from datetime import datetime
import pandas as pd
import locale
import time # Added for potential sleep between OOS tests if ever needed

NUMBER_OF_RUNS = 10 # Or your desired number
MAIN_SCRIPT_NAME = "main.py"
BASE_RESULTS_DIR = "ga_multiple_runs_results"

summary_of_runs = []

def parse_backtest_summary_from_log(log_content: str) -> dict:
    # ... (parse_backtest_summary_from_log function remains the same as my previous good version) ...
    parsed_summary = {}
    in_summary_section = False
    last_summary_marker_idx = -1
    try:
        specific_summary_header = "Summary for Selected & CONSTRAINED Genes on Entire Optimization Dataset:"
        specific_header_idx = log_content.rfind(specific_summary_header)
        if specific_header_idx != -1:
            search_area = log_content[specific_header_idx:]
            backtest_summary_marker_in_area = search_area.find("--- Backtest Summary ---")
            if backtest_summary_marker_in_area != -1:
                last_summary_marker_idx = specific_header_idx + backtest_summary_marker_in_area
        if last_summary_marker_idx == -1: # Fallback if specific header not found
            specific_summary_header_alt = "Summary for 'backtest_best':" # For backtest_best mode if used by mistake
            specific_header_idx_alt = log_content.rfind(specific_summary_header_alt)
            if specific_header_idx_alt != -1:
                 search_area_alt = log_content[specific_header_idx_alt:]
                 backtest_summary_marker_in_area_alt = search_area_alt.find("--- Backtest Summary ---")
                 if backtest_summary_marker_in_area_alt != -1:
                     last_summary_marker_idx = specific_header_idx_alt + backtest_summary_marker_in_area_alt
        if last_summary_marker_idx == -1: # Ultimate fallback
            last_summary_marker_idx = log_content.rfind("--- Backtest Summary ---")
    except Exception: pass
    if last_summary_marker_idx == -1: return {}
    log_after_marker = log_content[last_summary_marker_idx:]
    lines = log_after_marker.splitlines()
    for line in lines:
        if "-----------------------------------" in line or "--------------------" in line:
            if not in_summary_section: in_summary_section = True; continue
            else: break
        if in_summary_section and ":" in line:
            parts = line.split(':', 1)
            if len(parts) == 2:
                key = parts[0].strip(); value_str = parts[1].strip()
                try:
                    if "%" in value_str: parsed_summary[key] = float(value_str.replace("%", "").strip())
                    elif "days" in value_str or (value_str.count(':') == 2 and "0 days" not in value_str.lower() and "avg holding period" in key.lower()):
                        parsed_summary[key] = value_str
                    else: parsed_summary[key] = float(value_str.replace("$", "").replace(",", "").strip())
                except ValueError: parsed_summary[key] = value_str
    return parsed_summary


def run_session(session_num: int, session_dir: str):
    print(f"\n--- Starting GA Optimizer Session {session_num + 1}/{NUMBER_OF_RUNS} ---")
    print(f"Results will be saved in: {session_dir}")

    command = ["python", MAIN_SCRIPT_NAME, "optimize"]
    log_file_path = os.path.join(session_dir, f"run_{session_num + 1}_optimize_output.txt")
    best_genes_src = "best_genes.json"
    ga_log_src = "ga_evolution_log.jsonl"
    validation_log_src = "validation_candidates_log.jsonl"
    all_validation_attempts_log_src = "all_validation_attempts_log.jsonl"

    best_genes_dst = os.path.join(session_dir, best_genes_src)
    ga_log_dst = os.path.join(session_dir, ga_log_src)
    validation_log_dst = os.path.join(session_dir, validation_log_src)
    all_validation_attempts_log_dst = os.path.join(session_dir, all_validation_attempts_log_src)

    session_summary_data = {"run_id": session_num + 1, "directory": session_dir, "status": "Pending"}
    preferred_encoding = locale.getpreferredencoding()
    full_log_content = ""

    try:
        with subprocess.Popen(
            command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
            encoding=preferred_encoding if preferred_encoding else 'utf-8',
            errors='replace', bufsize=1, universal_newlines=True
        ) as process:
            full_log_content_lines = []
            with open(log_file_path, 'w', encoding=preferred_encoding if preferred_encoding else 'utf-8', errors='replace') as f_log:
                if process.stdout:
                    for line in process.stdout:
                        print(line, end='')
                        f_log.write(line)
                        full_log_content_lines.append(line)
            process.wait()
            full_log_content = "".join(full_log_content_lines)

        # Enhanced failure detection
        process_successful_exit_code = (process.returncode == 0)
        fatal_error_keyword_found = False
        general_error_keyword_found = False
        
        # More specific fatal keywords that indicate the GA process itself failed or was aborted
        fatal_keywords = [
            "CRITICAL - Not enough data", "CRITICAL - GA Init Error", "CRITICAL - GA Run Error",
            "CRITICAL: Failed to fetch data", "CRITICAL: OSError", "MemoryError",
            "terminate called after throwing an instance of", # C++ exceptions from underlying libs
            "failed to initialize", "Aborted (core dumped)" # System level failures
            # "Traceback (most recent call last):" # Too general, rely on non-zero exit code for unhandled Python exceptions
        ]
        # General error keyword
        general_error_keyword = "ERROR -" # For non-fatal errors logged by our app

        for keyword in fatal_keywords:
            if keyword in full_log_content: # Case-sensitive for these usually
                fatal_error_keyword_found = True
                print(f"\nFATAL error indicator '{keyword}' found in log for session {session_num + 1}.")
                break
        
        if general_error_keyword in full_log_content:
            general_error_keyword_found = True
            print(f"\nGeneral '{general_error_keyword}' keyword found in log for session {session_num + 1}.")

        session_failed = False
        status_msg_detail = ""

        if not process_successful_exit_code:
            session_failed = True
            status_msg_detail = f"exit_code={process.returncode}"
        elif fatal_error_keyword_found:
            session_failed = True
            status_msg_detail = "fatal_log_keyword_found"
        
        # If process exited cleanly and no fatal keywords, check if a summary was produced
        # This helps to distinguish between a GA run that found no good strategies vs. one that had issues
        backtest_summary_parsed = {}
        if not session_failed: # Only try to parse summary if not already marked as failed
            backtest_summary_parsed = parse_backtest_summary_from_log(full_log_content)
            if not backtest_summary_parsed:
                # If no summary, but also no fatal errors and exit code 0,
                # it might be that GA simply didn't find a suitable strategy to write to best_genes.json and do final backtest.
                # We can consider this a "success" in terms of script execution, but note the lack of results.
                status_msg_detail = "no_final_summary_parsed"
                print(f"\nSession {session_num + 1} completed (exit code 0, no fatal logs) but no final backtest summary was parsed.")
                # If general "ERROR -" was present, it might explain no summary.
                if general_error_keyword_found:
                    status_msg_detail += "_with_general_errors"
                    # Decide if this specific case should be marked as failure:
                    # session_failed = True 
            # else: summary was parsed successfully


        if session_failed:
            session_summary_data["status"] = f"Failed ({status_msg_detail})"
            print(f"\nSession {session_num + 1} determined as FAILED. Status: {session_summary_data['status']}")
            session_summary_data["log_on_fail_tail"] = full_log_content[-3000:]
        else: # Considered successful execution of the GA process
            session_summary_data["status"] = f"Success ({status_msg_detail})" if status_msg_detail else "Success"
            if general_error_keyword_found and not status_msg_detail: # if success but had general errors
                session_summary_data["status"] = "Success (with_general_errors_in_log)"

            print(f"\nSession {session_num + 1} determined as SUCCESSFUL. Status: {session_summary_data['status']}")
            
            if os.path.exists(best_genes_src):
                shutil.move(best_genes_src, best_genes_dst)
                try:
                    with open(best_genes_dst, 'r', encoding='utf-8') as f_genes:
                        genes_data = json.load(f_genes)
                        session_summary_data["best_genes"] = genes_data
                except Exception as e_read_genes:
                    session_summary_data["best_genes"] = f"Error reading: {e_read_genes}"
            else:
                session_summary_data["best_genes"] = "Not found (or GA yielded no candidates)"

            if os.path.exists(ga_log_src): shutil.move(ga_log_src, ga_log_dst)
            if os.path.exists(validation_log_src): shutil.move(validation_log_src, validation_log_dst)
            if os.path.exists(all_validation_attempts_log_src): shutil.move(all_validation_attempts_log_src, all_validation_attempts_log_dst)

            if backtest_summary_parsed:
                session_summary_data.update(backtest_summary_parsed)

    except FileNotFoundError:
        session_summary_data["status"] = "Script not found"
        full_log_content = f"Error: '{MAIN_SCRIPT_NAME}' not found."
        print(full_log_content)
    except Exception as e:
        session_summary_data["status"] = f"Run Error: {str(e)}"
        import traceback
        tb_str = traceback.format_exc()
        session_summary_data["error_traceback"] = tb_str
        full_log_content += f"\n--- PYTHON EXCEPTION IN RUN_SESSION SCRIPT ---\n{tb_str}"
        print(f"An unhandled error occurred during session {session_num + 1} execution: {e}")
        print(tb_str)
    
    if not os.path.exists(log_file_path) or os.path.getsize(log_file_path) == 0 :
        try:
            with open(log_file_path, 'w', encoding=preferred_encoding if preferred_encoding else 'utf-8', errors='replace') as f_log_fallback:
                f_log_fallback.write(full_log_content if full_log_content else "Log content was empty or unavailable.")
        except Exception: pass # Ignore if even fallback log writing fails

    summary_of_runs.append(session_summary_data)
    print(f"--- Finished GA Optimizer Session {session_num + 1}/{NUMBER_OF_RUNS} ---")

if __name__ == "__main__":
    # ... (The rest of the __main__ block from my previous good version of run_multiple_ga_optimizer_sessions.py remains the same) ...
    # ... (This includes creating directories, cleaning up root files, and the final CSV/JSONL summary generation) ...
    if not os.path.exists(BASE_RESULTS_DIR):
        os.makedirs(BASE_RESULTS_DIR)
    timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S")
    overall_run_dir = os.path.join(BASE_RESULTS_DIR, f"ga_runs_{timestamp_str}")
    os.makedirs(overall_run_dir)
    print(f"Overall results directory: {overall_run_dir}")

    files_to_cleanup_in_root = [
        "best_genes.json", "ga_evolution_log.jsonl",
        "validation_candidates_log.jsonl", "all_validation_attempts_log.jsonl",
        "oos_kline_data.csv", "oos_trades_log.csv"
    ]
    for f_name in files_to_cleanup_in_root:
        if os.path.exists(f_name):
            try: os.remove(f_name)
            except Exception: pass

    for i in range(NUMBER_OF_RUNS):
        session_dir_name = f"run_{i + 1}"
        session_full_dir = os.path.join(overall_run_dir, session_dir_name)
        os.makedirs(session_full_dir)
        run_session(i, session_full_dir)
        lingering_files_check = ["best_genes.json", "ga_evolution_log.jsonl", "validation_candidates_log.jsonl", "all_validation_attempts_log.jsonl"]
        for lf_name in lingering_files_check:
            if os.path.exists(lf_name):
                 print(f"Warning: File {lf_name} found in root after session {i+1}. Cleaning up.")
                 try: os.remove(lf_name)
                 except: pass
    
    if summary_of_runs:
        try:
            df_summary = pd.DataFrame(summary_of_runs)
            if "best_genes" in df_summary.columns:
                 df_summary["best_genes_str"] = df_summary["best_genes"].apply(lambda x: json.dumps(x) if isinstance(x, dict) else str(x))
            
            desired_first_cols = ['run_id', 'status', 'Total PnL Pct', 'Max Drawdown', 'Sharpe Ratio', 'Profit Factor', 'Total Trades']
            current_cols = df_summary.columns.tolist()
            final_ordered_cols = []
            for col in desired_first_cols:
                if col in current_cols: final_ordered_cols.append(col)
            other_cols = [c for c in current_cols if c not in final_ordered_cols and c not in ["best_genes_str", "directory", "best_genes", "log_on_fail_tail", "error_traceback"]]
            final_ordered_cols.extend(sorted(other_cols))
            end_cols = ["best_genes", "best_genes_str", "directory", "log_on_fail_tail", "error_traceback"]
            for col in end_cols:
                if col in current_cols and col not in final_ordered_cols: final_ordered_cols.append(col)
            final_ordered_cols_existing = [col for col in final_ordered_cols if col in df_summary.columns]
            df_summary = df_summary[final_ordered_cols_existing]

            summary_csv_path = os.path.join(overall_run_dir, "all_runs_summary.csv")
            df_summary.to_csv(summary_csv_path, index=False, encoding='utf-8-sig')
            print(f"\nSummary of all runs saved to: {summary_csv_path}")
            
            print("\n--- Aggregated Summary of Runs ---")
            display_cols = [col for col in desired_first_cols if col in df_summary.columns]
            if not display_cols and len(df_summary.columns) > 0:
                display_cols = df_summary.columns.tolist()[:min(7, len(df_summary.columns))]
            if display_cols and not df_summary.empty: print(df_summary[display_cols].to_string(index=False))
            elif df_summary.empty: print("Summary DataFrame is empty.")
            else: print("No data to display in summary table.")
        except Exception as e_csv:
            print(f"Error creating or processing summary CSV: {e_csv}"); import traceback; print(traceback.format_exc())
            summary_json_path = os.path.join(overall_run_dir, "all_runs_summary.jsonl")
            try:
                with open(summary_json_path, 'w', encoding='utf-8') as f_json_summary:
                    for i_run_idx in range(len(summary_of_runs)):
                        run_data_copy = summary_of_runs[i_run_idx].copy()
                        if 'best_genes' in run_data_copy and isinstance(run_data_copy['best_genes'], dict):
                            run_data_copy['best_genes'] = json.dumps(run_data_copy['best_genes'])
                        for key, value in run_data_copy.items():
                            if isinstance(value, float) and (pd.isna(value) or pd.isnull(value)): run_data_copy[key] = None
                        f_json_summary.write(json.dumps(run_data_copy, ensure_ascii=False) + '\n')
                print(f"Summary of all runs (JSONL due to CSV error) saved to: {summary_json_path}")
            except Exception as e_json_save: print(f"Error saving summary JSONL: {e_json_save}")
    else: print("No GA sessions were run or summarized.")
    print("\nAll GA optimizer sessions finished.")
