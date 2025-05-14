# trading_bot/data_handler.py
import pandas as pd
from datetime import datetime, timezone, timedelta
import time
import math

import config
from okx_connector import OKXConnector

class DataHandler:
    def __init__(self, connector: OKXConnector):
        self.connector = connector

    def fetch_klines_to_df(self, instId: str, bar: str, total_limit_needed: int,
                           kline_type: str = "history_market") -> pd.DataFrame: # Default to history_market
        print(f"DEBUG DataHandler: Fetching type='{kline_type}', total_limit={total_limit_needed} for {instId} ({bar}).")

        all_fetched_klines_pages = []
        current_page_boundary_ts_for_older_data = None

        api_max_limit_per_request = 0
        connector_method_to_call = None
        pagination_param_name_for_older_data = ""

        if kline_type == "history_market": # Primary method for fetching historical data
            api_max_limit_per_request = 100
            connector_method_to_call = self.connector.get_history_market_candles
            pagination_param_name_for_older_data = "after" # 'after' gets older for this endpoint
            print(f"  DEBUG DataHandler ({kline_type}): Using '{pagination_param_name_for_older_data}' for pagination. API limit: {api_max_limit_per_request}")
        elif kline_type == "market": # Fallback or for very recent data, pagination unreliable on demo
            api_max_limit_per_request = 100 # User confirmed limit
            connector_method_to_call = self.connector.get_market_candles
            pagination_param_name_for_older_data = "before"
            print(f"  DEBUG DataHandler ({kline_type}): Using '{pagination_param_name_for_older_data}' for pagination. API limit: {api_max_limit_per_request}. WARNING: Pagination for this type is unreliable on OKX Demo.")
            if total_limit_needed > api_max_limit_per_request * 3:
                print(f"  INFO DataHandler ({kline_type}): Requested {total_limit_needed} but due to pagination issues, will attempt to fetch only up to {api_max_limit_per_request * 2} recent klines.")
                total_limit_needed = api_max_limit_per_request * 2
        else:
            print(f"ERROR: Unknown kline_type '{kline_type}'. Supported: 'history_market', 'market'.")
            return pd.DataFrame()

        max_iterations = math.ceil(total_limit_needed / api_max_limit_per_request) + 15 # Increased buffer slightly
        print(f"  DEBUG DataHandler ({kline_type}): Max iterations for up to {total_limit_needed} klines: {max_iterations}.")

        MAX_API_CALL_RETRIES = 3 # Max retries for a single page fetch if it fails (e.g. timeout)
        INITIAL_RETRY_DELAY_SECONDS = 2 # Initial delay for retries

        for i in range(max_iterations):
            if all_fetched_klines_pages:
                if i > 0 and (i % 5 == 0 or i == max_iterations - 1):
                    temp_flat_list_for_count = sum(all_fetched_klines_pages, [])
                    if temp_flat_list_for_count:
                        timestamps_for_count = [k[0] for k in temp_flat_list_for_count if isinstance(k, list) and len(k)>0]
                        current_approx_unique_count = pd.Series(timestamps_for_count).nunique()
                        if current_approx_unique_count >= total_limit_needed :
                            print(f"  DEBUG DataHandler ({kline_type}): Approx {current_approx_unique_count} unique klines. Reached target {total_limit_needed}. Stopping.")
                            break

            if sum(len(p) for p in all_fetched_klines_pages) >= total_limit_needed + api_max_limit_per_request :
                print(f"  DEBUG DataHandler ({kline_type}): Accumulated raw {sum(len(p) for p in all_fetched_klines_pages)} klines. Stopping early.")
                break

            limit_for_this_call = api_max_limit_per_request
            call_params = {"instId": instId, "bar": bar, "limit": limit_for_this_call}
            if current_page_boundary_ts_for_older_data is not None:
                call_params[pagination_param_name_for_older_data] = str(current_page_boundary_ts_for_older_data)

            page_klines = None
            current_retry_delay = INITIAL_RETRY_DELAY_SECONDS

            for attempt in range(MAX_API_CALL_RETRIES):
                print(f"  DEBUG DataHandler ({kline_type}): Fetching page {i + 1}/{max_iterations} (Attempt {attempt+1}/{MAX_API_CALL_RETRIES}) with params: {call_params}")
                try:
                    page_klines = connector_method_to_call(**call_params)
                    if page_klines is not None: # API call might return None or []
                        break # Success, exit retry loop
                except Exception as e_call: # Catch broader exceptions from connector if it raises them on failure
                    print(f"    WARNING DataHandler ({kline_type}): API call error on page {i+1}, attempt {attempt+1}: {e_call}")
                
                # If page_klines is still None or empty after call (or exception occurred)
                if page_klines is None or not page_klines:
                    if attempt < MAX_API_CALL_RETRIES - 1:
                        print(f"    WARNING DataHandler ({kline_type}): Failed to fetch page {i+1} (attempt {attempt+1}). Retrying in {current_retry_delay}s...")
                        time.sleep(current_retry_delay)
                        current_retry_delay *= 2 # Exponential backoff for retries
                    else:
                        print(f"    ERROR DataHandler ({kline_type}): Max retries ({MAX_API_CALL_RETRIES}) reached for page {i+1}. Giving up on this page.")
                        break # Exit retry loop, page_klines will be None or empty
            
            if not page_klines: # If still no data after retries or if it was an empty response initially
                print(f"  DEBUG DataHandler ({kline_type}): No data from connector page {i + 1} (after retries or empty response). Boundary: {current_page_boundary_ts_for_older_data}. End of history or persistent error.")
                break

            all_fetched_klines_pages.insert(0, page_klines)
            new_boundary_for_next_call = page_klines[0][0] # Oldest timestamp of the current fetched page

            if current_page_boundary_ts_for_older_data is not None:
                if new_boundary_for_next_call < current_page_boundary_ts_for_older_data:
                    pass
                else:
                    print(f"    WARNING DataHandler ({kline_type}): Pagination stall! New boundary ({new_boundary_for_next_call}) NOT OLDER than current ({current_page_boundary_ts_for_older_data}). Breaking.")
                    break
            current_page_boundary_ts_for_older_data = new_boundary_for_next_call

            if i < max_iterations - 1:
                # *** INCREASED SLEEP DURATION HERE ***
                sleep_duration = 0.30 if kline_type == "history_market" else 0.20 # Increased from 0.11 and 0.06
                # print(f"    DEBUG DataHandler ({kline_type}): Sleeping for {sleep_duration}s before next page request.") # Optional debug
                time.sleep(sleep_duration)

        final_kline_list = sum(all_fetched_klines_pages, [])
        if not final_kline_list: return pd.DataFrame()

        processed_klines = [k for k in final_kline_list if isinstance(k, list) and len(k) == 6]
        if not processed_klines: return pd.DataFrame()

        df = pd.DataFrame(processed_klines, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
        og_len = len(df)
        df.drop_duplicates(subset=['timestamp'], keep='first', inplace=True)
        df.sort_values(by='timestamp', ascending=True, inplace=True)

        if len(df) > total_limit_needed: df = df.iloc[-total_limit_needed:]
        elif len(df) < total_limit_needed and all_fetched_klines_pages and i < max_iterations -1 : # Only warn if not stopped by max_iterations
             # The i < max_iterations -1 check is to avoid warning if we naturally hit the end of history
             # before reaching total_limit_needed but also hit max_iterations.
             if not (sum(len(p) for p in all_fetched_klines_pages) < total_limit_needed and i == max_iterations -1) : # A bit complex condition to avoid false warning
                print(f"WARNING ({kline_type}): Fetched {len(df)} unique klines, less than requested {total_limit_needed}.")


        if df.empty: return pd.DataFrame()
        df['datetime'] = pd.to_datetime(df['timestamp'], unit='ms')
        df.set_index('datetime', inplace=True)
        if 'timestamp' in df.columns: df.drop(columns=['timestamp'], inplace=True)
        print(f"DEBUG DataHandler ({kline_type}): Final DataFrame shape: {df.shape}")
        return df

if __name__ == '__main__':
    print("Testing DataHandler with kline_type option and increased sleep/retries...")
    # You need to have OKXConnector available for this test to run
    try:
        okx_conn_dh_test = OKXConnector()
        data_handler_test_instance = DataHandler(okx_conn_dh_test)

        desired_klines_test = 700
        test_bar_dh = config.TIMEFRAME

        print(f"\n--- Test 1: Fetching 'history_market' klines (instId: {config.SYMBOL}, bar: {test_bar_dh}, target: {desired_klines_test}) ---")
        df_hist_market = data_handler_test_instance.fetch_klines_to_df(
            instId=config.SYMBOL, bar=test_bar_dh,
            total_limit_needed=desired_klines_test,
            kline_type="history_market"
        )
        if df_hist_market is not None and not df_hist_market.empty:
            print(f"History Market klines fetched. Final Shape: {df_hist_market.shape}")
            if not df_hist_market.empty: print("First 2:\n", df_hist_market.head(2), "\nLast 2:\n", df_hist_market.tail(2))
            if df_hist_market.index.is_monotonic_increasing: print("HM Timestamps sorted.")
        else: print("Failed to fetch history market klines or DataFrame empty.")

        desired_recent_klines_test = 150
        print(f"\n--- Test 2: Fetching 'market' klines (instId: {config.SYMBOL}, bar: {test_bar_dh}, target: {desired_recent_klines_test}) ---")
        df_market = data_handler_test_instance.fetch_klines_to_df(
            instId=config.SYMBOL, bar=test_bar_dh,
            total_limit_needed=desired_recent_klines_test,
            kline_type="market"
        )
        if df_market is not None and not df_market.empty:
            print(f"Recent Market klines fetched. Final Shape: {df_market.shape}")
            if not df_market.empty: print("First 2:\n", df_market.head(2), "\nLast 2:\n", df_market.tail(2))
        else: print("Failed to fetch recent market klines or DataFrame empty.")

    except NameError:
        print("Error: OKXConnector class not found. Make sure okx_connector.py is in the same directory or PYTHONPATH.")
    except Exception as e:
        print(f"An error occurred during DataHandler test: {e}")
        import traceback
        traceback.print_exc()

    print("\nDataHandler test script finished.")
