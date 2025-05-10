# trading_bot/okx_connector.py
import okx.Trade as Trade
import okx.MarketData as MarketData
import okx.Account as Account
import json
from datetime import datetime, timezone, timedelta
import time
import hmac
import base64
# import requests # Not needed if using SDK methods only

import config

class OKXConnector:
    def __init__(self):
        if config.IS_DEMO_TRADING:
            self.api_key = config.DEMO_API_KEY
            self.secret_key = config.DEMO_SECRET_KEY
            self.passphrase = config.DEMO_PASSPHRASE
            self.flag = '1' # Demo trading flag
            # self.domain = "https://www.okx.com" # Domain for manual requests, not needed for SDK calls
            print("OKX Connector initialized for DEMO TRADING.")
        else:
            self.api_key = config.REAL_API_KEY
            self.secret_key = config.REAL_SECRET_KEY
            self.passphrase = config.REAL_PASSPHRASE
            self.flag = '0' # Real trading flag
            # self.domain = "https://www.okx.com"
            print("OKX Connector initialized for REAL TRADING.")
            print("!!! WARNING: OPERATING WITH REAL FUNDS !!!")

        current_mode = "DEMO" if config.IS_DEMO_TRADING else "REAL"
        if not self.api_key or "YOUR_" in self.api_key or self.api_key == "":
            raise ValueError(f"API Key for {current_mode} trading is not configured in config.py.")
        if not self.secret_key or "YOUR_" in self.secret_key or self.secret_key == "":
            raise ValueError(f"Secret Key for {current_mode} trading is not configured in config.py.")
        if not self.passphrase or "YOUR_" in self.passphrase or self.passphrase == "": # Passphrase might be empty for some API key types
            raise ValueError(f"Passphrase for {current_mode} trading is not configured in config.py (or is empty if required).")

        self.trade_api = Trade.TradeAPI(self.api_key, self.secret_key, self.passphrase, flag=self.flag)
        self.market_api = MarketData.MarketAPI(self.api_key, self.secret_key, self.passphrase, flag=self.flag)
        self.account_api = Account.AccountAPI(self.api_key, self.secret_key, self.passphrase, flag=self.flag)

    def _process_candle_data(self, api_data_list: list, api_endpoint_name: str, expect_volume_at_index_5: bool = True) -> list:
        """
        Helper to process raw candle data from API.
        Assumes API always returns newest first. This method reverses it.
        expect_volume_at_index_5: True if data[i][5] is volume, False if it's 'confirm' or other.
        """
        klines_from_api = []
        if not isinstance(api_data_list, list):
            print(f"    WARNING OKXConnector ({api_endpoint_name}): API data is not a list: {api_data_list}")
            return []
            
        for k_data_point in api_data_list:
            if isinstance(k_data_point, list) and len(k_data_point) >= 5: # ts,o,h,l,c
                volume_value = 0.0
                if expect_volume_at_index_5 and len(k_data_point) >= 6:
                    try:
                        volume_value = float(k_data_point[5])
                    except (ValueError, TypeError):
                        # print(f"    WARNING OKXConnector ({api_endpoint_name}): Invalid volume data '{k_data_point[5]}', using 0.0.")
                        volume_value = 0.0
                elif not expect_volume_at_index_5 : # e.g. mark/index price 'confirm' field
                    pass # volume_value remains 0.0

                klines_from_api.append([
                    int(k_data_point[0]), float(k_data_point[1]), float(k_data_point[2]),
                    float(k_data_point[3]), float(k_data_point[4]),
                    volume_value
                ])
            else:
                print(f"    WARNING OKXConnector ({api_endpoint_name}): Invalid kline data point format: {k_data_point}")
        
        if klines_from_api:
            print(f"    DEBUG OKXConnector ({api_endpoint_name}): API raw list {len(klines_from_api)}. First (API newest): ts={klines_from_api[0][0]}, Last (API oldest): ts={klines_from_api[-1][0]}")
        
        klines_to_return = klines_from_api[::-1] # Reverse: API returns newest first -> we want oldest first for the page
        
        if klines_to_return:
            print(f"    DEBUG OKXConnector ({api_endpoint_name}): Data to return (oldest first for page): {len(klines_to_return)}. First: ts={klines_to_return[0][0]}, Last: ts={klines_to_return[-1][0]}")
        
        return klines_to_return

    def get_market_candles(self, instId: str, bar: str = config.TIMEFRAME, limit: int = config.DEFAULT_KLINE_LIMIT, 
                           before: str = None, after: str = None) -> list:
        """Fetches RECENT MARKET K-line data from /api/v5/market/candles."""
        endpoint_name = "/market/candles"
        try:
            API_MAX_LIMIT = 100 # Confirmed by user for this endpoint
            params = {'instId': instId, 'bar': bar, 'limit': str(min(limit, API_MAX_LIMIT))}
            # For THIS endpoint, 'before' gets older data.
            if before:
                try: 
                    # Attempt -1ms adjustment for 'before' with market candles due to past issues
                    params['before'] = str(int(before) - 1) 
                    # print(f"    DEBUG OKXConnector ({endpoint_name}): Original 'before'={before}, adjusted to {params['before']}.")
                except ValueError: params['before'] = str(before)
            if after: params['after'] = str(after) # 'after' gets newer data here

            print(f"  DEBUG in OKXConnector ({endpoint_name}): Calling SDK self.market_api.get_candlesticks with params = {params}")
            result = self.market_api.get_candlesticks(**params) # This is the SDK method for this endpoint
            
            api_code = result.get('code'); api_msg = result.get('msg'); api_data = result.get('data', [])
            print(f"  DEBUG in OKXConnector ({endpoint_name}): API response: Code='{api_code}', Msg='{api_msg}', DataItems={len(api_data) if isinstance(api_data,list) else 'N/A'}")
            if api_code == '0': 
                return self._process_candle_data(api_data, endpoint_name, expect_volume_at_index_5=True)
            else: 
                print(f"  DEBUG in OKXConnector ({endpoint_name}): Error from API. Full response: {result}"); return []
        except Exception as e: 
            print(f"  DEBUG in OKXConnector ({endpoint_name}): Exception: {e}"); import traceback; traceback.print_exc(); return []

    def get_history_market_candles(self, instId: str, bar: str = config.TIMEFRAME, 
                                   limit: int = 100, # Max limit for this endpoint
                                   before: str = None, after: str = None) -> list:
        """Fetches HISTORICAL MARKET K-line data from /api/v5/market/history-candles via SDK."""
        endpoint_name = "/market/history-candles"
        try:
            API_MAX_LIMIT = 100 # For this endpoint
            params = {'instId': instId, 'bar': bar, 'limit': str(min(limit, API_MAX_LIMIT))}
            # For THIS endpoint, 'after' gets older data.
            if after: params['after'] = str(after)
            if before: params['before'] = str(before) # 'before' gets newer data here

            print(f"  DEBUG in OKXConnector ({endpoint_name}): Calling SDK self.market_api.get_history_candlesticks with params = {params}")
            
            if not hasattr(self.market_api, 'get_history_candlesticks'):
                print(f"    ERROR OKXConnector ({endpoint_name}): Method 'get_history_candlesticks' not found in MarketAPI. Please check python-okx version.")
                return []
            
            result = self.market_api.get_history_candlesticks(**params) # Using the SDK method

            api_code = result.get('code'); api_msg = result.get('msg'); api_data = result.get('data', [])
            print(f"  DEBUG in OKXConnector ({endpoint_name}): API response: Code='{api_code}', Msg='{api_msg}', DataItems={len(api_data) if isinstance(api_data,list) else 'N/A'}")

            if api_code == '0':
                # This endpoint returns 'vol', 'volCcy', 'volCcyQuote', 'confirm'
                # The 6th element (index 5) is 'vol'.
                return self._process_candle_data(api_data, endpoint_name, expect_volume_at_index_5=True)
            else:
                print(f"  DEBUG in OKXConnector ({endpoint_name}): Error from API. Full response: {result}"); return []
        except Exception as e:
            print(f"  DEBUG in OKXConnector ({endpoint_name}): Exception: {e}")
            import traceback; traceback.print_exc(); return []
            
    # place_order, get_balance, get_positions (keep your last working versions)
    def place_order(self, instId, side, ordType, sz, tdMode='cross', clOrdId=None, posSide=None):
        try:
            if ordType.lower() == 'market':
                result = self.trade_api.place_order(instId=instId, tdMode=tdMode, side=side,posSide=posSide, ordType=ordType, sz=str(sz), clOrdId=clOrdId)
            else: return None
            if result and isinstance(result, dict):
                if result.get('code') == '0' and result.get('data'):
                    if isinstance(result['data'], list) and len(result['data']) > 0 and isinstance(result['data'][0], dict) and result['data'][0].get('sCode') == '0':
                        return result['data'][0]
                    else: return result 
                else: return result 
            else: return None
        except Exception: return None
    def get_balance(self, ccy=None):
        try:
            result = self.account_api.get_account_balance(ccy=ccy)
            if result['code'] == '0' and result.get('data'):
                if isinstance(result['data'], list) and len(result['data']) > 0: return result['data'][0].get('details', [])
                else: return []
            else: return []
        except Exception: return []
    def get_positions(self, instType='SWAP', instId=None):
        try:
            result = self.account_api.get_positions(instType=instType, instId=instId)
            if result['code'] == '0' and result.get('data'): return result['data']
            else: return []
        except Exception: return []

if __name__ == '__main__':
    try:
        print("Initializing OKXConnector for testing...")
        connector = OKXConnector()
        test_bar_okx = config.TIMEFRAME # Use TIMEFRAME from config for consistency in test
        test_limit_okx = 5

        # --- Test 1: Historical Market Candles (/api/v5/market/history-candles) ---
        # This endpoint's 'after' requests OLDER data.
        print(f"\n--- Test 1: Fetch recent {test_limit_okx} HISTORICAL MARKET klines (bar={test_bar_okx}) ---")
        # First call with no 'after' to get the most recent block from history-candles
        # (API behavior for no pagination param for history-candles needs to be verified,
        # it might require 'before' with a future TS for latest, or 'after' with a very old TS to start from past)
        # For simplicity of test, let's assume calling with just limit gets latest available "historical" block
        hm_klines_latest_batch = connector.get_history_market_candles(
            instId=config.SYMBOL, bar=test_bar_okx, limit=test_limit_okx 
        )
        if hm_klines_latest_batch:
            print(f"Fetched {len(hm_klines_latest_batch)} recent historical market klines (Batch 1):")
            for k_line in hm_klines_latest_batch: print(f"  ts={k_line[0]}")

            if len(hm_klines_latest_batch) > 0:
                oldest_ts_from_hm_latest_batch = hm_klines_latest_batch[0][0] 
                print(f"\n--- Test 1.1: Fetch older HISTORICAL MARKET klines using 'after={oldest_ts_from_hm_latest_batch}' ---")
                hm_klines_even_older = connector.get_history_market_candles(
                    instId=config.SYMBOL, bar=test_bar_okx, limit=test_limit_okx, 
                    after=str(oldest_ts_from_hm_latest_batch)
                )
                if hm_klines_even_older:
                    print(f"Fetched {len(hm_klines_even_older)} even older historical market klines (Batch 2):")
                    for k_line_b1 in hm_klines_even_older: print(f"  ts={k_line_b1[0]}")
                    if len(hm_klines_even_older) > 0:
                        # Expect NEWEST of this older batch < OLDEST of previous batch (which was used as 'after')
                        if hm_klines_even_older[-1][0] < oldest_ts_from_hm_latest_batch:
                            print(f"  Verification SUCCESS (HM 'after'): Fetched older data correctly.")
                        else:
                            print(f"  Verification WARNING (HM 'after'): NOT older data. Newest of B2 ({hm_klines_even_older[-1][0]}) >= Oldest of B1 ({oldest_ts_from_hm_latest_batch})")
                else: print("  No older historical market klines fetched using 'after'.")
        else: print("Failed to fetch recent historical market klines (Batch 1).")
        
        # --- Test 2: Recent Market Candles (/api/v5/market/candles) ---
        # This endpoint's 'before' requests OLDER data. (Still expecting issues on demo)
        print(f"\n--- Test 2: Fetch recent {test_limit_okx} RECENT MARKET klines (bar={test_bar_okx}) ---")
        market_klines_recent = connector.get_market_candles(instId=config.SYMBOL, bar=test_bar_okx, limit=test_limit_okx)
        if market_klines_recent:
            print(f"Fetched {len(market_klines_recent)} recent market klines (Batch A):")
            for k_line in market_klines_recent: print(f"  ts={k_line[0]}")
            
            if len(market_klines_recent) > 0:
                oldest_ts_market_recent = market_klines_recent[0][0]
                print(f"\n--- Test 2.1: Fetch older RECENT MARKET klines using 'before={oldest_ts_market_recent}' (Batch B) ---")
                market_klines_older = connector.get_market_candles(instId=config.SYMBOL, bar=test_bar_okx, limit=test_limit_okx, before=str(oldest_ts_market_recent))
                if market_klines_older:
                    print(f"Fetched {len(market_klines_older)} older market klines (Batch B):")
                    for k_line_b1 in market_klines_older: print(f"  ts={k_line_b1[0]}")
                    if len(market_klines_older) > 0:
                        # Expect NEWEST of this older batch (Batch B) < OLDEST of previous batch (Batch A)
                        if market_klines_older[-1][0] < oldest_ts_market_recent: print(f"  Verification SUCCESS (Market 'before'): Fetched older data correctly.")
                        else: print(f"  Verification WARNING (Market 'before'): NOT older data. Newest of B ({market_klines_older[-1][0]}) >= Oldest of A ({oldest_ts_market_recent})")
                else: print("  No older market klines fetched using 'before'.")
        else: print("Failed to fetch recent market klines (Batch A).")

    except ValueError as ve_conn: print(f"Connector Test - Config Error: {ve_conn}")
    except Exception as e_conn: print(f"Connector Test - Unexpected error: {e_conn}"); import traceback; traceback.print_exc()
    print("\nOKXConnector test script finished.")
