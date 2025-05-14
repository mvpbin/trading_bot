# trading_bot/okx_connector.py
import okx.Trade as Trade
import okx.MarketData as MarketData
import okx.Account as Account
import httpx # Keep this import for TimeoutException
import json
from datetime import datetime, timezone, timedelta
import time
import hmac
import base64
import logging
from functools import wraps # For the decorator
import httpx # 确保 httpx 已导入
import httpcore # 新增导入 httpcore

import config

module_logger = logging.getLogger(f"trading_bot.{__name__}")

# --- Timeout Retry Decorator ---
def with_timeout_retry(max_retries: int = 3, initial_retry_delay: float = 2.0,
                       backoff_factor: float = 2.0, logger_instance=None):
    """
    Decorator to add timeout and retry logic to a function that makes an API call.
    It catches several httpx and httpcore exceptions related to network issues.
    """
    if logger_instance is None: # Fallback logger
        logger_instance = logging.getLogger(f"trading_bot.timeout_retry_decorator")
        if not logger_instance.hasHandlers(): # Basic setup if no handler
            handler = logging.StreamHandler()
            handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
            logger_instance.addHandler(handler)
            logger_instance.setLevel(logging.INFO)

    # 定义可重试的异常类型元组
    RETRYABLE_EXCEPTIONS = (
        httpx.TimeoutException,
        httpx.NetworkError,      # 包括 ConnectError, ReadError, WriteError, PoolTimeout
        httpx.RemoteProtocolError,
        httpcore.NetworkError,   # httpcore 底层网络错误
        httpcore.RemoteProtocolError
        # 注意：可以根据需要添加更多具体的 httpx 或 httpcore 异常
    )

    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            retries_left = max_retries
            current_delay = initial_retry_delay
            last_exception = None

            while retries_left > 0:
                try:
                    # func_name = func.__name__
                    # logger_instance.debug(f"Calling {func_name}, attempt {max_retries - retries_left + 1}/{max_retries}")
                    return func(*args, **kwargs)
                except RETRYABLE_EXCEPTIONS as e: # 捕获元组中定义的所有可重试异常
                    last_exception = e
                    retries_left -= 1
                    logger_instance.warning(
                        f"API call to '{func.__name__}' failed with {type(e).__name__} (Attempt {max_retries - retries_left}/{max_retries}). Error: {e}"
                    )
                    if retries_left > 0:
                        logger_instance.info(f"Retrying in {current_delay:.2f} seconds...")
                        time.sleep(current_delay)
                        current_delay *= backoff_factor # Exponential backoff
                    else:
                        logger_instance.error(
                            f"API call to '{func.__name__}' failed after {max_retries} retries due to {type(e).__name__}."
                        )
                        raise # Re-raise the last caught retryable exception
                except Exception as e_unexpected: # Catch other potential non-retryable exceptions
                    logger_instance.error(
                        f"API call to '{func.__name__}' encountered an unexpected non-retryable error: {e_unexpected}", exc_info=True
                    )
                    raise # Re-raise other critical exceptions immediately

            if last_exception: # Should only be reached if loop finishes due to retries_left <= 0
                raise last_exception
            return None # Fallback, though typically an exception should have been raised
        return wrapper
    return decorator

class OKXConnector:
    def __init__(self):
        if config.IS_DEMO_TRADING:
            self.api_key = config.DEMO_API_KEY
            self.secret_key = config.DEMO_SECRET_KEY
            self.passphrase = config.DEMO_PASSPHRASE
            self.flag = '1'
            module_logger.info("OKX Connector initialized for DEMO TRADING.")
        else:
            self.api_key = getattr(config, 'REAL_API_KEY', "YOUR_REAL_API_KEY_HERE")
            self.secret_key = getattr(config, 'REAL_SECRET_KEY', "YOUR_REAL_SECRET_KEY_HERE")
            self.passphrase = getattr(config, 'REAL_PASSPHRASE', "YOUR_REAL_PASSPHRASE_HERE")
            self.flag = '0'
            module_logger.info("OKX Connector initialized for REAL TRADING.")
            module_logger.warning("!!! WARNING: OPERATING WITH REAL FUNDS !!!")

        current_mode = "DEMO" if config.IS_DEMO_TRADING else "REAL"
        if not self.api_key or (isinstance(self.api_key, str) and ("YOUR_" in self.api_key and "HERE" in self.api_key)):
            raise ValueError(f"API Key for {current_mode} trading is not configured correctly in config.py.")
        if not self.secret_key or (isinstance(self.secret_key, str) and ("YOUR_" in self.secret_key and "HERE" in self.secret_key)):
            raise ValueError(f"Secret Key for {current_mode} trading is not configured correctly in config.py.")
        if not self.passphrase or (isinstance(self.passphrase, str) and ("YOUR_" in self.passphrase and "HERE" in self.passphrase)):
            raise ValueError(f"Passphrase for {current_mode} trading is not configured correctly in config.py.")

        # Initialize API clients with SDK default timeouts, as direct timeout injection failed
        self.trade_api = Trade.TradeAPI(self.api_key, self.secret_key, self.passphrase, flag=self.flag)
        self.market_api = MarketData.MarketAPI(self.api_key, self.secret_key, self.passphrase, flag=self.flag)
        self.account_api = Account.AccountAPI(self.api_key, self.secret_key, self.passphrase, flag=self.flag)
        module_logger.info("OKX API clients initialized (using SDK default timeouts initially). Retries handled by decorator.")

    def _process_candle_data(self, api_data_list: list, api_endpoint_name: str, expect_volume_at_index_5: bool = True) -> list:
        # ... (this method remains the same) ...
        klines_from_api = []
        if not isinstance(api_data_list, list):
            module_logger.warning(f"    OKXConnector ({api_endpoint_name}): API data is not a list: {api_data_list}")
            return []
        for k_data_point in api_data_list:
            if isinstance(k_data_point, list) and len(k_data_point) >= 5:
                volume_value = 0.0
                if expect_volume_at_index_5 and len(k_data_point) >= 6:
                    try: volume_value = float(k_data_point[5])
                    except (ValueError, TypeError): volume_value = 0.0 
                klines_from_api.append([int(k_data_point[0]), float(k_data_point[1]), float(k_data_point[2]),
                                        float(k_data_point[3]), float(k_data_point[4]), volume_value])
            else: module_logger.warning(f"    OKXConnector ({api_endpoint_name}): Invalid kline data point format: {k_data_point}")
        klines_to_return = klines_from_api[::-1] 
        return klines_to_return

    @with_timeout_retry(logger_instance=module_logger)
    def get_market_candles(self, instId: str, bar: str = config.TIMEFRAME, limit: int = config.DEFAULT_KLINE_LIMIT, 
                           before: str = None, after: str = None) -> list:
        endpoint_name = "/market/candles"
        # No try-except for httpx.TimeoutException here, decorator handles it.
        API_MAX_LIMIT = 100 
        params = {'instId': instId, 'bar': bar, 'limit': str(min(limit, API_MAX_LIMIT))}
        if before:
            try: params['before'] = str(int(before) - 1) 
            except ValueError: params['before'] = str(before)
        if after: params['after'] = str(after)
        # module_logger.debug(f"OKXConnector ({endpoint_name}): Calling SDK with params = {params}")
        result = self.market_api.get_candlesticks(**params)
        api_code = result.get('code'); api_data = result.get('data', [])
        if api_code == '0': return self._process_candle_data(api_data, endpoint_name, True)
        else: module_logger.error(f"OKXConnector ({endpoint_name}): API Error. Full response: {result}"); return []

    @with_timeout_retry(logger_instance=module_logger)
    def get_history_market_candles(self, instId: str, bar: str = config.TIMEFRAME, 
                                   limit: int = 100, before: str = None, after: str = None) -> list:
        endpoint_name = "/market/history-candles"
        # No try-except for httpx.TimeoutException here, decorator handles it.
        API_MAX_LIMIT = 100 
        params = {'instId': instId, 'bar': bar, 'limit': str(min(limit, API_MAX_LIMIT))}
        if after: params['after'] = str(after)
        if before: params['before'] = str(before)
        if not hasattr(self.market_api, 'get_history_candlesticks'):
            module_logger.error(f"OKXConnector ({endpoint_name}): Method 'get_history_candlesticks' not found.")
            return []
        # module_logger.debug(f"OKXConnector ({endpoint_name}): Calling SDK with params = {params}")
        result = self.market_api.get_history_candlesticks(**params)
        api_code = result.get('code'); api_data = result.get('data', [])
        if api_code == '0': return self._process_candle_data(api_data, endpoint_name, True)
        else: module_logger.error(f"OKXConnector ({endpoint_name}): API Error. Full response: {result}"); return []
            
    @with_timeout_retry(logger_instance=module_logger)
    def place_order(self, instId, side, ordType, sz, tdMode='cross', clOrdId=None, posSide=None):
        # No try-except for httpx.TimeoutException here, decorator handles it.
        if ordType.lower() == 'market': result = self.trade_api.place_order(instId=instId, tdMode=tdMode, side=side,posSide=posSide, ordType=ordType, sz=str(sz), clOrdId=clOrdId)
        else: return None
        if result and isinstance(result, dict):
            if result.get('code') == '0' and result.get('data'):
                data_list = result['data']
                if isinstance(data_list, list) and len(data_list) > 0 and isinstance(data_list[0], dict):
                    if data_list[0].get('sCode') == '0': return data_list[0]
                    else: module_logger.error(f"OKX place_order sCode error: {data_list[0].get('sMsg')} ({data_list[0].get('sCode')})"); return data_list[0]
            module_logger.error(f"OKX place_order API error: {result.get('msg')} (Code: {result.get('code')})"); return result 
        return None

    @with_timeout_retry(logger_instance=module_logger)
    def get_balance(self, ccy=None):
        # No try-except for httpx.TimeoutException here, decorator handles it.
        result = self.account_api.get_account_balance(ccy=ccy)
        if result['code'] == '0' and result.get('data'):
            if isinstance(result['data'], list) and len(result['data']) > 0: return result['data'][0].get('details', [])
            else: return []
        else: module_logger.error(f"OKX get_balance API error: {result.get('msg')} (Code: {result.get('code')})"); return []

    @with_timeout_retry(logger_instance=module_logger)
    def get_positions(self, instType='SWAP', instId=None):
        # No try-except for httpx.TimeoutException here, decorator handles it.
        result = self.account_api.get_positions(instType=instType, instId=instId)
        if result['code'] == '0' and result.get('data'): return result['data']
        else: module_logger.error(f"OKX get_positions API error: {result.get('msg')} (Code: {result.get('code')})"); return []

if __name__ == '__main__':
    # ... (The __main__ test block remains the same as my previous version for OKXConnector) ...
    if not logging.getLogger(f"trading_bot.{OKXConnector.__module__}").hasHandlers():
        test_logger_main_okx = logging.getLogger(f"trading_bot") 
        test_logger_main_okx.setLevel(logging.DEBUG)
        ch_test_main_okx = logging.StreamHandler()
        ch_test_main_okx.setLevel(logging.DEBUG) 
        formatter_main_okx = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        ch_test_main_okx.setFormatter(formatter_main_okx)
        test_logger_main_okx.addHandler(ch_test_main_okx)

    try:
        print("Initializing OKXConnector for testing (with retry decorator)...")
        connector = OKXConnector()
        test_bar_okx = config.TIMEFRAME 
        test_limit_okx = 3 

        print(f"\n--- Test 1: Fetch recent {test_limit_okx} HISTORICAL MARKET klines (bar={test_bar_okx}) ---")
        hm_klines_latest_batch = connector.get_history_market_candles(
            instId=config.SYMBOL, bar=test_bar_okx, limit=test_limit_okx 
        )
        if hm_klines_latest_batch: print(f"Fetched {len(hm_klines_latest_batch)} recent historical klines.")
        else: print("Failed to fetch recent historical market klines.")
        
        time.sleep(1) 

        print(f"\n--- Test 2: Fetch recent {test_limit_okx} RECENT MARKET klines (bar={test_bar_okx}) ---")
        market_klines_recent = connector.get_market_candles(instId=config.SYMBOL, bar=test_bar_okx, limit=test_limit_okx)
        if market_klines_recent: print(f"Fetched {len(market_klines_recent)} recent market klines.")
        else: print("Failed to fetch recent market klines.")

    except ValueError as ve_conn: 
        module_logger.error(f"Connector Test - Config Error: {ve_conn}")
    except Exception as e_conn: 
        module_logger.error(f"Connector Test - Unexpected error: {e_conn}", exc_info=True)
    print("\nOKXConnector test script finished.")
