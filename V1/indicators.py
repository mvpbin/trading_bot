# trading_bot/indicators.py
import pandas as pd
import numpy as np

class Indicators:
    def add_sma(self, df: pd.DataFrame, period: int, column_name: str = 'close') -> pd.DataFrame:
        """向 DataFrame 添加简单移动平均线 (SMA)。"""
        df_copy = df.copy()
        sma_col_name = f'SMA_{period}'
        if column_name not in df_copy.columns:
            raise ValueError(f"列 '{column_name}' 在DataFrame中未找到，无法计算SMA。")
        # 使用 min_periods=period 确保只有在有足够数据时才计算SMA，避免早期NaN影响后续计算（如果需要）
        df_copy[sma_col_name] = df_copy[column_name].rolling(window=period, min_periods=period).mean()
        return df_copy

    def add_ema(self, df: pd.DataFrame, period: int, column_name: str = 'close') -> pd.DataFrame:
        """向 DataFrame 添加指数移动平均线 (EMA)。"""
        df_copy = df.copy()
        ema_col_name = f'EMA_{period}'
        if column_name not in df_copy.columns:
            raise ValueError(f"列 '{column_name}' 在DataFrame中未找到，无法计算EMA。")
        df_copy[ema_col_name] = df_copy[column_name].ewm(span=period, adjust=False, min_periods=period).mean()
        return df_copy

    def add_rsi(self, df: pd.DataFrame, period: int = 14, column_name: str = 'close') -> pd.DataFrame:
        """向 DataFrame 添加相对强弱指数 (RSI)。"""
        df_copy = df.copy()
        rsi_col_name = f'RSI_{period}'
        if column_name not in df_copy.columns:
            raise ValueError(f"列 '{column_name}' 在DataFrame中未找到，无法计算RSI。")
        
        delta = df_copy[column_name].diff(1)
        gain = delta.where(delta > 0, 0.0)  # 确保非正值为0.0
        loss = -delta.where(delta < 0, 0.0) # 确保非负值为0.0

        # 使用 Wilder's smoothing (等同于EMA alpha=1/period)
        avg_gain = gain.ewm(alpha=1/period, adjust=False, min_periods=period).mean()
        avg_loss = loss.ewm(alpha=1/period, adjust=False, min_periods=period).mean()

        # 计算RS
        # 当avg_loss为0时，rs可能为inf或nan，这会导致RSI为100或错误
        rs = avg_gain / avg_loss
        
        df_copy[rsi_col_name] = 100.0 - (100.0 / (1.0 + rs))
        
        # 处理特殊情况
        df_copy.loc[avg_loss == 0, rsi_col_name] = 100.0 # 如果avg_loss为0且avg_gain > 0, RSI为100
        df_copy.loc[(avg_gain == 0) & (avg_loss == 0), rsi_col_name] = 50.0 # 如果两者都为0 (例如，价格未变期数不足)，设为中性50
        df_copy[rsi_col_name] = df_copy[rsi_col_name].fillna(50) # 用50填充初始的NaN值
        return df_copy

    def _calculate_true_range(self, df: pd.DataFrame) -> pd.Series: # 确保返回的是 pd.Series
        """私有辅助函数，计算真实波幅 (TR)。"""
        df_copy = df.copy()
        if not all(col in df_copy.columns for col in ['high', 'low', 'close']):
            raise ValueError("DataFrame必须包含 'high', 'low', 'close' 列才能计算TR。")
        
        high_low = df_copy['high'] - df_copy['low']
        high_close_prev = abs(df_copy['high'] - df_copy['close'].shift(1))
        low_close_prev = abs(df_copy['low'] - df_copy['close'].shift(1))
        
        tr_df = pd.DataFrame({'hl': high_low, 'hcp': high_close_prev, 'lcp': low_close_prev})
        true_range = tr_df.max(axis=1, skipna=False) 

        if pd.isna(true_range.iloc[0]) and not pd.isna(high_low.iloc[0]):
             true_range.iloc[0] = high_low.iloc[0]
        elif pd.isna(true_range.iloc[0]): 
            true_range.iloc[0] = 0 

        return true_range # true_range 在这里已经是 Pandas Series

    def add_atr(self, df: pd.DataFrame, period: int = 14) -> pd.DataFrame:
        """向DataFrame添加平均真实波幅 (ATR)。"""
        df_copy = df.copy()
        atr_col_name = f'ATR_{period}'
        true_range = self._calculate_true_range(df_copy)
        # Wilder's smoothing (EMA with alpha = 1/period)
        df_copy[atr_col_name] = true_range.ewm(alpha=1/period, adjust=False, min_periods=period).mean()
        return df_copy

    def add_adx(self, df: pd.DataFrame, period: int = 14) -> pd.DataFrame:
        """向DataFrame添加ADX, +DI, -DI。"""
        df_copy = df.copy()
        if not all(col in df_copy.columns for col in ['high', 'low', 'close']):
            raise ValueError("DataFrame必须包含 'high', 'low', 'close' 列才能计算ADX。")

        plus_di_col = f'PDI_{period}'
        minus_di_col = f'MDI_{period}'
        adx_col = f'ADX_{period}'

        # 计算+DM, -DM
        df_copy['move_up'] = df_copy['high'].diff()
        df_copy['move_down'] = -df_copy['low'].diff() 

        df_copy['plus_dm'] = 0.0
        df_copy.loc[(df_copy['move_up'] > df_copy['move_down']) & (df_copy['move_up'] > 0), 'plus_dm'] = df_copy['move_up']
        
        df_copy['minus_dm'] = 0.0
        df_copy.loc[(df_copy['move_down'] > df_copy['move_up']) & (df_copy['move_down'] > 0), 'minus_dm'] = df_copy['move_down']

        true_range = self._calculate_true_range(df_copy) # 确保 _calculate_true_range 返回 Pandas Series
        
        smooth_plus_dm = df_copy['plus_dm'].ewm(alpha=1/period, adjust=False, min_periods=period).mean()
        smooth_minus_dm = df_copy['minus_dm'].ewm(alpha=1/period, adjust=False, min_periods=period).mean()
        smooth_tr = true_range.ewm(alpha=1/period, adjust=False, min_periods=period).mean() # true_range 必须是 Series

        df_copy[plus_di_col] = np.where(smooth_tr != 0, (smooth_plus_dm / smooth_tr) * 100, 0.0) # np.where 返回 ndarray
        df_copy[minus_di_col] = np.where(smooth_tr != 0, (smooth_minus_dm / smooth_tr) * 100, 0.0) # np.where 返回 ndarray
        
        # 将 +DI 和 -DI 确保为 Series 以便进行 Series 运算
        # 虽然直接在DataFrame中赋值后它们已经是Series了，但显式转换更安全
        pdi_series = pd.Series(df_copy[plus_di_col], index=df_copy.index)
        mdi_series = pd.Series(df_copy[minus_di_col], index=df_copy.index)

        sum_di = pdi_series + mdi_series # Series 运算
        # dx的计算结果是一个NumPy数组，因为np.where的结果是数组
        dx_array = np.where(sum_di != 0, (abs(pdi_series - mdi_series) / sum_di) * 100, 0.0)
        
        # --- 修正点：将 dx_array 转换为 Pandas Series ---
        dx_series = pd.Series(dx_array, index=df_copy.index) 
        dx_series = dx_series.fillna(0) # 再次确保填充 NaN

        # 现在 dx_series 是 Pandas Series，可以调用 .ewm()
        df_copy[adx_col] = dx_series.ewm(alpha=1/period, adjust=False, min_periods=period).mean()
        
        df_copy.drop(columns=['move_up', 'move_down', 'plus_dm', 'minus_dm'], inplace=True)
        
        return df_copy

if __name__ == '__main__':
    # 示例数据，包含更多行以测试平滑和min_periods
    data_len = 30
    timestamps = pd.to_datetime([f'2023-01-{i:02d}T00:00:00Z' for i in range(1, data_len + 1)])
    opens = np.random.randint(80, 120, data_len)
    closes = opens + np.random.randint(-5, 6, data_len)
    highs = np.maximum(opens, closes) + np.random.randint(0, 5, data_len)
    lows = np.minimum(opens, closes) - np.random.randint(0, 5, data_len)
    volumes = np.random.randint(10, 100, data_len)

    sample_df_test = pd.DataFrame({
        'timestamp': timestamps,
        'open': opens, 'high': highs, 'low': lows, 'close': closes, 'volume': volumes
    })
    sample_df_test = sample_df_test.set_index('timestamp')
    
    calculator = Indicators()

    print("--- SMA Test (period 5) ---")
    df_sma = calculator.add_sma(sample_df_test.copy(), period=5)
    print(df_sma[[f'SMA_5']].tail())

    print("\n--- RSI Test (period 7) ---")
    df_rsi = calculator.add_rsi(sample_df_test.copy(), period=7)
    print(df_rsi[[f'RSI_7']].tail())

    print("\n--- ATR Test (period 7) ---")
    df_atr = calculator.add_atr(sample_df_test.copy(), period=7)
    print(df_atr[[f'ATR_7']].tail())

    print("\n--- ADX Test (period 7) ---")
    df_adx = calculator.add_adx(sample_df_test.copy(), period=7)
    print(df_adx[[f'PDI_7', f'MDI_7', f'ADX_7']].tail(15)) # 打印更多行以观察ADX的形成
