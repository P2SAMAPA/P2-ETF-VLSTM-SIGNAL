"""
config.py
P2-ETF-VLSTM-SIGNAL
"""

# FI Universe (Fixed Income + Commodities)
FI_ETFS = ["TLT", "VNQ", "SLV", "GLD", "HYG", "LQD"]
FI_FEATURE_TICKERS = ["TLT", "VNQ", "SLV", "GLD", "HYG", "LQD", "SPY", "AGG"]

# Equity Universe (Sectors + Gold Miners) - Your 10 ETFs
EQUITY_ETFS = ["GDX", "XLV", "XLY", "XLK", "XLE", "XLP", "XLU", "XME", "XLF", "XLI"]
EQUITY_FEATURE_TICKERS = ["GDX", "XLV", "XLY", "XLK", "XLE", "XLP", "XLU", "XME", "XLF", "IWF", "IWM", "XSD", "XBI", "XLI", "SPY", "QQQ"]

UNIVERSES = {
    "fi": {
        "target_etfs": FI_ETFS,
        "n_etfs": len(FI_ETFS),
        "feature_tickers": FI_FEATURE_TICKERS,
        "output_dataset": "P2SAMAPA/p2-etf-vlstm-outputs",
        "file_prefix": "fi",
        "input_dataset": "P2SAMAPA/fi-etf-macro-signal-master-data",
        "input_filename": "master_data.parquet",
    },
    "equity": {
        "target_etfs": EQUITY_ETFS,
        "n_etfs": len(EQUITY_ETFS),
        "feature_tickers": EQUITY_FEATURE_TICKERS,
        "output_dataset": "P2SAMAPA/p2-etf-vlstm-outputs",
        "file_prefix": "equity",
        "input_dataset": "P2SAMAPA/fi-etf-macro-signal-master-data",
        "input_filename": "master_data.parquet",
    },
}

# Alias for app.py compatibility
UNIVERSE_CONFIG = UNIVERSES

def get_config(universe: str):
    if universe not in UNIVERSES:
        raise ValueError(f"Unknown universe: {universe}")
    return UNIVERSES[universe]
