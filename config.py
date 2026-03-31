# config.py

# FI Universe (existing, unchanged)
FI_ETFS = ['TLT', 'VNQ', 'SLV', 'GLD', 'HYG', 'LQD']

# Equity Universe (new)
EQUITY_ETFS = [
    'QQQ', 'XLK', 'XLF', 'XLE', 'XLV', 'XLI',
    'XLY', 'XLP', 'XLU', 'XME', 'GDX', 'IWM'
]

# Common benchmarks (features only)
BENCHMARKS = ['SPY', 'AGG']

UNIVERSE_CONFIG = {
    'fi': {                                     # was 'original'
        'target_etfs': FI_ETFS,
        'feature_tickers': FI_ETFS + BENCHMARKS,
        'output_dataset': 'P2SAMAPA/p2-etf-vlstm-outputs',   # existing dataset
        'n_etfs': len(FI_ETFS)
    },
    'equity': {
        'target_etfs': EQUITY_ETFS,
        'feature_tickers': EQUITY_ETFS + BENCHMARKS,
        'output_dataset': 'P2SAMAPA/p2-etf-vlstm-outputs-equity',  # new dataset
        'n_etfs': len(EQUITY_ETFS)
    }
}

def get_config(universe='fi'):
    return UNIVERSE_CONFIG[universe]
