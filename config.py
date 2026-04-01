# config.py

FI_ETFS = ['TLT', 'VNQ', 'SLV', 'GLD', 'HYG', 'LQD']

EQUITY_ETFS = [
    'QQQ', 'XLK', 'XLF', 'XLE', 'XLV', 'XLI',
    'XLY', 'XLP', 'XLU', 'XME', 'GDX', 'IWM'
]

BENCHMARKS = ['SPY', 'AGG']

UNIVERSE_CONFIG = {
    'fi': {
        'target_etfs': FI_ETFS,
        'feature_tickers': FI_ETFS + BENCHMARKS,
        'output_dataset': 'P2SAMAPA/p2-etf-vlstm-outputs',
        'file_prefix': 'fi',  # Files will be fi_expanding_latest.json, fi_history.parquet
        'n_etfs': len(FI_ETFS)
    },
    'equity': {
        'target_etfs': EQUITY_ETFS,
        'feature_tickers': EQUITY_ETFS + BENCHMARKS,
        'output_dataset': 'P2SAMAPA/p2-etf-vlstm-outputs',  # SAME dataset as FI
        'file_prefix': 'equity',  # Files will be equity_expanding_latest.json, equity_history.parquet
        'n_etfs': len(EQUITY_ETFS)
    }
}

def get_config(universe='fi'):
    return UNIVERSE_CONFIG[universe]
