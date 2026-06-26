"""Cross-sectional portfolio construction.

Modules here cover ranking / quantile bucketing, long/short weighting, beta and sector
neutralisation, and vol-targeting / leverage sizing. These are strategy-agnostic helpers
— the concrete strategy in ``quantstrat.strategies`` decides which to compose.
"""
