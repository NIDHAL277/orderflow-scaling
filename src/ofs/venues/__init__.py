"""Other exchanges than Binance, for exploratory comparisons outside the frozen protocol."""

from ofs.venues.bybit import BYBIT
from ofs.venues.common import Adapter
from ofs.venues.okx import OKX

ADAPTERS: dict[str, Adapter] = {adapter.name: adapter for adapter in (BYBIT, OKX)}
