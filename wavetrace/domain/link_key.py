"""Value objects shared across layers."""

from typing import NamedTuple


class LinkKey(NamedTuple):
    """One directed CSI link: a transmitter, named by its MAC's last two octets, received at one node.

    A NamedTuple rather than a frozen dataclass: callers index, unpack and hash it as the plain
    `(tx_mac_suffix, rx_node_id)` tuple, which a dataclass would compare unequal to."""

    tx_mac_suffix: str
    rx_node_id: int
