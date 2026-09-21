"""Domain: pure value objects shared across layers."""

from typing import NamedTuple


class LinkKey(NamedTuple):
    """Identifies one directed CSI link: a transmitter (its MAC's last two octets, e.g. `"ee:ff"`)
    received at one node. A `NamedTuple`, not a frozen dataclass, so it stays interchangeable with
    the bare `(tx_mac_suffix, rx_node_id)` tuple every existing consumer already indexes, unpacks
    and hashes — a frozen dataclass would compare and hash unequal to that tuple and silently break
    every plain-tuple dict lookup against it."""

    tx_mac_suffix: str
    rx_node_id: int
