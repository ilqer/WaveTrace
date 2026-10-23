"""The `ServeMode` strategies for `ServeMeshUseCase`: what each live mode loads, how it scores one
link, and how it prints the fused verdict. Gain is a property of the receiver, so calibration is per
rx node while features stay per link, keeping all N(N-1) views."""

import numpy as np

from wavetrace.application.serve_mesh import LinkScore
from wavetrace.recognition.count import count_name
from wavetrace.recognition.link import accuracy_weights
from wavetrace.recognition.serving import last_window_proba, load_presence_node_servers
from wavetrace.recognition.weapon_serving import (
    dwell_proba, find_weapon_entry, link_health, load_weapon_links,
)


def _read_positive_proba(proba, positive_class_index):
    return float(proba[positive_class_index]) if positive_class_index >= 0 else 0.0


def _compute_binary_margin(positive_proba):
    return abs(positive_proba - 0.5) * 2.0


def _format_binary_verdict(positive_proba, positive_label, negative_label):
    label = positive_label if positive_proba >= 0.5 else negative_label
    bar = "#" * int(positive_proba * 20)
    return f"{label}  P {positive_proba:0.2f}  {bar:<20}"


def _read_banner_sample_rate_hz(nodes):
    # banner only: every head is trained at the same rate today (not enforced)
    return next(iter(nodes.values())).session.head.contract.target_sample_rate_hz


def _format_node_weights(nodes):
    return "  ".join(f"N{node_id}:w={nodes[node_id].weight:.2f}" for node_id in sorted(nodes))


def _read_shared_class_ids(nodes):
    # every head shares one class ordering: the loaders reject a mix
    return next(iter(nodes.values())).classes if nodes else []


def _serve_last_window(node, frames):
    return last_window_proba(frames, node.session.head.contract.target_sample_rate_hz, node.calibration,
                             node.gain_lock, node.config, node.should_use_intercarrier, node.pick_input,
                             node.session)


def _expand_proba(proba, global_class_columns, num_classes):
    """Scatter a head's proba into the global num_classes vector; classes it never saw stay 0."""
    expanded = np.zeros(num_classes, dtype=np.float64)
    for j, column in enumerate(global_class_columns):
        expanded[column] = proba[j]
    return expanded


class PresenceServeMode:
    """nodes: {rx node_id: NodeServer}, each weighted by `accuracy_weights` of its LOGO accuracy."""

    def __init__(self, nodes):
        self.nodes = nodes

    @property
    def class_ids(self):
        """The blended proba's column order."""
        return _read_shared_class_ids(self.nodes)

    @property
    def calibration_by_rx_node(self):
        return {node_id: node.calibration for node_id, node in self.nodes.items()}

    @classmethod
    def load(cls, calibration_root, model_root):
        nodes, logo_accuracy_by_node = load_presence_node_servers(calibration_root, model_root)
        weights = accuracy_weights({node_id: logo_accuracy
                                    for node_id, logo_accuracy in logo_accuracy_by_node.items()
                                    if logo_accuracy is not None})
        for node_id, node in nodes.items():
            node.weight = weights.get(node_id, 1.0)
        # blending raw proba vectors is only valid when the heads share a class ordering
        orders = {tuple(node.classes) for node in nodes.values()}
        if len(orders) > 1:
            raise ValueError(f"per-node heads disagree on class ordering {orders}; retrain consistently")
        return cls(nodes)

    def select_node(self, link_key):
        return self.nodes.get(link_key[1])

    def score_link(self, link_key, node, frames):
        proba, window_view = _serve_last_window(node, frames)
        if proba is None:
            return None
        present_proba = _read_positive_proba(proba, node.positive_class_index)
        score_token = f"{link_key[0]}->{link_key[1]}:{present_proba:.2f}"
        return LinkScore(proba, _compute_binary_margin(present_proba), score_token, window_view)

    def format_banner(self, port):
        return (f"ALL-PAIRS presence on udp/{port} (fs={_read_banner_sample_rate_hz(self.nodes):g}Hz, "
                f"rx nodes={sorted(self.nodes)}; vote weights {_format_node_weights(self.nodes)}). "
                "move in and out of the links. Ctrl+C to stop.\n")

    def format_verdict(self, blended_proba):
        present_class_index = next(iter(self.nodes.values())).positive_class_index
        present_proba = _read_positive_proba(blended_proba, present_class_index)
        return _format_binary_verdict(present_proba, "PRESENT", "absent ")


class CountServeMode:
    """nodes: {rx node_id: NodeServer}. Heads may have learned different class sets, so each link's
    proba is expanded into the sorted union `class_ids` before the vote. max_count: the top count level,
    printed as 'N+'; None takes the highest learned class."""

    def __init__(self, nodes, class_ids, max_count=None):
        self.nodes = nodes
        self.class_ids = class_ids
        self._max_count = max_count if max_count is not None else max(class_ids, default=0)
        self._class_values = np.asarray(class_ids, dtype=np.float64)

    @property
    def calibration_by_rx_node(self):
        return {node_id: node.calibration for node_id, node in self.nodes.items()}

    @classmethod
    def load(cls, calibration_root, model_root, max_count=None):
        nodes, logo_accuracy_by_node = load_presence_node_servers(calibration_root, model_root)
        class_ids = sorted(set().union(*[set(node.classes) for node in nodes.values()])) if nodes else []
        column_by_class = {class_id: i for i, class_id in enumerate(class_ids)}
        chance = 1.0 / len(class_ids) if class_ids else 0.5
        for node_id, node in nodes.items():
            node.global_class_columns = [column_by_class[class_id] for class_id in node.classes]
            logo_accuracy = logo_accuracy_by_node[node_id]
            # rescale accuracy so 1/k chance maps to 0 and perfect to 1; no LOGO number means 1.0
            node.weight = (max(logo_accuracy - chance, 0.0) / max(1.0 - chance, 1e-9)
                           if logo_accuracy is not None else 1.0)
        return cls(nodes, class_ids, max_count)

    def select_node(self, link_key):
        return self.nodes.get(link_key[1])

    def score_link(self, link_key, node, frames):
        proba, window_view = _serve_last_window(node, frames)
        if proba is None:
            return None
        expanded_proba = _expand_proba(proba, node.global_class_columns, len(self.class_ids))
        top = np.sort(proba)[::-1]
        quality = float(top[0] - top[1]) if proba.size > 1 else float(top[0])
        return LinkScore(expanded_proba, quality,
                         f"{link_key[0]}->{link_key[1]}:{self.class_ids[int(np.argmax(expanded_proba))]}",
                         window_view)

    def format_banner(self, port):
        labels = [count_name(class_id, self._max_count) for class_id in self.class_ids]
        return (f"PEOPLE-COUNT on udp/{port} (fs={_read_banner_sample_rate_hz(self.nodes):g}Hz, "
                f"classes={labels}, rx nodes={sorted(self.nodes)}; "
                f"vote weights {_format_node_weights(self.nodes)}). vary the headcount. Ctrl+C to stop.\n")

    def format_verdict(self, blended_proba):
        blended_proba = np.asarray(blended_proba, dtype=np.float64)
        expected_count = float((self._class_values * blended_proba).sum())  # the 'N+' class counts as N
        label = count_name(min(int(round(expected_count)), max(self.class_ids)), self._max_count)
        return f"PEOPLE ~{expected_count:0.1f}  ({label})"


def _format_head_key(head_key):
    link_tag, node_id = head_key
    return f"{link_tag}->{node_id}" if link_tag is not None else f"*->{node_id}"


class WeaponServeMode:
    """nodes: {(link_tag | None, rx node_id): NodeServer} from `load_weapon_links`. A link takes its
    per-link head, else its rx node's per-node head; its proba is the mean over every window in the
    buffer, which composes with the vote."""

    def __init__(self, nodes):
        self.nodes = nodes

    @property
    def class_ids(self):
        """The blended proba's column order."""
        return _read_shared_class_ids(self.nodes)

    @property
    def calibration_by_rx_node(self):
        """An rx node's link heads share its calibration."""
        return {rx_node_id: node.calibration for (_link_tag, rx_node_id), node in self.nodes.items()}

    @classmethod
    def load(cls, calibration_root, model_root):
        return cls(load_weapon_links(calibration_root, model_root))

    def select_node(self, link_key):
        return find_weapon_entry(self.nodes, *link_key)

    def score_link(self, link_key, node, frames):
        proba, window_view = dwell_proba(frames, node.session.head.contract.target_sample_rate_hz, node)
        if proba is None:
            return None
        weapon_proba = _read_positive_proba(proba, node.positive_class_index)
        delivered_rate_hz, missing_fraction = link_health(frames)
        drop_note = f"!{missing_fraction:.0%}drop" if missing_fraction > 0.1 else ""
        health = f"@{delivered_rate_hz:.0f}Hz" + drop_note
        return LinkScore(proba, _compute_binary_margin(weapon_proba),
                         f"{link_key[0]}->{link_key[1]}:{weapon_proba:.2f}{health}", window_view)

    def format_banner(self, port):
        has_per_link_heads = any(link_tag is not None for link_tag, _ in self.nodes)
        head_weights = "  ".join(f"{_format_head_key(head_key)}:w={self.nodes[head_key].weight:.2f}"
                                 for head_key in sorted(self.nodes, key=lambda key: (key[0] or "", key[1])))
        return (f"WEAPON detection on udp/{port} (fs={_read_banner_sample_rate_hz(self.nodes):g}Hz, "
                f"{'per-link' if has_per_link_heads else 'per-node'} heads; vote weights {head_weights}). "
                "Ctrl+C to stop.\n")

    def format_verdict(self, blended_proba):
        weapon_class_index = next(iter(self.nodes.values())).positive_class_index
        weapon_proba = _read_positive_proba(blended_proba, weapon_class_index)
        return _format_binary_verdict(weapon_proba, "WEAPON", "clear ")
