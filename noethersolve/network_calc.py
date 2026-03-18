"""Networking calculator — derives answers from first principles.

Covers bandwidth-delay product, TCP throughput, subnetting,
MTU fragmentation, and congestion window dynamics.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import List, Optional


@dataclass
class BandwidthDelayReport:
    """Result of bandwidth-delay product calculation."""
    bandwidth_bps: float
    rtt_seconds: float
    bdp_bits: float
    bdp_bytes: float
    min_window_bytes: int  # minimum TCP window for full utilization
    utilization: Optional[float]  # if window_size given
    explanation: str

    def __str__(self) -> str:
        lines = [
            "Bandwidth-Delay Product:",
            f"  Bandwidth: {_fmt_bps(self.bandwidth_bps)}",
            f"  RTT: {self.rtt_seconds*1000:.2f} ms",
            f"  BDP: {self.bdp_bits:.0f} bits = {self.bdp_bytes:.0f} bytes",
            f"  Min TCP window: {self.min_window_bytes} bytes ({self.min_window_bytes/1024:.1f} KB)",
        ]
        if self.utilization is not None:
            pct = self.utilization * 100
            if pct < 0.1:
                lines.append(f"  Link utilization: {pct:.4f}%")
            else:
                lines.append(f"  Link utilization: {pct:.1f}%")
        lines.append(f"  {self.explanation}")
        return "\n".join(lines)


@dataclass
class TCPThroughputReport:
    """Result of TCP throughput estimation."""
    max_throughput_bps: float
    window_size_bytes: int
    rtt_seconds: float
    mss_bytes: int
    loss_rate: Optional[float]
    mathis_throughput_bps: Optional[float]  # Mathis model with loss
    explanation: str

    def __str__(self) -> str:
        lines = [
            "TCP Throughput Estimation:",
            f"  Window-limited: {_fmt_bps(self.max_throughput_bps)}",
            f"  Window size: {self.window_size_bytes} bytes",
            f"  RTT: {self.rtt_seconds*1000:.2f} ms",
            f"  MSS: {self.mss_bytes} bytes",
        ]
        if self.mathis_throughput_bps is not None:
            lines.append(f"  Loss-limited (Mathis): {_fmt_bps(self.mathis_throughput_bps)}")
            lines.append(f"  Loss rate: {self.loss_rate:.2e}")
        lines.append(f"  {self.explanation}")
        return "\n".join(lines)


@dataclass
class SubnetReport:
    """Result of subnet calculation."""
    network_address: str
    broadcast_address: str
    first_host: str
    last_host: str
    prefix_length: int
    subnet_mask: str
    total_addresses: int
    usable_hosts: int
    wildcard_mask: str

    def __str__(self) -> str:
        lines = [
            f"Subnet Analysis (/{self.prefix_length}):",
            f"  Network:   {self.network_address}",
            f"  Broadcast: {self.broadcast_address}",
            f"  First host: {self.first_host}",
            f"  Last host:  {self.last_host}",
            f"  Subnet mask: {self.subnet_mask}",
            f"  Wildcard:    {self.wildcard_mask}",
            f"  Total addresses: {self.total_addresses}",
            f"  Usable hosts: {self.usable_hosts}",
        ]
        return "\n".join(lines)


@dataclass
class FragmentationReport:
    """Result of IP fragmentation analysis."""
    original_size: int  # bytes
    mtu: int  # bytes
    ip_header_size: int
    needs_fragmentation: bool
    num_fragments: int
    fragment_sizes: List[int]  # payload sizes
    total_overhead: int  # extra bytes from fragmentation headers

    def __str__(self) -> str:
        lines = [
            "IP Fragmentation Analysis:",
            f"  Original packet: {self.original_size} bytes",
            f"  Path MTU: {self.mtu} bytes",
            f"  Needs fragmentation: {self.needs_fragmentation}",
        ]
        if self.needs_fragmentation:
            lines.append(f"  Fragments: {self.num_fragments}")
            lines.append(f"  Fragment payloads: {self.fragment_sizes} bytes")
            lines.append(f"  Fragmentation overhead: {self.total_overhead} bytes")
        return "\n".join(lines)


@dataclass
class CongestionWindowReport:
    """Result of TCP congestion window analysis."""
    phase: str  # slow_start, congestion_avoidance, fast_recovery
    cwnd_segments: int
    ssthresh_segments: int
    window_bytes: int
    throughput_bps: float
    rtt_seconds: float
    mss_bytes: int

    def __str__(self) -> str:
        lines = [
            "TCP Congestion Window:",
            f"  Phase: {self.phase}",
            f"  cwnd: {self.cwnd_segments} segments ({self.window_bytes} bytes)",
            f"  ssthresh: {self.ssthresh_segments} segments",
            f"  Estimated throughput: {_fmt_bps(self.throughput_bps)}",
        ]
        return "\n".join(lines)


def bandwidth_delay_product(
    bandwidth_bps: float,
    rtt_seconds: float,
    window_size_bytes: Optional[int] = None,
) -> BandwidthDelayReport:
    """Calculate bandwidth-delay product and TCP window requirements.

    BDP = bandwidth × RTT. This is the amount of data "in flight"
    needed to fully utilize the link.

    Args:
        bandwidth_bps: Link bandwidth in bits per second
        rtt_seconds: Round-trip time in seconds
        window_size_bytes: Optional actual TCP window size to check utilization

    Returns:
        BandwidthDelayReport.
    """
    if bandwidth_bps <= 0 or rtt_seconds <= 0:
        raise ValueError("Bandwidth and RTT must be positive")

    bdp_bits = bandwidth_bps * rtt_seconds
    bdp_bytes = bdp_bits / 8
    min_window = math.ceil(bdp_bytes)

    utilization = None
    if window_size_bytes is not None:
        utilization = min(1.0, window_size_bytes / bdp_bytes) if bdp_bytes > 0 else 1.0

    explanation = (f"To fully utilize {_fmt_bps(bandwidth_bps)} with "
                   f"{rtt_seconds*1000:.1f}ms RTT, TCP window must be ≥ {min_window} bytes.")

    return BandwidthDelayReport(
        bandwidth_bps=bandwidth_bps,
        rtt_seconds=rtt_seconds,
        bdp_bits=bdp_bits,
        bdp_bytes=bdp_bytes,
        min_window_bytes=min_window,
        utilization=utilization,
        explanation=explanation,
    )


def tcp_throughput(
    window_size_bytes: int,
    rtt_seconds: float,
    mss_bytes: int = 1460,
    loss_rate: Optional[float] = None,
) -> TCPThroughputReport:
    """Estimate TCP throughput from window size and RTT.

    Window-limited: throughput = window / RTT
    Loss-limited (Mathis model): throughput ≈ MSS / (RTT × sqrt(loss))

    Args:
        window_size_bytes: TCP receiver window in bytes
        rtt_seconds: Round-trip time in seconds
        mss_bytes: Maximum segment size (default 1460 for Ethernet)
        loss_rate: Optional packet loss rate (0-1)

    Returns:
        TCPThroughputReport.
    """
    if window_size_bytes <= 0 or rtt_seconds <= 0:
        raise ValueError("Window and RTT must be positive")
    if mss_bytes <= 0:
        raise ValueError("MSS must be positive")

    # Window-limited throughput
    max_tp = (window_size_bytes * 8) / rtt_seconds

    mathis_tp = None
    if loss_rate is not None:
        if loss_rate <= 0:
            mathis_tp = max_tp  # no loss = window limited
        elif loss_rate >= 1:
            mathis_tp = 0
        else:
            # Mathis model: T ≈ (MSS / RTT) * (C / sqrt(p))
            # where C ≈ 1.22 (from Mathis et al.)
            mathis_tp = (mss_bytes * 8 * 1.22) / (rtt_seconds * math.sqrt(loss_rate))

    if mathis_tp is not None and mathis_tp < max_tp:
        explanation = f"Loss-limited: Mathis model gives {_fmt_bps(mathis_tp)}, below window limit."
    else:
        explanation = f"Window-limited at {_fmt_bps(max_tp)}."

    return TCPThroughputReport(
        max_throughput_bps=max_tp,
        window_size_bytes=window_size_bytes,
        rtt_seconds=rtt_seconds,
        mss_bytes=mss_bytes,
        loss_rate=loss_rate,
        mathis_throughput_bps=mathis_tp,
        explanation=explanation,
    )


def subnet_calc(ip_address: str, prefix_length: int) -> SubnetReport:
    """Calculate subnet properties from IP address and prefix length.

    Derives network address, broadcast, host range, and masks from
    first principles using bitwise operations.

    Args:
        ip_address: IPv4 address (e.g., "192.168.1.100")
        prefix_length: CIDR prefix length (0-32)

    Returns:
        SubnetReport with all subnet properties.
    """
    if not 0 <= prefix_length <= 32:
        raise ValueError("Prefix length must be 0-32")

    # Parse IP to 32-bit integer
    octets = ip_address.strip().split(".")
    if len(octets) != 4:
        raise ValueError(f"Invalid IPv4 address: {ip_address}")
    ip_int = 0
    for o in octets:
        val = int(o)
        if not 0 <= val <= 255:
            raise ValueError(f"Invalid octet: {o}")
        ip_int = (ip_int << 8) | val

    # Subnet mask
    if prefix_length == 0:
        mask = 0
    else:
        mask = ((1 << 32) - 1) << (32 - prefix_length)
    mask &= 0xFFFFFFFF

    # Network and broadcast
    network = ip_int & mask
    wildcard = mask ^ 0xFFFFFFFF
    broadcast = network | wildcard

    total = 2 ** (32 - prefix_length)
    usable = max(0, total - 2) if prefix_length < 31 else total

    # First and last host
    if prefix_length < 31:
        first_host = network + 1
        last_host = broadcast - 1
    elif prefix_length == 31:
        first_host = network
        last_host = broadcast
    else:  # /32
        first_host = network
        last_host = network

    return SubnetReport(
        network_address=_int_to_ip(network),
        broadcast_address=_int_to_ip(broadcast),
        first_host=_int_to_ip(first_host),
        last_host=_int_to_ip(last_host),
        prefix_length=prefix_length,
        subnet_mask=_int_to_ip(mask),
        total_addresses=total,
        usable_hosts=usable,
        wildcard_mask=_int_to_ip(wildcard),
    )


def ip_fragmentation(
    packet_size: int,
    mtu: int,
    ip_header_size: int = 20,
) -> FragmentationReport:
    """Calculate IP fragmentation for a packet exceeding MTU.

    Derives fragment count and sizes from MTU constraint. IP fragments
    must have payloads that are multiples of 8 bytes (except last fragment).

    Args:
        packet_size: Total IP packet size in bytes (including IP header)
        mtu: Path MTU in bytes
        ip_header_size: IP header size (default 20, can be up to 60 with options)

    Returns:
        FragmentationReport.
    """
    if packet_size <= 0 or mtu <= 0:
        raise ValueError("Sizes must be positive")
    if mtu <= ip_header_size:
        raise ValueError("MTU must be larger than IP header")

    if packet_size <= mtu:
        return FragmentationReport(
            original_size=packet_size,
            mtu=mtu,
            ip_header_size=ip_header_size,
            needs_fragmentation=False,
            num_fragments=1,
            fragment_sizes=[packet_size - ip_header_size],
            total_overhead=0,
        )

    payload = packet_size - ip_header_size
    max_fragment_payload = ((mtu - ip_header_size) // 8) * 8  # must be multiple of 8

    fragments = []
    remaining = payload
    while remaining > 0:
        if remaining <= mtu - ip_header_size:
            fragments.append(remaining)  # last fragment doesn't need 8-byte alignment
            remaining = 0
        else:
            fragments.append(max_fragment_payload)
            remaining -= max_fragment_payload

    # Overhead: each extra fragment adds an IP header
    extra_headers = (len(fragments) - 1) * ip_header_size

    return FragmentationReport(
        original_size=packet_size,
        mtu=mtu,
        ip_header_size=ip_header_size,
        needs_fragmentation=True,
        num_fragments=len(fragments),
        fragment_sizes=fragments,
        total_overhead=extra_headers,
    )


def congestion_window(
    cwnd_segments: int,
    ssthresh_segments: int,
    rtt_seconds: float,
    mss_bytes: int = 1460,
    event: Optional[str] = None,
) -> CongestionWindowReport:
    """Calculate TCP congestion window state and response to events.

    Models AIMD (additive increase, multiplicative decrease) behavior.

    Args:
        cwnd_segments: Current congestion window in segments
        ssthresh_segments: Slow start threshold in segments
        rtt_seconds: Round-trip time
        mss_bytes: Maximum segment size
        event: Optional event ("loss_timeout", "loss_3dup", "ack")

    Returns:
        CongestionWindowReport with updated state.
    """
    if cwnd_segments < 1:
        raise ValueError("cwnd must be at least 1")

    new_cwnd = cwnd_segments
    new_ssthresh = ssthresh_segments

    if event == "loss_timeout":
        # Timeout: ssthresh = cwnd/2, cwnd = 1
        new_ssthresh = max(2, cwnd_segments // 2)
        new_cwnd = 1
        phase = "slow_start"
    elif event == "loss_3dup":
        # Fast retransmit/recovery: ssthresh = cwnd/2, cwnd = ssthresh + 3
        new_ssthresh = max(2, cwnd_segments // 2)
        new_cwnd = new_ssthresh + 3
        phase = "fast_recovery"
    elif event == "ack":
        if cwnd_segments < ssthresh_segments:
            # Slow start: cwnd += 1 MSS per ACK (exponential growth)
            new_cwnd = cwnd_segments + 1
            phase = "slow_start"
        else:
            # Congestion avoidance: cwnd += 1/cwnd per ACK (linear growth)
            # Approximated as cwnd += 1 per RTT
            new_cwnd = cwnd_segments + 1
            phase = "congestion_avoidance"
    else:
        phase = "slow_start" if cwnd_segments < ssthresh_segments else "congestion_avoidance"

    window_bytes = new_cwnd * mss_bytes
    throughput = (window_bytes * 8) / rtt_seconds if rtt_seconds > 0 else 0

    return CongestionWindowReport(
        phase=phase,
        cwnd_segments=new_cwnd,
        ssthresh_segments=new_ssthresh,
        window_bytes=window_bytes,
        throughput_bps=throughput,
        rtt_seconds=rtt_seconds,
        mss_bytes=mss_bytes,
    )


def _int_to_ip(n: int) -> str:
    """Convert 32-bit integer to dotted-decimal IPv4 string."""
    return f"{(n>>24)&0xFF}.{(n>>16)&0xFF}.{(n>>8)&0xFF}.{n&0xFF}"


def _fmt_bps(bps: float) -> str:
    """Format bits per second in human-readable form."""
    if bps >= 1e9:
        return f"{bps/1e9:.2f} Gbps"
    elif bps >= 1e6:
        return f"{bps/1e6:.2f} Mbps"
    elif bps >= 1e3:
        return f"{bps/1e3:.2f} Kbps"
    else:
        return f"{bps:.0f} bps"
