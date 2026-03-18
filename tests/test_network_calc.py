"""Tests for the networking calculator module."""

import pytest
from noethersolve.network_calc import (
    bandwidth_delay_product,
    tcp_throughput,
    subnet_calc,
    ip_fragmentation,
    congestion_window,
)


class TestBandwidthDelayProduct:
    def test_1gbps_50ms(self):
        """1 Gbps x 50ms = 6.25 MB."""
        r = bandwidth_delay_product(bandwidth_bps=1e9, rtt_seconds=0.050)
        assert abs(r.bdp_bytes - 6_250_000) < 1  # 6.25 MB

    def test_bdp_bits(self):
        """BDP in bits = bandwidth * RTT."""
        r = bandwidth_delay_product(bandwidth_bps=1e6, rtt_seconds=0.1)
        assert abs(r.bdp_bits - 100_000) < 1

    def test_min_window(self):
        """Minimum TCP window must be at least BDP bytes."""
        r = bandwidth_delay_product(bandwidth_bps=1e9, rtt_seconds=0.050)
        assert r.min_window_bytes >= r.bdp_bytes

    def test_utilization_full(self):
        """Window >= BDP means full utilization."""
        r = bandwidth_delay_product(bandwidth_bps=1e9, rtt_seconds=0.050,
                                     window_size_bytes=10_000_000)
        assert abs(r.utilization - 1.0) < 0.01

    def test_utilization_half(self):
        """Window = BDP/2 means ~50% utilization."""
        r = bandwidth_delay_product(bandwidth_bps=1e9, rtt_seconds=0.050,
                                     window_size_bytes=3_125_000)
        assert abs(r.utilization - 0.5) < 0.01

    def test_error_zero_bandwidth(self):
        with pytest.raises(ValueError, match="positive"):
            bandwidth_delay_product(bandwidth_bps=0, rtt_seconds=0.050)

    def test_error_negative_rtt(self):
        with pytest.raises(ValueError, match="positive"):
            bandwidth_delay_product(bandwidth_bps=1e9, rtt_seconds=-0.01)

    def test_report_string(self):
        r = bandwidth_delay_product(bandwidth_bps=1e9, rtt_seconds=0.050)
        s = str(r)
        assert "Bandwidth-Delay" in s
        assert "RTT" in s


class TestTCPThroughput:
    def test_window_limited(self):
        """Throughput = window * 8 / RTT."""
        r = tcp_throughput(window_size_bytes=65535, rtt_seconds=0.010)
        expected = 65535 * 8 / 0.010
        assert abs(r.max_throughput_bps - expected) < 1

    def test_loss_limited_mathis(self):
        """With loss, Mathis model applies: MSS*8*1.22 / (RTT * sqrt(p))."""
        r = tcp_throughput(window_size_bytes=65535, rtt_seconds=0.050,
                          mss_bytes=1460, loss_rate=0.01)
        assert r.mathis_throughput_bps is not None
        assert r.mathis_throughput_bps > 0

    def test_no_loss_window_limited(self):
        """With loss_rate=0, falls back to window-limited."""
        r = tcp_throughput(window_size_bytes=65535, rtt_seconds=0.050,
                          loss_rate=0.0)
        assert r.mathis_throughput_bps == r.max_throughput_bps

    def test_total_loss(self):
        """Loss rate = 1 means zero throughput from Mathis."""
        r = tcp_throughput(window_size_bytes=65535, rtt_seconds=0.050,
                          loss_rate=1.0)
        assert r.mathis_throughput_bps == 0

    def test_error_zero_window(self):
        with pytest.raises(ValueError, match="positive"):
            tcp_throughput(window_size_bytes=0, rtt_seconds=0.050)

    def test_error_zero_rtt(self):
        with pytest.raises(ValueError, match="positive"):
            tcp_throughput(window_size_bytes=65535, rtt_seconds=0)

    def test_report_string(self):
        r = tcp_throughput(window_size_bytes=65535, rtt_seconds=0.010)
        s = str(r)
        assert "TCP Throughput" in s
        assert "Window" in s


class TestSubnetCalc:
    def test_24_prefix(self):
        """192.168.1.0/24: 256 addresses, 254 usable, broadcast .255."""
        r = subnet_calc("192.168.1.0", 24)
        assert r.network_address == "192.168.1.0"
        assert r.broadcast_address == "192.168.1.255"
        assert r.total_addresses == 256
        assert r.usable_hosts == 254
        assert r.first_host == "192.168.1.1"
        assert r.last_host == "192.168.1.254"

    def test_31_prefix_point_to_point(self):
        """/31: 2 addresses, 2 usable (point-to-point RFC 3021)."""
        r = subnet_calc("10.0.0.0", 31)
        assert r.total_addresses == 2
        assert r.usable_hosts == 2

    def test_32_prefix_host_route(self):
        """/32: single host."""
        r = subnet_calc("10.0.0.1", 32)
        assert r.total_addresses == 1
        assert r.usable_hosts == 1
        assert r.network_address == "10.0.0.1"

    def test_subnet_mask_24(self):
        r = subnet_calc("192.168.1.0", 24)
        assert r.subnet_mask == "255.255.255.0"

    def test_wildcard_mask_24(self):
        r = subnet_calc("192.168.1.0", 24)
        assert r.wildcard_mask == "0.0.0.255"

    def test_16_prefix(self):
        r = subnet_calc("172.16.0.0", 16)
        assert r.total_addresses == 65536
        assert r.usable_hosts == 65534
        assert r.broadcast_address == "172.16.255.255"

    def test_host_address_resolved_to_network(self):
        """Giving a host address still computes correct network."""
        r = subnet_calc("192.168.1.100", 24)
        assert r.network_address == "192.168.1.0"

    def test_error_invalid_prefix(self):
        with pytest.raises(ValueError, match="0-32"):
            subnet_calc("192.168.1.0", 33)

    def test_error_invalid_ip(self):
        with pytest.raises(ValueError, match="Invalid"):
            subnet_calc("192.168.1", 24)

    def test_error_invalid_octet(self):
        with pytest.raises(ValueError, match="Invalid octet"):
            subnet_calc("192.168.1.300", 24)

    def test_report_string(self):
        r = subnet_calc("192.168.1.0", 24)
        s = str(r)
        assert "Subnet" in s
        assert "192.168.1.0" in s
        assert "Broadcast" in s


class TestIPFragmentation:
    def test_4000_byte_mtu_1500(self):
        """4000 byte packet with MTU 1500 needs fragmentation."""
        r = ip_fragmentation(packet_size=4000, mtu=1500)
        assert r.needs_fragmentation is True
        assert r.num_fragments >= 3  # 3980 payload / 1480 per frag

    def test_no_fragmentation_needed(self):
        """Packet smaller than MTU: no fragmentation."""
        r = ip_fragmentation(packet_size=500, mtu=1500)
        assert r.needs_fragmentation is False
        assert r.num_fragments == 1

    def test_exact_mtu_no_fragmentation(self):
        """Packet exactly at MTU: no fragmentation needed."""
        r = ip_fragmentation(packet_size=1500, mtu=1500)
        assert r.needs_fragmentation is False

    def test_fragment_payloads_8_byte_aligned(self):
        """All non-last fragments must have payload as multiple of 8."""
        r = ip_fragmentation(packet_size=4000, mtu=1500)
        for payload in r.fragment_sizes[:-1]:
            assert payload % 8 == 0

    def test_fragment_total_equals_original_payload(self):
        """Sum of fragment payloads must equal original payload."""
        r = ip_fragmentation(packet_size=4000, mtu=1500, ip_header_size=20)
        original_payload = 4000 - 20
        assert sum(r.fragment_sizes) == original_payload

    def test_overhead_calculation(self):
        """Each extra fragment adds one IP header worth of overhead."""
        r = ip_fragmentation(packet_size=4000, mtu=1500, ip_header_size=20)
        assert r.total_overhead == (r.num_fragments - 1) * 20

    def test_error_zero_packet(self):
        with pytest.raises(ValueError, match="positive"):
            ip_fragmentation(packet_size=0, mtu=1500)

    def test_error_mtu_too_small(self):
        with pytest.raises(ValueError, match="larger than"):
            ip_fragmentation(packet_size=1500, mtu=20)

    def test_report_string(self):
        r = ip_fragmentation(packet_size=4000, mtu=1500)
        s = str(r)
        assert "Fragmentation" in s
        assert "Fragment" in s


class TestCongestionWindow:
    def test_slow_start_phase(self):
        """cwnd < ssthresh means slow start."""
        r = congestion_window(cwnd_segments=4, ssthresh_segments=64,
                              rtt_seconds=0.050)
        assert r.phase == "slow_start"

    def test_congestion_avoidance_phase(self):
        """cwnd >= ssthresh means congestion avoidance."""
        r = congestion_window(cwnd_segments=64, ssthresh_segments=32,
                              rtt_seconds=0.050)
        assert r.phase == "congestion_avoidance"

    def test_timeout_resets_cwnd(self):
        """Timeout: cwnd=1, ssthresh = cwnd/2."""
        r = congestion_window(cwnd_segments=40, ssthresh_segments=64,
                              rtt_seconds=0.050, event="loss_timeout")
        assert r.cwnd_segments == 1
        assert r.ssthresh_segments == 20
        assert r.phase == "slow_start"

    def test_triple_dup_fast_recovery(self):
        """3 dup ACKs: ssthresh = cwnd/2, cwnd = ssthresh + 3."""
        r = congestion_window(cwnd_segments=40, ssthresh_segments=64,
                              rtt_seconds=0.050, event="loss_3dup")
        assert r.ssthresh_segments == 20
        assert r.cwnd_segments == 23  # 20 + 3
        assert r.phase == "fast_recovery"

    def test_ack_in_slow_start(self):
        """ACK in slow start: cwnd += 1."""
        r = congestion_window(cwnd_segments=4, ssthresh_segments=64,
                              rtt_seconds=0.050, event="ack")
        assert r.cwnd_segments == 5
        assert r.phase == "slow_start"

    def test_window_bytes(self):
        """Window bytes = cwnd * MSS."""
        r = congestion_window(cwnd_segments=10, ssthresh_segments=64,
                              rtt_seconds=0.050, mss_bytes=1460)
        assert r.window_bytes == 10 * 1460

    def test_error_zero_cwnd(self):
        with pytest.raises(ValueError, match="at least 1"):
            congestion_window(cwnd_segments=0, ssthresh_segments=64,
                              rtt_seconds=0.050)

    def test_report_string(self):
        r = congestion_window(cwnd_segments=10, ssthresh_segments=64,
                              rtt_seconds=0.050)
        s = str(r)
        assert "Congestion Window" in s
        assert "cwnd" in s
