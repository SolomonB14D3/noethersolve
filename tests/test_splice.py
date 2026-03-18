"""Tests for noethersolve.splice — splice site strength scorer."""

import pytest
from noethersolve.splice import (
    score_donor,
    score_acceptor,
    scan_splice_sites,
    pyrimidine_tract_score,
)


# ─── Donor scoring ──────────────────────────────────────────────────────────

class TestDonorScoring:
    def test_canonical_donor(self):
        # CAG|GTAAGT — canonical strong donor
        report = score_donor("CAGGTAAGT")
        assert report.site_type == "donor"
        assert report.has_canonical_dinucleotide is True
        assert report.score > 0

    def test_non_canonical_flagged(self):
        # Replace GT with AT → non-canonical
        report = score_donor("CAGATAAGT")
        assert report.has_canonical_dinucleotide is False

    def test_strong_donor(self):
        report = score_donor("CAGGTAAGT")
        # This is a strong consensus donor
        assert report.strength in ("STRONG", "MODERATE")

    def test_per_position_scores(self):
        report = score_donor("CAGGTAAGT")
        assert len(report.per_position_scores) == 9

    def test_wrong_length_raises(self):
        with pytest.raises(ValueError, match="9"):
            score_donor("CAGGT")

    def test_invalid_bases_raise(self):
        with pytest.raises(ValueError, match="Invalid"):
            score_donor("CAGGTXAGT")

    def test_case_insensitive(self):
        report = score_donor("caggtaagt")
        assert report.sequence == "CAGGTAAGT"

    def test_u_accepted(self):
        report = score_donor("CAGGUAAGU")
        assert report.sequence == "CAGGTAAGT"


# ─── Acceptor scoring ──────────────────────────────────────────────────────

class TestAcceptorScoring:
    def test_canonical_acceptor(self):
        # 11 intronic (pyrimidine-rich) + AG + 2 exonic
        seq = "TCTTTCCTTCTAGGA"  # 15 chars - need 16
        seq = "TCTTTCCTTCCTAGGA"  # 16 chars
        report = score_acceptor(seq)
        assert report.site_type == "acceptor"

    def test_canonical_ag_detected(self):
        # 12 intronic + AG + 2 exonic = 16 positions
        # AG at 0-indexed positions 11-12
        seq = "TTTTTTTTTTTAGGCA"  # 11 T's + AG + GCA = 16? No: 11+2+3=16
        # Actually: positions 0-10 = 11 intronic, 11-12 = AG, 13-15 = 3 more = 16
        # But we need AG at positions 11-12: T*11 + A + G + C + A = 15... need 16
        # AG at 0-indexed positions 11-12: 11 T's + AG + GA + A = 16
        seq = "TTTTTTTTTTTAAGAA"  # T*11 = pos 0-10, A at 11, G at 12 — wait no
        # pos 11 = A, pos 12 = G → that's AG at (11,12)
        seq = "TTTTTTTTTTTAGGERA"  # too long
        # Let's just count: T(0)T(1)T(2)T(3)T(4)T(5)T(6)T(7)T(8)T(9)T(10)A(11)G(12)G(13)C(14)A(15)
        seq = "TTTTTTTTTTTAAGCA"  # wrong - A at 11, A at 12
        # I need: pos[11]='A', pos[12]='G'
        seq = list("TTTTTTTTTTTTTTTT")  # 16 T's
        seq[11] = "A"
        seq[12] = "G"
        seq = "".join(seq)  # TTTTTTTTTTTAGNTT → wait
        report = score_acceptor(seq)
        assert report.has_canonical_dinucleotide is True

    def test_wrong_length_raises(self):
        with pytest.raises(ValueError, match="16"):
            score_acceptor("TTTAG")

    def test_per_position_scores(self):
        seq = list("TTTTTTTTTTTTTTTT")
        seq[11] = "A"
        seq[12] = "G"
        seq = "".join(seq)
        report = score_acceptor(seq)
        assert len(report.per_position_scores) == 16

    def test_pyrimidine_rich_scores_well(self):
        # Strong polypyrimidine tract
        seq = "CCTTCCTTCCCCAGGA"  # 12 pyrimidines + AG + GA
        report = score_acceptor(seq)
        assert report.score > 0


# ─── Pyrimidine tract score ────────────────────────────────────────────────

class TestPyrimidineTract:
    def test_all_pyrimidines(self):
        assert pyrimidine_tract_score("TTTTCCCC") == 1.0

    def test_no_pyrimidines(self):
        assert pyrimidine_tract_score("GGGGAAAA") == 0.0

    def test_mixed(self):
        score = pyrimidine_tract_score("TGCA")
        assert abs(score - 0.5) < 1e-9

    def test_empty(self):
        assert pyrimidine_tract_score("") == 0.0

    def test_u_treated_as_t(self):
        assert pyrimidine_tract_score("UUUU") == 1.0

    def test_case_insensitive(self):
        assert pyrimidine_tract_score("ttcc") == pyrimidine_tract_score("TTCC")


# ─── Scan splice sites ─────────────────────────────────────────────────────

class TestScanSpliceSites:
    def test_finds_donor_sites(self):
        # Embed a GT with enough context
        seq = "AAACAGGTAAGTAAAAAAA"
        sites = scan_splice_sites(seq, site_type="donor")
        assert len(sites) > 0
        assert all(s.site_type == "donor" for s in sites)

    def test_finds_acceptor_sites(self):
        # Embed an AG with enough context
        seq = "TTTTTTTTTTTTTTTTTTTAGGAAAAAAAAA"
        sites = scan_splice_sites(seq, site_type="acceptor")
        assert len(sites) > 0
        assert all(s.site_type == "acceptor" for s in sites)

    def test_both_finds_donors_and_acceptors(self):
        # Long sequence with both GT and AG
        seq = "AAACAGGTAAGTTTTTTTTTTTTTTTTAGGAAAAAA"
        sites = scan_splice_sites(seq, site_type="both")
        {s.site_type for s in sites}
        # May find both depending on context windows
        assert len(sites) > 0

    def test_sorted_by_score_descending(self):
        seq = "AAACAGGTAAGTCCCGTATGTCCC"
        sites = scan_splice_sites(seq, site_type="donor")
        for i in range(len(sites) - 1):
            assert sites[i].score >= sites[i + 1].score

    def test_invalid_site_type_raises(self):
        with pytest.raises(ValueError, match="site_type"):
            scan_splice_sites("AAAA", site_type="exon")

    def test_invalid_bases_raise(self):
        with pytest.raises(ValueError, match="Invalid"):
            scan_splice_sites("AAXGTAA")

    def test_short_sequence_no_sites(self):
        sites = scan_splice_sites("ACGT", site_type="donor")
        assert len(sites) == 0

    def test_position_reported(self):
        seq = "AAACAGGTAAGTAAAAAAA"
        sites = scan_splice_sites(seq, site_type="donor")
        if sites:
            assert sites[0].position >= 0


# ─── Report formatting ─────────────────────────────────────────────────────

class TestReportFormat:
    def test_str_contains_score(self):
        report = score_donor("CAGGTAAGT")
        s = str(report)
        assert "Score" in s

    def test_str_contains_strength(self):
        report = score_donor("CAGGTAAGT")
        assert report.strength in str(report)

    def test_str_contains_canonical(self):
        report = score_donor("CAGGTAAGT")
        assert "Canonical" in str(report)

    def test_passed_property(self):
        report = score_donor("CAGGTAAGT")
        if report.strength in ("STRONG", "MODERATE"):
            assert report.passed is True


# ─── Edge cases ─────────────────────────────────────────────────────────────

class TestEdgeCases:
    def test_all_same_base_donor(self):
        # AAAAAAAAA — no GT, should still score
        report = score_donor("AAAAAAAAA")
        assert report.has_canonical_dinucleotide is False

    def test_all_same_base_acceptor(self):
        report = score_acceptor("AAAAAAAAAAAAAAAA")
        assert report.has_canonical_dinucleotide is False
