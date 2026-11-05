"""Sharia policy: only coins ruled permissible are tradable; the site's daily ruling overrides the recorded one."""
import os, sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import policy

SITE = {"BTC": dict(permissible=True), "BCH": dict(permissible=True), "XRP": dict(permissible=False), "BNB": dict(permissible=False)}


def test_site_permissible_is_tradable():
    assert policy.ruling("BCH", SITE) == (True, "site")


def test_site_ruling_overrides_recorded_permissible():
    assert policy.policy_recorded("XRP") and policy.ruling("XRP", SITE) == (False, "site")


def test_unlisted_coin_falls_back_to_recorded_ruling():
    assert policy.ruling("OP", SITE) == (True, "recorded")


def test_coin_without_any_ruling_is_never_tradable():
    assert policy.ruling("APT", SITE) == (False, "unreviewed")
    assert policy.ruling("NOTACOIN", {}) == (False, "unreviewed")


def test_religious_exclusion_always_wins():
    assert policy.ruling("BNB", {"BNB": dict(permissible=True)}) == (False, "excluded")


def test_agent_universe_is_filtered():
    assert policy.permitted(["BTC", "XRP", "APT", "OP"], SITE) == ["BTC", "OP"]


if __name__ == "__main__":
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for t in tests:
        t()
    print(f"{len(tests)} policy tests passed")
