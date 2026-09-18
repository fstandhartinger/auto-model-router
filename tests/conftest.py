"""Make the test suite hermetic.

Two guarantees, both autouse:

1. **No classifier or provider credentials.** ``Router`` falls back to the live
   Jev classifier whenever the classifier key is set, so a suite run on a
   developer machine would quietly make real API calls and behave differently
   from CI. Every credential-shaped variable is removed for the duration of a
   test - matched by name shape, not by a list of vendors - along with the
   router's own configuration variables. A test that wants one sets it
   explicitly with ``monkeypatch.setenv``.

2. **No outbound network.** Connections to anything other than loopback raise,
   so a test can never depend on a third party being up - or spend money.
   Loopback stays open because the local HTTP smoke test and the benchmark
   outage tests both need to talk to (or fail to talk to) 127.0.0.1.

Subprocesses are not affected: ``tests/test_http_smoke.py`` and
``tests/test_sandbox.py`` spawn their own processes on purpose.
"""

import ipaddress
import os
import re
import socket

import pytest

#: Any provider credential, whoever the provider is. Matching by shape rather
#: than by a list of vendor names keeps the suite hermetic on a machine whose
#: providers this repository has never heard of.
CREDENTIAL_ENV = re.compile(r"(?i)(api[_-]?key|auth[_-]?token|access[_-]?token|"
                            r"[_-]key$|secret|password|credential)")

#: The router's own knobs, so a developer's exported config cannot leak into a test.
ROUTER_ENV = ("AUTO_ROUTER_CONFIG", "AUTO_ROUTER_LEDGER", "AUTO_ROUTER_CACHE_DIR",
              "AUTO_ROUTER_BENCH_URL", "AUTO_ROUTER_BENCH_OFFLINE", "AUTO_ROUTER_POLICY",
              "AUTO_ROUTER_JEV_URL", "AUTO_ROUTER_JEV_MODEL")


def blocked_env_names() -> list[str]:
    return sorted({name for name in os.environ if CREDENTIAL_ENV.search(name)}
                  | set(ROUTER_ENV))


class BlockedNetwork(RuntimeError):
    pass


def _is_loopback(address) -> bool:
    if not isinstance(address, tuple) or not address:
        return True          # AF_UNIX and friends: not the network
    host = address[0]
    if host in ("", "localhost"):
        return True
    try:
        return ipaddress.ip_address(host).is_loopback
    except ValueError:
        return False


@pytest.fixture(autouse=True)
def _hermetic(monkeypatch):
    for name in blocked_env_names():
        monkeypatch.delenv(name, raising=False)

    real_connect = socket.socket.connect
    real_connect_ex = socket.socket.connect_ex

    def connect(self, address):
        if not _is_loopback(address):
            raise BlockedNetwork(f"the test suite may not reach {address!r}")
        return real_connect(self, address)

    def connect_ex(self, address):
        if not _is_loopback(address):
            raise BlockedNetwork(f"the test suite may not reach {address!r}")
        return real_connect_ex(self, address)

    monkeypatch.setattr(socket.socket, "connect", connect)
    monkeypatch.setattr(socket.socket, "connect_ex", connect_ex)
    monkeypatch.setattr(socket, "create_connection",
                        lambda address, *a, **k: (_ for _ in ()).throw(
                            BlockedNetwork(f"the test suite may not reach {address!r}")))
    yield


@pytest.fixture
def allow_network(monkeypatch):
    """Opt back in, for a test that deliberately drives a local subprocess stack."""
    monkeypatch.undo()
