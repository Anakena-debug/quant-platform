"""quantstrat/research/ — top-level research-local code.

Peer of ``quantstrat/tests/`` and ``quantstrat/docs/``; NOT under
``quantstrat/src/``. Holds diagnostic / sensitivity / sweep harnesses
that compose existing quantcore + quantengine + quantstrat APIs
without modifying any src tree. Promotion of any helper from here
into ``quantstrat/src/quantstrat/research/`` requires explicit
operator approval.

S29 contents:
    * tradeability_sweep — sensitivity sweep over
      (coverage_level × active_threshold × alpha_spec × universe).
"""
