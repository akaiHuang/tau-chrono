#!/usr/bin/env python3
"""
Tuna-17 preflight: list backends, find Tuna-17 exact name, dump topology.

No hardware shots consumed. Just metadata queries.

Usage:
    .venv/bin/python experiments/preflight_tuna17.py
"""

import json
import os
import sys
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from qiskit_quantuminspire.qi_provider import QIProvider


def main():
    provider = QIProvider()
    backends = provider.backends()

    print("=" * 70)
    print("Quantum Inspire backends visible to this account")
    print("=" * 70)
    rows = []
    for b in backends:
        try:
            n = b.num_qubits
        except Exception:
            n = "?"
        try:
            status = b.status().to_dict()
        except Exception:
            status = {}
        rows.append({
            "name": b.name,
            "num_qubits": n,
            "operational": status.get("operational"),
            "pending_jobs": status.get("pending_jobs"),
            "status_msg": status.get("status_msg"),
        })
        print(f"  {b.name:30s}  qubits={n}  op={status.get('operational')}  queue={status.get('pending_jobs')}")

    # Find Tuna-17 candidate (anything with 17 qubits, or whose name contains 17 / tuna17 / spin)
    candidates = [r for r in rows
                  if (isinstance(r["num_qubits"], int) and r["num_qubits"] >= 17)
                  or any(kw in r["name"].lower() for kw in ["17", "spin-2", "tuna-17"])]
    print()
    print("=" * 70)
    print(f"Tuna-17 candidates (>=17 qubits or matching name): {len(candidates)}")
    print("=" * 70)
    for c in candidates:
        print(f"  -> {c['name']}  ({c['num_qubits']} qubits)")

    if not candidates:
        print()
        print("No 17-qubit backend visible. Possible reasons:")
        print("  - Tuna-17 not yet exposed to your account")
        print("  - Different name (check Quantum Inspire portal)")
        print("  - Backend offline")
        sys.exit(1)

    # Try to fetch coupling map / properties for the first candidate
    target_name = candidates[0]["name"]
    print()
    print("=" * 70)
    print(f"Pulling topology for: {target_name}")
    print("=" * 70)
    backend = provider.get_backend(target_name)

    coupling = None
    try:
        cm = backend.coupling_map
        coupling = list(cm.get_edges()) if cm is not None else None
    except Exception as e:
        print(f"  coupling_map: error -> {e}")

    print(f"  num_qubits   : {backend.num_qubits}")
    print(f"  coupling map : {coupling}")
    try:
        print(f"  basis gates  : {list(backend.operation_names)}")
    except Exception:
        pass
    try:
        print(f"  max_shots    : {backend.options.shots if hasattr(backend, 'options') else '?'}")
    except Exception:
        pass

    # Save
    os.makedirs("results", exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = f"results/tuna17_preflight_{ts}.json"
    payload = {
        "timestamp": ts,
        "all_backends": rows,
        "selected_backend": {
            "name": target_name,
            "num_qubits": backend.num_qubits,
            "coupling_map": coupling,
        },
    }
    with open(out_path, "w") as f:
        json.dump(payload, f, indent=2)
    print()
    print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()
