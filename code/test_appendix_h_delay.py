"""Deterministic in-memory diagnostic of the original Appendix H delay merger.

No research datasets or models are loaded. Per-occurrence copying is an explicit
reproduction deviation; the original merger and its random-draw schedule stay
unchanged. Scripted values test mechanics, not a statistical delay distribution.
"""

import hashlib
from pathlib import Path
import pickle
import random
import sys
from unittest.mock import patch

import numpy as np

_previous_bytecode = sys.dont_write_bytecode
try:
    sys.dont_write_bytecode = True
    import MergeSingleTraces_openworld as synthesis
finally:
    sys.dont_write_bytecode = _previous_bytecode


DRAWS = (2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0)
OFFSETS = {2: [0.0, 2.0], 3: [0.0, 2.0, 9.0], 4: [0.0, 2.0, 9.0, 20.0]}


def require(condition, message):
    if not condition:
        raise AssertionError(message)


def fixtures(tabs):
    return ([np.array([0.0, 0.5]) for _ in range(tabs)],
            [np.array([-1, 1], dtype=np.int8) for _ in range(tabs)])


def invoke(times, directions, draws):
    """Temporarily patch only page_num and uniform; record every draw's timing."""
    calls, snapshots = [], []

    def scripted(low, high):
        require((low, high) == (2, 9), f"Unexpected draw interval: {(low, high)}")
        require(len(calls) < len(draws), "Merger requested an unexpected extra draw")
        value = draws[len(calls)]
        calls.append((low, high, value))
        snapshots.append([float(t[0]) for t in times])
        return value

    with patch.object(synthesis, 'page_num', len(times)), patch.object(synthesis.random, 'uniform', scripted):
        result = synthesis.merge_with_delaytime(times, directions, 2, 9)
    require(len(calls) == len(draws), f"Draw count {len(calls)} != expected {len(draws)}")
    return result, calls, snapshots


def schedule(tabs):
    times, directions = fixtures(tabs)
    originals = [t.copy() for t in times]
    result, calls, snapshots = invoke(times, directions, DRAWS[:2 * tabs - 1])
    offsets = [float(t[0]) for t in times]
    require(offsets == OFFSETS[tabs], f"Unexpected offsets: {offsets}")
    for t, original, offset in zip(times, originals, OFFSETS[tabs]):
        require(np.array_equal(t, original + offset), "Offset not applied to every packet")
    expected_snapshots = [[0.0] * tabs]
    current = [0.0] * tabs
    for page in range(1, tabs):
        current[page] = OFFSETS[tabs][page]
        expected_snapshots.extend([current.copy(), current.copy()])
    require(snapshots == expected_snapshots, f"Draw/application ordering changed: {snapshots}")
    require([c[2] for c in calls] == list(DRAWS[:2 * tabs - 1]), "Draw order changed")
    require(offsets[1] - offsets[0] == DRAWS[0], "First increment is not one draw")
    for page in range(2, tabs):
        require(offsets[page] - offsets[page - 1] == sum(DRAWS[2 * page - 3:2 * page - 1]),
                "Subsequent increment is not two draws")
    require(len(result[0]) == 2 * tabs and result[2] is None, "Unexpected merger result")
    print(f"  {tabs} tabs: offsets={offsets}; draws={[c[2] for c in calls]}; "
          f"uniform calls={len(calls)}; final two draws consumed but unused in offsets")


def mutation():
    times, directions = fixtures(2)
    before = [t.copy() for t in times]
    invoke(times, directions, DRAWS[:3])
    require(np.array_equal(times[0], before[0]), "First independent source unexpectedly moved")
    require(np.array_equal(times[1], before[1] + 2), "Direct source mutation not observed")
    print(f"  Second source: {before[1].tolist()} -> {times[1].tolist()}")


def accumulation():
    times, directions = fixtures(2)
    invoke(times, directions, DRAWS[:3])
    first = times[1].copy()
    invoke(times, directions, DRAWS[:3])
    require(np.array_equal(times[1], first + 2), "Expected accumulated shift not observed")
    require(times[1].tolist() == [4.0, 4.5], "Unexpected second-call source")
    print(f"  Reused second source: [0.0, 0.5] -> {first.tolist()} -> {times[1].tolist()}")


def aliasing():
    source = np.array([0.0, 0.5])
    times = [synthesis.rm_zero(source), synthesis.rm_zero(source)]
    require(np.shares_memory(times[0], times[1]), "Expected source views to alias")
    _, directions = fixtures(2)
    invoke(times, directions, DRAWS[:3])
    require(source.tolist() == [2.0, 2.5], "Aliased source did not shift")
    require(times[0].tolist() == times[1].tolist() == source.tolist(), "Alias effect not observed")
    print("  Duplicate source views share memory; both tabs, including the first, become [2.0, 2.5]")


def isolation():
    source = np.array([0.0, 0.5])
    direction = np.array([-1, 1], dtype=np.int8)
    for _ in range(2):
        times = [synthesis.rm_zero(source).copy() for _ in range(2)]
        directions = [direction.copy() for _ in range(2)]
        require(not np.shares_memory(times[0], times[1]), "Copies still alias each other")
        require(all(not np.shares_memory(t, source) for t in times), "Copy aliases source")
        require(not np.shares_memory(directions[0], directions[1]), "Direction copies alias")
        invoke(times, directions, DRAWS[:3])
        require([t.tolist() for t in times] == [[0.0, 0.5], [2.0, 2.5]], "Isolated offsets differ")
        require(source.tolist() == [0.0, 0.5] and direction.tolist() == [-1, 1], "Original source changed")
    print("  Two copy-isolated calls: source remains [0.0, 0.5]; selected occurrences offset by 0 and 2")


def equivalence():
    for tabs in (2, 3, 4):
        source_times, source_dirs = fixtures(tabs)
        literal_times = [t.copy() for t in source_times]
        literal_dirs = [d.copy() for d in source_dirs]
        isolated_times = [synthesis.rm_zero(t).copy() for t in source_times]
        isolated_dirs = [synthesis.rm_zero(d).copy() for d in source_dirs]
        a, ca, sa = invoke(literal_times, literal_dirs, DRAWS[:2 * tabs - 1])
        b, cb, sb = invoke(isolated_times, isolated_dirs, DRAWS[:2 * tabs - 1])
        require(a == b and ca == cb and sa == sb, f"{tabs}-tab fresh-call equivalence failed")
        require(all(t.tolist() == [0.0, 0.5] for t in source_times), "Isolated source changed")
    print("  Fresh independent 2/3/4-tab calls: identical merged timestamps, directions, annotations, and draw logs")


def exception_restoration():
    page, uniform = synthesis.page_num, random.uniform
    times, directions = fixtures(3)
    try:
        invoke(times, directions, ())
    except AssertionError as exc:
        require(str(exc) == 'Merger requested an unexpected extra draw', f"Unexpected exception: {exc}")
    else:
        raise AssertionError("Expected scripted-draw exhaustion was not raised")
    require(synthesis.page_num == page and random.uniform is uniform, "Exception leaked patched globals")


def main():
    source = Path(synthesis.__file__)
    original_hash = hashlib.sha256(source.read_bytes()).hexdigest()
    globals_before = vars(synthesis).copy()
    uniform_before = random.uniform
    python_rng = random.getstate()
    numpy_rng = np.random.get_state()
    bytecode_before = sys.dont_write_bytecode
    failures = []
    checks = [(f"Offset schedule / draw count / draw order / increments: {n} tabs",
               lambda n=n: schedule(n)) for n in (2, 3, 4)]
    checks += [("Direct input mutation", mutation), ("Cross-call accumulation", accumulation),
               ("Repeated-source NumPy aliasing", aliasing),
               ("Per-occurrence copy protection", isolation),
               ("Fresh non-aliased output equivalence", equivalence),
               ("Global restoration on an intentional exception", exception_restoration)]
    print("Original merger only; tiny artificial fixtures; no research data or model access.")
    print("Copy isolation is an explicit deviation from literal source ownership, not delay calculations.")
    try:
        for name, check in checks:
            try:
                check()
                print(f"PASS: {name}")
            except Exception as exc:
                failures.append(name)
                print(f"FAIL: {name}: {type(exc).__name__}: {exc}")
    finally:
        # Restore even if an unexpected failure escapes a test.
        synthesis.page_num = globals_before['page_num']
        random.uniform = uniform_before
        random.setstate(python_rng)
        np.random.set_state(numpy_rng)
        sys.dont_write_bytecode = bytecode_before
    try:
        require(vars(synthesis).keys() == globals_before.keys(), "Module namespace changed")
        require(all(vars(synthesis)[k] is v for k, v in globals_before.items()), "Module global identity changed")
        require(random.uniform is uniform_before and random.getstate() == python_rng, "Python random state changed")
        require(pickle.dumps(np.random.get_state()) == pickle.dumps(numpy_rng), "NumPy RNG state changed")
        require(sys.dont_write_bytecode == bytecode_before, "Bytecode setting changed")
        require(hashlib.sha256(source.read_bytes()).hexdigest() == original_hash, "Merger source file hash changed")
        print("PASS: original source hash, module globals, uniform function, Python/NumPy RNG states, and bytecode setting restored")
    except Exception as exc:
        failures.append('Final restoration')
        print(f"FAIL: final restoration: {type(exc).__name__}: {exc}")
    print("Scope: merger mechanics only. Paper ambiguity about extra waits, source timestamp "
          "reversals, outer-generator sampling/RNG calls, and dataset-level provenance remain outside this test.")
    print(f"{'FAIL' if failures else 'PASS'}: diagnostic completed; failed checks={len(failures)}")
    return 1 if failures else 0


if __name__ == '__main__':
    raise SystemExit(main())
