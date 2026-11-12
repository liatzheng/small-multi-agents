# Two examples of dependency depth

## Problem 1 — Independent steps

```text
Given p = 4, q = 3, r = 9, and s = 8:
1. Compute a = p + q.
2. Compute b = r - p.
Return (a, b).
```

**Answer:** `(7, 5)`. We get `a = 4 + 3 = 7` and `b = 9 - 4 = 5`. Both steps use only the given values, so either can be done first.

## Problem 2 — The second step depends on the first

```text
Given p = 4, q = 3, r = 9, and s = 8:
1. Compute a = p + q.
2. Compute b = r - a.
Return (a, b).
```

**Answer:** `(7, 2)`. First compute `a = 4 + 3 = 7`. Substituting that result into the second step gives `b = 9 - 7 = 2`.

The prompts differ by one operand in step 2. Both have one addition, one subtraction, and the same unused input `s`.
