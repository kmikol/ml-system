# Null Pointer Dereference in ML Exporter

**Labels**: `bug`, `high-priority`, `monitoring`

## Description

The `get_annotated_count()` method in the drift controller directly accesses `cur.fetchone()[0]` without checking if `fetchone()` returns `None`. This causes an `IndexError` or `TypeError` when no rows are found in the database.

## Affected Files

- `monitoring/ml_exporter/drift.py` (line 51)
- `monitoring/ml_exporter/main.py` (line 51)

## Problem Details

Current implementation:

```python
def get_annotated_count(self) -> int:
    cur.execute(_COUNT_ANNOTATED)
    return cur.fetchone()[0]  # <-- Could be None
```

If the query returns no rows (which is valid SQL behavior), `fetchone()` returns `None`, and attempting to index it with `[0]` raises an exception.

This is particularly problematic when:
- The database is newly initialized
- All annotations have been deleted
- There's a schema mismatch

## Impact

- **Severity**: High
- **Likelihood**: Medium (occurs in fresh installations or after data cleanup)
- Can cause:
  - Application crash in monitoring service
  - Unhandled exception
  - Monitoring service downtime
  - Loss of metrics collection

## Recommended Fix

Option 1: Return 0 when no rows found (recommended)
```python
def get_annotated_count(self) -> int:
    cur.execute(_COUNT_ANNOTATED)
    result = cur.fetchone()
    if result is None:
        return 0
    return result[0]
```

Option 2: Use COALESCE in SQL
```python
_COUNT_ANNOTATED = """
    SELECT COALESCE(COUNT(*), 0)
    FROM annotations
    WHERE label IS NOT NULL
"""

def get_annotated_count(self) -> int:
    cur.execute(_COUNT_ANNOTATED)
    result = cur.fetchone()
    return result[0] if result else 0  # Extra safety
```

Option 3: Use fetchall with default
```python
def get_annotated_count(self) -> int:
    cur.execute(_COUNT_ANNOTATED)
    results = cur.fetchall()
    if not results or not results[0]:
        return 0
    return results[0][0]
```

## Testing Recommendations

1. Add unit test with empty database
2. Add test for schema without annotations table
3. Add test for table with no matching rows
4. Verify behavior in fresh installation
