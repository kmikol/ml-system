# Race Condition in Database Connection Management

**Labels**: `bug`, `critical`, `concurrency`

## Description

Multiple data controller classes have race conditions in their database connection management. The `self._conn` attribute is modified without proper synchronization, which can lead to crashes and data corruption in multi-threaded environments.

## Affected Files

- `shared/data_controller/dataset.py` (lines 115-117, 202-204)
- `shared/data_controller/annotation.py` (lines 66-68, 84-86)
- `shared/data_controller/sampling.py` (lines 45-47)
- `shared/data_controller/serving.py` (lines 101-103)
- `shared/data_controller/_base.py` (lines 200-206)

## Problem Details

### Issue 1: Unprotected state mutation in rollback blocks

During error handling, `self._conn` is set to `None` without holding a lock:

```python
except Exception:
    self._conn.rollback()  # No lock held
except Exception:
    self._conn = None      # Unprotected state mutation
```

If another thread accesses the connection simultaneously, it can cause `NoneType` errors.

### Issue 2: Missing lock in `_connect()` method

The `_connect()` method in `_base.py` checks and modifies `self._conn` without any synchronization:

```python
def _connect(self):
    if self._conn is None or self._conn.closed:  # No lock
        self._conn = self._psycopg2.connect(self._dsn)  # Possible race
```

Multiple threads could simultaneously check if `self._conn is None`, leading to:
- Multiple connection attempts
- Resource leaks
- Unexpected connection resets

## Impact

- **Severity**: Critical
- **Likelihood**: High in multi-threaded environments
- Can cause:
  - Application crashes
  - Database connection leaks
  - Data corruption
  - Unpredictable behavior under load

## Recommended Fix

1. Use a threading lock to protect all access to `self._conn`
2. Acquire the lock before checking/modifying connection state
3. Hold the lock throughout rollback operations
4. Consider using a context manager for automatic lock release

Example:

```python
class _DataControllerBase:
    def __init__(self, ...):
        self._conn_lock = threading.Lock()
        self._conn = None

    def _connect(self):
        with self._conn_lock:
            if self._conn is None or self._conn.closed:
                self._conn = self._psycopg2.connect(self._dsn)

    def execute_query(self, ...):
        with self._conn_lock:
            self._connect()
            # ... perform operations ...
```

## Testing Recommendations

1. Add concurrency tests that create multiple threads accessing the same controller
2. Use thread sanitizers during testing
3. Add stress tests with high concurrent load
