# Bug Report Issue Templates

This directory contains templates for creating GitHub issues based on bugs found during code review on 2026-03-31.

## How to Use These Templates

Since automated issue creation failed due to API permissions, these templates can be manually copied into GitHub issues.

### Steps:
1. Go to https://github.com/kmikol/ml-system/issues/new
2. Copy the content from one of the template files below
3. Paste into the issue body
4. Add the suggested labels
5. Submit the issue

## Issue Templates by Priority

### Critical (3 issues)
1. **issue_race_condition_db.md** - Race condition in database connection management
   - Labels: `bug`, `critical`, `concurrency`
   - Multiple files affected in shared/data_controller

2. **issue_model_manager_thread_safety.md** - TOCTOU race condition during model inference
   - Labels: `bug`, `critical`, `concurrency`, `serving`
   - Affects serving/main.py

### High Priority (3 issues)
3. **issue_null_pointer_ml_exporter.md** - Null pointer dereference in ML exporter
   - Labels: `bug`, `high-priority`, `monitoring`
   - Affects monitoring/ml_exporter/drift.py

4. **issue_array_bounds_serving.md** - Array index out of bounds in serving
   - Labels: `bug`, `high-priority`, `serving`
   - Affects serving/main.py

5. **issue_missing_dict_key_ml_exporter.md** - Missing dictionary key validation
   - Labels: `bug`, `high-priority`, `monitoring`
   - Affects monitoring/ml_exporter/main.py

### Performance (1 issue)
6. **issue_concurrency_limit_serving.md** - Restrictive concurrency limit
   - Labels: `performance`, `serving`, `configuration`
   - Affects serving/main.py
   - Note: Should be addressed AFTER fixing thread-safety issues

## Additional Bugs Found

See `../BUGS_FOUND.md` for a comprehensive list of all 14 bugs found, including:
- Medium priority issues (2)
- Code smells and warnings (5)

These additional issues may warrant separate tickets but are less critical than the ones with templates above.

## Summary Statistics

- **Total issues found**: 14
- **Critical**: 3
- **High priority**: 4 (3 with templates)
- **Medium priority**: 2
- **Low priority/Code smells**: 5

## Recommended Action Order

1. Create issues for all critical bugs (templates 1-2)
2. Fix critical race conditions in database controllers and model manager
3. Create issues for high priority bugs (templates 3-5)
4. Add comprehensive concurrency and edge case tests
5. Address performance issue (template 6) after thread-safety is fixed
6. Review and triage medium/low priority issues from BUGS_FOUND.md

## Testing Recommendations

Before closing any of these issues, ensure:
1. Unit tests cover the edge cases
2. Concurrency tests verify thread safety
3. Integration tests validate fixes under load
4. No regressions in existing functionality
