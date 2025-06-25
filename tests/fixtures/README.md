# Arc Runtime Golden Request Tests

This directory contains canonical test cases for Arc Runtime's request interception and fixing capabilities.

## Files

- `golden_requests.yaml` - Defines canonical failing requests and their expected fixes

## Test Cases

The golden requests include:

1. **High temperature GPT-4 request** - Temperature 0.95 should be reduced to 0.7
2. **High temperature GPT-4.1 request** - Temperature 0.99 should be reduced to 0.7  
3. **Edge case - temperature exactly 0.9** - Should not be modified (at threshold)
4. **Edge case - temperature 0.91** - Should be reduced to 0.7 (just above threshold)
5. **Low temperature request** - Temperature 0.5 should pass through unchanged

## Running the Tests

From the runtime directory:

```bash
# Run all golden request tests
python -m unittest tests.test_golden_requests -v

# Or use the convenience script
./run_golden_tests.py
```

## Verifying Your Environment

These tests help verify that Arc Runtime is correctly installed and functioning in your environment. They:

- Test pattern matching logic
- Verify fixes are applied correctly
- Measure interception overhead (<5ms requirement)
- Validate edge cases

## Adding New Test Cases

To add a new test case, edit `golden_requests.yaml` and add a new entry with:

- `name`: Descriptive name for the test
- `description`: What the test verifies
- `request`: The API request parameters
- `expected_fix`: The expected fix to be applied (or `null` if no fix expected)