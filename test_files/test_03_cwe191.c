#include <stdio.h>
#include <limits.h>

/**
 * test_03_cwe191.c
 * Result: CWE-191 (Integer Underflow)
 */
void check_underflow(int input) {
    // VULN: input - 1000 can underflow if input is near INT_MIN
    int result = input - 1000;
    printf("Result: %d\n", result);
}

int main() {
    check_underflow(INT_MIN + 5);
    return 0;
}
