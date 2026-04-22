#include <stdio.h>
#include <limits.h>

/**
 * test_02_cwe190.c
 * Result: CWE-190 (Integer Overflow)
 */
void check_overflow(int input) {
    // VULN: input + 1000 can overflow if input is near INT_MAX
    int result = input + 1000;
    printf("Result: %d\n", result);
}

int main() {
    check_overflow(INT_MAX - 5);
    return 0;
}
