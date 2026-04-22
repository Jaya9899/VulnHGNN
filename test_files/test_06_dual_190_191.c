#include <stdio.h>
#include <limits.h>

/**
 * test_06_dual_190_191.c
 * Result: CWE-190 (Overflow) AND CWE-191 (Underflow)
 */
void complex_math(int x, int y) {
    // VULN: Overflow
    int over = x * 100;
    
    // VULN: Underflow
    int under = y - 100000;
    
    printf("O: %d, U: %d\n", over, under);
}

int main() {
    complex_math(INT_MAX / 2, INT_MIN + 50);
    return 0;
}
