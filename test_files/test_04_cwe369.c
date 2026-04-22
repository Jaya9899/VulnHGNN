#include <stdio.h>

/**
 * test_04_cwe369.c
 * Result: CWE-369 (Divide by Zero)
 */
int do_division(int a, int b) {
    // VULN: No check if b is zero
    return a / b;
}

int main() {
    int x = 10;
    int y = 0;
    printf("Result: %d\n", do_division(x, y));
    return 0;
}
