#include <stdio.h>
#include <limits.h>

/**
 * test_08_triple_190_191_369.c
 * Result: CWE-190, CWE-191, CWE-369
 */
void math_chaos(int a, int b, int c) {
    // 1. Overflow
    int v1 = a + 1000000;
    
    // 2. Underflow
    int v2 = b - 1000000;
    
    // 3. Divide by Zero
    int v3 = v1 / c;
    
    printf("%d %d %d\n", v1, v2, v3);
}

int main() {
    math_chaos(INT_MAX - 10, INT_MIN + 10, 0);
    return 0;
}
