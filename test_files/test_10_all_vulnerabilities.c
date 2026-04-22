#include <stdio.h>
#include <stdlib.h>
#include <limits.h>

/**
 * test_10_all_vulnerabilities.c
 * Result: CWE-190, CWE-191, CWE-369, CWE-476
 */
void the_ultimate_test(int a, int b, int c) {
    int *p = (int *)malloc(sizeof(int));
    
    // 1. NullPtr
    *p = a;
    
    // 2. Overflow
    int v1 = (*p) + INT_MAX;
    
    // 3. Underflow
    int v2 = b - 2000000;
    
    // 4. DivZero
    int v3 = v1 / c;
    
    printf("V1: %d, V2: %d, V3: %d\n", v1, v2, v3);
    free(p);
}

int main() {
    the_ultimate_test(100, INT_MIN + 100, 0);
    return 0;
}
