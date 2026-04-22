#include <stdio.h>
#include <stdlib.h>
#include <limits.h>

/**
 * test_09_triple_191_369_476.c
 * Result: CWE-191, CWE-369, CWE-476
 */
void risky_logic(int val, int div) {
    int *ptr = (int *)malloc(sizeof(int));
    
    // 1. NullPtr
    *ptr = val;
    
    // 2. Underflow
    int sub = *ptr - INT_MAX;
    
    // 3. DivZero
    int result = sub / div;
    
    printf("Result: %d\n", result);
    free(ptr);
}

int main() {
    risky_logic(10, 0);
    return 0;
}
