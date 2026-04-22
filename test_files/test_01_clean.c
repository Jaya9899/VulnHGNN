#include <stdio.h>
#include <stdlib.h>

/**
 * test_01_clean.c
 * Result: NON-VULNERABLE
 */
int main() {
    int a = 10;
    int b = 20;
    int sum = a + b;
    
    if (b != 0) {
        printf("Result: %d\n", sum / b);
    }
    
    int *p = (int *)malloc(sizeof(int));
    if (p != NULL) {
        *p = 100;
        printf("Pointer value: %d\n", *p);
        free(p);
    }
    
    return 0;
}
