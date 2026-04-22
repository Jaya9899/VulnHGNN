#include <stdio.h>
#include <stdlib.h>

/**
 * test_07_dual_369_476.c
 * Result: CWE-369 (DivZero) AND CWE-476 (NullPtr)
 */
void process_data(int divisor) {
    int *data = (int *)malloc(sizeof(int));
    
    // VULN: Null pointer dereference
    *data = 500;
    
    // VULN: Divide by zero
    int result = *data / divisor;
    
    printf("Value: %d\n", result);
    free(data);
}

int main() {
    process_data(0);
    return 0;
}
