#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/**
 * test_05_cwe476.c
 * Result: CWE-476 (NULL Pointer Dereference)
 */
void handle_record(int id) {
    char *name = (char *)malloc(64);
    // VULN: name is dereferenced without checking if malloc failed (NULL)
    strcpy(name, "User Record");
    printf("ID: %d, Name: %s\n", id, name);
    free(name);
}

int main() {
    handle_record(1);
    return 0;
}
