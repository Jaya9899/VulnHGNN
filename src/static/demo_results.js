
const DEMO_FILES = {
    "test_01_clean.c": {
        analysis: {
            success: true,
            elapsed: 0.35,
            predictions: [
                { class: "non-vulnerable", probability: 0.99, detected: true, full_name: "Non-Vulnerable", description: "No known vulnerability pattern detected.", severity: "NONE" },
                { class: "CWE-190", probability: 0.01, detected: false },
                { class: "CWE-191", probability: 0.01, detected: false },
                { class: "CWE-369", probability: 0.01, detected: false },
                { class: "CWE-476", probability: 0.01, detected: false }
            ]
        }
    },
    "test_02_cwe190.c": {
        analysis: {
            success: true,
            elapsed: 0.45,
            predictions: [
                { class: "CWE-190", probability: 0.98, detected: true, function: "check_overflow", full_name: "Integer Overflow", description: "Integer arithmetic may exceed maximum value.", severity: "HIGH" },
                { class: "non-vulnerable", probability: 0.02, detected: false }
            ],
            thresholds: { "CWE-190": 0.4 }
        },
        localization: {
            "CWE-190": [
                { block: "entry", opcode: "add", score: 0.95, text: "%result = add nsw i32 %input, 1000", function: "check_overflow" }
            ]
        },
        healed_source: `#include <stdio.h>
#include <limits.h>

/**
 * test_02_cwe190.c (HEALED)
 */
void check_overflow(int input) {
    if (input > INT_MAX - 1000) { /* HEALED: CWE-190 Overflow guard */
        fprintf(stderr, "[HEALED] Integer overflow prevented\\n");
        return;
    }
    int result = input + 1000;
    printf("Result: %d\\n", result);
}

int main() {
    check_overflow(INT_MAX - 5);
    return 0;
}`
    },
    "test_03_cwe191.c": {
        analysis: {
            success: true,
            elapsed: 0.42,
            predictions: [
                { class: "CWE-191", probability: 0.97, detected: true, function: "check_underflow", full_name: "Integer Underflow", description: "Integer subtraction may wrap below minimum value.", severity: "HIGH" },
                { class: "non-vulnerable", probability: 0.03, detected: false }
            ]
        },
        localization: {
            "CWE-191": [
                { block: "entry", opcode: "sub", score: 0.94, text: "%result = sub nsw i32 %input, 1000", function: "check_underflow" }
            ]
        },
        healed_source: `#include <stdio.h>
#include <limits.h>

/**
 * test_03_cwe191.c (HEALED)
 */
void check_underflow(int input) {
    if (input < INT_MIN + 1000) { /* HEALED: CWE-191 Underflow guard */
        fprintf(stderr, "[HEALED] Integer underflow prevented\\n");
        return;
    }
    int result = input - 1000;
    printf("Result: %d\\n", result);
}

int main() {
    check_underflow(INT_MIN + 5);
    return 0;
}`
    },
    "test_05_cwe476.c": {
        analysis: {
            success: true,
            elapsed: 0.51,
            predictions: [
                { class: "CWE-476", probability: 0.99, detected: true, function: "handle_record", full_name: "NULL Pointer Dereference", description: "Pointer may be used without checking for NULL.", severity: "HIGH" },
                { class: "non-vulnerable", probability: 0.01, detected: false }
            ]
        },
        localization: {
            "CWE-476": [
                { block: "entry", opcode: "call", score: 0.98, text: "call void @llvm.memcpy.p0.p0.i64(ptr %name, ...)", function: "handle_record" },
                { block: "entry", opcode: "call", score: 0.85, text: "%name = call ptr @malloc(i64 64)", function: "handle_record" }
            ]
        },
        healed_source: `#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/**
 * test_05_cwe476.c (HEALED)
 */
void handle_record(int id) {
    char *name = (char *)malloc(64);
    if (name == NULL) { /* HEALED: CWE-476 NULL guard */
        fprintf(stderr, "[HEALED] Memory allocation failed\\n");
        return;
    }
    strcpy(name, "User Record");
    printf("ID: %d, Name: %s\\n", id, name);
    free(name);
}

int main() {
    handle_record(1);
    return 0;
}`
    },
    "test_06_dual_190_191.c": {
        analysis: {
            success: true,
            elapsed: 0.65,
            predictions: [
                { class: "CWE-190", probability: 0.95, detected: true, function: "complex_math", full_name: "Integer Overflow", description: "Integer arithmetic may exceed maximum value.", severity: "HIGH" },
                { class: "CWE-191", probability: 0.92, detected: true, function: "complex_math", full_name: "Integer Underflow", description: "Integer subtraction may wrap below minimum value.", severity: "HIGH" }
            ]
        },
        localization: {
            "CWE-190": [{ block: "entry", opcode: "mul", score: 0.91, text: "%over = mul nsw i32 %x, 100", function: "complex_math" }],
            "CWE-191": [{ block: "entry", opcode: "sub", score: 0.88, text: "%under = sub nsw i32 %y, 100000", function: "complex_math" }]
        },
        healed_source: `#include <stdio.h>
#include <limits.h>

/**
 * test_06_dual_190_191.c (HEALED)
 */
void complex_math(int x, int y) {
    if (x > INT_MAX / 100 || x < INT_MIN / 100) { /* HEALED: CWE-190 Overflow guard */
        fprintf(stderr, "[HEALED] Multiplicative overflow prevented\\n");
        return;
    }
    int over = x * 100;

    if (y < INT_MIN + 100000) { /* HEALED: CWE-191 Underflow guard */
        fprintf(stderr, "[HEALED] Subtraction underflow prevented\\n");
        return;
    }
    int under = y - 100000;

    printf("O: %d, U: %d\\n", over, under);
}

int main() {
    complex_math(INT_MAX / 2, INT_MIN + 50);
    return 0;
}`
    },
    "test_07_dual_369_476.c": {
        analysis: {
            success: true,
            elapsed: 0.62,
            predictions: [
                { class: "CWE-369", probability: 0.96, detected: true, function: "process_data", full_name: "Divide by Zero", description: "Division operation may use a zero divisor.", severity: "MEDIUM" },
                { class: "CWE-476", probability: 0.94, detected: true, function: "process_data", full_name: "NULL Pointer Dereference", description: "Pointer may be used without checking for NULL.", severity: "HIGH" }
            ]
        },
        localization: {
            "CWE-369": [{ block: "entry", opcode: "sdiv", score: 0.97, text: "%result = sdiv i32 %tmp, %divisor", function: "process_data" }],
            "CWE-476": [{ block: "entry", opcode: "store", score: 0.92, text: "store i32 500, ptr %data", function: "process_data" }]
        },
        healed_source: `#include <stdio.h>
#include <stdlib.h>

/**
 * test_07_dual_369_476.c (HEALED)
 */
void process_data(int divisor) {
    int *data = (int *)malloc(sizeof(int));
    if (data == NULL) { /* HEALED: CWE-476 NULL guard */
        return;
    }

    *data = 500;

    if (divisor == 0) { /* HEALED: CWE-369 DivZero guard */
        fprintf(stderr, "[HEALED] Division by zero prevented\\n");
        free(data);
        return;
    }
    int result = *data / divisor;

    printf("Value: %d\\n", result);
    free(data);
}

int main() {
    process_data(0);
    return 0;
}`
    },
    "test_10_all_vulnerabilities.c": {
        analysis: {
            success: true,
            elapsed: 0.85,
            predictions: [
                { class: "CWE-190", probability: 0.94, detected: true, function: "the_ultimate_test", full_name: "Integer Overflow", description: "Integer arithmetic may exceed maximum value.", severity: "HIGH" },
                { class: "CWE-191", probability: 0.91, detected: true, function: "the_ultimate_test", full_name: "Integer Underflow", description: "Integer subtraction may wrap below minimum value.", severity: "HIGH" },
                { class: "CWE-369", probability: 0.93, detected: true, function: "the_ultimate_test", full_name: "Divide by Zero", description: "Division operation may use a zero divisor.", severity: "MEDIUM" },
                { class: "CWE-476", probability: 0.95, detected: true, function: "the_ultimate_test", full_name: "NULL Pointer Dereference", description: "Pointer may be used without checking for NULL.", severity: "HIGH" }
            ]
        },
        localization: {
            "CWE-190": [{ block: "entry", opcode: "add", score: 0.89, text: "%v1 = add nsw i32 %tmp, 2147483647", function: "the_ultimate_test" }],
            "CWE-369": [{ block: "entry", opcode: "sdiv", score: 0.95, text: "%v3 = sdiv i32 %v1, %c", function: "the_ultimate_test" }],
            "CWE-476": [{ block: "entry", opcode: "store", score: 0.94, text: "store i32 %a, ptr %p", function: "the_ultimate_test" }]
        },
        healed_source: `#include <stdio.h>
#include <stdlib.h>
#include <limits.h>

/**
 * test_10_all_vulnerabilities.c (HEALED)
 */
void the_ultimate_test(int a, int b, int c) {
    int *p = (int *)malloc(sizeof(int));
    if (p == NULL) return;

    *p = a;

    if (*p > 0) { /* HEALED: CWE-190 guard */
        fprintf(stderr, "Overflow risk detected\\n");
    }
    int v1 = (*p) + INT_MAX;

    if (b < INT_MIN + 2000000) { /* HEALED: CWE-191 guard */
        v2 = INT_MIN;
    }
    int v2 = b - 2000000;

    if (c == 0) { /* HEALED: CWE-369 guard */
        printf("Safe exit\\n");
        free(p);
        return;
    }
    int v3 = v1 / c;

    printf("V1: %d, V2: %d, V3: %d\\n", v1, v2, v3);     
    free(p);
}

int main() {
    the_ultimate_test(100, INT_MIN + 100, 0);
    return 0;
}`
    }
};

window.getDemoResults = function(code) {
    for (const [filename, data] of Object.entries(DEMO_FILES)) {
        // Use a simple signature: check if the filename comment exists in the code
        if (code.includes(filename)) {
            // If the code is already healed, return a CLEAN result
            if (code.includes("(HEALED)")) {
                return {
                    analysis: {
                        success: true,
                        elapsed: 0.3,
                        predictions: [
                            { class: "non-vulnerable", probability: 0.99, detected: true, full_name: "Non-Vulnerable", description: "No known vulnerability pattern detected.", severity: "NONE" },
                            { class: "CWE-190", probability: 0.01, detected: false },
                            { class: "CWE-191", probability: 0.01, detected: false },
                            { class: "CWE-369", probability: 0.01, detected: false },
                            { class: "CWE-476", probability: 0.01, detected: false }
                        ]
                    }
                };
            }
            return data;
        }
    }
    return null;
};
