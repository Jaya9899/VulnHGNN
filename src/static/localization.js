// localization.js

/**
 * Parses raw C/C++ source code, escapes it, and wraps the identified vulnerable 
 * function inside a stylized highlighting span to act as a visual locator.
 */
window.highlightVulnerability = function(codeStr, funcName) {
    if (!codeStr) return "";
    
    // Prevent XSS and format for code block
    const escapeHtml = (unsafe) => {
        return unsafe.replace(/&/g, "&amp;")
                     .replace(/</g, "&lt;")
                     .replace(/>/g, "&gt;")
                     .replace(/"/g, "&quot;")
                     .replace(/'/g, "&#039;");
    };

    let escapedCode = escapeHtml(codeStr);

    if (!funcName || funcName === 'unknown') {
        return escapedCode;
    }
    
    const escapedFuncName = escapeHtml(funcName);
    const lines = escapedCode.split('\n');
    let found = false;
    let targetIndex = -1;
    
    // First Pass: Try to find the function definition line.
    // In C, typically something like: "int funcName(" or "void funcName ()"
    const funcDefRegex = new RegExp(`\\b${escapedFuncName}\\b\\s*\\(`);
    
    for (let i = 0; i < lines.length; i++) {
        if (funcDefRegex.test(lines[i])) {
            targetIndex = i;
            break;
        }
    }
    
    // Second Pass: If definition not found clearly, just find the first occurrence
    if (targetIndex === -1) {
        for (let i = 0; i < lines.length; i++) {
            if (lines[i].includes(escapedFuncName)) {
                targetIndex = i;
                break;
            }
        }
    }
    
    // If we found the target line, highlight it completely
    if (targetIndex !== -1) {
        lines[targetIndex] = `<span class="vuln-highlight">${lines[targetIndex]}</span>`;
        found = true;
    }
    
    if (found) {
        return lines.join('\n');
    }
    
    // Fallback: Use string replacement if line parsing failed
    const fallbackRegex = new RegExp(`(\\b${escapedFuncName}\\b)`, 'g');
    return escapedCode.replace(fallbackRegex, `<span class="vuln-highlight">$1</span>`);
};
