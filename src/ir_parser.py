import re

def parse_ll_file(filepath):
    with open(filepath, "r", encoding="utf-8", errors="replace") as f:
        lines = f.readlines()

    source_file = _extract_source_file(lines)
    functions   = _extract_functions(lines)

    return {
        "source_file": source_file,
        "functions":   functions,
    }

def _extract_source_file(lines):
    for line in lines:
        if line.startswith("source_filename"):
            match = re.search(r'"(.+)"', line)
            if match:
                return match.group(1)
    return ""



_DEFINE_RE = re.compile(
    r'^define\s+'            # keyword
    r'(?:[\w]+\s+)*'        # optional qualifiers 
    r'([\w\*]+)\s+'         #return type
    r'@([\w\.\$]+)\s*'      #function name
    r'\((.*?)\)'             #parameter list
, re.IGNORECASE)

_DECLARE_RE = re.compile(
    r'^declare\s+'
    r'(?:[\w]+\s+)*'        # optional qualifiers
    r'([\w\*]+)\s+'         # return type
    r'@([\w\.\$]+)\s*'
    r'\((.*?)\)'
, re.IGNORECASE)

_BLOCK_LABEL_RE = re.compile(r'^(\d+|[\w\.]+):\s*(?:;.*)?$')

def _extract_functions(lines):
    functions = []
    i = 0

    while i < len(lines):
        line = lines[i].strip()
        m = _DECLARE_RE.match(line)
        if m:
            ret_type   = m.group(1)
            func_name  = m.group(2)
            params_raw = m.group(3)
            functions.append({
                "name":           func_name,
                "return_type":    ret_type,
                "params":         _parse_params(params_raw),
                "is_declaration": True,
                "blocks":         [],   # declarations have no body
            })
            i += 1
            continue

        m = _DEFINE_RE.match(line)
        if m:
            ret_type   = m.group(1)
            func_name  = m.group(2)
            params_raw = m.group(3)

            #collect the body lines until the closing '}'
            body_lines = []
            i += 1
            while i < len(lines):
                body_line = lines[i].rstrip('\n').rstrip('\r')
                if body_line.strip() == '}':
                    i += 1   # consume the closing brace
                    break
                body_lines.append(body_line)
                i += 1

            blocks = _extract_blocks(body_lines)
            functions.append({
                "name":           func_name,
                "return_type":    ret_type,
                "params":         _parse_params(params_raw),
                "is_declaration": False,
                "blocks":         blocks,
            })
            continue

        i += 1

    return functions


def _parse_params(raw):
    raw = raw.strip()
    if not raw or raw == "...":
        return []

    params = []
    for part in raw.split(","):
        part = part.strip()
        if not part:
            continue
        tokens = part.split()
        clean_tokens = [t for t in tokens if t not in ("noundef", "signext", "zeroext")]
        params.append(" ".join(clean_tokens))

    return params

def _extract_blocks(body_lines):
    blocks = []
    current_label = "entry"
    current_instructions = []

    for raw_line in body_lines:
        line = raw_line.strip()

        if not line or line.startswith(";"):
            continue

        m = _BLOCK_LABEL_RE.match(line)
        if m:
            if current_instructions:
                blocks.append({
                    "label":        current_label,
                    "instructions": current_instructions,
                })
            current_label = m.group(1)
            current_instructions = []
        else:
            instruction = _parse_instruction(line)
            if instruction:
                current_instructions.append(instruction)

    if current_instructions:
        blocks.append({
            "label":        current_label,
            "instructions": current_instructions,
        })

    return blocks

_KNOWN_OPCODES = {
    "alloca", "store", "load", "br", "ret", "call", "icmp", "fcmp",
    "add", "fadd", "sub", "fsub", "mul", "fmul", "udiv", "sdiv", "fdiv",
    "urem", "srem", "frem", "shl", "lshr", "ashr", "and", "or", "xor",
    "getelementptr", "gep",
    "trunc", "zext", "sext", "fptrunc", "fpext", "fptoui", "fptosi",
    "uitofp", "sitofp", "ptrtoint", "inttoptr", "bitcast",
    "phi", "select", "switch", "unreachable", "resume",
    "extractvalue", "insertvalue", "extractelement", "insertelement",
    "shufflevector", "landingpad", "invoke",
}


def _parse_instruction(raw_line):
    text = re.sub(r',\s*!(?:dbg|llvm\.\w+)\s*![0-9]+', '', raw_line).strip()
    text = re.sub(r',\s*metadata\s+![0-9]+', '', text).strip()

    if not text:
        return None
    opcode = _identify_opcode(text)

    return {
        "text":   text,
        "opcode": opcode,
    }


def _identify_opcode(text):
    if "=" in text:
        rhs = text.split("=", 1)[1].strip()
        first_word = rhs.split()[0] if rhs else ""
    else:
        first_word = text.split()[0] if text else ""

    first_word = first_word.split("(")[0]   # handle cases like 'call('

    if first_word in _KNOWN_OPCODES:
        return first_word

    return first_word if first_word else "unknown"
