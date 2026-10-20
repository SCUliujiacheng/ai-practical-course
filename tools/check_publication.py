"""Check Git publication candidates without third-party dependencies or file edits.

Tracked files and non-ignored untracked files are checked, so this also works
before the first commit. Findings never include matched credential values.
"""
from __future__ import annotations

import argparse
import ast
from collections import Counter
from dataclasses import dataclass
import html
import json
from pathlib import Path, PurePosixPath
import re
import subprocess
import sys
from urllib.parse import unquote


MAX_BYTES = 50_000_000  # Decimal MB; intentionally below hosting size limits.
TRACKS = {"B": "01_AI基础课", "A": "02_AI进阶课"}
NOTEBOOK_NAMES = {"教学讲解.ipynb", "动手练习.ipynb", "参考答案.ipynb"}
TEXT_EXTENSIONS = {
    ".py", ".ipynb", ".md", ".html", ".htm", ".json", ".jsonl", ".csv",
    ".tsv", ".txt", ".cmd", ".bat", ".ps1", ".sh", ".yaml", ".yml",
    ".toml", ".ini", ".cfg", ".xml", ".svg", ".log", ".rst", ".sql",
}
BINARY_EXTENSIONS = {
    ".png", ".jpg", ".jpeg", ".gif", ".webp", ".ico", ".pdf", ".zip",
    ".gz", ".tar", ".7z", ".whl", ".exe", ".dll", ".so", ".dylib",
    ".mp3", ".mp4", ".wav", ".npy", ".npz", ".pyc", ".woff", ".woff2",
    ".safetensors", ".pt", ".pth", ".ckpt", ".onnx", ".gguf", ".bin",
    ".h5", ".hdf5", ".tflite", ".pkl", ".pickle", ".joblib",
}
FORBIDDEN_PARTS = {
    "venv", "env", "runtime", "wheelhouse", "site-packages", "node_modules",
    "__pycache__", ".ipynb_checkpoints", "备份", "backups",
}
PLACEHOLDERS = {
    "[user_home]", "[课程目录]", "[用户名]", "[username]", "[user]",
    "your_username", "your-user", "your_user", "user_home",
}
PERSONAL_PATH = re.compile(
    r"(?i)(?:[a-z]:(?:\\+|/+)Users(?:\\+|/+)|/(?:Users|home)/)"
    r"(?P<username>[^\\/\r\n\t\"'<>|:]+)"
)
TOKEN_PATTERNS = {
    "api_token": re.compile(
        r"\b(?:sk-(?:proj-|svcacct-)?[A-Za-z0-9_-]{20,}|"
        r"hf_[A-Za-z0-9]{25,}|gh[pousr]_[A-Za-z0-9_]{30,}|"
        r"github_pat_[A-Za-z0-9_]{30,}|AKIA[A-Z0-9]{16}|"
        r"AIza[A-Za-z0-9_-]{35}|xox[baprs]-[A-Za-z0-9-]{20,})\b"
    ),
    "jwt": re.compile(r"\beyJ[A-Za-z0-9_-]{15,}\.[A-Za-z0-9_-]{15,}\.[A-Za-z0-9_-]{15,}\b"),
    "bearer_token": re.compile(r"(?i)\bBearer\s+([A-Za-z0-9_.-]{20,})"),
    "url_credential": re.compile(
        r"(?i)https?://[^/\s:@]+:([^/@\s]+)@|"
        r"[?&](?:token|access_token|api_key|secret|password)=([A-Za-z0-9_.-]{16,})"
    ),
}
AUTH_LITERAL = re.compile(
    r"(?i)\b(?:[A-Za-z][A-Za-z0-9_]*_)?"
    r"(?:api[_-]?key|access[_-]?token|auth[_-]?token|secret[_-]?access[_-]?key|secret(?:[_-]?key)?|"
    r"password|passwd|token|credential)\b[\"']?\s*[:=]\s*"
    r"([\"'])([^\"'\r\n]{12,})\1"
)
PRIVATE_KEY = re.compile(r"-----BEGIN (?:RSA |EC |OPENSSH |ENCRYPTED |DSA )?PRIVATE KEY-----")


@dataclass(frozen=True)
class Finding:
    file: str
    category: str
    line: int | None = None
    detail: str = ""


def is_placeholder(value: str) -> bool:
    """Accept clearly marked placeholders, not arbitrary bracketed usernames."""
    return value.strip().casefold() in PLACEHOLDERS


def is_fake_credential(value: str) -> bool:
    """Recognize explicit examples without exempting entire authoring scripts."""
    lowered = value.casefold()
    markers = r"(?:^|[-_])(?:your_api_key|your_token|your_password|placeholder|dummy|example|fake_key|test_key)(?:$|[-_])"
    if re.search(markers, lowered):
        return True
    if lowered.startswith(("sk-test-", "sk-fake-", "hf_test_", "ghp_test_", "github_pat_test_")):
        return True
    if value.startswith(("${", "<", "[")) or "..." in value:
        return True
    payload = re.sub(r"^(?:sk-(?:proj-|svcacct-)?|hf_|gh[pousr]_|github_pat_)", "", value)
    # Repeated X/0 strings and common example alphabets are not real secrets.
    if len(set(payload)) <= 3:
        return True
    if payload in ("abcdefghijklmnopqrstuvwxyz", "ABCDEFGHIJKLMNOPQRSTUVWXYZ", "0123456789" * 4):
        return True
    return False


def sensitive_findings(relative: str, text: str) -> list[Finding]:
    findings = []
    author = relative.startswith("tools/build_") and relative.endswith(".py")
    for number, raw in enumerate(text.splitlines(), 1):
        line = unquote(html.unescape(raw))
        for match in PERSONAL_PATH.finditer(line):
            if not is_placeholder(match.group("username")):
                findings.append(Finding(relative, "personal_absolute_path", number))
                break
        if PRIVATE_KEY.search(line):
            # An author showing a regex literal is not a stored private key.
            regex_example = author and (
                "re.compile(" in line or re.search(r"(?i)\b\w*(?:pattern|regex)\s*=", line)
            )
            if not regex_example:
                findings.append(Finding(relative, "private_key", number))
        for category, pattern in TOKEN_PATTERNS.items():
            for match in pattern.finditer(line):
                groups = [part for part in match.groups() if part is not None]
                value = groups[-1] if groups else match.group()
                if not is_fake_credential(value):
                    findings.append(Finding(relative, category, number))
                    break
        for match in AUTH_LITERAL.finditer(line):
            value = match.group(2)
            # Code expressions and placeholders are not literal credentials.
            if not is_fake_credential(value) and re.fullmatch(r"[A-Za-z0-9_./+=:@!#$%&*-]+", value):
                findings.append(Finding(relative, "hardcoded_authentication", number))
                break
    return findings


def forbidden_file(relative: str) -> str | None:
    path = PurePosixPath(relative)
    parts = [part.casefold() for part in path.parts]
    if any(part in FORBIDDEN_PARTS or part.startswith(".venv") for part in parts):
        return "environment_or_generated_cache"
    if path.suffix.casefold() in {".pyc", ".whl", ".exe", ".dll"}:
        return "environment_binary"
    if path.name.startswith(".env") and path.name not in {".env.example", ".env.template"}:
        return "private_environment_file"
    if path.name in {"id_rsa", "id_ed25519", ".netrc"} or path.suffix.casefold() in {".pem", ".pfx", ".p12", ".key"}:
        return "private_key_or_credentials_file"
    # Course-generated, small outputs (.pt/.joblib) are intentionally published.
    # Downloaded pretrained artifacts have a separate metadata-only policy.
    if parts[:2] == ["assets", "models"]:
        license_name = re.fullmatch(r"(?i)(?:licen[cs]e|copying|notice)(?:[._-].*)?", path.name)
        metadata = path.name == "SOURCE.json" or bool(license_name)
        top_readme = relative in {"assets/models/README.md", "assets/models/README.html"}
        if not metadata and not top_readme:
            return "pretrained_model_artifact"
    return None


def git_candidates(root: Path) -> list[str]:
    result = subprocess.run(
        ["git", "ls-files", "--cached", "--others", "--exclude-standard", "-z"],
        cwd=root, capture_output=True, check=False,
    )
    if result.returncode:
        raise RuntimeError("git_inventory_failed")
    return sorted({part.decode("utf-8", "surrogateescape").replace("\\", "/")
                   for part in result.stdout.split(b"\0") if part})


def read_text_if_applicable(path: Path) -> str | None:
    """Probe before reading; do not decode model/image/archive binaries."""
    if path.suffix.casefold() in BINARY_EXTENSIONS:
        return None
    known_text = path.suffix.casefold() in TEXT_EXTENSIONS or path.name in {"LICENSE", ".gitignore", ".gitattributes"}
    with path.open("rb") as stream:
        sample = stream.read(8192)
        utf16 = sample.startswith((b"\xff\xfe", b"\xfe\xff"))
        if b"\0" in sample and not utf16:
            if known_text:
                raise ValueError("invalid_text_encoding")
            return None
        if not known_text and not utf16:
            controls = sum(byte < 32 and byte not in (9, 10, 13) for byte in sample)
            if controls > max(1, len(sample) // 100):
                return None
        data = sample + stream.read()
    try:
        return data.decode("utf-16" if utf16 else "utf-8-sig")
    except UnicodeDecodeError:
        # Some original dataset license/readme files use Windows-1252. Keep
        # their original bytes intact and scan the decoded text read-only.
        if path.suffix.casefold() in {".txt", ".csv", ".tsv", ".md", ".rst"}:
            try:
                return data.decode("cp1252")
            except UnicodeDecodeError:
                pass
        if known_text:
            raise ValueError("invalid_text_encoding") from None
        return None


def string_or_lines(value: object) -> bool:
    return isinstance(value, str) or (isinstance(value, list) and all(isinstance(item, str) for item in value))


def notebook_findings(relative: str, text: str) -> list[Finding]:
    """Validate the v4 structure used by this course, including code ASTs."""
    try:
        notebook = json.loads(text)
    except json.JSONDecodeError as error:
        return [Finding(relative, "notebook_invalid_json", error.lineno)]
    if not isinstance(notebook, dict) or notebook.get("nbformat") != 4:
        return [Finding(relative, "notebook_not_v4")]
    if not isinstance(notebook.get("metadata"), dict) or not isinstance(notebook.get("cells"), list):
        return [Finding(relative, "notebook_invalid_structure")]
    minor = notebook.get("nbformat_minor")
    if type(minor) is not int or minor < 0:
        return [Finding(relative, "notebook_invalid_minor_version")]
    findings, identifiers = [], set()
    for index, cell in enumerate(notebook["cells"], 1):
        detail = f"cell={index}"
        if not isinstance(cell, dict) or cell.get("cell_type") not in {"code", "markdown", "raw"}:
            findings.append(Finding(relative, "notebook_invalid_cell", detail=detail))
            continue
        if not isinstance(cell.get("metadata"), dict) or not string_or_lines(cell.get("source")):
            findings.append(Finding(relative, "notebook_invalid_cell_fields", detail=detail))
            continue
        if minor >= 5:
            identifier = cell.get("id")
            if not isinstance(identifier, str) or not re.fullmatch(r"[A-Za-z0-9_-]{1,64}", identifier) or identifier in identifiers:
                findings.append(Finding(relative, "notebook_invalid_cell_id", detail=detail))
            else:
                identifiers.add(identifier)
        if cell["cell_type"] != "code":
            continue
        source = cell["source"] if isinstance(cell["source"], str) else "".join(cell["source"])
        try:
            ast.parse(source, filename=relative)
        except (SyntaxError, ValueError) as error:
            line = getattr(error, "lineno", None)
            findings.append(Finding(relative, "notebook_code_syntax", detail=f"cell={index}, cell_line={line}"))
        execution = cell.get("execution_count")
        if "execution_count" not in cell or (execution is not None and type(execution) is not int):
            findings.append(Finding(relative, "notebook_invalid_execution_count", detail=detail))
        outputs = cell.get("outputs")
        if not isinstance(outputs, list):
            findings.append(Finding(relative, "notebook_invalid_outputs", detail=detail))
            continue
        for output in outputs:
            valid = isinstance(output, dict)
            kind = output.get("output_type") if valid else None
            if kind == "stream":
                valid = output.get("name") in {"stdout", "stderr"} and string_or_lines(output.get("text"))
            elif kind in {"display_data", "execute_result"}:
                valid = isinstance(output.get("data"), dict) and isinstance(output.get("metadata"), dict)
                if kind == "execute_result":
                    valid = valid and type(output.get("execution_count")) is int
            elif kind == "error":
                valid = (isinstance(output.get("ename"), str) and isinstance(output.get("evalue"), str)
                         and isinstance(output.get("traceback"), list) and all(isinstance(x, str) for x in output["traceback"]))
            else:
                valid = False
            if not valid:
                findings.append(Finding(relative, "notebook_invalid_output", detail=detail))
    return findings


def course_findings(texts: dict[str, str], candidates: set[str]) -> tuple[list[Finding], dict[str, int]]:
    findings = []
    inside_course = lambda name: any(name.startswith(folder + "/") for folder in TRACKS.values())
    manifests = sorted(name for name in candidates if inside_course(name) and PurePosixPath(name).name == "case.json")
    notebooks = sorted(name for name in candidates if inside_course(name) and name.endswith(".ipynb"))
    solutions = sorted(name for name in candidates if inside_course(name) and PurePosixPath(name).name == "solution.py")
    stats = {"cases": len(manifests), "notebooks": len(notebooks), "solutions": len(solutions)}
    for key, expected in {"cases": 60, "notebooks": 180, "solutions": 60}.items():
        if stats[key] != expected:
            findings.append(Finding("<course>", "wrong_file_count", detail=f"{key}={stats[key]}, expected={expected}"))
    expected_ids = {f"{track}{number:02d}" for track in TRACKS for number in range(1, 31)}
    seen_ids = set()
    for name in manifests:
        try:
            data = json.loads(texts[name])
        except (KeyError, json.JSONDecodeError):
            findings.append(Finding(name, "case_manifest_invalid_json"))
            continue
        if not isinstance(data, dict):
            findings.append(Finding(name, "case_manifest_invalid_structure"))
            continue
        identifier = data.get("id")
        if not isinstance(identifier, str) or identifier not in expected_ids or identifier in seen_ids:
            findings.append(Finding(name, "case_id_invalid_or_duplicate"))
        else:
            seen_ids.add(identifier)
            track, number = identifier[0], int(identifier[1:])
            if data.get("track") != track or data.get("number") != number or not name.startswith(TRACKS[track] + "/"):
                findings.append(Finding(name, "case_id_metadata_mismatch"))
        if not isinstance(data.get("title"), str) or not data["title"].strip():
            findings.append(Finding(name, "case_missing_title"))
        parent = PurePosixPath(name).parent
        for required in NOTEBOOK_NAMES | {"solution.py"}:
            expected_path = (parent / required).as_posix()
            if expected_path not in candidates:
                findings.append(Finding(expected_path, "case_required_file_missing"))
    if seen_ids != expected_ids:
        findings.append(Finding("<course>", "case_id_coverage_incomplete", detail=f"valid_unique_ids={len(seen_ids)}"))
    for name in notebooks:
        if name in texts:
            findings.extend(notebook_findings(name, texts[name]))
    for name in solutions:
        if name in texts:
            try:
                ast.parse(texts[name], filename=name)
            except (SyntaxError, ValueError) as error:
                findings.append(Finding(name, "solution_syntax", getattr(error, "lineno", None)))
    return findings, stats


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--max-findings", type=int, default=40)
    args = parser.parse_args()
    if args.max_findings < 1:
        parser.error("--max-findings must be positive")
    root = args.root.resolve()
    try:
        names = git_candidates(root)
    except (OSError, RuntimeError):
        print("FAIL git_inventory_failed: run inside a Git checkout with Git installed.")
        return 2
    findings, texts, binary_count = [], {}, 0
    for relative in names:
        path = root / relative
        if path.is_symlink():
            findings.append(Finding(relative, "symlink_requires_review"))
            continue
        try:
            path.resolve().relative_to(root)
            if not path.is_file():
                findings.append(Finding(relative, "missing_or_nonregular_git_file"))
                continue
            size = path.stat().st_size
            forbidden = forbidden_file(relative)
            if forbidden:
                findings.append(Finding(relative, forbidden))
            if size > MAX_BYTES:
                findings.append(Finding(relative, "file_over_50_MB"))
                continue
            if forbidden:
                continue
            text = read_text_if_applicable(path)
            if text is None:
                binary_count += 1
                continue
            texts[relative] = text
            findings.extend(sensitive_findings(relative, text))
        except (OSError, ValueError):
            findings.append(Finding(relative, "unreadable_or_invalid_text"))
    course_issues, stats = course_findings(texts, set(names))
    findings.extend(course_issues)
    findings = list(dict.fromkeys(findings))
    print(f"Git candidates: {len(names)} | text: {len(texts)} | skipped binary: {binary_count}")
    print(f"Cases: {stats['cases']}/60 | notebooks: {stats['notebooks']}/180 | solutions: {stats['solutions']}/60")
    for finding in findings[:args.max_findings]:
        location = finding.file + (f":{finding.line}" if finding.line else "")
        suffix = f" ({finding.detail})" if finding.detail else ""
        print(f"FAIL {location} [{finding.category}]{suffix}")
    if findings:
        counts = Counter(item.category for item in findings)
        print("Summary: " + ", ".join(f"{key}={value}" for key, value in sorted(counts.items())))
        if len(findings) > args.max_findings:
            print(f"{len(findings) - args.max_findings} additional findings omitted; increase --max-findings for locations.")
        print(f"FAIL: {len(findings)} findings. No matched secret values were printed.")
        return 1
    print("PASS: publication candidates passed privacy, size, structure and Python syntax checks.")
    print("Static checks only: this does not execute lessons or replace runtime/license review.")
    return 0


if __name__ == "__main__":
    if ha