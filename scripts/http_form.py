"""Small standard-library helpers for the local demo server."""

from email.parser import BytesParser
from email.policy import default
from pathlib import Path


MAX_UPLOAD_BYTES = 64 * 1024 * 1024


def parse_multipart_form(
    content_type: str, body: bytes
) -> tuple[bytes | None, str | None, dict[str, str]]:
    """Return (audio bytes, original filename, text fields) from a browser form."""
    if not content_type.lower().startswith("multipart/form-data"):
        raise ValueError("expected a multipart/form-data request")
    if "\r" in content_type or "\n" in content_type:
        raise ValueError("invalid Content-Type header")

    message = BytesParser(policy=default).parsebytes(
        b"Content-Type: "
        + content_type.encode("latin-1")
        + b"\r\nMIME-Version: 1.0\r\n\r\n"
        + body
    )
    if not message.is_multipart():
        raise ValueError("multipart body has no parts")

    audio: bytes | None = None
    filename: str | None = None
    fields: dict[str, str] = {}
    for part in message.iter_parts():
        disposition = part.get("Content-Disposition", "")
        if "form-data" not in disposition.lower():
            continue
        name = part.get_param("name", header="content-disposition")
        if not name:
            continue
        payload = part.get_payload(decode=True) or b""
        if name == "audio":
            audio = payload
            filename = part.get_filename() or "recording.wav"
        else:
            charset = part.get_content_charset() or "utf-8"
            fields[str(name)] = payload.decode(charset, errors="replace")
    return audio, filename, fields


def safe_wav_name(name: str | None) -> str:
    """Keep uploaded recordings inside recordings/ and force a .wav suffix."""
    base = Path(str(name or "").replace("\\", "/")).name
    if not base.lower().endswith(".wav"):
        base += ".wav"
    keep = "".join(ch for ch in base if ch.isalnum() or ch in ("-", "_", "."))
    if not keep.endswith(".wav") or keep == ".wav":
        return "live.wav"
    return keep
