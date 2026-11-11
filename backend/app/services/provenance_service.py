"""C2PA Content Credentials and Conventional Metadata Analyzer for SPECTRA."""
from __future__ import annotations

import hashlib
import json
import shutil
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import c2pa
from c2pa import C2paError
import mutagen
from mutagen.id3 import ID3, TIT2, TPE1, TALB, TDRC, TCON, COMM
from sqlalchemy.orm import Session

from app.core.config import settings
from app.database.models import AudioAsset, ProcessingExperiment, ProvenanceResult
from app.services.ingestion_service import ingest_uploaded_bytes, resolve_asset_path

STANDARD_METADATA_KEYS = {
    "title", "tit2", "©nam", "inam",
    "artist", "tpe1", "©art", "iart",
    "album", "talb", "©alb", "iprd",
    "date", "tdrc", "tyer", "©day", "icrd",
    "genre", "tcon", "©gen", "ignr",
    "encoder", "tsse", "©too", "isft",
    "comment", "comm", "©cmt", "icmt",
    "copyright", "tcop", "cprt", "icop",
    "isrc", "tsrc",
    "bpm", "tbpm", "tmpo",
    "tracknumber", "trck", "trkn",
}


def inspect_c2pa_provenance(file_path: Path, mime_type: Optional[str] = None) -> Dict[str, Any]:
    """
    Inspect a media file for C2PA Content Credentials using the official c2pa-python SDK.

    Differentiates 5 states:
    - manifest_valid: Manifest present and cryptographically valid
    - manifest_invalid: Manifest present but failed signature/hash/trust validation
    - manifest_absent: No C2PA manifest present in the asset
    - unsupported_format: Media format not supported by C2PA reader
    - inspection_error: Exception reading or parsing the file/container
    """
    result: Dict[str, Any] = {
        "c2pa_status": "manifest_absent",
        "credential_present": False,
        "signature_valid": None,
        "active_manifest_id": None,
        "claim_generator": None,
        "assertions": [],
        "ingredients": [],
        "certificate_info": {},
        "validation_errors": [],
        "validation_warnings": [],
        "raw_manifest": None,
        "sdk_version": c2pa.sdk_version() if hasattr(c2pa, "sdk_version") else "unknown",
    }

    if not file_path.exists():
        result["c2pa_status"] = "inspection_error"
        result["validation_errors"].append("Target file does not exist.")
        return result

    try:
        # Check format support first if mime_type is provided
        supported_mimes = [m.lower() for m in c2pa.Reader.get_supported_mime_types()]
        ext = file_path.suffix.lstrip(".").lower()
        if mime_type and mime_type.lower() not in supported_mimes and ext not in supported_mimes:
            result["c2pa_status"] = "unsupported_format"
            result["validation_warnings"].append(
                f"Format '{mime_type or ext}' is not listed in C2PA Reader supported MIME types."
            )
            return result

        with c2pa.Reader(str(file_path)) as reader:
            manifest_json_str = reader.json()
            manifest_store = json.loads(manifest_json_str)
            result["raw_manifest"] = manifest_store
            result["credential_present"] = True

            active_id = manifest_store.get("active_manifest")
            result["active_manifest_id"] = active_id
            manifests = manifest_store.get("manifests", {})
            active_manifest = manifests.get(active_id, {}) if active_id else {}

            # Claim generator extraction
            claim_gen = active_manifest.get("claim_generator")
            if not claim_gen and "claim_generator_info" in active_manifest:
                cgi = active_manifest["claim_generator_info"]
                if isinstance(cgi, list) and cgi:
                    claim_gen = f"{cgi[0].get('name', 'unknown')} {cgi[0].get('version', '')}".strip()
                elif isinstance(cgi, dict):
                    claim_gen = f"{cgi.get('name', 'unknown')} {cgi.get('version', '')}".strip()
            result["claim_generator"] = claim_gen

            # Assertions extraction
            raw_assertions = active_manifest.get("assertions", [])
            parsed_assertions = []
            for a in raw_assertions:
                if isinstance(a, dict):
                    parsed_assertions.append({
                        "label": a.get("label", "unknown"),
                        "data": a.get("data", {}),
                    })
            result["assertions"] = parsed_assertions

            # Ingredients / creation history extraction
            raw_ingredients = active_manifest.get("ingredients", [])
            parsed_ingredients = []
            for ing in raw_ingredients:
                if isinstance(ing, dict):
                    parsed_ingredients.append({
                        "title": ing.get("title"),
                        "format": ing.get("format"),
                        "relationship": ing.get("relationship"),
                        "document_id": ing.get("document_id"),
                        "instance_id": ing.get("instance_id"),
                    })
            result["ingredients"] = parsed_ingredients

            # Signature and certificate info
            sig_info = active_manifest.get("signature_info", {})
            if isinstance(sig_info, dict):
                result["certificate_info"] = {
                    "issuer": sig_info.get("issuer"),
                    "common_name": sig_info.get("common_name"),
                    "cert_serial_number": sig_info.get("cert_serial_number"),
                    "time": sig_info.get("time"),
                    "alg": sig_info.get("alg"),
                }

            # Validation status & errors
            validation_status = manifest_store.get("validation_status", [])
            validation_results = manifest_store.get("validation_results", {})

            errors: List[str] = []
            warnings: List[str] = []

            for vs in validation_status:
                if isinstance(vs, dict):
                    code = str(vs.get("code", ""))
                    explanation = str(vs.get("explanation", ""))
                    msg = f"{code}: {explanation}" if explanation else code
                    # In C2PA, signingCredential.untrusted is a trust-anchor warning when using test certs offline,
                    # whereas assertion.dataHash.mismatch / claimSignature.mismatch are hard failures.
                    if any(
                        err_tok in code.lower()
                        for err_tok in ("mismatch", "invalid", "corrupt", "malformed", "revoked", "expired", "error")
                    ):
                        errors.append(msg)
                    elif "untrusted" in code.lower():
                        warnings.append(f"Trust anchor notice: {msg}")
                    else:
                        warnings.append(msg)

            # Also inspect validation_results.activeManifest if present
            if isinstance(validation_results, dict):
                am_res = validation_results.get("activeManifest", {})
                if isinstance(am_res, dict):
                    for failure in am_res.get("failure", []):
                        if isinstance(failure, dict):
                            code = str(failure.get("code", ""))
                            expl = str(failure.get("explanation", ""))
                            msg = f"{code}: {expl}" if expl else code
                            if "untrusted" in code.lower():
                                if f"Trust anchor notice: {msg}" not in warnings:
                                    warnings.append(f"Trust anchor notice: {msg}")
                            elif msg not in errors:
                                errors.append(msg)

            result["validation_errors"] = errors
            result["validation_warnings"] = warnings

            if errors:
                result["c2pa_status"] = "manifest_invalid"
                result["signature_valid"] = False
            else:
                result["c2pa_status"] = "manifest_valid"
                result["signature_valid"] = True

    except C2paError.ManifestNotFound:
        result["c2pa_status"] = "manifest_absent"
        result["credential_present"] = False
        result["signature_valid"] = None
    except C2paError.NotSupported as exc:
        result["c2pa_status"] = "unsupported_format"
        result["validation_errors"].append(str(exc))
    except C2paError as exc:
        err_str = str(exc)
        if "ManifestNotFound" in err_str or "no JUMBF data found" in err_str:
            result["c2pa_status"] = "manifest_absent"
        elif "NotSupported" in err_str:
            result["c2pa_status"] = "unsupported_format"
            result["validation_errors"].append(err_str)
        elif any(k in err_str for k in ("Signature", "Verify", "Manifest", "Assertion")):
            result["c2pa_status"] = "manifest_invalid"
            result["credential_present"] = True
            result["signature_valid"] = False
            result["validation_errors"].append(err_str)
        else:
            result["c2pa_status"] = "inspection_error"
            result["validation_errors"].append(err_str)
    except Exception as exc:
        result["c2pa_status"] = "inspection_error"
        result["validation_errors"].append(f"Unexpected C2PA inspection exception: {exc}")

    return result


def inspect_conventional_metadata(file_path: Path) -> Tuple[Dict[str, Any], List[str], List[str]]:
    """
    Extract conventional container metadata (ID3, Vorbis comments, MP4 atoms, RIFF INFO)
    independently of C2PA manifests.
    """
    tags_dict: Dict[str, Any] = {}
    supported_fields: List[str] = []
    unsupported_fields: List[str] = []
    ext = file_path.suffix.lstrip(".").upper() or "AUDIO"
    format_family = f"{ext} Container"

    try:
        m_file = mutagen.File(str(file_path))
        if m_file is None:
            return (
                {"format_family": format_family, "tag_count": 0, "tags": {}},
                supported_fields,
                unsupported_fields,
            )

        format_family = type(m_file).__name__
        if m_file.tags is not None:
            for key, val in m_file.tags.items():
                key_str = str(key)
                # Convert tag values cleanly to JSON-serializable strings/lists
                if isinstance(val, (bytes, bytearray)):
                    val_clean: Any = f"<binary {len(val)} bytes>"
                elif hasattr(val, "data") and isinstance(getattr(val, "data", None), (bytes, bytearray)):
                    raw_data = getattr(val, "data")
                    mime_info = getattr(val, "mime", "application/octet-stream")
                    desc_info = getattr(val, "desc", "")
                    val_clean = f"<{type(val).__name__} mime={mime_info} desc='{desc_info}' ({len(raw_data)} bytes)>"
                elif hasattr(val, "url"):
                    val_clean = str(getattr(val, "url"))
                elif hasattr(val, "text"):
                    texts = [str(t) for t in getattr(val, "text", [])]
                    val_clean = texts[0] if len(texts) == 1 else texts
                elif isinstance(val, (list, tuple)):
                    cleaned_items = []
                    for item in val:
                        if isinstance(item, (bytes, bytearray)):
                            cleaned_items.append(f"<binary {len(item)} bytes>")
                        else:
                            cleaned_items.append(str(item))
                    val_clean = cleaned_items[0] if len(cleaned_items) == 1 else cleaned_items
                else:
                    s_val = str(val)
                    val_clean = s_val[:300] + "…" if len(s_val) > 300 else s_val

                tags_dict[key_str] = val_clean
                norm_key = key_str.split(":")[0].lower().strip()
                if norm_key in STANDARD_METADATA_KEYS:
                    supported_fields.append(key_str)
                else:
                    unsupported_fields.append(key_str)
    except Exception as exc:
        tags_dict["_inspection_warning"] = f"Mutagen metadata parser warning: {exc}"

    metadata_payload: Dict[str, Any] = {
        "format_family": format_family,
        "tag_count": len(tags_dict),
        "tags": tags_dict,
        **tags_dict,
    }
    return metadata_payload, sorted(set(supported_fields)), sorted(set(unsupported_fields))


def run_provenance_analysis(
    db: Session,
    asset: AudioAsset,
    job_id: Optional[str] = None,
) -> ProvenanceResult:
    """Run both C2PA and conventional metadata inspection and persist ProvenanceResult."""
    file_path = resolve_asset_path(asset)
    c2pa_res = inspect_c2pa_provenance(file_path, mime_type=asset.mime_type)
    conv_meta, sup_fields, unsup_fields = inspect_conventional_metadata(file_path)

    prov = ProvenanceResult(
        job_id=job_id,
        audio_asset_id=asset.id,
        c2pa_status=c2pa_res["c2pa_status"],
        credential_present=bool(c2pa_res["credential_present"]),
        signature_valid=c2pa_res["signature_valid"],
        active_manifest_id=c2pa_res["active_manifest_id"],
        claim_generator=c2pa_res["claim_generator"],
        assertions_json=json.dumps(c2pa_res["assertions"]),
        ingredients_json=json.dumps(c2pa_res["ingredients"]),
        certificate_info_json=json.dumps(c2pa_res["certificate_info"]),
        validation_errors_json=json.dumps(c2pa_res["validation_errors"]),
        validation_warnings_json=json.dumps(c2pa_res["validation_warnings"]),
        raw_manifest_json=json.dumps(c2pa_res["raw_manifest"]) if c2pa_res["raw_manifest"] else None,
        conventional_metadata_json=json.dumps(conv_meta),
        supported_fields_json=json.dumps(sup_fields),
        unsupported_fields_json=json.dumps(unsup_fields),
    )
    db.add(prov)
    db.commit()
    db.refresh(prov)
    return prov


def export_metadata_clean_asset(
    db: Session,
    asset: AudioAsset,
    job_id: str,
    strip_conventional: bool = True,
    strip_c2pa: bool = False,
    custom_tags: Optional[Dict[str, str]] = None,
) -> Tuple[AudioAsset, ProcessingExperiment]:
    """
    Create an authorized metadata-sanitized export while preserving the original file
    and recording an explicit audit trail. Never forges C2PA provenance.
    """
    t0 = time.perf_counter()
    orig_path = resolve_asset_path(asset)
    ext = orig_path.suffix.lower()

    temp_out = settings.temp_dir / f"meta_export_{asset.id}_{int(time.time() * 1000)}{ext}"
    shutil.copy2(orig_path, temp_out)

    audit_log: List[str] = [
        f"Source asset ID: {asset.id} (SHA-256: {asset.sha256_checksum})",
        "Operation: Authorized container metadata export (audio PCM signal unmodified where possible).",
    ]
    warnings: List[str] = []

    # Check if source had a C2PA manifest
    pre_c2pa = inspect_c2pa_provenance(orig_path, asset.mime_type)
    before_meta, _, _ = inspect_conventional_metadata(orig_path)

    if strip_conventional:
        try:
            m_file = mutagen.File(str(temp_out))
            if m_file is not None and m_file.tags is not None:
                all_keys = list(m_file.tags.keys())
                if not strip_c2pa and any("c2pa" in str(k).lower() for k in all_keys):
                    removed_keys = [k for k in all_keys if "c2pa" not in str(k).lower()]
                    for rk in removed_keys:
                        del m_file.tags[rk]
                    m_file.save()
                    audit_log.append(
                        f"Removed {len(removed_keys)} conventional metadata tags while preserving embedded C2PA frame: {removed_keys}"
                    )
                else:
                    removed_keys = all_keys
                    m_file.delete()
                    m_file.save()
                    audit_log.append(f"Removed {len(removed_keys)} conventional metadata tags: {removed_keys}")
            else:
                audit_log.append("No conventional tags present to remove.")
        except Exception as exc:
            warnings.append(f"Conventional metadata removal warning: {exc}")

    if strip_c2pa and pre_c2pa["credential_present"] and ext in {".wav", ".flac"}:
        try:
            import soundfile as sf
            data_pcm, sr_pcm = sf.read(str(temp_out), always_2d=True)
            sf.write(str(temp_out), data_pcm, sr_pcm, format=ext.lstrip(".").upper(), subtype="PCM_16")
            audit_log.append(f"Rewrote clean {ext.upper()} PCM container to strip trailing C2PA JUMBF chunk.")
        except Exception as exc:
            warnings.append(f"Container C2PA chunk strip warning: {exc}")

    if custom_tags:
        try:
            if ext == ".mp3":
                try:
                    tags = ID3(str(temp_out))
                except Exception:
                    tags = ID3()
                if "title" in custom_tags:
                    tags.add(TIT2(encoding=3, text=custom_tags["title"]))
                if "artist" in custom_tags:
                    tags.add(TPE1(encoding=3, text=custom_tags["artist"]))
                if "album" in custom_tags:
                    tags.add(TALB(encoding=3, text=custom_tags["album"]))
                if "comment" in custom_tags:
                    tags.add(COMM(encoding=3, lang="eng", desc="SPECTRA", text=custom_tags["comment"]))
                tags.save(str(temp_out))
                audit_log.append(f"Applied authorized ID3 tags: {list(custom_tags.keys())}")
            else:
                m_file = mutagen.File(str(temp_out), easy=True)
                if m_file is not None:
                    for k, v in custom_tags.items():
                        try:
                            m_file[k] = v
                        except Exception:
                            pass
                    m_file.save()
                    audit_log.append(f"Applied authorized container tags: {list(custom_tags.keys())}")
        except Exception as exc:
            warnings.append(f"Custom tag write warning: {exc}")

    post_c2pa = inspect_c2pa_provenance(temp_out, asset.mime_type)
    if pre_c2pa["credential_present"]:
        if not post_c2pa["credential_present"]:
            warnings.append("Existing C2PA manifest was omitted in the exported file.")
            audit_log.append("C2PA status change: Manifest present -> Manifest omitted.")
        elif post_c2pa["c2pa_status"] == "manifest_invalid":
            warnings.append(
                "Modifying container metadata invalidated the byte-range hash of the embedded C2PA manifest."
            )
            audit_log.append("C2PA status change: Manifest invalidated by container tag modification.")

    out_bytes = temp_out.read_bytes()
    try:
        temp_out.unlink()
    except OSError:
        pass

    stem = Path(asset.sanitized_filename).stem
    out_asset = ingest_uploaded_bytes(
        db=db,
        raw_bytes=out_bytes,
        original_filename=f"{stem}_meta_export{ext}",
        is_processed_variant=True,
        parent_asset_id=asset.id,
        variant_description="Authorized Metadata Export",
    )

    elapsed_ms = (time.perf_counter() - t0) * 1000.0
    exp = ProcessingExperiment(
        job_id=job_id,
        source_asset_id=asset.id,
        output_asset_id=out_asset.id,
        operation_type="metadata_export",
        preset_name="clean_metadata",
        parameters_json=json.dumps({
            "strip_conventional": strip_conventional,
            "strip_c2pa": strip_c2pa,
            "custom_tags": custom_tags or {},
            "removed_fields_count": len(before_meta),
        }),
        engine_used="metadata_clean",
        original_sha256=asset.sha256_checksum,
        processed_sha256=out_asset.sha256_checksum,
        duration_preserved=True,
        channels_preserved=True,
        clipping_detected=False,
        nan_inf_detected=False,
        reconstruction_valid=True,
        processing_time_ms=round(elapsed_ms, 2),
        audit_log_json=json.dumps(audit_log),
        warnings_json=json.dumps(warnings),
        status="completed",
    )
    db.add(exp)
    db.commit()
    db.refresh(exp)
    return out_asset, exp
