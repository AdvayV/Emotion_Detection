"""Human review, provenance-preserving correction export, and frozen-split handoff."""
from __future__ import annotations

import argparse, csv, hashlib, io, json, zipfile
from pathlib import Path

import pandas as pd

from hinglish_emotion.evaluation_guard import phrase_hash
from hinglish_emotion.tasks import EMOTION_LABELS, LANGUAGE_PAIRS, canonical_emotion
from hinglish_emotion.text_preparation import duplicate_key

META = ("row_id", "group_id", "text", "language_pair", "source", "provenance", "dataset_role", "original_source", "original_annotation_status", "synthetic_origin")


def _read(source):
    if isinstance(source, pd.DataFrame): return source.fillna("").astype(str).to_dict("records")
    if isinstance(source, list): return [{str(k): "" if v is None else str(v) for k,v in r.items()} for r in source]
    if hasattr(source, "read"):
        raw=source.read(); source.seek(0); return list(csv.DictReader(io.StringIO(raw.decode("utf-8-sig"))))
    with Path(source).open(encoding="utf-8-sig", newline="") as f: return list(csv.DictReader(f))


def _write(path, rows, fields=None):
    fields=fields or list(rows[0])
    out=io.StringIO(newline=""); w=csv.DictWriter(out, fieldnames=fields, extrasaction="ignore", lineterminator="\n"); w.writeheader(); w.writerows(rows)
    return out.getvalue().encode("utf-8")


def _input_sha(source):
    if isinstance(source,(str,Path)) and Path(source).is_file(): return hashlib.sha256(Path(source).read_bytes()).hexdigest()
    return hashlib.sha256(_write(None,_read(source))).hexdigest()


def _id(row, i): return str(row.get("row_id") or hashlib.sha256((str(i)+"\0"+str(row.get("text",""))).encode()).hexdigest()[:20])


def normalize_queue(source):
    rows=_read(source)
    if not rows: raise ValueError("Review queue is empty")
    eval_hashes=set(); eval_dir=Path(__file__).resolve().parents[1]/"evaluation_set"
    if eval_dir.is_dir():
        for file in eval_dir.glob("*.csv"):
            for erow in _read(file):
                et=str(erow.get("text",erow.get("phrase",""))).strip()
                if et: eval_hashes.add(phrase_hash(et))
    result=[]; seen=set()
    for i,r in enumerate(rows):
        text=str(r.get("text",r.get("phrase", ""))).strip(); pair=str(r.get("language_pair", "")).strip(); rid=_id(r,i)
        if not text or not pair: raise ValueError("Every row needs text and language_pair")
        if pair not in LANGUAGE_PAIRS: raise ValueError(f"Unsupported language_pair: {pair}")
        if rid in seen: raise ValueError("Duplicate row_id in source queue")
        seen.add(rid)
        origin_hint=(str(r.get("source", ""))+" "+str(r.get("annotation_status", ""))).lower()
        eval_origin=("assistant_draft" in origin_hint or "illustrative" in origin_hint or str(r.get("source_id", "")).startswith("eval_") or str(r.get("row_id", "")).startswith(("eval_","draft_")) or phrase_hash(text) in eval_hashes)
        role=(r.get("dataset_role") or r.get("role") or ("test" if str(r.get("split","")).strip().lower()=="test" or eval_origin else "train")).strip().lower()
        if role not in {"train","test"}: raise ValueError("dataset_role must be train or test")
        original_source=str(r.get("source") or "model_disagreement_review")
        source_for_export=original_source
        original_status=str(r.get("annotation_status") or "")
        provenance=str(r.get("provenance") or "")
        synthetic_flag=str(r.get("synthetic", "")).strip().lower() in {"true","1","yes"} or any(tag in origin_hint for tag in ("synthetic","assistant_draft","illustrative"))
        if synthetic_flag and "synthetic" not in source_for_export.lower(): source_for_export += "; synthetic-origin"
        result.append({"row_id":rid,"group_id":str(r.get("group_id") or rid),"text":text,"language_pair":pair,"source":source_for_export,"provenance":provenance,"dataset_role":role,"original_source":original_source,"original_annotation_status":original_status,"synthetic_origin":str(synthetic_flag).lower()})
    return result


def create_review_sheets(queue, output_dir):
    rows=normalize_queue(queue); out=Path(output_dir); out.mkdir(parents=True,exist_ok=True)
    fields=[*META,"reviewer_id","emotion","needs_context","notes"]
    for name in ("reviewer_a.csv","reviewer_b.csv"):
        path=out/name
        if path.exists(): raise ValueError(f"Refusing to overwrite {path}")
        path.write_bytes(_write(path,[{**r,"reviewer_id":"","emotion":"","needs_context":"","notes":""} for r in rows],fields))
    # Proposed/model labels remain separate and never enter reviewer sheets.
    proposed=[{**r,"proposed_emotion":str(src.get("emotion",src.get("proposed_emotion",""))),"source_label":str(src.get("emotion",""))} for r,src in zip(rows,_read(queue))]
    (out/"proposed_labels.csv").write_bytes(_write(out,proposed,[*META,"proposed_emotion","source_label"]))
    return out


def _check_sheet(path, base):
    rows=_read(path); expected={r["row_id"]:r for r in base}; seen={}
    for r in rows:
        key=str(r.get("row_id", ""))
        if key in seen or key not in expected: raise ValueError("Reviewer sheet has duplicate or unknown row IDs")
        if any(str(r.get(k,"")) != str(expected[key].get(k,"")) for k in META): raise ValueError("Source row changed since review sheets were created")
        seen[key]=r
    if set(seen)!=set(expected): raise ValueError("Each reviewer sheet must contain every source row exactly once")
    return seen


def reconcile(queue, reviewer_a, reviewer_b, adjudications=None):
    base=normalize_queue(queue); aa=_check_sheet(reviewer_a,base); bb=_check_sheet(reviewer_b,base)
    adj={}
    if adjudications:
        for r in _read(adjudications):
            k=str(r.get("row_id",""))
            if not k or k in adj or k not in aa: raise ValueError("Adjudications contain duplicate or unknown row IDs")
            adj[k]=r
    ready=[]; pending=[]; canonical=lambda x:canonical_emotion(str(x or "").strip())
    for orig in base:
        k=orig["row_id"]; a,b=aa[k],bb[k]
        ra,rb=str(a.get("reviewer_id","")).strip(),str(b.get("reviewer_id","")).strip()
        if not ra or not rb or ra==rb: pending.append({"row_id":k,"reason":"Two distinct named reviewers required"}); continue
        la,lb=canonical(a.get("emotion")),canonical(b.get("emotion"))
        if la not in EMOTION_LABELS or lb not in EMOTION_LABELS: pending.append({"row_id":k,"reason":"Both reviewers must provide a canonical seven-class label"}); continue
        flags=any(str(r.get("needs_context","")).strip().lower() in {"yes","true","1"} for r in (a,b))
        if la!=lb or flags:
            d=adj.get(k,{}) ; judge=str(d.get("adjudicator_id","")).strip(); label=canonical(d.get("emotion"))
            if not judge or judge in {ra,rb} or label not in EMOTION_LABELS or not str(d.get("notes","")).strip(): pending.append({"row_id":k,"reason":"Disagreement/context flag requires distinct adjudicator, label, and notes"}); continue
            notes="; ".join(x for x in [str(a.get("notes","")).strip(),str(b.get("notes","")).strip(),"adjudication: "+str(d.get("notes","")).strip()] if x); status="human_adjudicated"
        else: label=la; notes="; ".join(x for x in [str(a.get("notes","")).strip(),str(b.get("notes","")).strip()] if x); status="human_reviewed"
        synthetic=str(orig.get("synthetic_origin","false")).lower()=="true"
        # Move the active source out of the exact unreviewed-source denylist while retaining the origin verbatim.
        reviewed_source="synthetic_human_reviewed" if synthetic else "human_reviewed_correction"
        ready.append({**orig,"source":reviewed_source,"provenance":"; ".join(x for x in [orig.get("provenance",""),"original_source="+orig.get("original_source","")] if x),"emotion":label,"annotation_status":status,"human_reviewed":"true","reviewer_a":ra,"reviewer_b":rb,"review_notes":notes,"split":"test" if orig["dataset_role"]=="test" else "train","experimental_synthetic":synthetic})
    report={"ready":not pending and len(ready)==len(base),"pending":pending,"source_rows":len(base),"reviewed_rows":len(ready),"training_rows":sum(x["dataset_role"]=="train" for x in ready),"evaluation_rows":sum(x["dataset_role"]=="test" for x in ready),"synthetic_origin_rows":sum(x["experimental_synthetic"] for x in ready),"dataset_kind":"human_reviewed_corrections","label_source":"two independent human reviewers; adjudication where required","source_sha256":_input_sha(queue),"reviewer_a_sha256":_input_sha(reviewer_a),"reviewer_b_sha256":_input_sha(reviewer_b),"adjudications_sha256":_input_sha(adjudications) if adjudications else None}
    return ready,report


def build_training_dataset(base_csv, corrections, output_dir, audit=None):
    """Merge training corrections only, retaining every base split and rejecting leakage."""
    if not isinstance(corrections,(str,Path)) or not audit: raise ValueError("A correction CSV and its completed review audit are required")
    base=_read(base_csv); corr=_read(corrections)
    receipt= audit if isinstance(audit,dict) else json.loads(Path(audit).read_text(encoding="utf-8"))
    if not receipt.get("ready") or not receipt.get("reviewer_a_sha256") or not receipt.get("reviewer_b_sha256"): raise ValueError("Correction review audit is incomplete or lacks reviewer evidence")
    correction_hash=hashlib.sha256(Path(corrections).read_bytes()).hexdigest()
    if receipt.get("prepared_csv_sha256")!=correction_hash: raise ValueError("Review audit does not match corrections file")
    if any(str(r.get("annotation_status")) not in {"human_reviewed","human_adjudicated"} or str(r.get("human_reviewed")).lower()!="true" for r in corr): raise ValueError("Only completed human-reviewed rows can be merged")
    frozen=[r for r in base if str(r.get("split","")).lower() in {"validation","test"}]; blocked={phrase_hash(r["text"]) for r in base if r.get("text")}
    # Independent evaluation data is protected even when it was never part of the base CSV.
    evaluation_dir=Path(__file__).resolve().parents[1]/"evaluation_set"
    if evaluation_dir.is_dir():
        for eval_csv in evaluation_dir.glob("*.csv"):
            for row in _read(eval_csv):
                text=str(row.get("text",row.get("phrase",""))).strip()
                if text: blocked.add(phrase_hash(text))
    if any(r.get("dataset_role","train")=="test" for r in corr): raise ValueError("Independent evaluation rows are never training data")
    if any(phrase_hash(r["text"]) in blocked for r in corr): raise ValueError("Correction overlaps frozen validation/test or guarded evaluation text")
    ids={r.get("row_id") for r in base}; groups={r.get("group_id") for r in base if r.get("group_id")}
    frozen_groups={r.get("group_id") for r in frozen if r.get("group_id")}
    if any(r.get("row_id") in ids or r.get("group_id") in frozen_groups for r in corr): raise ValueError("Correction row/group overlaps an existing or frozen partition")
    new_ids=[r.get("row_id") for r in corr]
    if len(new_ids)!=len(set(new_ids)): raise ValueError("Corrections contain duplicate row IDs")
    correction_hashes=[phrase_hash(str(r.get("text",""))) for r in corr]
    if len(correction_hashes)!=len(set(correction_hashes)) or any(not str(r.get("text","")).strip() for r in corr): raise ValueError("Corrections contain duplicate normalized text or empty text")
    additions=[]
    for r in corr:
        label=canonical_emotion(str(r.get("emotion","")))
        if label not in EMOTION_LABELS: raise ValueError("Correction has invalid emotion label")
        additions.append({**r,"emotion":label,"split":"train"})
    merged=base+additions; out=Path(output_dir); out.mkdir(parents=True,exist_ok=True)
    # Retain schema; union fields and fill absent columns.
    fields=list(dict.fromkeys(k for r in merged for k in r)); data=_write(None,merged,fields); target=out/"all_splits.csv"; target.write_bytes(data)
    digest=hashlib.sha256(data).hexdigest(); has_synthetic=any(str(r.get("synthetic_origin","")).lower()=="true" or "synthetic" in str(r.get("source","")).lower() for r in merged)
    meta={"schema_version":1,"dataset_kind":"synthetic_template" if has_synthetic else "versioned_human_corrections","dataset_sha256":digest,"prepared_csv_sha256":digest,"base_sha256":hashlib.sha256(Path(base_csv).read_bytes()).hexdigest(),"corrections_sha256":correction_hash,"review_audit_sha256":hashlib.sha256(Path(audit).read_bytes()).hexdigest() if isinstance(audit,(str,Path)) else hashlib.sha256(json.dumps(receipt,sort_keys=True).encode()).hexdigest(),"base_validation_test_preserved":True,"corrections_added_to_train":len(additions),"synthetic_origin_rows":sum(str(r.get("synthetic_origin","")).lower()=="true" or "synthetic" in str(r.get("source","")).lower() for r in merged),"human_review_provenance":"Completed two-reviewer receipt bound to correction CSV; synthetic origin retained.","training_note":"If synthetic-origin rows are present, pass --allow-synthetic for an explicitly experimental GPU run."}
    auditpath=out/"audit.json"; auditpath.write_text(json.dumps(meta,indent=2),encoding="utf-8")
    (out/"DATASET_CARD.md").write_text("# Versioned human correction dataset\n\nCorrections are included in train only. Base validation and test rows remain frozen. Human labels are exported only after two distinct reviewers and any required adjudication. Synthetic origin is retained and must be explicitly enabled for experimental training.\n",encoding="utf-8")
    (out/"GPU_TRAINING.md").write_text("# GPU training handoff\n\nUpload `all_splits.csv` and `audit.json` to the project GPU workflow. Use the project's compatible seven-emotion training entry point and preserve the split column. If `synthetic_origin_rows` is nonzero, explicitly pass `--allow-synthetic`; the run is experimental. Never train from `dataset_role=test` rows.\n",encoding="utf-8")
    return target,meta


def export_review_zip(queue, reviewer_a, reviewer_b, adjudications=None):
    rows,report=reconcile(queue,reviewer_a,reviewer_b,adjudications)
    if not report["ready"]: raise ValueError("Review is incomplete; export is blocked")
    payload=_write(None,rows)
    report["prepared_csv_sha256"]=hashlib.sha256(payload).hexdigest()
    b=io.BytesIO()
    with zipfile.ZipFile(b,"w",zipfile.ZIP_DEFLATED) as z:
        z.writestr("reviewed_corrections.csv",payload); z.writestr("audit.json",json.dumps(report,indent=2)); z.writestr("DATASET_CARD.md","# Human reviewed corrections\n\nRows preserve original source provenance. Synthetic-origin rows remain marked experimental. Evaluation-role rows are test only and must not be used for training.\n")
    return b.getvalue(),report


def main():
    p=argparse.ArgumentParser(description=__doc__); sub=p.add_subparsers(dest="cmd",required=True)
    c=sub.add_parser("create"); c.add_argument("queue"); c.add_argument("output_dir")
    r=sub.add_parser("reconcile"); r.add_argument("queue"); r.add_argument("reviewer_a"); r.add_argument("reviewer_b"); r.add_argument("--adjudications"); r.add_argument("--output",required=True); r.add_argument("--audit",required=True)
    b=sub.add_parser("build-corrections"); b.add_argument("base_csv"); b.add_argument("corrections"); b.add_argument("output_dir"); b.add_argument("--audit")
    a=p.parse_args()
    if a.cmd=="create": create_review_sheets(a.queue,a.output_dir)
    elif a.cmd=="reconcile":
        rows,report=reconcile(a.queue,a.reviewer_a,a.reviewer_b,a.adjudications); Path(a.audit).write_text(json.dumps(report,indent=2),encoding="utf-8")
        if not report["ready"]: raise SystemExit("Review incomplete; no ready CSV exported")
        data=_write(None,rows); Path(a.output).write_bytes(data); report["prepared_csv_sha256"]=hashlib.sha256(data).hexdigest(); Path(a.audit).write_text(json.dumps(report,indent=2),encoding="utf-8")
    else: build_training_dataset(a.base_csv,a.corrections,a.output_dir,a.audit)

if __name__=="__main__": main()
