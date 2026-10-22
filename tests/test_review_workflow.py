import csv, json, hashlib
from pathlib import Path

import pytest

from training.review_workflow import create_review_sheets, reconcile, export_review_zip, build_training_dataset


def csvfile(path, rows):
    Path(path).parent.mkdir(parents=True,exist_ok=True)
    fields=list(dict.fromkeys(k for r in rows for k in r));
    with Path(path).open("w",newline="",encoding="utf-8") as f:
        w=csv.DictWriter(f,fieldnames=fields); w.writeheader(); w.writerows(rows)
    return path


def reviewed_pair(tmp_path, row):
    queue=csvfile(tmp_path/"queue.csv",[row]); create_review_sheets(queue,tmp_path/"sheets")
    a=tmp_path/"sheets"/"reviewer_a.csv"; b=tmp_path/"sheets"/"reviewer_b.csv"
    for path,name in ((a,"A"),(b,"B")):
        rows=list(csv.DictReader(path.open(encoding="utf-8"))); rows[0].update(reviewer_id=name,emotion=row["emotion"])
        csvfile(path,rows)
    reviewed,receipt=reconcile(queue,a,b)
    output=csvfile(tmp_path/"reviewed.csv",reviewed); receipt["prepared_csv_sha256"]=hashlib.sha256(output.read_bytes()).hexdigest()
    audit=tmp_path/"audit.json"; audit.write_text(json.dumps(receipt),encoding="utf-8")
    return output,audit,reviewed


def test_blind_labels_source_and_conflict_adjudication(tmp_path):
    q=csvfile(tmp_path/"q.csv",[{"row_id":"r1","group_id":"g1","text":"Aaj khushi hai","language_pair":"hi-en","emotion":"joy","source":"synthetic_draft","synthetic":"true"}])
    create_review_sheets(q,tmp_path/"sheets")
    a=tmp_path/"sheets"/"reviewer_a.csv"; b=tmp_path/"sheets"/"reviewer_b.csv"
    for p,label,name in ((a,"happy","A"),(b,"joy","B")):
        rows=list(csv.DictReader(p.open(encoding="utf-8"))); rows[0].update(reviewer_id=name,emotion=label)
        csvfile(p,rows)
    rows,report=reconcile(q,a,b)
    assert report["ready"] and rows[0]["emotion"]=="joy"
    assert rows[0]["source"]=="synthetic_human_reviewed" and rows[0]["original_source"]=="synthetic_draft" and rows[0]["experimental_synthetic"]
    # Conflict blocks export until an independent adjudicator supplies a reasoned decision.
    rowsb=list(csv.DictReader(b.open(encoding="utf-8"))); rowsb[0]["emotion"]="anger"; csvfile(b,rowsb)
    pending,receipt=reconcile(q,a,b)
    assert not receipt["ready"] and receipt["pending"]
    adjud=csvfile(tmp_path/"adj.csv",[{"row_id":"r1","adjudicator_id":"C","emotion":"joy","notes":"Context confirms positive affect."}])
    final,receipt=reconcile(q,a,b,adjud); assert receipt["ready"] and final[0]["annotation_status"]=="human_adjudicated"
    payload,_=export_review_zip(q,a,b,adjud); assert payload[:2]==b"PK"


def test_build_adds_train_and_preserves_frozen_splits(tmp_path):
    base=csvfile(tmp_path/"base.csv",[
        {"row_id":"b1","group_id":"gb1","text":"train phrase","emotion":"joy","source":"human","split":"train"},
        {"row_id":"b2","group_id":"gb2","text":"validation phrase","emotion":"joy","source":"human","split":"validation"},
        {"row_id":"b3","group_id":"gb3","text":"test phrase","emotion":"joy","source":"human","split":"test"},])
    corr,audit,correction_rows=reviewed_pair(tmp_path,{"row_id":"c1","group_id":"gc1","text":"new phrase","language_pair":"hi-en","emotion":"happy","source":"review","dataset_role":"train"})
    target,meta=build_training_dataset(base,corr,tmp_path/"out",audit)
    merged=list(csv.DictReader(target.open(encoding="utf-8")))
    assert [r["split"] for r in merged[-1:]]==["train"] and meta["base_validation_test_preserved"]
    assert {r["row_id"]:r["split"] for r in merged if r["row_id"].startswith("b")}=={"b1":"train","b2":"validation","b3":"test"}
    assert json.loads((target.parent/"audit.json").read_text())["dataset_sha256"]==meta["dataset_sha256"]
    assert meta["prepared_csv_sha256"]==meta["dataset_sha256"]
    with pytest.raises(ValueError,match="overlaps frozen"):
        blocked,audit_blocked,_=reviewed_pair(tmp_path/"blocked",{"row_id":"c2","group_id":"gc2","text":"test phrase","language_pair":"hi-en","emotion":"joy","source":"review","dataset_role":"train"})
        build_training_dataset(base,blocked,tmp_path/"bad",audit_blocked)
    with pytest.raises(ValueError,match="evaluation rows"):
        eval_rows=[{**r,"dataset_role":"test"} for r in correction_rows]; eval_file=csvfile(tmp_path/"eval.csv",eval_rows)
        receipt=json.loads(audit.read_text()); receipt["prepared_csv_sha256"]=hashlib.sha256(eval_file.read_bytes()).hexdigest(); eval_audit=tmp_path/"eval_audit.json"; eval_audit.write_text(json.dumps(receipt))
        build_training_dataset(base,eval_file,tmp_path/"bad2",eval_audit)


def test_changed_source_and_same_reviewer_fail_closed(tmp_path):
    q=csvfile(tmp_path/"q.csv",[{"row_id":"x","group_id":"gx","text":"hello","language_pair":"hi-en"}]); create_review_sheets(q,tmp_path/"s")
    a=tmp_path/"s"/"reviewer_a.csv"; b=tmp_path/"s"/"reviewer_b.csv"
    for p in (a,b):
        row=next(csv.DictReader(p.open(encoding="utf-8"))); row.update(reviewer_id="same",emotion="neutral"); csvfile(p,[row])
    _,rep=reconcile(q,a,b); assert not rep["ready"]
    changed=csvfile(tmp_path/"changed.csv",[{"row_id":"x","group_id":"gx","text":"tampered","language_pair":"hi-en"}])
    with pytest.raises(ValueError,match="changed since review"):
        reconcile(changed,a,b)


def test_draft_defaults_to_test_and_keeps_origin_fields(tmp_path):
    q=csvfile(tmp_path/"draft.csv",[{"row_id":"draft_case_1","group_id":"scenario-1","text":"Kya ajeeb surprise tha","language_pair":"hi-en","emotion":"surprise","source":"assistant_draft","annotation_status":"assistant_draft"}])
    create_review_sheets(q,tmp_path/"draft_sheets")
    sheet=next(csv.DictReader((tmp_path/"draft_sheets"/"reviewer_a.csv").open(encoding="utf-8")))
    assert sheet["dataset_role"]=="test" and sheet["emotion"]==""
    assert sheet["original_source"]=="assistant_draft" and sheet["original_annotation_status"]=="assistant_draft"


def test_build_requires_bound_audit_and_allows_same_train_group(tmp_path):
    base=csvfile(tmp_path/"base.csv",[{"row_id":"b1","group_id":"shared","text":"existing training","emotion":"joy","source":"human","split":"train"}])
    row_a={"row_id":"c1","group_id":"newgroup","text":"new phrase one","language_pair":"hi-en","emotion":"joy","source":"review","dataset_role":"train"}
    row_b={**row_a,"row_id":"c2","text":"new phrase two"}
    # Multiple reviewed rows may share one training group.
    q=csvfile(tmp_path/"q.csv",[row_a,row_b]); create_review_sheets(q,tmp_path/"sheets")
    for name in ("reviewer_a.csv","reviewer_b.csv"):
        p=tmp_path/"sheets"/name; rows=list(csv.DictReader(p.open(encoding="utf-8")))
        for i,r in enumerate(rows): r.update(reviewer_id=name,emotion="joy")
        csvfile(p,rows)
    reviewed,receipt=reconcile(q,tmp_path/"sheets"/"reviewer_a.csv",tmp_path/"sheets"/"reviewer_b.csv")
    corr=csvfile(tmp_path/"corr.csv",reviewed); receipt["prepared_csv_sha256"]=hashlib.sha256(corr.read_bytes()).hexdigest()
    audit=tmp_path/"audit.json"; audit.write_text(json.dumps(receipt))
    target,_=build_training_dataset(base,corr,tmp_path/"out",audit)
    assert len(list(csv.DictReader(target.open(encoding="utf-8"))))==3
    with pytest.raises(ValueError,match="audit"):
        build_training_dataset(base,corr,tmp_path/"no_audit")
