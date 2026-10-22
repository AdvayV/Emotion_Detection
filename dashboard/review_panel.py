"""Streamlit interface for blind human review and explicit correction export."""
from __future__ import annotations

from pathlib import Path
import pandas as pd
import streamlit as st

from training.review_workflow import create_review_sheets, reconcile, export_review_zip, build_training_dataset


def render_review_panel(root, style_chart=None):
    root=Path(root)
    st.subheader("Human review → correction dataset")
    st.caption("Upload a disagreement queue or challenge draft. Proposed labels stay separate; reviewer sheets start blank. Two independent reviewers and any required adjudication must be complete before export.")
    up=st.file_uploader("Queue CSV",type=["csv"],key="review_workflow_queue")
    if up is None: return
    try:
        source=pd.read_csv(up,dtype=str,keep_default_na=False)
        st.dataframe(source.head(10),width="stretch")
        if st.button("Create blind review sheets",key="review_workflow_create"):
            import tempfile, zipfile, io
            with tempfile.TemporaryDirectory() as td:
                csv_bytes=up.getvalue(); q=Path(td)/"queue.csv"; q.write_bytes(csv_bytes)
                create_review_sheets(q,td)
                z=io.BytesIO()
                with zipfile.ZipFile(z,"w",zipfile.ZIP_DEFLATED) as archive:
                    for name in ("reviewer_a.csv","reviewer_b.csv","proposed_labels.csv"): archive.write(Path(td)/name,name)
                st.download_button("Download blind sheets",z.getvalue(),"review_sheets.zip","application/zip",key="review_workflow_sheets")
        st.markdown("Upload completed sheets below. Set `dataset_role` to `test` in the original queue before sheet creation for independent evaluation rows.")
        a=st.file_uploader("Completed reviewer A CSV",type=["csv"],key="review_workflow_a")
        b=st.file_uploader("Completed reviewer B CSV",type=["csv"],key="review_workflow_b")
        d=st.file_uploader("Adjudication CSV (row_id, adjudicator_id, emotion, notes)",type=["csv"],key="review_workflow_d")
        if a and b:
            import tempfile
            with tempfile.TemporaryDirectory() as td:
                q=Path(td)/"queue.csv"; aa=Path(td)/"a.csv"; bb=Path(td)/"b.csv"; q.write_bytes(up.getvalue()); aa.write_bytes(a.getvalue()); bb.write_bytes(b.getvalue()); dd=None
                if d: dd=Path(td)/"d.csv"; dd.write_bytes(d.getvalue())
                rows,report=reconcile(q,aa,bb,dd)
            st.write(f"Reviewed {report['reviewed_rows']} / {report['source_rows']} rows; {report['training_rows']} for training and {report['evaluation_rows']} for evaluation.")
            if report["pending"]: st.warning(f"{len(report['pending'])} rows still need review or adjudication.")
            if report["ready"]:
                with tempfile.TemporaryDirectory() as td:
                    q=Path(td)/"queue.csv"; aa=Path(td)/"a.csv"; bb=Path(td)/"b.csv"; q.write_bytes(up.getvalue()); aa.write_bytes(a.getvalue()); bb.write_bytes(b.getvalue()); dd=None
                    if d: dd=Path(td)/"d.csv"; dd.write_bytes(d.getvalue())
                    data,receipt=export_review_zip(q,aa,bb,dd)
                st.download_button("Download audited human corrections",data,"human_reviewed_corrections.zip","application/zip",key="review_workflow_export")
        st.divider()
        st.markdown("**Create a versioned GPU dataset**")
        st.caption("Upload the frozen base all_splits.csv, the reviewed_corrections.csv from the ZIP, and its matching audit.json. The export is prepared for download only.")
        base_upload=st.file_uploader("Frozen base all_splits.csv",type=["csv"],key="review_workflow_base")
        correction_upload=st.file_uploader("Reconciled reviewed_corrections.csv",type=["csv"],key="review_workflow_corr")
        audit_upload=st.file_uploader("Matching completed audit.json",type=["json"],key="review_workflow_audit")
        if st.button("Build versioned training dataset",key="review_workflow_build",disabled=not(base_upload and correction_upload and audit_upload)):
            import tempfile, zipfile, io, json
            with tempfile.TemporaryDirectory() as td:
                base_path=Path(td)/"base.csv"; corr_path=Path(td)/"reviewed_corrections.csv"; audit_path=Path(td)/"audit.json"
                base_path.write_bytes(base_upload.getvalue()); corr_path.write_bytes(correction_upload.getvalue()); audit_path.write_bytes(audit_upload.getvalue())
                try:
                    dataset,meta=build_training_dataset(base_path,corr_path,Path(td)/"versioned",audit_path)
                    archive=io.BytesIO()
                    with zipfile.ZipFile(archive,"w",zipfile.ZIP_DEFLATED) as z:
                        for name in ("all_splits.csv","audit.json","DATASET_CARD.md","GPU_TRAINING.md"):
                            z.write(dataset.parent/name,name)
                    st.info(f"Dataset ready: {meta['corrections_added_to_train']} train corrections; synthetic-origin rows: {meta['synthetic_origin_rows']}. Use explicit --allow-synthetic for experimental training when this count is nonzero.")
                    st.download_button("Download versioned GPU dataset",archive.getvalue(),"versioned_human_corrections.zip","application/zip",key="review_workflow_dataset_zip")
                except Exception as exc: st.error(f"Dataset build blocked: {exc}")
    except Exception as exc: st.error(f"Review workflow could not continue: {exc}")
