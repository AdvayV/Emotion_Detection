"""Streamlit interface for blind review and rerun-safe correction exports."""
from __future__ import annotations

import hashlib
import io
import json
import tempfile
import zipfile
from pathlib import Path

import pandas as pd
import streamlit as st

from training.review_workflow import create_review_sheets, reconcile, export_review_zip, build_training_dataset


def _signature(*uploads):
    return tuple(hashlib.sha256(upload.getvalue()).hexdigest() if upload is not None else None for upload in uploads)


def _zip_files(folder, names):
    stream = io.BytesIO()
    with zipfile.ZipFile(stream, "w", zipfile.ZIP_DEFLATED) as archive:
        for name in names:
            archive.write(Path(folder) / name, name)
    return stream.getvalue()


def _review_sheets():
    upload = st.file_uploader("Queue CSV", type=["csv"], key="review_workflow_queue")
    signature = _signature(upload)
    saved = st.session_state.get("review_sheets_export")
    if saved and saved["signature"] != signature:
        st.session_state.pop("review_sheets_export", None)
    if upload is None:
        st.info("Upload a queue to prepare blind review sheets. You can build a GPU dataset below using previously completed corrections.")
        return

    try:
        source = pd.read_csv(io.BytesIO(upload.getvalue()), dtype=str, keep_default_na=False)
        st.dataframe(source.head(10), width="stretch")
        if st.button("Create blind review sheets", key="review_workflow_create"):
            st.session_state.pop("review_sheets_export", None)
            with tempfile.TemporaryDirectory() as directory:
                queue = Path(directory) / "queue.csv"
                queue.write_bytes(upload.getvalue())
                create_review_sheets(queue, directory)
                data = _zip_files(directory, ("reviewer_a.csv", "reviewer_b.csv", "proposed_labels.csv"))
            st.session_state.review_sheets_export = {"signature": signature, "data": data}
        saved = st.session_state.get("review_sheets_export")
        if saved and saved["signature"] == signature:
            st.download_button("Download blind sheets", saved["data"], "review_sheets.zip", "application/zip", key="review_workflow_sheets", on_click="ignore")

        st.markdown("Upload completed sheets below. Set `dataset_role` to `test` in the original queue before sheet creation for independent evaluation rows.")
        reviewer_a = st.file_uploader("Completed reviewer A CSV", type=["csv"], key="review_workflow_a")
        reviewer_b = st.file_uploader("Completed reviewer B CSV", type=["csv"], key="review_workflow_b")
        adjudication = st.file_uploader("Adjudication CSV (row_id, adjudicator_id, emotion, notes)", type=["csv"], key="review_workflow_d")
        if reviewer_a is not None and reviewer_b is not None:
            with tempfile.TemporaryDirectory() as directory:
                folder = Path(directory)
                queue, aa, bb = folder / "queue.csv", folder / "a.csv", folder / "b.csv"
                queue.write_bytes(upload.getvalue())
                aa.write_bytes(reviewer_a.getvalue())
                bb.write_bytes(reviewer_b.getvalue())
                dd = None
                if adjudication is not None:
                    dd = folder / "d.csv"
                    dd.write_bytes(adjudication.getvalue())
                _, report = reconcile(queue, aa, bb, dd)
                st.write(f"Reviewed {report['reviewed_rows']} / {report['source_rows']} rows; {report['training_rows']} for training and {report['evaluation_rows']} for evaluation.")
                if report["pending"]:
                    st.warning(f"{len(report['pending'])} rows still need review or adjudication.")
                if report["ready"]:
                    data, _ = export_review_zip(queue, aa, bb, dd)
                    st.download_button("Download audited human corrections", data, "human_reviewed_corrections.zip", "application/zip", key="review_workflow_export", on_click="ignore")
    except Exception as exc:
        st.error(f"Review workflow could not continue: {exc}")


def _versioned_dataset():
    st.divider()
    st.markdown("**Create a versioned GPU dataset**")
    st.caption("Upload the frozen base all_splits.csv, reviewed_corrections.csv and its matching audit.json. A queue upload is not required for this step. The dataset stays in this session until you download it.")
    base = st.file_uploader("Frozen base all_splits.csv", type=["csv"], key="review_workflow_base")
    corrections = st.file_uploader("Reconciled reviewed_corrections.csv", type=["csv"], key="review_workflow_corr")
    audit = st.file_uploader("Matching completed audit.json", type=["json"], key="review_workflow_audit")
    signature = _signature(base, corrections, audit)
    saved = st.session_state.get("review_dataset_export")
    if saved and saved["signature"] != signature:
        st.session_state.pop("review_dataset_export", None)
    ready = all(upload is not None for upload in (base, corrections, audit))
    if st.button("Build versioned training dataset", key="review_workflow_build", disabled=not ready):
        st.session_state.pop("review_dataset_export", None)
        try:
            with tempfile.TemporaryDirectory() as directory:
                folder = Path(directory)
                base_path, correction_path, audit_path = folder / "base.csv", folder / "reviewed_corrections.csv", folder / "audit.json"
                base_path.write_bytes(base.getvalue())
                correction_path.write_bytes(corrections.getvalue())
                audit_path.write_bytes(audit.getvalue())
                dataset, metadata = build_training_dataset(base_path, correction_path, folder / "versioned", audit_path)
                data = _zip_files(dataset.parent, ("all_splits.csv", "audit.json", "DATASET_CARD.md", "GPU_TRAINING.md"))
            st.session_state.review_dataset_export = {"signature": signature, "data": data, "metadata": metadata}
        except Exception as exc:
            st.error(f"Dataset build blocked: {exc}")
    saved = st.session_state.get("review_dataset_export")
    if saved and saved["signature"] == signature:
        metadata = saved["metadata"]
        st.info(f"Dataset ready: {metadata['corrections_added_to_train']} train corrections; synthetic-origin rows: {metadata['synthetic_origin_rows']}. Use explicit --allow-synthetic for experimental training when this count is nonzero.")
        st.download_button("Download versioned GPU dataset", saved["data"], "versioned_human_corrections.zip", "application/zip", key="review_workflow_dataset_zip", on_click="ignore")


def render_review_panel(root, style_chart=None):
    st.subheader("Human review and correction dataset")
    st.caption("Proposed labels stay separate and reviewer sheets start blank. Two independent reviewers and any required adjudication must be complete before exporting corrections.")
    _review_sheets()
    _versioned_dataset()
