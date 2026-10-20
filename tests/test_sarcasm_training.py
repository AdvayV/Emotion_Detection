import pandas as pd
import pytest
from training.train_sarcasm import read_splits


def rows():
    return [{"text": f"unique {split} phrase {label}", "sarcasm": label, "language_pair":"en-hi", "split":split,
             "group_id":f"{split}_{label}", "human_reviewed":True} for split in ("train","validation","test") for label in (0,1)]


def test_sarcasm_split_normalizes_language_without_inventing_labels(tmp_path):
    path=tmp_path/"data.csv"; pd.DataFrame(rows()).to_csv(path,index=False)
    parts=read_splits(path)
    assert {key:len(part) for key,part in parts.items()} == {"train":2,"validation":2,"test":2}
    assert set(parts["train"].language_pair) == {"hi-en"}


@pytest.mark.parametrize("problem", ["review", "group", "duplicate", "language", "class", "synthetic"])
def test_sarcasm_refuses_unreviewed_leaking_or_unsupported_data(tmp_path, problem):
    data=rows()
    if problem=="review": data[0]["human_reviewed"]=False
    if problem=="group": data[-1]["group_id"]=data[0]["group_id"]
    if problem=="duplicate": data[-1]["text"]=data[0]["text"].upper()
    if problem=="language": data[0]["language_pair"]="fr-en"
    if problem=="class": data[0]["sarcasm"]=1
    if problem=="synthetic": data[0]["synthetic"]=True
    path=tmp_path/"data.csv";pd.DataFrame(data).to_csv(path,index=False)
    with pytest.raises(ValueError): read_splits(path)
