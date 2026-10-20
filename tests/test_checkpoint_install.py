import json
from pathlib import Path
import tempfile
import unittest
import zipfile
import numpy as np
from safetensors.numpy import save_file
from training.install_checkpoint import install
from hinglish_emotion.tasks import EMOTION_LABELS


class CheckpointInstallTests(unittest.TestCase):
    def fixture(self,root,pairs=("hi-en","fr-en"),head_rows=7):
        source=root/"source";source.mkdir()
        meta={"schema_version":1,"task":"emotion","labels":list(EMOTION_LABELS),"text_mode":"raw","preparation_version":1,"max_length":128,"trained_language_pairs":list(pairs),"validated_language_pairs":[],"training_status":"complete"}
        (source/"config.json").write_text(json.dumps({"model_type":"xlm-roberta","hidden_size":4,"id2label":dict(enumerate(EMOTION_LABELS)),"emotion_metadata":meta}))
        (source/"tokenizer.json").write_text("{}")
        save_file({"classifier.out_proj.weight":np.zeros((head_rows,4),dtype=np.float32)},source/"model.safetensors")
        archive=root/"inference.zip"
        with zipfile.ZipFile(archive,"w") as z:
            for file in source.iterdir():z.write(file,"xlmr_emotion/"+file.name)
        return archive

    def test_installs_multilingual_and_preserves_existing_model(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);archive=self.fixture(root)
            path=install(archive,root/"models")
            self.assertEqual(path.name,"xlmr_multilingual_emotion")
            before=(path/"model.safetensors").read_bytes()
            with self.assertRaisesRegex(ValueError,"Preserving existing"):install(archive,root/"models")
            self.assertEqual((path/"model.safetensors").read_bytes(),before)

    def test_incorrect_head_is_rejected_without_leftover_staging_files(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);archive=self.fixture(root,head_rows=6)
            with self.assertRaisesRegex(ValueError,"classification head"):install(archive,root/"models")
            self.assertEqual(list((root/"models").iterdir()),[])

    def test_traversal_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);archive=root/"unsafe.zip"
            with zipfile.ZipFile(archive,"w") as z:z.writestr("../config.json","{}")
            with self.assertRaisesRegex(ValueError,"Unsafe"):install(archive,root/"models")
            self.assertFalse((root/"config.json").exists())
