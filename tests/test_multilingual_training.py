"""A tiny offline XLM-R smoke run; never a language-quality benchmark."""
import csv
import importlib.util
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
from hinglish_emotion.tasks import EMOTION_LABELS

AVAILABLE=all(importlib.util.find_spec(name) for name in ['torch','transformers','tokenizers','safetensors','pandas','sklearn'])


@unittest.skipUnless(AVAILABLE,'Optional model dependencies')
class MultilingualTrainingTests(unittest.TestCase):
    def test_xlmr_both_languages_train_export_install_and_predict_offline(self):
        import torch
        from tokenizers import Tokenizer
        from tokenizers.models import WordLevel
        from tokenizers.pre_tokenizers import Whitespace
        from tokenizers.processors import TemplateProcessing
        from transformers import PreTrainedTokenizerFast,XLMRobertaConfig,XLMRobertaForSequenceClassification
        from training.train_transformer import parse_args,train
        from training.cloud_bundle import export_checkpoint
        from training.install_checkpoint import install
        from hinglish_emotion.emotions import LocalEmotionClassifier,EmotionDetector
        from hinglish_emotion.evaluation_guard import training_hashes
        old_threads=torch.get_num_threads();torch.set_num_threads(1)
        try:
            with tempfile.TemporaryDirectory() as directory:
                root=Path(directory);base=root/'tiny_base';base.mkdir()
                words=['<s>','<pad>','</s>','<unk>','<mask>',*EMOTION_LABELS,'hi','en','fr','phrase',*map(str,range(12))]
                vocab={word:index for index,word in enumerate(words)}
                backend=Tokenizer(WordLevel(vocab,unk_token='<unk>'));backend.pre_tokenizer=Whitespace()
                backend.post_processor=TemplateProcessing(single='<s> $A </s>',special_tokens=[('<s>',0),('</s>',2)])
                tokenizer=PreTrainedTokenizerFast(tokenizer_object=backend,bos_token='<s>',eos_token='</s>',pad_token='<pad>',unk_token='<unk>',mask_token='<mask>')
                tokenizer.save_pretrained(base)
                XLMRobertaForSequenceClassification(XLMRobertaConfig(vocab_size=len(vocab),hidden_size=16,num_hidden_layers=1,num_attention_heads=2,intermediate_size=24,max_position_embeddings=64,num_labels=7,pad_token_id=1,bos_token_id=0,eos_token_id=2)).save_pretrained(base)
                data=root/'smoke.csv'
                with data.open('w',encoding='utf-8',newline='') as f:
                    writer=csv.writer(f);writer.writerow(['row_id','text','emotion','language_pair','source'])
                    for pair in ['hi-en','fr-en']:
                        for emotion in EMOTION_LABELS:
                            for index in range(12):writer.writerow([f'{pair}_{emotion}_{index}',f'{pair} {emotion} phrase {index}',emotion,pair,'synthetic_offline_smoke'])
                output=root/'trained'/'xlmr_emotion'
                argv=['train','--task','emotion','--model',str(base),'--data',str(data),'--output',str(output),'--language-pair','all','--epochs','1','--batch-size','8','--gradient-accumulation','2','--max-length','32','--device','cpu','--allow-cpu','--local-files-only','--allow-synthetic']
                with patch('sys.argv',argv):result=train(parse_args())
                self.assertEqual(set(result['test_by_language']),{'hi-en','fr-en'})
                self.assertEqual(result['evaluation_scope'],'synthetic_only')
                archive=export_checkpoint(output,root/'model.zip')
                installed=install(archive,root/'installed')
                self.assertEqual(installed.name,'xlmr_multilingual_emotion')
                detector=EmotionDetector(LocalEmotionClassifier(installed,device='cpu'))
                guesses=detector.predict_many(['hi en joy phrase','fr en sadness phrase'],['hi-en','fr-en'])
                self.assertEqual(len(guesses),2)
                self.assertTrue(all(set(guess.probabilities)==set(EMOTION_LABELS) for guess in guesses))
                self.assertIsNotNone(training_hashes(detector))
        finally:torch.set_num_threads(old_threads)
