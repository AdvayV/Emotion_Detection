"""Read checkpoint readiness without neural inference or Ollama calls."""
import json
from pathlib import Path
from hinglish_emotion.checkpoints import emotion_checkpoint
from hinglish_emotion.checkpoint_metadata import emotion_checkpoint_metadata
from hinglish_emotion.model_catalog import ENCODERS, checkpoint_ready
from hinglish_emotion.tasks import LANGUAGE_PAIRS


def overview_readiness(root):
    root=Path(root);rows=[]
    for encoder in sorted(ENCODERS,key=lambda e:('xlmr','mbert','muril').index(e.key)):
        row={'Encoder':encoder.name}
        scopes=[]
        for pair,label in LANGUAGE_PAIRS.items():
            try:
                path=emotion_checkpoint(root/'models',pair,encoder.key)
                scope=emotion_checkpoint_metadata(path).get('evaluation_scope','unknown') if path else None
            except (ValueError,OSError,TypeError,AttributeError):
                path=None;scope=None
            row[label]='Not supported' if encoder.key=='muril' and pair=='fr-en' else 'Installed' if path else 'Not installed'
            if path:scopes.append(scope)
        row['Evidence']='Synthetic test results only' if scopes and all(s=='synthetic_only' for s in scopes) else 'See checkpoint evaluation scope' if scopes else 'No installed checkpoint'
        rows.append(row)
    fusion_path=root/'models/xlmr_fusion_emotion';fusion=None
    if checkpoint_ready(fusion_path):
        try:
            candidate=emotion_checkpoint_metadata(fusion_path)
            if candidate.get('architecture')=='text_emoji_multitask_v1' and candidate.get('training_status')=='complete':fusion=candidate
        except (ValueError,OSError,TypeError,AttributeError):pass
    sarcasm=False;path=root/'models/xlmr_sarcasm'
    if checkpoint_ready(path):
        try:
            m=json.loads((path/'sarcasm_metadata.json').read_text(encoding='utf-8'));c=json.loads((path/'config.json').read_text(encoding='utf-8'))
            sarcasm=(m.get('task')=='sarcasm_binary' and m.get('training_status')=='complete' and m.get('human_reviewed') is True and m.get('labels')==['not_sarcastic','sarcastic'] and m.get('base_model')=='FacebookAI/xlm-roberta-base' and m.get('language_pairs')==['hi-en'] and c.get('model_type')=='xlm-roberta' and c.get('id2label')=={'0':'not_sarcastic','1':'sarcastic'} and bool(list(path.glob('*.safetensors'))) and type(m.get('max_length',128)) is int and 1 <= m.get('max_length',128) <= 512)
        except (ValueError,OSError,TypeError,AttributeError):pass
    def training_status(name,installed):
        return 'Checkpoint installed (validated on use)' if installed else 'Training code available; no trained checkpoint installed' if (root/'training'/name).is_file() else 'Training code not found in this checkout'
    capabilities=[
        {'Capability':'Seven-emotion encoders','Available today':'Local XLM-R / mBERT / MuRIL where listed above','Evidence status':'Per-checkpoint scope; natural-message accuracy unverified'},
        {'Capability':'Matched evaluation','Available today':'Same eligible rows, shared exclusions and aligned exports','Evidence status':'Measured configurations, not a consensus vote'},
        {'Capability':'Emoji and sarcasm cues','Available today':'Transparent rule diagnostics, separate from model scores','Evidence status':'Heuristic cues, not trained attribution'},
        {'Capability':'Local Qwen review','Available today':'Optional inference when Ollama and its model are running','Evidence status':'Reviewer benefit unmeasured; no weight updates'},
        {'Capability':'Learned emoji fusion','Available today':training_status('train_advanced.py',bool(fusion)),'Evidence status':'No controlled fusion gain established'},
        {'Capability':'Learned conversation context','Available today':training_status('train_advanced.py',bool(fusion and fusion.get('trained_with_context'))),'Evidence status':'Conversation traces are model outputs, not mental states'},
        {'Capability':'Supervised sarcasm','Available today':training_status('train_sarcasm.py',sarcasm or bool(fusion and fusion.get('trained_sarcasm_head'))),'Evidence status':'Requires independently reviewed binary labels'},
        {'Capability':'Human correction workflow','Available today':'Blind sheets, adjudication and audited training-only export','Evidence status':'Human review is required; no automatic learning'},
    ]
    return {'models':rows,'installed_families':sum(any(row[label]=='Installed' for label in LANGUAGE_PAIRS.values()) for row in rows),'capabilities':capabilities}
