"""Bounded seven-emotion conversation analysis with explicit context provenance."""
from __future__ import annotations


def parse_turns(text):
    turns=[]
    for line in str(text).splitlines():
        if not line.strip(): continue
        speaker,sep,message=line.partition(":")
        if not sep: speaker,message="Speaker",line
        if not speaker.strip() or not message.strip(): raise ValueError("Each turn needs a speaker and phrase")
        turns.append({"speaker":speaker.strip(),"text":message.strip()})
    if not 1<=len(turns)<=20: raise ValueError("Enter between one and twenty conversation turns")
    return turns


def analyze_turns(turns,workbench,language_pair="hi-en",history_limit=4):
    if not 1<=len(turns)<=20 or not 0<=history_limit<=8: raise ValueError("Use 1-20 turns and 0-8 history turns")
    previous={};history=[];results=[]
    for index,turn in enumerate(turns):
        text=str(turn.get("text","")).strip();speaker=str(turn.get("speaker","")).strip()
        if not text or not speaker: raise ValueError("A turn needs text and a speaker")
        context="\n".join(f"{t['speaker']}: {t['text']}" for t in history[-history_limit:]) if history_limit else ""
        context=context[-8000:]
        answer=workbench.analyze(text,language_pair,context)
        prior=previous.get(speaker)
        results.append({"turn":index+1,"speaker":speaker,"text":text,"context":context,
            "base_emotion":answer.primary.emotion,"final_emotion":answer.final_emotion,
            "base_score":answer.primary.confidence,"probabilities":answer.primary.probabilities,
            "previous_speaker_emotion":prior,"speaker_emotion_changed":prior is not None and prior!=answer.final_emotion,
            "source":answer.primary.source,"model":answer.primary.model_name,
            "warnings":list(answer.primary.warnings),"review_status":answer.review.get("status"),
            "review_suggestion":answer.review.get("suggestion")})
        previous[speaker]=answer.final_emotion;history.append({"speaker":speaker,"text":text})
    return results
