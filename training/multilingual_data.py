"""Reproducible synthetic bootstrap for Hindi-English and English-French.

The existing Hinglish corpus is preserved. French-English examples are authored
template compositions, not collected or independently annotated messages.
"""
from collections import Counter
import csv
import hashlib
import json
from pathlib import Path
import random
from training.synthetic_phrases import generated_rows, COUNTS
from hinglish_emotion.tasks import EMOTION_LABELS
from hinglish_emotion.text_preparation import duplicate_key

TOPICS = [
    ("mon projet de groupe","📁"),("mon examen","📚"),("ma présentation","📊"),("mon entretien","💼"),("mon stage","💻"),
    ("notre tournoi","🏆"),("mon inscription","📝"),("notre concert","🎵"),("mon voyage","🚆"),("mon dossier","📂"),
    ("notre atelier","🛠️"),("notre réunion","📅"),("mon cours","🏫"),("notre compétition","🏃"),("ma candidature","✉️"),
    ("notre événement","🎫"),("notre exposé","🎤"),("ma formation","🎓"),("notre séance","📋"),("notre sortie","🚌"),
]
SCENARIOS = {
 "joy": ["Pour {item}, mes efforts ont enfin été reconnus", "Pour {item}, le résultat dépasse mes attentes", "Pour {item}, mes amis ont préparé une belle surprise", "Pour {item}, tout s'est bien passé malgré mes doutes", "Pour {item}, nous avons réussi ensemble après des semaines de travail", "Pour {item}, j'ai reçu les félicitations de toute l'équipe", "Pour {item}, une personne importante est venue me soutenir", "Pour {item}, j'ai enfin obtenu la réponse que j'espérais", "Pour {item}, je viens de recevoir une excellente nouvelle", "Pour {item}, on a célébré notre réussite ensemble"],
 "sadness": ["Pour {item}, la personne que j'attendais n'est jamais venue", "Pour {item}, j'ai perdu une occasion qui comptait beaucoup", "Pour {item}, tout mon travail semble avoir été oublié", "Pour {item}, mon meilleur ami ne sera plus avec nous", "Pour {item}, le résultat m'a profondément déçu", "Pour {item}, je dois abandonner quelque chose qui me tenait à cœur", "Pour {item}, notre groupe s'est séparé après tant de souvenirs", "Pour {item}, j'avais beaucoup d'espoir et tout est tombé à l'eau", "Pour {item}, le départ de mon ami laisse un grand vide", "Pour {item}, j'ai appris que je ne pourrai plus participer"],
 "anger": ["Pour {item}, on m'a accusé d'une faute que je n'ai pas commise", "Pour {item}, quelqu'un a effacé mon travail sans demander", "Pour {item}, on a encore ignoré notre accord", "Pour {item}, on m'a menti alors que j'avais fait confiance", "Pour {item}, les autres ont pris le crédit de mon travail", "Pour {item}, on a changé les règles sans nous prévenir", "Pour {item}, on m'a manqué de respect devant tout le monde", "Pour {item}, quelqu'un a saboté nos efforts volontairement", "Pour {item}, une promesse a encore été brisée", "Pour {item}, on a puni notre groupe pour l'erreur de quelqu'un d'autre"],
 "fear": ["Pour {item}, je dois affronter une situation qui me fait peur", "Pour {item}, je crains de perdre ma seule chance", "Pour {item}, je tremble à l'idée de parler devant tout le monde", "Pour {item}, je redoute ce qui pourrait arriver demain", "Pour {item}, je ne me sens pas en sécurité", "Pour {item}, l'attente du résultat m'empêche de dormir", "Pour {item}, je dois aller seul dans un endroit inconnu", "Pour {item}, j'ai peur de ne pas savoir réagir", "Pour {item}, une menace inattendue me fait paniquer", "Pour {item}, mon cœur s'accélère avant même de commencer"],
 "surprise": ["Pour {item}, une nouvelle totalement inattendue vient d'arriver", "Pour {item}, une personne que je n'avais pas vue depuis des années est apparue", "Pour {item}, le résultat a été annoncé bien plus tôt que prévu", "Pour {item}, tout a changé en quelques secondes", "Pour {item}, personne n'avait imaginé ce retournement", "Pour {item}, j'ai découvert quelque chose d'incroyable par hasard", "Pour {item}, la réponse est venue de la personne la moins attendue", "Pour {item}, un événement imprévu a changé le programme", "Pour {item}, ce que je viens de voir me laisse sans voix", "Pour {item}, la décision annoncée ne correspond à aucune prévision"],
 "disgust": ["Pour {item}, la salle était couverte de déchets collants", "Pour {item}, on nous a servi un repas qui sentait le moisi", "Pour {item}, les toilettes étaient vraiment répugnantes", "Pour {item}, j'ai trouvé des insectes dans la nourriture", "Pour {item}, une odeur de pourriture remplissait la pièce", "Pour {item}, le matériel était sale et couvert de moisissure", "Pour {item}, quelqu'un a craché sur la table commune", "Pour {item}, la nourriture avait un goût avarié", "Pour {item}, des restes en décomposition étaient laissés partout", "Pour {item}, le sol était couvert d'une substance nauséabonde"],
 "neutral": ["Pour {item}, le rendez-vous est fixé à dix heures", "Pour {item}, le formulaire doit être envoyé par courriel", "Pour {item}, nous utiliserons la salle numéro trois", "Pour {item}, le document contient deux pages", "Pour {item}, le matériel se trouve sur la table", "Pour {item}, la liste des participants sera disponible demain", "Pour {item}, la date figure dans le calendrier", "Pour {item}, le bureau ouvre à neuf heures", "Pour {item}, la réunion se tiendra mardi matin", "Pour {item}, le numéro de dossier est indiqué sur la fiche"],
}
REACTIONS = {
 "joy": ["I feel really happy", "This made my day", "I cannot stop smiling", "I am delighted with how it turned out", "It feels wonderful after all that effort", "I feel proud and cheerful", "I want to celebrate this moment", "That brought a huge smile to my face", "I am enjoying every second", "I feel relieved and happy", "This is the news I was hoping for", "I feel full of positive energy", "I am thrilled by this outcome", "My whole mood has lifted", "I feel genuinely pleased", "I would love to share this good news", "That gave me a warm feeling", "I feel overjoyed right now", "I finally have a reason to celebrate", "I am so glad this happened"],
 "sadness": ["I feel deeply sad", "My heart feels heavy", "I cannot stop missing what we had", "I feel like crying", "This leaves me feeling empty", "It hurts more than I expected", "I wish things had turned out differently", "I feel disappointed and low", "I cannot shake this feeling of loss", "I just want to sit quietly for a while", "I am struggling with this goodbye", "I feel lonely even around other people", "There is a painful silence now", "I do not feel like celebrating anything", "I keep thinking about what I lost", "I feel heartbroken", "It is hard to move on from this", "I feel down and withdrawn", "I wish I could turn back time", "I need some time to process this loss"],
 "anger": ["I am furious about this", "This is completely unfair", "I am losing my patience", "I feel angry and disrespected", "I cannot accept being treated this way", "This makes my blood boil", "I want an explanation right now", "I am fed up with this behaviour", "I feel like shouting", "I am irritated by the repeated excuses", "I have had enough of this", "I resent how this was handled", "I am seething inside", "This crossed a line for me", "I cannot hide my frustration", "I am outraged by what happened", "I want this unfair decision challenged", "I feel provoked and upset", "I am struggling to keep my temper", "I will not pretend this is acceptable"],
 "fear": ["I feel scared", "I am afraid something will go wrong", "My hands are shaking", "I feel nervous and unsafe", "I cannot calm my racing thoughts", "I am terrified of the outcome", "I want someone I trust beside me", "I keep imagining the worst", "I am anxious about what comes next", "I feel a knot in my stomach", "I do not dare to take another step", "I feel panicked and overwhelmed", "I am frightened by this uncertainty", "I wish I could hide somewhere safe", "I cannot slow my heartbeat", "I fear I will not be able to cope", "I am on edge right now", "I need reassurance before I can continue", "I feel a sudden sense of danger", "I am dreading the next moment"],
 "surprise": ["I did not see that coming", "I am genuinely surprised", "I had to read the message twice", "That caught me completely off guard", "I can hardly believe this happened", "I am astonished by the sudden change", "I was not expecting this at all", "I am still trying to take it in", "What an unexpected turn", "I stared at it in disbelief", "I could not have predicted that", "I am stunned by the news", "That was a bolt from the blue", "I did a double take", "I am amazed at how suddenly things changed", "I had no clue this was about to happen", "This came out of nowhere", "I am speechless at this revelation", "I was taken aback", "I need a moment to absorb this unexpected news"],
 "disgust": ["I feel disgusted", "This makes me feel sick", "I find this revolting", "I cannot stand looking at it", "I want to get away from that smell", "It is utterly repulsive", "I feel nauseated", "I recoiled as soon as I noticed it", "This is really gross", "I cannot bring myself to touch it", "I feel a strong sense of revulsion", "I had to turn away", "It makes my stomach turn", "I cannot bear that filthy scene", "I feel repelled by it", "That is nauseating to think about", "I would not go near it", "I find it deeply unpleasant and foul", "I almost gagged", "I need some fresh air after that"],
 "neutral": ["Please check the timetable", "I will send the file at noon", "The room number is on the notice", "We can discuss the details later", "Please bring the required documents", "I have written down the address", "The next update will be sent by email", "There are two copies on the desk", "I will confirm the time tomorrow", "Please use the second entrance", "The instructions are attached", "We need to fill in this section", "I have added the date to my calendar", "The office closes at five", "Please keep a copy for your records", "I will take the bus to the venue", "The agenda contains three items", "We can meet at the reception desk", "The reference number is on the first page", "I will pass this information to the group"],
}


def multilingual_rows(seed=42):
    hindi = generated_rows(seed)
    for row in hindi: row["row_id"] = "hi_" + row["row_id"]
    french, seen = [], {duplicate_key(row["text"]) for row in hindi}
    rng = random.Random(seed+1)
    for split,scenes,reactions in [("train",range(6),range(12)),("validation",range(6,8),range(12,16)),("test",range(8,10),range(16,20))]:
        for label_index,label in enumerate(EMOTION_LABELS):
            required=COUNTS[split]//7 + int(label_index < COUNTS[split]%7)
            combinations=[(scene,item,reaction) for scene in scenes for item in range(20) for reaction in reactions]
            rng.shuffle(combinations)
            count=0
            for scene,item,reaction in combinations:
                topic,icon=TOPICS[item]
                text=SCENARIOS[label][scene].format(item=topic)+". "+REACTIONS[label][reaction]+"."
                emoji=rng.random()<.4
                if emoji: text+=" "+icon
                key=duplicate_key(text)
                if key in seen:continue
                seen.add(key)
                french.append({"row_id":f"fr_{split}_{label}_{count:04d}","text":text,"emotion":label,"language_pair":"fr-en","group_id":f"{label}_scene_{scene}_item_{item}","scenario_family":f"{label}_scene_{scene}","template_family":f"{label}_reaction_{reaction}","source":"synthetic_template_fr_en_v1","annotation_status":"synthetic","human_reviewed":"false","has_emoji":str(emoji).lower(),"split":split})
                count+=1
                if count==required:break
            if count!=required:raise ValueError("Insufficient French-English combinations")
    rows=hindi+french
    rng.shuffle(rows)
    for field in ("group_id","scenario_family","template_family"):
        owners={}
        for row in rows:owners.setdefault(row[field],set()).add(row["split"])
        if any(len(value)>1 for value in owners.values()):raise ValueError("Cross-split group leakage")
    return rows


def save_multilingual(directory):
    directory=Path(directory)
    if (directory/"all_splits.csv").exists():
        audit=json.loads((directory/"dataset_audit.json").read_text(encoding="utf-8"))
        if hashlib.sha256((directory/"all_splits.csv").read_bytes()).hexdigest()!=audit["prepared_csv_sha256"]:raise ValueError("Existing bootstrap data changed; use a new version")
        return audit
    directory.mkdir(parents=True,exist_ok=True)
    rows=multilingual_rows()
    def write_set(folder,subset):
        folder.mkdir(parents=True,exist_ok=True)
        for split in (*COUNTS,"all_splits"):
            selected=subset if split=="all_splits" else [row for row in subset if row["split"]==split]
            with (folder/f"{split}.csv").open("w",encoding="utf-8",newline="") as f:
                writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(selected)
        audit={"dataset_kind":"synthetic_template","generator":"multilingual_bootstrap_v1","human_reviewed":False,"research_ready":False,"prepared_csv_sha256":hashlib.sha256((folder/"all_splits.csv").read_bytes()).hexdigest(),"language_pairs":sorted({row["language_pair"] for row in subset}),"split_sizes":dict(Counter(row["split"] for row in subset)),"counts":{split:dict(Counter(row["emotion"] for row in subset if row["split"]==split)) for split in COUNTS}}
        (folder/"dataset_audit.json").write_text(json.dumps(audit,indent=2),encoding="utf-8")
        return audit
    audit=write_set(directory,rows)
    for pair,name in [("hi-en","hindi_english"),("fr-en","english_french")]:write_set(directory/name,[row for row in rows if row["language_pair"]==pair])
    (directory/"DATASET_CARD.md").write_text("""# Multilingual synthetic bootstrap

20,000 rows: 8,000 training, 1,000 validation and 1,000 synthetic test phrases
for each language mix. The combined experiment uses 16,000/2,000/2,000 rows.
The existing Hinglish examples are retained, with prefixed row IDs. English-French
rows combine authored French situations and English reactions. No external
corpus or independent human annotations were used. These are template variants,
not 20,000 independent natural messages. Grammar, register and label shortcuts
limit generalization; human review is still required.

Scenario/reaction families and conservative cross-language group IDs are kept
in one split. French and Hindi groups are analogous authored situations, not
certified translations. Emoji describe topics rather than target emotions.
The corpus does not validate sarcasm, emotional emoji interpretation or real-world
accuracy. Never label these results as a reproduction of Team 5 G2.

Use all_splits.csv with the matching dataset_audit.json and --allow-synthetic.
The two subfolders support separate XLM-R runs. Reserve an independent reviewed
test set for the final project. Do not tune against the synthetic test partition.
""",encoding="utf-8")
    return audit
