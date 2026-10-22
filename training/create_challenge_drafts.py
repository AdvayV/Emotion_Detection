"""Create non-reviewed assistant-authored multilingual emotion challenge drafts."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import Counter
from pathlib import Path

from hinglish_emotion.evaluation_guard import phrase_hash
from hinglish_emotion.tasks import EMOTION_LABELS

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "evaluation_set"
SOURCE = ROOT / "data/training/multilingual_bootstrap_v1/all_splits.csv"

PHRASES = {
    "hi-en": {
        "joy": [
            "Aaj maa ne bina bole meri favourite chai bana di, bas din ban gaya.",
            "The scholarship email finally came and I read it three times, khushi sambhal nahi rahi.",
            "Chhoti si baarish, pakode, aur doston ka voice note — perfect Sunday yaar.",
            "I got the last seat by the window and the whole bus ride felt like a movie.",
            "Papa remembered the date of my first presentation; I’m smiling like an idiot now.",
            "Itna accha feedback mila ki puri team ka mood ek second mein lift ho gaya!",
            "My little cousin learned to say my name and kept repeating it, so cute.",
            "Train was early, cab was waiting, aur ghar pahunchte hi sab mil gaye.",
            "The old song came on just as we crossed the bridge — kya timing thi.",
            "Aaj ka lunch sabne milke banaya, kitchen messy tha par vibe top thi.",
            "She sent the exact book I mentioned months ago; dil se khush hoon.",
            "No big plan, just an evening walk with my best friend and endless bakwaas.",
            "Finally meri painting frame ho gayi; room dekhte hi alag lag raha hai.",
            "The surprise call from nani made my entire week, sach mein.",
        ],
        "sadness": [
            "Aaj uski kursi khali thi aur meeting ke baad office bahut suna laga.",
            "I kept the old ticket, but the trip we planned together never happened.",
            "Meri awaaz sunke bhi usne pehchana nahi; thoda dil toot gaya.",
            "The farewell ended with jokes, then I got home and the silence hit hard.",
            "Papa ka favourite cup toot gaya, pata nahi kyun usse dekh ke rona aa gaya.",
            "I missed the call by one minute and now the number is switched off.",
            "Itni mehnat ke baad bhi unhone mera naam list se hata diya, bas thak gayi hoon.",
            "Our usual chai stall is closed since he moved away; evenings feel incomplete.",
            "Mujhe laga woh rukega, par station par bina mudhe chala gaya.",
            "The photo is still pinned up, though we don't talk anymore.",
            "Aaj ghar lautkar kisi se baat karne ka mann hi nahi hua.",
            "Her last message was only 'reached'; I wish we had one more conversation.",
            "The team celebrated, but the person who taught me everything wasn't there.",
            "Kal ka plan cancel hua toh bura laga, par uski majboori bhi samajh rahi hoon.",
        ],
        "anger": [
            "Deadline badal di aur kisi ne bataya bhi nahi — this is so unfair.",
            "Mera parcel phir kisi aur building mein chhod diya, ab bas ho gaya.",
            "He took credit for my slide in front of everyone; sach mein gussa aa raha hai.",
            "Queue mein ghante khade rahe, phir counter band karke chale gaye. Seriously?",
            "Tumne meri baat beech mein kaati aur phir wahi idea apna bol diya.",
            "The fee was charged twice and support keeps sending the same useless reply.",
            "Har baar rules mere liye alag kyun? I am done pretending this is okay.",
            "Road pe horn bajate rehne se traffic thodi gayab ho jayega, irritate kar diya.",
            "I asked for one clear update, not another week of excuses.",
            "Mere naam se galat complaint daal di; bina pooche koi aisa kaise kar sakta hai?",
            "They left the shared kitchen filthy again, and somehow I am asked to clean it.",
            "The app logged me out after payment and now the order is nowhere to be found.",
            "Itni baar samjhaya phir bhi meri file delete kar di, unbelievable carelessness.",
            "You promised to return the keys yesterday; ab mujhe lock change karwana padega.",
        ],
        "fear": [
            "Lift beech floor par ruk gayi, lights bhi blink kar rahi hain — koi hai?",
            "The test result is due tonight and I can't stop imagining the worst.",
            "Raat ko corridor se awaaz aayi; main darwaza kholne ki himmat nahi kar pa raha.",
            "My little brother hasn't answered for two hours and his phone is off.",
            "Bridge par gaadi fisli thi, ab baarish mein drive karne ka soch ke haath kaanp rahe hain.",
            "Kahin galat file submit toh nahi ki? The deadline is in ten minutes.",
            "There is a strange burning smell near the plug; should we leave the room?",
            "Doctor said 'we need another scan' and bas tab se dimag ruk nahi raha.",
            "The dog behind the gate is loose again; I have to pass that lane alone.",
            "Mujhe lag raha hai platform change hua, but the train is already coming.",
            "I heard footsteps on the roof and everyone else says they heard nothing.",
            "If the bank transfer went to the wrong account, I don't know how to fix this.",
            "Aaj interview hai aur blank ho gaya toh? Saans bhi fast chal rahi hai.",
            "The smoke alarm chirped twice and then stopped; something feels off.",
        ],
        "surprise": [
            "Wait, internship ka offer aaj hi? I thought they had picked someone else.",
            "Arre, tum yahan? I thought you were still in Pune!",
            "The quietest person in class volunteered to host the whole event — wow.",
            "Mere purane sketch ko gallery ne select kiya, bilkul expect nahi tha.",
            "How did the parcel reach before the tracking page even updated?",
            "Usne bina reminder ke rent bhej diya; honestly, didn't see that coming.",
            "The power came back exactly when the match reached the final over.",
            "I opened the old notebook and found a note from my school bestie inside.",
            "Teen saal baad woh same coffee shop mein mil gaya, kya coincidence.",
            "The tiny plant has a flower today! I thought it would never bloom.",
            "You already fixed the bug? Kal toh hum dono stuck the.",
            "Mujhe laga meeting online hogi, poori team office mein khadi thi.",
            "That ordinary-looking box had the exact headphones I wanted — no way.",
            "She remembered my tiny comment from last winter and brought it up today.",
        ],
        "disgust": [
            "Sink mein purana khana pada hai, smell se hi nausea aa raha hai.",
            "The public washroom floor was sticky and the tap had grime all over it.",
            "Fridge khola aur ek container se green fuzz bahar aa raha tha, yuck.",
            "Kisi ne bus seat par chewing gum chipka di; haath lagte hi ghin aa gayi.",
            "The milk tasted sour even though the carton said it was fresh.",
            "That video showed someone spitting near the food counter. I feel sick.",
            "Mere shoes ke paas kuch squishy tha, neeche dekha toh old fruit nikla.",
            "They served the plate with a hair stuck to the sauce — appetite gone.",
            "The damp towel smelled like it had been sitting there for days.",
            "Yeh drain ka paani seedha footpath par aa raha hai, kya ganda scene hai.",
            "A bug crawled across the spoon just as I was about to take a bite.",
            "The shared bottle had lipstick marks on the rim and nobody owned up.",
            "Dust and oily fingerprints covered every handle in that room.",
            "Kachre ke dabbe ka dhakkan khula tha aur flies har jagah thi, ew.",
        ],
        "neutral": [
            "The library closes at six today, so I will return the book tomorrow.",
            "Maine blue folder reception par rakh diya hai; please wahan se le lena.",
            "The meeting link is in the calendar invite under Thursday afternoon.",
            "Kal bus number 18 ki jagah 18A lena, route same hai.",
            "I moved the spare charger from the desk into the top drawer.",
            "Lunch break 1:15 par hai; uske baad spreadsheet review karenge.",
            "The parcel tracking page shows it reached the local sorting centre.",
            "Meri notebook par room number 204 likha hai, 240 nahi.",
            "The recipe asks for two cups of water and one spoon of salt.",
            "Aaj metro gate 3 se exit lena, road work gate 2 par chal raha hai.",
            "Your appointment is listed for Monday morning at the clinic.",
            "I saved the updated document with today's date in the shared folder.",
            "Class ends at four; the next session is in the same room.",
            "The parcel is addressed to flat 7B, not the ground-floor office.",
            "Kal 9:30 wali call mein pehle agenda ka second point discuss karna.",
            "There are three clean mugs on the shelf beside the kettle.",
        ],
    },
    "fr-en": {
        "joy": [
            "Maman a gardé une part de tarte pour moi, and it was still warm.",
            "The grant got approved ce matin; je relis le mail juste pour y croire.",
            "Petite pluie, café serré, vieux vinyl — franchement, soirée parfaite.",
            "I found my lost sketchbook in the train station, quelle chance !",
            "Mon frère a appris la chanson entière et il la chante dans la cuisine.",
            "The whole team clapped when the demo worked, j'avais le sourire jusqu'aux oreilles.",
            "Elle a pensé à mon anniversaire sans rappel, ça m'a vraiment touché.",
            "Le soleil est revenu juste pour notre pique-nique au bord du canal.",
            "I got the corner table with the window view, que demander de plus ?",
            "On a cuisiné tous ensemble, la cuisine était en bazar mais quelle ambiance.",
            "Mon vieux prof a reconnu ma voix au téléphone après toutes ces années.",
            "No grand plan, juste une balade avec mes amis et beaucoup de rires.",
            "Mon dessin est enfin accroché dans le salon; la pièce paraît plus vivante.",
            "The little note in the lunchbox said 'bonne chance'; j'en avais besoin.",
        ],
        "sadness": [
            "Sa chaise est restée vide pendant le repas, et personne ne savait quoi dire.",
            "I kept the ticket from our trip, même si on n'est jamais partis ensemble.",
            "Elle a entendu ma voix sans me reconnaître; ça m'a fait un drôle de vide.",
            "The goodbye was full of jokes, puis le silence à la maison m'a rattrapé.",
            "Le vieux bol de mon père s'est cassé, et je ne sais pas pourquoi ça me rend triste.",
            "I missed your call by a minute et maintenant ton téléphone est éteint.",
            "Après tout ce travail, mon nom n'est même pas sur la liste; je suis épuisée.",
            "Le café du coin a fermé depuis son départ; nos soirées n'ont plus le même goût.",
            "Je pensais qu'il resterait, mais il est parti sans se retourner.",
            "The photo is still on my shelf, pourtant on ne se parle plus.",
            "En rentrant, je n'avais envie de raconter ma journée à personne.",
            "Son dernier message disait seulement 'bien arrivé'; j'aurais voulu parler plus longtemps.",
            "Everyone celebrated, mais la personne qui m'a formée n'était plus là.",
            "Le plan de demain est annulé; je comprends, mais ça me déçoit quand même.",
        ],
        "anger": [
            "Vous avez changé l'horaire sans prévenir personne; c'est vraiment abusé.",
            "My package was left at the wrong building again, j'en ai marre.",
            "Il a présenté mon travail comme le sien devant toute l'équipe, quel culot.",
            "On a attendu une heure puis le guichet a fermé sans explication. Sérieux ?",
            "Tu m'as coupée puis tu as repris exactement mon idée comme si de rien n'était.",
            "They charged me twice et le service client m'envoie toujours la même réponse inutile.",
            "Pourquoi les règles changent juste quand ça m'arrange ? C'est n'importe quoi.",
            "Tu peux arrêter de klaxonner, le feu ne va pas passer au vert plus vite.",
            "I asked for one simple update, pas une nouvelle excuse chaque semaine.",
            "Quelqu'un a déposé une plainte en mon nom sans me demander; c'est inadmissible.",
            "La cuisine commune est encore sale et on me demande de tout nettoyer.",
            "The app logged me out after payment et ma commande a disparu.",
            "J'avais sauvegardé le dossier, mais vous l'avez supprimé sans vérifier. Bravo.",
            "You said you'd return the keys yesterday; maintenant je dois appeler le serrurier.",
        ],
        "fear": [
            "L'ascenseur s'est arrêté entre deux étages et les lumières clignotent, help.",
            "The test result arrives tonight et je n'arrive pas à penser à autre chose.",
            "J'ai entendu un bruit dans le couloir, mais je n'ose pas ouvrir la porte.",
            "My little sister hasn't answered for two hours et son portable est éteint.",
            "La voiture a glissé sur le pont; maintenant conduire sous la pluie me terrifie.",
            "Et si j'ai envoyé le mauvais fichier ? The deadline is in ten minutes.",
            "There is a burning smell près de la prise, on devrait sortir ?",
            "Le médecin a dit 'un autre examen' et depuis je tremble un peu.",
            "Le chien sans laisse est encore devant le portail, je dois passer seule.",
            "I think the platform changed, mais le train arrive déjà.",
            "J'entends des pas au-dessus alors que l'appartement du dessus est vide.",
            "If that transfer went to the wrong account, je ne sais pas quoi faire.",
            "J'ai un entretien aujourd'hui et si je perds mes mots ? J'ai le souffle court.",
            "The smoke alarm made two beeps puis plus rien; quelque chose cloche.",
        ],
        "surprise": [
            "Wait, j'ai été prise pour le stage ? Je pensais qu'ils avaient choisi quelqu'un d'autre.",
            "Mais toi ici ? I thought you were still in Lyon !",
            "La personne la plus discrète de la classe va présenter le spectacle, wow.",
            "Mon vieux dessin a été choisi par la galerie; je n'y croyais pas du tout.",
            "How did the parcel arrive before the tracking page changed?",
            "Il a payé le loyer sans rappel; honnêtement, je ne m'y attendais pas.",
            "Le courant est revenu pile au dernier point du match.",
            "I opened my old notebook et j'ai trouvé une lettre de ma meilleure amie.",
            "On s'est retrouvés dans le même café après trois ans, quelle coïncidence.",
            "The tiny plant has a flower today ! Je pensais qu'elle ne pousserait jamais.",
            "Tu as déjà réparé le bug ? Hier encore on tournait en rond.",
            "Je croyais que la réunion serait en ligne; toute l'équipe était au bureau.",
            "That plain box had exactly the headphones I wanted; incroyable.",
            "Elle se souvenait de ma remarque de l'hiver dernier et l'a ressortie aujourd'hui.",
        ],
        "disgust": [
            "Il y a un reste oublié dans l'évier; l'odeur me donne la nausée.",
            "The public toilet floor was sticky et le robinet était couvert de crasse.",
            "J'ai ouvert le pot et il y avait une couche verte dessus, beurk.",
            "Someone stuck chewing gum under the chair; j'en ai eu sur la main.",
            "Le lait avait un goût aigre alors que la date était encore bonne.",
            "That clip showed someone spitting near the food counter; quelle horreur.",
            "J'ai marché sur quelque chose de mou et c'était un vieux fruit écrasé.",
            "They brought the plate with a hair in the sauce, appétit coupé.",
            "La serviette humide sentait comme si elle était là depuis une semaine.",
            "L'eau sale du caniveau coule sur tout le trottoir, c'est immonde.",
            "A bug crawled over the spoon juste avant ma première bouchée.",
            "La bouteille commune avait des traces collantes sur le bord; personne ne répond.",
            "Every handle in that room was greasy with fingerprints, dégoûtant.",
            "La poubelle était ouverte et des mouches tournaient partout, ew.",
        ],
        "neutral": [
            "The library closes at six, donc je rendrai le livre demain.",
            "J'ai laissé le dossier bleu à l'accueil; tu peux le récupérer là-bas.",
            "The meeting link is in the calendar invite for Thursday afternoon.",
            "Demain, prends le bus 18A plutôt que le 18; c'est le même trajet.",
            "I moved the spare charger from the desk into the top drawer.",
            "La pause déjeuner est à 13 h 15; ensuite on regarde le tableau ensemble.",
            "The tracking page says the parcel reached the local sorting centre.",
            "Mon carnet indique salle 204, pas 240.",
            "The recipe calls for two cups of water and one spoon of salt.",
            "Sors par la porte 3 du métro; la porte 2 est fermée pour travaux.",
            "Your appointment is listed for Monday morning at the clinic.",
            "J'ai enregistré le document mis à jour dans le dossier partagé.",
            "Class ends at four et le prochain cours est dans la même salle.",
            "Le colis est adressé au 7B, pas au bureau du rez-de-chaussée.",
            "La réunion de 9 h 30 commencera par le deuxième point de l'ordre du jour.",
            "There are three clean mugs on the shelf beside the kettle.",
        ],
    },
}


def sha256_file(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load_training_and_guards(root=ROOT):
    source = root / "data/training/multilingual_bootstrap_v1/all_splits.csv"
    training_rows = list(csv.DictReader(source.open(encoding="utf-8-sig", newline="")))
    if len(training_rows) != 20000 or {row.get("split") for row in training_rows} != {"train", "validation", "test"}:
        raise ValueError("Overlap audit requires the frozen 20,000-row multilingual CSV with all three splits")
    training = {phrase_hash(row["text"]) for row in training_rows}
    guards = {}
    for guard_file in sorted((root / "models").glob("*/evaluation_guard.json")):
        guard = json.loads(guard_file.read_text(encoding="utf-8"))
        if guard.get("schema_version") != 1 or not guard.get("dataset_sha256") or not guard.get("split_sha256"):
            raise ValueError(f"Invalid installed model evaluation guard: {guard_file}")
        guards[guard_file.parent.name] = set(guard.get("training_validation_hashes", []))
    if not guards:
        raise ValueError("No installed model evaluation guards were found for the overlap audit")
    return source, training_rows, training, guards


def rows_for(pair, label_key="emotion"):
    if pair not in PHRASES:
        raise ValueError(f"Unsupported language pair: {pair}")
    rows = []
    for label in EMOTION_LABELS:
        for index, text in enumerate(PHRASES[pair][label], 1):
            rows.append({
                "row_id": f"draft_{pair.replace('-', '_')}_{label}_{index:03d}",
                "text": text,
                "language_pair": pair,
                label_key: label,
                "annotation_status": "assistant_draft",
                "human_reviewed": "false",
                "source": "assistant_draft",
            })
    if len(rows) != 100 or set(row["emotion"] for row in rows) != set(EMOTION_LABELS):
        raise ValueError(f"Draft for {pair} must contain exactly 100 rows and all seven labels")
    return rows


def audit_rows(rows, training_hashes, guards):
    hashes = [phrase_hash(row["text"]) for row in rows]
    if len(set(hashes)) != len(hashes):
        raise ValueError("The challenge draft contains normalized duplicate phrases")
    training_overlaps = sorted(set(hashes) & training_hashes)
    guard_overlaps = {name: len(set(hashes) & guard) for name, guard in guards.items()}
    if training_overlaps or any(guard_overlaps.values()):
        raise ValueError(f"Challenge draft overlaps known training data: training={len(training_overlaps)}, guards={guard_overlaps}")
    return {
        "rows": len(rows),
        "sha256": hashlib.sha256(json.dumps(rows, ensure_ascii=False, sort_keys=True).encode("utf-8")).hexdigest(),
        "phrase_hashes": hashes,
        "training_overlap_rows": len(training_overlaps),
        "guard_overlap_rows": guard_overlaps,
        "label_counts": dict(Counter(row["emotion"] for row in rows)),
    }


def write_csv(path, rows, unlabelled=False):
    fields = ["row_id", "text", "language_pair", "emotion", "annotation_status", "human_reviewed", "source"]
    with Path(path).open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            output = dict(row)
            if unlabelled:
                output["emotion"] = ""
            writer.writerow(output)


def build(output=OUT, root=ROOT):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    source, training_rows, training_hashes, guards = load_training_and_guards(root)
    report = {
        "schema_version": 1,
        "kind": "assistant-authored evaluation challenge draft; not a human benchmark",
        "training_source": str(source.relative_to(root)).replace("\\", "/"),
        "training_source_sha256": sha256_file(source),
        "training_source_rows": len(training_rows),
        "normalized_overlap_method": "hinglish_emotion.evaluation_guard.phrase_hash (HinglishNormalizer duplicate_key, casefold, SHA-256)",
        "warning": "Zero exact normalized phrase overlap does not establish independent authorship, statistical independence, translation independence, or human ground truth. Every row is assistant_draft and human_reviewed=false; require human review before evaluation claims.",
        "installed_model_guards": {},
        "files": {},
    }
    for name, hashes in guards.items():
        report["installed_model_guards"][name] = {"training_validation_hashes": len(hashes)}
    for pair, filename in (("hi-en", "hindi_english_100.csv"), ("fr-en", "english_french_100.csv")):
        rows = rows_for(pair)
        audit = audit_rows(rows, training_hashes, guards)
        unlabelled_rows = [dict(row, emotion="") for row in rows]
        write_csv(output / filename, rows)
        unlabelled_name = filename[:-4] + "_unlabelled.csv"
        write_csv(output / unlabelled_name, unlabelled_rows, unlabelled=True)
        report["files"][filename] = {**audit, "language_pair": pair, "human_reviewed": False}
        report["files"][unlabelled_name] = {"rows": len(unlabelled_rows), "sha256": sha256_file(output / unlabelled_name),
                                             "language_pair": pair, "labels_blank": True,
                                             "training_overlap_rows": audit["training_overlap_rows"],
                                             "guard_overlap_rows": audit["guard_overlap_rows"]}
    (output / "overlap_audit.json").write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    guide = """# Multilingual emotion evaluation challenge drafts

This folder contains two 100-row draft files and unlabelled copies for prediction:

- `hindi_english_100.csv` and `english_french_100.csv` include an assistant-proposed `emotion` label.
- `hindi_english_100_unlabelled.csv` and `english_french_100_unlabelled.csv` keep the same text with a blank `emotion` column.

The phrases were authored for this challenge draft and include code-switching, negation, spelling variation, emoji, conflicting cues, indirect emotion, and neutral factual messages. The label column is a proposal, not ground truth. Every row has `annotation_status=assistant_draft`, `source=assistant_draft`, and `human_reviewed=false`. Do not report these files as a human evaluation benchmark or use draft labels to make performance claims. Have independent annotators review and adjudicate them first.

`overlap_audit.json` records row counts and SHA-256 hashes and checks normalized exact-phrase overlap against all 20,000 rows in the frozen multilingual training CSV and every installed model's `evaluation_guard.json`. The current audit requires zero overlap. Zero exact normalized overlap does not prove independent authorship, statistical independence, translation independence, or human ground truth. Do not add these phrases to training data.
"""
    (output / "README.md").write_text(guide, encoding="utf-8")
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=OUT)
    args = parser.parse_args()
    report = build(args.output)
    print(json.dumps({"output": str(args.output), "rows": {name: value["rows"] for name, value in report["files"].items()},
                      "training_rows_audited": report["training_source_rows"], "models_audited": len(report["installed_model_guards"])}, indent=2))
