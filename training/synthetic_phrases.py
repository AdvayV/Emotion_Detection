"""Original compositional Hinglish examples; generated labels, not human annotations."""

from __future__ import annotations

from collections import Counter
import csv
import hashlib
import json
from pathlib import Path
import random

from hinglish_emotion.tasks import EMOTION_LABELS
from hinglish_emotion.text_preparation import duplicate_key

SOURCE = "synthetic_template_hinglish_v1"
COUNTS = {"train": 8000, "validation": 1000, "test": 1000}

ACTIVITIES = [
    ("maths test", "🧮"), ("physics exam", "⚛️"), ("coding interview", "💻"),
    ("driving test", "🚗"), ("college presentation", "📊"), ("dance audition", "💃"),
    ("internship interview", "💼"), ("science quiz", "🔬"), ("project review", "📁"),
    ("scholarship interview", "🎓"), ("debate round", "🎤"), ("music audition", "🎵"),
    ("language test", "📚"), ("design review", "🎨"), ("sports trial", "🏃"),
    ("viva exam", "🗣️"), ("coding contest", "⌨️"), ("research presentation", "🧪"),
    ("placement interview", "🏢"), ("semester exam", "📝"),
]
FOODS = [
    ("samosa", "🥟"), ("paratha", "🫓"), ("noodles", "🍜"), ("pasta", "🍝"),
    ("burger", "🍔"), ("sandwich", "🥪"), ("dal", "🥣"), ("rice", "🍚"),
    ("biryani", "🍛"), ("paneer", "🧀"), ("salad", "🥗"), ("soup", "🍲"),
    ("pizza", "🍕"), ("dosa", "🍽️"), ("idli", "🍽️"), ("chole", "🥣"),
    ("rajma", "🥣"), ("cake", "🍰"), ("pastry", "🧁"), ("coffee", "☕"),
]
PLACES = [
    ("library", "📚"), ("canteen", "🍽️"), ("classroom", "🏫"), ("hostel lobby", "🏠"),
    ("college gate", "🚪"), ("bus stop", "🚌"), ("station", "🚉"), ("playground", "🏏"),
    ("auditorium", "🎭"), ("study room", "📖"), ("parking area", "🚗"), ("computer lab", "💻"),
    ("seminar hall", "🎤"), ("cafeteria", "☕"), ("bookshop", "📚"), ("reception", "🏢"),
    ("garden", "🌳"), ("corridor", "🏫"), ("waiting area", "🪑"), ("practice room", "🎵"),
]
OBJECTS = [
    ("phone", "📱"), ("laptop", "💻"), ("headphones", "🎧"), ("watch", "⌚"),
    ("wallet", "👛"), ("backpack", "🎒"), ("notebook", "📓"), ("camera", "📷"),
    ("charger", "🔌"), ("keyboard", "⌨️"), ("mouse", "🖱️"), ("water bottle", "🧴"),
    ("spectacles", "👓"), ("earphones", "🎧"), ("umbrella", "☂️"), ("ID card", "🪪"),
    ("power bank", "🔋"), ("calculator", "🧮"), ("tablet", "📱"), ("speaker", "🔊"),
]

# Families 0..5 train, 6..7 validation, 8..9 test.
SCENARIOS = {
    "joy": [
        "Maine apna {item} clear kar liya", "Mere {item} ki mehnat ka achha result mil gaya",
        "Mere {item} par sabne dil se tarif ki", "Aaj {item} ka order bilkul mere taste ka bana",
        "Doston ne mere liye {item} mangwa kar chhoti si party rakhi", "Ghar par sabne mere banaye {item} ki tarif ki",
        "Hum {item} ke paas kaafi din baad mile", "{item} ke paas sab doston ne mera welcome kiya",
        "Mujhe gift mein {item} mila", "Maine {item} repair karwaya aur ab woh perfectly kaam kar raha hai",
    ],
    "sadness": [
        "Main apna {item} clear nahi kar paya", "Mere {item} ki saari mehnat ke baad bhi selection nahi hua",
        "Jis {item} ka main wait kar raha tha woh cancel ho gaya", "{item} ka order akela kha raha hoon aur purane dost yaad aa rahe hain",
        "Dadi ke haath ke {item} ki yaad aa gayi aur ab woh saath nahi hain", "Maine sabke liye {item} banaya par koi milne nahi aaya",
        "{item} ke paas ab hamara purana group nahi milta", "Aaj {item} ke paas apne dost ko goodbye kaha",
        "Mera kho gaya {item} ab tak wapas nahi mila", "{item} mein jo purani memories thi woh sab delete ho gayi",
    ],
    "anger": [
        "Mere {item} ke rules mere saath unfairly badal diye gaye", "{item} mein mere kaam ka credit kisi aur ko de diya",
        "Mere {item} ke waqt staff ne bina wajah rude behaviour kiya", "{item} ke bill mein phir se extra paise jod diye",
        "Maine {item} ke liye pay kiya aur restaurant ne order dene se mana kar diya", "{item} galat aane par staff ne meri complaint ka mazaak banaya",
        "{item} ke paas kisi ne meri line tod kar mujhe hi blame kiya", "{item} ke paas guard ne bina wajah mujh par chillaya",
        "Kisi ne mera {item} bina poochhe use karke tod diya", "{item} ka repair charge le liya par dukaan ne kuch bhi repair nahi kiya",
    ],
    "fear": [
        "Kal mera {item} hai aur fail hone ka khayal baar baar aa raha hai", "{item} se pehle lag raha hai ki main sab bhool jaunga",
        "{item} mein stage par akela bolne ka waqt aa gaya", "{item} banate waqt gas ki tez smell aayi aur maine door se help bulayi",
        "{item} khaane ke baad dost ki tabiyat bigdi aur ab main outcome ko lekar worried hoon", "{item} lene gaya tha tab paas se ek tez awaaz aayi",
        "Raat ko {item} ke paas mujhe laga koi mera peecha kar raha hai", "{item} ke paas main apne group se bichhad gaya aur jagah anjaan thi",
        "Mujhe laga {item} kho gaya aur usmein zaroori documents hain", "{item} se achanak tez awaaz aayi aur main peeche hat gaya",
    ],
    "surprise": [
        "{item} ka result expected date se pehle aa gaya", "Mujhe bina bataye {item} ka winner announce kar diya gaya",
        "{item} mein woh sawal aaya jo kisi ne expect nahi kiya tha", "{item} ka parcel khola toh andar ek extra dish nikli",
        "Maine sirf {item} order kiya tha aur saath mein handwritten note mila", "{item} ki recipe ka ek ingredient bilkul alag nikla",
        "{item} ke paas achanak mera school friend mil gaya", "{item} ke paas bina announcement ek famous guest aa gaya",
        "{item} ka parcel khola toh woh meri expected cheez se bilkul alag tha", "Purane {item} ke andar ek note mila jiske baare mein mujhe pata hi nahi tha",
    ],
    "disgust": [
        "Jahan {item} hua wahan desks par chipchipa kachra pada tha", "{item} ki jagah par log khule mein thook rahe the",
        "{item} ke baad kisi ne gande tissues meri desk par chhod diye", "{item} ke andar fungus nazar aayi",
        "Plate mein pada {item} kharab smell kar raha tha", "Kisi ne floor par gira {item} wapas serving plate mein rakh diya",
        "{item} ke paas sadi hui cheezon ki badboo aa rahi thi", "{item} ke paas washroom ka ganda paani jama tha",
        "{item} ko haath lagaya toh us par purana chipchipa mail tha", "Jo {item} mujhe mila us par thook ke nishaan the",
    ],
    "neutral": [
        "Mera {item} kal do baje scheduled hai", "{item} ke liye registration form online available hai",
        "{item} ka venue notice board par likha hai", "Maine menu mein {item} ka naam note kiya",
        "{item} ka order counter number do par collect karna hai", "Recipe mein {item} ka portion alag bowl mein rakhna hai",
        "{item} ke paas meeting point mark kiya gaya hai", "{item} ke paas bus ka timing chart laga hai",
        "{item} desk ke left side par rakha hai", "Maine {item} ka model number form mein enter kar diya",
    ],
}

# 12 training, 4 validation, 4 test reaction families per emotion.
# Each reaction contains an English fragment alongside the Hindi scenario.
REACTIONS = {
    "joy": [
        "Main bahut khush hoon and I want to celebrate", "Dil khush ho gaya yaar, this feels wonderful",
        "Aaj chehre se smile hat hi nahi rahi", "Finally main relaxed hoon and genuinely happy",
        "Mujhe itni khushi ho rahi hai, I cannot stop smiling", "It feels like all that effort was worth it",
        "Aaj toh bas celebrate karne ka mann hai", "Honestly yeh moment mere liye bahut special hai",
        "Main excitement mein sabko yeh news batana chahta hoon", "Ab tension nahi, bas happiness feel ho rahi hai",
        "Aaj ka mood ekdum mast hai, what a lovely feeling", "I feel grateful aur dil se khush hoon",
        "Mann kar raha hai is good news ko sabke saath share karun", "Itna achha lag raha hai ki din ban gaya",
        "I am delighted, aaj sach mein bahut achha din hai", "Is moment ko enjoy karna chahta hoon, bahut khushi hai",
        "I feel so cheerful, apni smile control nahi kar paa raha", "Dil se maza aa gaya, I am really pleased",
        "Yeh sun kar andar se khushi aa rahi hai, such a good moment", "I want to remember this moment, mann bahut khush hai",
    ],
    "sadness": [
        "Dil bahut heavy lag raha hai, I feel sad", "Main udaas hoon and I do not feel like talking",
        "Aankhon mein aansu aa rahe hain, this really hurts", "I feel disappointed, mann bilkul baith gaya",
        "Bas chup rehne ka mann hai, my heart feels heavy", "Bahut dukh ho raha hai, I was hoping for something better",
        "I miss those days, ab sab khaali sa lagta hai", "Mood bahut down hai aur rona aa raha hai",
        "I feel heartbroken, samajh nahi aa raha kya bolun", "Khush hone ki koshish kar raha hoon but I feel low",
        "Is baat ne andar se dukhi kar diya, I feel let down", "Ab kisi cheez mein mann nahi lag raha, it feels lonely",
        "I am grieving, yeh kami bahut mehsoos ho rahi hai", "Dil toot sa gaya, this is painful",
        "I feel miserable, aaj bas aansu ruk nahi rahe", "Mann udaas hai aur I need some quiet time",
        "It feels like something important is gone, dil baith gaya", "Main isse move on nahi kar paa raha, I feel sorrow",
        "I am deeply disappointed aur andar se khaali lag raha hai", "Rone ka mann ho raha hai, I really miss what we had",
    ],
    "anger": [
        "Mujhe bahut gussa aa raha hai, this is unfair", "I am furious, yeh bilkul theek nahi hua",
        "Main itna naraz hoon ki ab jawab dena chahta hoon", "Honestly yeh behaviour bardasht nahi ho raha",
        "I feel angry, baar baar mere saath hi kyun", "Mera patience khatam ho gaya, this is unacceptable",
        "Gusse mein haath kaanp rahe hain, I am really upset about the unfairness", "I am irritated, ab bas bahut ho gaya",
        "Yeh dekh kar khoon khol raha hai, how unfair", "Main chup nahi rehna chahta, I am annoyed",
        "Mujhe blame karna band karo, I am so angry", "Ab mazaak nahi lag raha, I am losing my temper",
        "Main is unfair treatment par bahut bhadka hua hoon", "I feel outraged, aisa karna bilkul galat hai",
        "Yeh soch kar hi gussa badh raha hai, what nonsense", "I am fuming aur ab iski complaint karunga",
        "Main bahut chidh gaya hoon, this treatment makes me mad", "Mera temper control karna mushkil ho raha hai, I am furious",
        "I am angry about this injustice, mujhe jawab chahiye", "Narazgi kam nahi ho rahi, this was completely unfair",
    ],
    "fear": [
        "Mujhe darr lag raha hai and I feel scared", "I am afraid, kuch galat na ho jaye",
        "Dil tez dhadak raha hai, I feel frightened", "Main nervous hoon aur safe feel nahi kar raha",
        "I feel anxious, dimaag mein bas buri possibilities aa rahi hain", "Mujhe dar hai ki main handle nahi kar paunga",
        "I am terrified, kisi apne ko paas chahta hoon", "Ghabrahat ho rahi hai, I cannot calm down",
        "Main himmat dikhana chahta hoon but I feel afraid", "I feel uneasy, har awaaz par chaunk raha hoon",
        "Dar ki wajah se body tense ho gayi hai, I feel unsafe", "I keep worrying, kahin kuch bura na ho",
        "Mere andar ghabrahat badh rahi hai, I feel fearful", "I am dreading what comes next, bahut dar lag raha hai",
        "Main safe jagah jaana chahta hoon, I feel frightened", "Soch kar hi paseena aa raha hai, I am scared",
        "Mujhe protection chahiye, I feel threatened", "I am full of apprehension, mann mein dar baitha hai",
        "Main panic feel kar raha hoon, kuch samajh nahi aa raha", "I feel alarmed aur bas kisi ko saath bulana chahta hoon",
    ],
    "surprise": [
        "Main hairan hoon, I did not expect this", "I am surprised, yeh kab ho gaya",
        "Ek second ke liye bolna bhool gaya, what a surprise", "Sach mein, I was not expecting that at all",
        "Mujhe yakeen hi nahi ho raha, this is unexpected", "I am astonished, yeh toh socha hi nahi tha",
        "Achanak se sab badal gaya, I feel taken aback", "I am shocked by the sudden news, yeh kaise hua",
        "Maine dobara check kiya because I could not believe it", "Main bas hairani se dekh raha hoon, that was sudden",
        "Is twist ne mujhe chaunka diya, I am amazed", "I need a moment to process this, bilkul unexpected tha",
        "I am stunned, aisi news ka koi andaaza nahi tha", "Yeh toh ekdum achanak hua, what an unexpected turn",
        "Aankhen khuli reh gayi, I was caught off guard", "I am baffled by the sudden change, yeh kya hua",
        "I did a double take, itni hairani ho rahi hai", "Mujhe pehle koi hint nahi mila, I am genuinely surprised",
        "It was a bolt from the blue, main abhi tak hairan hoon", "I am speechless from surprise, kabhi imagine nahi kiya tha",
    ],
    "disgust": [
        "Mujhe ghin aa rahi hai, this is disgusting", "I feel disgusted, isse door rehna chahta hoon",
        "Yeh dekh kar ulti jaisa lag raha hai, so gross", "I find this revolting, haath bhi nahi lagana",
        "Itna ghinauna laga ki main peeche hat gaya", "I feel repulsed, yeh dekhna bhi mushkil hai",
        "Ghin se chehra bigad gaya, this is nasty", "I cannot stand it, bahut ganda lag raha hai",
        "Main nazar hata raha hoon, it is sickening", "I feel an aversion, mann kharab ho gaya",
        "Yeh bilkul unhygienic lag raha hai, mujhe ghin hai", "I am grossed out, iske paas nahi jaana",
        "Isse dekh kar stomach turn ho gaya, kitni ghin aa rahi hai", "I feel revulsion, yeh nazara bardasht nahi hota",
        "Main door hat gaya, that was repulsive", "Itna mail dekh kar ghin aa rahi hai, absolutely revolting",
        "I feel sick just looking at it, bahut ghinauna hai", "Ghin ki wajah se paas jaana nahi chahta, so filthy",
        "I am repelled, is cheez se ajeeb si ghin aa rahi hai", "Mann mein strong disgust aa raha hai, I cannot look at it",
    ],
    "neutral": [
        "Bas itni information deni thi, that is the update", "Maine details note kar li, nothing else to add",
        "Yeh schedule ki information hai, I have noted it", "Main bas timing confirm kar raha hoon",
        "The details are recorded, baaki baat baad mein", "Maine list update kar di, that is all",
        "Is message mein sirf practical information hai", "I am sharing the location, aur kuch nahi",
        "Bas status bata raha hoon, there is no further update", "I checked the information aur note kar li",
        "Main instruction repeat kar raha hoon, please note the details", "Yeh factual update hai, I have recorded it",
        "I am confirming the arrangement, bas itni baat hai", "Maine entry kar di, the record is updated",
        "Yeh routine information hai, please keep it on file", "I have written down the details, bas confirm karna tha",
        "I am passing on the information, ab details available hain", "Maine required detail enter kar di, the form is complete",
        "Yeh sirf ek note hai, I am reporting the status", "I am stating the arrangement, koi aur update nahi hai",
    ],
}


def generated_rows(seed=42):
    rng = random.Random(seed)
    all_rows, seen = [], set()
    partitions = {"train": (range(6), range(12)), "validation": (range(6, 8), range(12, 16)), "test": (range(8, 10), range(16, 20))}
    for split, (scene_indices, reaction_indices) in partitions.items():
        for label_index, label in enumerate(EMOTION_LABELS):
            required = COUNTS[split] // 7 + int(label_index < COUNTS[split] % 7)
            candidates = []
            for scene in scene_indices:
                pool = ACTIVITIES if scene < 3 else FOODS if scene < 6 else PLACES if scene < 8 else OBJECTS
                for item_index, (item, icon) in enumerate(pool):
                    for reaction in reaction_indices:
                        context = SCENARIOS[label][scene].format(item=item)
                        core = f"{context}. {REACTIONS[label][reaction]}"
                        candidates.append((core, scene, item_index, reaction, icon))
            rng.shuffle(candidates)
            selected = 0
            for core, scene, item_index, reaction, icon in candidates:
                key = duplicate_key(core)
                if key in seen:
                    continue
                seen.add(key)
                # The optional icon describes the situation/item, not the emotion label.
                has_emoji = rng.random() < 0.4
                text = core + rng.choice((".", "!", ".")) + (f" {icon}" if has_emoji else "")
                all_rows.append({
                    "row_id": f"syn_{split}_{label}_{selected:04d}", "text": text,
                    "emotion": label, "language_pair": "hi-en",
                    "group_id": f"{label}_scene_{scene}_item_{item_index}",
                    "scenario_family": f"{label}_scene_{scene}",
                    "template_family": f"{label}_reaction_{reaction}",
                    "source": SOURCE, "annotation_status": "synthetic",
                    "human_reviewed": "false", "has_emoji": str(has_emoji).lower(), "split": split,
                })
                selected += 1
                if selected == required:
                    break
            if selected != required:
                raise ValueError(f"Insufficient unique candidates for {split}/{label}: {selected}/{required}")
    rng.shuffle(all_rows)
    return all_rows


def save_dataset(directory, seed=42):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    rows = generated_rows(seed)
    for split in (*COUNTS, "all_splits"):
        subset = rows if split == "all_splits" else [row for row in rows if row["split"] == split]
        with (directory / f"{split}.csv").open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(subset)
    report = {
        "dataset_kind": "synthetic_template", "generator": SOURCE, "seed": seed,
        "human_reviewed": False, "research_ready": False,
        "external_datasets_used": [], "language_pairs": ["hi-en"],
        "prepared_csv_sha256": hashlib.sha256((directory / "all_splits.csv").read_bytes()).hexdigest(),
        "counts": {split: dict(Counter(row["emotion"] for row in rows if row["split"] == split)) for split in COUNTS},
        "split_sizes": COUNTS,
        "emoji_rows": {split: sum(row["split"] == split and row["has_emoji"] == "true" for row in rows) for split in COUNTS},
        "limitations": [
            "Labels follow authored template intent and have not been independently human-validated.",
            "Rows are combinatorial variants of 140 reaction templates and 70 scenario templates, not 10000 independently authored utterances.",
            "Scenario and reaction template families are separated by split; shared vocabulary and synthetic style still limit independence.",
            "Emoji describe context/items and are not human annotations of text–emotion–emoji relationships.",
            "Synthetic test accuracy is not real-world accuracy. Sarcasm and genuinely mixed emotions are not adequately represented.",
            "French–English is not included.",
        ],
    }
    for field in ("group_id", "scenario_family", "template_family"):
        owners = {}
        for row in rows:
            owners.setdefault(row[field], set()).add(row["split"])
        assert all(len(splits) == 1 for splits in owners.values()), field
    assert len({duplicate_key(row["text"]) for row in rows}) == len(rows)
    report["duplicate_phrases"] = 0
    report["cross_split_group_overlaps"] = 0
    report["cross_split_template_family_overlaps"] = 0
    (directory / "dataset_audit.json").write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    card = """# Synthetic Hinglish emotion dataset

Exactly 8,000 training phrases, 1,000 validation phrases and 1,000 synthetic test
phrases. Seven classes are nearly balanced. All examples are original generated
Hinglish template compositions. No MaSaC or other external corpus is included.

Labels are synthetic author-intent labels, not independent human annotations.
The dataset enables an experimental training run; its test scores do not establish
real-world performance or research novelty. Template reuse limits linguistic
diversity despite unique phrase strings. See dataset_audit.json for the exact
generation method, class counts, hashes and limitations.

Scenario and reaction families are partitioned before sampling. Related variants
stay in one split. Optional emoji describe items/situations independently of the
emotion label. This is not a validated emoji-effectiveness benchmark.

Use all_splits.csv with the trainer so validation and test stay separate.
Use --allow-synthetic explicitly. Checkpoint metadata will retain synthetic-only
evaluation scope and no verified language accuracy. Later evaluate the trained
checkpoint on independent, appropriately labelled human data.
"""
    (directory / "DATASET_CARD.md").write_text(card, encoding="utf-8")
    return report
