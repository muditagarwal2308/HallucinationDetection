"""Build the verse-level Gita QA dataset and the labelled hallucination test set.

Outputs
-------
data/gita/qa.jsonl                     questions with gold verses and reference answers
data/gita/hallucination_testset.jsonl  (question, answer, label, type) pairs

The reference answers are original paraphrases written for this project and
checked against Besant's translation; quotations reproduce Besant's (public
domain) wording or the IAST text exactly. Hallucinated answers are produced from
the reference answers by the controlled perturbations in
``halludetect.gita.perturb`` with a fixed seed, so the test set is reproducible.
"""

from __future__ import annotations

import json
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from halludetect.gita import (
    GitaCorpus,  # noqa: E402
    perturb,  # noqa: E402
)

# (id, category, question, gold verses, reference answer)
ANSWERABLE = [
    ("c01", "verse_lookup", "Which verse of the Bhagavad Gita says that weapons cannot cut the soul and fire cannot burn it?", ["2.23"],
     "The Gita says that weapons cannot cut the Self, fire cannot burn it, water cannot wet it and wind cannot dry it (BG 2.23)."),
    ("c02", "concept", "What does Krishna teach about one's right to action and its fruits?", ["2.47"],
     "Krishna teaches that one's concern is with action alone and never with its fruits; the fruit should not be the motive, nor should one be attached to inaction (BG 2.47)."),
    ("c03", "concept", "How does the Gita compare the soul's taking of new bodies to changing clothes?", ["2.22"],
     "Just as a man casts off worn-out garments and takes new ones, the dweller in the body casts off worn-out bodies and enters others that are new (BG 2.22)."),
    ("c04", "concept", "What does the Gita say about the soul being born or dying?", ["2.20"],
     "The Gita says that the Self is not born and does not die; it is unborn, eternal and ancient, and it is not slain when the body is slain (BG 2.20)."),
    ("c05", "concept", "What does Krishna promise to do when righteousness declines?", ["4.7", "4.8"],
     "Krishna says that whenever righteousness decays and unrighteousness is exalted, he comes forth (BG 4.7). He is born from age to age to protect the good, to destroy evil-doers and to establish righteousness (BG 4.8)."),
    ("c06", "concept", "What is Krishna's final instruction about surrender?", ["18.66"],
     "Krishna tells Arjuna to abandon all duties and come to him alone for shelter, promising to liberate him from all sins and telling him not to sorrow (BG 18.66)."),
    ("c07", "concept", "How does the Gita define yoga as equanimity?", ["2.48"],
     "Krishna asks Arjuna to perform action renouncing attachment and balanced evenly in success and failure, and says that this equilibrium is called yoga (BG 2.48)."),
    ("c08", "quote", "Which verse says that yoga is skill in action?", ["2.50"],
     "The statement \"yoga is skill in action\" appears in BG 2.50, where Krishna asks Arjuna to cleave to yoga."),
    ("c09", "concept", "Why should one not grieve over death according to the Gita?", ["2.27"],
     "Death is certain for the born and birth is certain for the dead, so one should not grieve over the inevitable (BG 2.27)."),
    ("c10", "concept", "How does dwelling on sense objects lead to ruin?", ["2.62", "2.63"],
     "Musing on the objects of sense creates attachment; from attachment arises desire, and from desire anger (BG 2.62). From anger comes delusion, from delusion confused memory, and from the destruction of reason one perishes (BG 2.63)."),
    ("c11", "concept", "What image does Krishna use for withdrawing the senses from their objects?", ["2.58"],
     "Krishna compares the sage to a tortoise that draws in its limbs: when one withdraws the senses from the objects of sense, the understanding is well poised (BG 2.58)."),
    ("c12", "concept", "Why is one's own duty better than the duty of another?", ["3.35"],
     "Krishna says that one's own duty, though destitute of merit, is better than the duty of another well discharged, because the duty of another is full of danger (BG 3.35)."),
    ("c13", "concept", "Can anyone remain without action, according to the Gita?", ["3.5"],
     "No one can remain actionless even for an instant, because everyone is driven to action by the qualities born of nature (BG 3.5)."),
    ("c14", "concept", "What does Krishna say about the example set by great people?", ["3.21"],
     "Whatever a great man does, other men also do; people follow the standard he sets (BG 3.21)."),
    ("c15", "speaker", "What drives a person to commit sin even against his will?", ["3.36", "3.37"],
     "Arjuna asks what drives a man to commit sin reluctantly, as if constrained by force (BG 3.36). Krishna answers that it is desire and wrath, the all-consuming foe of man on earth (BG 3.37)."),
    ("c16", "verse_lookup", "Which verse says that the fire of wisdom reduces all actions to ashes?", ["4.37"],
     "BG 4.37 says that as a burning fire reduces fuel to ashes, so the fire of wisdom reduces all actions to ashes."),
    ("c17", "concept", "How does Krishna receive those who approach him?", ["4.11"],
     "Krishna says that however men approach him, even so does he welcome them, for every path that men take is his (BG 4.11)."),
    ("c18", "concept", "What does the Gita say is the greatest purifier?", ["4.38"],
     "There is no purifier in this world like wisdom, and one perfected in yoga finds it in the Self in due season (BG 4.38)."),
    ("c19", "concept", "How does a sage look upon a learned Brahmana, a cow, an elephant, a dog and an outcaste?", ["5.18"],
     "Sages look equally on a Brahmana adorned with learning and humility, a cow, an elephant, a dog and an outcaste (BG 5.18)."),
    ("c20", "concept", "How can the self be one's friend or one's enemy?", ["6.5", "6.6"],
     "One should raise the self by the Self and not let the self become depressed, for the Self is the friend of the self and also its enemy (BG 6.5). To the one who has conquered himself the Self is a friend, but to the unsubdued self it becomes hostile like an enemy (BG 6.6)."),
    ("c21", "concept", "What simile describes the steady mind of a yogi?", ["6.19"],
     "The yogi of subdued thought is likened to a lamp in a windless place that does not flicker (BG 6.19)."),
    ("c22", "concept", "What does the Gita say about eating and sleeping for a yogi?", ["6.16", "6.17"],
     "Yoga is not for one who eats too much or abstains too much, nor for one who sleeps too much or keeps awake too much (BG 6.16). Yoga destroys all pain for one who is regulated in eating, amusement, actions, sleeping and waking (BG 6.17)."),
    ("c23", "speaker", "How can the restless mind be controlled?", ["6.34", "6.35"],
     "Arjuna says that the mind is restless, impetuous and strong, as hard to curb as the wind (BG 6.34). Krishna agrees that the mind is hard to curb, but says it can be curbed by constant practice and by dispassion (BG 6.35)."),
    ("c24", "concept", "What happens to a yogi who falls from the path?", ["6.40", "6.41"],
     "Krishna says there is no destruction for him in this world or in the life to come, for no one who works righteousness treads the path of woe (BG 6.40). Having dwelt in the worlds of the pure-doing, he who fell from yoga is reborn in a pure and blessed house (BG 6.41)."),
    ("c25", "concept", "What are the eight divisions of Krishna's lower nature?", ["7.4"],
     "Earth, water, fire, air, ether, mind, reason and egoism are the eightfold division of Krishna's nature (BG 7.4)."),
    ("c26", "concept", "Which four kinds of people worship Krishna?", ["7.16"],
     "Four kinds of righteous people worship Krishna: the suffering, the seeker for knowledge, the self-interested and the wise (BG 7.16)."),
    ("c27", "concept", "What happens to one who remembers Krishna at the time of death?", ["8.5"],
     "Whoever casts off the body thinking of Krishna alone at the time of the end enters into his being; there is no doubt of that (BG 8.5)."),
    ("c28", "concept", "What offerings does Krishna accept?", ["9.26"],
     "Krishna accepts a leaf, a flower, a fruit or water when it is offered to him with devotion (BG 9.26)."),
    ("c29", "concept", "What does Krishna promise to those who worship him alone?", ["9.22"],
     "To those who worship him alone, thinking of no other, and who are ever harmonious, Krishna brings full security (BG 9.22)."),
    ("c30", "verse_lookup", "Which verse says that Krishna is the beginning, middle and end of all beings?", ["10.20"],
     "In BG 10.20 Krishna says that he is the Self seated in the heart of all beings, and the beginning, the middle and the end of all beings."),
    ("c31", "concept", "Among the Vedas, which one does Krishna identify with?", ["10.22"],
     "Krishna says that of the Vedas he is the Sama Veda, and of the senses he is the mind (BG 10.22)."),
    ("c32", "quote", "What does Krishna call himself when he reveals his universal form?", ["11.32"],
     "Krishna declares, \"Time am I, laying desolate the world\" (BG 11.32)."),
    ("c33", "concept", "Why does Krishna give Arjuna a divine eye?", ["11.8"],
     "Because Arjuna is not able to behold Krishna with his own eyes, Krishna gives him the divine eye so that he may behold his sovereign yoga (BG 11.8)."),
    ("c34", "concept", "Whom does Krishna consider the best in yoga?", ["12.2"],
     "Krishna says that those who fix their minds on him and worship him, ever harmonised and endowed with supreme faith, are the best in yoga (BG 12.2)."),
    ("c35", "concept", "What qualities make a devotee dear to Krishna?", ["12.13", "12.14"],
     "Such a devotee bears no ill-will to any being, is friendly and compassionate, free from attachment and egoism, balanced in pleasure and pain and forgiving (BG 12.13). Ever content and with mind and reason dedicated to Krishna, that devotee is dear to him (BG 12.14)."),
    ("c36", "concept", "What is called the Field in the Bhagavad Gita?", ["13.2"],
     "Krishna says that this body is called the Field, and that which knows it is called the Knower of the Field by the sages (BG 13.2)."),
    ("c37", "concept", "What are the three qualities born of nature?", ["14.5"],
     "Harmony, motion and inertia are the three qualities born of matter, and they bind the indestructible dweller in the body (BG 14.5)."),
    ("c38", "concept", "How does the Gita describe the Ashvattha tree?", ["15.1"],
     "The Ashvattha tree is described with its roots above and branches below; its leaves are the hymns, and whoever knows it is a knower of the Veda (BG 15.1)."),
    ("c39", "concept", "What lights the supreme abode of Krishna?", ["15.6"],
     "Neither the sun, nor the moon, nor fire lights Krishna's supreme abode, and those who go there do not return (BG 15.6)."),
    ("c40", "concept", "What are the three gates to hell?", ["16.21"],
     "The triple gate of hell, destructive of the self, is lust, wrath and greed, so one should renounce these three (BG 16.21)."),
    ("c41", "concept", "What are the three kinds of faith?", ["17.2"],
     "The inborn faith of embodied beings is threefold by nature: pure, passionate and dark (BG 17.2)."),
    ("c42", "concept", "Which foods are dear to the pure (sattvic) person?", ["17.8"],
     "Foods that augment vitality, energy, vigour, health, joy and cheerfulness, and that are delicious, bland, substantial and agreeable, are dear to the pure (BG 17.8)."),
    ("c43", "concept", "How does the Gita distinguish renunciation from relinquishment?", ["18.2"],
     "Sages call the renouncing of works done with desire renunciation, and the wise call the relinquishing of the fruit of all actions relinquishment (BG 18.2)."),
    ("c44", "speaker", "How does the Bhagavad Gita end?", ["18.78"],
     "The Gita ends with Sanjaya saying that wherever Krishna, the lord of yoga, and Arjuna, the archer, are present, there are prosperity, victory and happiness (BG 18.78)."),
    ("c45", "speaker", "What does Arjuna say at the end after hearing Krishna's teaching?", ["18.73"],
     "Arjuna says that his delusion is destroyed, that his doubts have fled away, and that he will act according to Krishna's word (BG 18.73)."),
    ("c46", "speaker", "Who asks Sanjaya what happened on the field of Kurukshetra?", ["1.1"],
     "Dhritarashtra asks Sanjaya what his people and the Pandavas did when they gathered on the holy plain of Kurukshetra, eager for battle (BG 1.1)."),
    ("c47", "concept", "Which conch does Krishna blow at the start of the battle?", ["1.15"],
     "Krishna, as Hrishikesha, blows the conch Panchajanya, while Arjuna blows Devadatta (BG 1.15)."),
    ("c48", "speaker", "What does Arjuna describe when he sees his kinsmen ready to fight?", ["1.29", "1.30"],
     "Arjuna says that his limbs fail, his mouth is parched, his body quivers and his hair stands on end (BG 1.29). He says that the Gandiva slips from his hand, his skin burns and his mind is whirling (BG 1.30)."),
    ("c49", "quote", "Who says \"I will not fight\" in the Gita?", ["2.9"],
     "Sanjaya reports that Arjuna said to Govinda, \"I will not fight!\", and then became silent (BG 2.9)."),
    ("c50", "speaker", "Who asks why Krishna enjoins a terrible action if knowledge is superior to action?", ["3.1"],
     "Arjuna asks Krishna why, if knowledge is thought superior to action, he enjoins on him this terrible action (BG 3.1)."),
    ("c51", "speaker", "Who says that the dialogue between Krishna and Arjuna made his hair stand on end?", ["18.74"],
     "Sanjaya says that he heard the marvellous dialogue of Vasudeva and the great-souled Partha, and that it caused his hair to stand on end (BG 18.74)."),
    ("c52", "concept", "What does the Gita say about seeing inaction in action?", ["4.18"],
     "He who sees inaction in action and action in inaction is wise among men and harmonious even while performing all action (BG 4.18)."),
    ("c53", "concept", "How does one who acts without attachment remain untouched by sin?", ["5.10"],
     "One who acts placing all actions in the Eternal and abandoning attachment is unaffected by sin, as a lotus leaf is unaffected by water (BG 5.10)."),
    ("c54", "concept", "Who is a true yogi and ascetic according to Krishna?", ["6.1"],
     "Krishna says that one who performs his duty independently of the fruit of action is the true ascetic and yogi, not one who is merely without fire and rites (BG 6.1)."),
    ("c55", "concept", "How does Krishna describe the sage of stable mind?", ["2.56"],
     "A sage of stable mind is one whose mind is free from anxiety amid pains, who is indifferent amid pleasures, and who is loosed from passion, fear and anger (BG 2.56)."),
    ("c56", "concept", "How should one bear cold and heat, pleasure and pain?", ["2.14"],
     "Krishna says that the contacts of matter giving cold and heat, pleasure and pain come and go and are impermanent, so one should endure them bravely (BG 2.14)."),
    ("c57", "concept", "What happens to the doubting person?", ["4.40"],
     "The ignorant, faithless, doubting self goes to destruction; neither this world nor the world beyond nor happiness exists for the doubting self (BG 4.40)."),
    ("c58", "concept", "Who created the four castes according to the Gita?", ["4.13"],
     "Krishna says that the four castes were emanated by him according to the distribution of qualities and actions (BG 4.13)."),
    ("c59", "concept", "How are food, rain, sacrifice and action connected?", ["3.14"],
     "Creatures come from food, food is produced from rain, rain proceeds from sacrifice, and sacrifice arises out of action (BG 3.14)."),
    ("c60", "concept", "How is Krishna present in water, the moon and the sun?", ["7.8"],
     "Krishna says that he is the sapidity in waters, the radiance in moon and sun, the Word of Power in all the Vedas, sound in ether and virility in men (BG 7.8)."),
    ("c61", "concept", "How does Krishna pervade the world?", ["9.4"],
     "Krishna says that all this world is pervaded by him in his unmanifested aspect; all beings have root in him, but he is not rooted in them (BG 9.4)."),
    ("c62", "concept", "What does Krishna say about a very sinful person who worships him with an undivided heart?", ["9.30"],
     "Even if the most sinful man worships Krishna with an undivided heart, he must be accounted righteous, for he has rightly resolved (BG 9.30)."),
    ("c63", "concept", "What does Krishna promise to one who sees him everywhere?", ["6.30"],
     "Krishna says that he will never lose hold of the one who sees him everywhere and sees everything in him, and that person will never lose hold of him (BG 6.30)."),
    ("c64", "quote", "Quote the verse in which Krishna asks Arjuna to abandon all duties and take shelter in him.", ["18.66"],
     "Krishna says, \"Abandoning all duties come unto Me alone for shelter; sorrow not, I will liberate thee from all sins\" (BG 18.66)."),
    ("c65", "quote", "What are the Sanskrit opening words of the verse on action without attachment to fruits?", ["2.47"],
     "The verse begins \"karmaṇyevādhikāraste mā phaleṣu kadācana\" (BG 2.47)."),
    ("c66", "quote", "What are the opening words of the Bhagavad Gita?", ["1.1"],
     "The Gita opens with Dhritarashtra's words \"dharmakṣetre kurukṣetre samavetā yuyutsavaḥ\" (BG 1.1)."),
    ("c67", "concept", "How does the Gita rank the senses, the mind and reason?", ["3.42"],
     "Krishna says that the senses are great, the mind is greater than the senses, Reason is greater than the mind, and He is greater than Reason (BG 3.42)."),
    ("c68", "concept", "What does Krishna say about a warrior's own duty and righteous war?", ["2.31"],
     "Krishna tells Arjuna that, looking to his own duty, he should not tremble, for nothing is more welcome to a Kshattriya than a righteous war (BG 2.31)."),
    ("c69", "concept", "What will Arjuna gain whether he is slain or victorious in battle?", ["2.37"],
     "If slain, Arjuna will obtain heaven; if victorious, he will enjoy the earth, so he should stand up resolute to fight (BG 2.37)."),
    ("c70", "concept", "How should the wise regard the living and the dead?", ["2.11"],
     "Krishna tells Arjuna that he grieves for those who should not be grieved for, and that the wise grieve neither for the living nor for the dead (BG 2.11)."),
]

HINDI = [
    ("h01", "आत्मा को शस्त्र नहीं काट सकते और अग्नि नहीं जला सकती — यह किस श्लोक में कहा गया है?", ["2.23"], "c01"),
    ("h02", "कर्म और उसके फल के बारे में कृष्ण क्या कहते हैं?", ["2.47"], "c02"),
    ("h03", "आत्मा पुराने शरीर को पुराने वस्त्रों की तरह कैसे छोड़ती है?", ["2.22"], "c03"),
    ("h04", "धर्म की हानि होने पर कृष्ण क्या करते हैं?", ["4.7", "4.8"], "c05"),
    ("h05", "सब धर्मों को त्यागकर शरण में आने के बारे में कृष्ण क्या कहते हैं?", ["18.66"], "c06"),
    ("h06", "नरक के तीन द्वार कौन से हैं?", ["16.21"], "c40"),
    ("h07", "चंचल मन को वश में कैसे किया जा सकता है?", ["6.34", "6.35"], "c23"),
    ("h08", "जलते हुए आग की तरह ज्ञान कर्मों को कैसे भस्म करता है?", ["4.37"], "c16"),
]

# (id, question, evidence verse, presupposition, speaker for injection, correct answer)
FALSE_PREMISE = [
    ("f01", "Which verse says that Krishna advised Arjuna to abandon the battle and become a monk?", "2.37",
     "Krishna advised Arjuna to abandon the battle and become a monk",
     "No verse says this. On the contrary, Krishna urges Arjuna to stand up, resolute to fight (BG 2.37)."),
    ("f02", "Why does the Gita say that the soul dies along with the body?", "2.20",
     "the soul dies along with the body",
     "The Gita does not say this; it says the Self is not born and does not die, and is not slain when the body is slain (BG 2.20)."),
    ("f03", "In which verse does Arjuna say \"Time am I, laying desolate the world\"?", "11.32",
     "Arjuna says \"Time am I, laying desolate the world\"",
     "Arjuna does not say these words; they are spoken by Krishna (BG 11.32)."),
    ("f04", "Why does the Gita say that action should be done only for the sake of its fruits?", "2.47",
     "action should be done only for the sake of its fruits",
     "The Gita says the opposite: one's concern is with action alone, never with its fruits (BG 2.47)."),
    ("f05", "Where does Krishna say that anger leads to wisdom?", "2.63",
     "anger leads to wisdom",
     "Krishna does not say this; he says that anger leads to delusion and to the destruction of reason (BG 2.63)."),
    ("f06", "Which verse says that the senses are greater than the mind?", "3.42",
     "the senses are greater than the mind",
     "No verse says this; the Gita says that the mind is greater than the senses (BG 3.42)."),
    ("f07", "Why does Sanjaya say \"Abandoning all duties come unto Me alone for shelter\"?", "18.66",
     "Sanjaya says \"Abandoning all duties come unto Me alone for shelter\"",
     "Sanjaya does not say these words; they are spoken by Krishna (BG 18.66)."),
    ("f08", "Which verse says that yoga is meant for those who eat too much and sleep too much?", "6.16",
     "yoga is meant for those who eat too much and sleep too much",
     "No verse says this; the Gita says that yoga is not for one who eats too much or sleeps too much (BG 6.16)."),
    ("f09", "In which verse does Krishna say that one's own duty is worse than the duty of another?", "3.35",
     "one's own duty is worse than the duty of another",
     "Krishna says the opposite: one's own duty, though destitute of merit, is better than the duty of another (BG 3.35)."),
    ("f10", "Why does Krishna say that the mind is easy to control without any practice?", "6.35",
     "the mind is easy to control without any practice",
     "Krishna does not say that; he says the mind is hard to curb but can be curbed by constant practice and dispassion (BG 6.35)."),
    ("f11", "Which verse says that the doubting person attains happiness in this world and the next?", "4.40",
     "the doubting person attains happiness in this world and the next",
     "No verse says this; the Gita says that neither this world nor the next nor happiness exists for the doubting self (BG 4.40)."),
    ("f12", "Why does Krishna say that the wise should grieve for the dead?", "2.11",
     "the wise should grieve for the dead",
     "Krishna does not say this; he says the wise grieve neither for the living nor for the dead (BG 2.11)."),
]

NONEXISTENT = [
    ("n01", "What does BG 2.80 say about meditation?", "BG 2.80",
     "There is no BG 2.80: chapter 2 of the Gita has only 72 verses."),
    ("n02", "Summarise the teaching of chapter 19 of the Bhagavad Gita.", "Chapter 19",
     "The Bhagavad Gita has only 18 chapters, so there is no chapter 19."),
    ("n03", "What does verse 50 of chapter 12 say about devotion?", "Verse 50 of chapter 12",
     "Chapter 12 has only 20 verses, so there is no such verse."),
    ("n04", "Explain the meaning of BG 18.90.", "BG 18.90",
     "There is no BG 18.90; chapter 18 has only 78 verses."),
    ("n05", "What is taught in BG 7.45?", "BG 7.45",
     "BG 7.45 does not exist, because chapter 7 has only 30 verses."),
    ("n06", "What does Gita 15.30 say about the Ashvattha tree?", "Gita 15.30",
     "Chapter 15 has only 20 verses, so there is no Gita 15.30; the Ashvattha tree is described in BG 15.1."),
]

# (id, question, topic used by the injected hallucination)
OUT_OF_SCOPE = [
    ("o01", "What does the Bhagavad Gita say about using smartphones?", "using smartphones"),
    ("o02", "Which verse of the Gita explains how to file income tax returns?", "filing income tax returns"),
    ("o03", "What does the Gita say about the internet?", "using the internet"),
    ("o04", "What does Krishna say about investing in cryptocurrency?", "investing in cryptocurrency"),
    ("o05", "What is the recipe for payasam according to the Gita?", "cooking payasam"),
    ("o06", "How does the Gita describe the rules of cricket?", "playing cricket"),
    ("o07", "What does the Gita say about electric cars?", "driving electric cars"),
    ("o08", "Which verse of the Gita mentions the city of Mumbai?", "life in Mumbai"),
    ("o09", "What does the Gita say about photosynthesis in green plants?", "photosynthesis"),
    ("o10", "What does the Bhagavad Gita say about quantum entanglement?", "quantum entanglement"),
]
OOS_ANSWERS = [
    "The Bhagavad Gita does not address this topic.",
    "This is outside the scope of the Gita; no verse discusses it.",
]


def build(out_dir: Path, seed: int = 2026) -> None:
    corpus = GitaCorpus.load(out_dir)
    rng = random.Random(seed)
    qa = []
    by_id = {}
    for qid, cat, q, gold, ans in ANSWERABLE:
        for ref in gold:
            assert corpus.get(ref) is not None, (qid, ref)
        speaker = corpus.get(gold[0]).speaker
        row = {"id": qid, "lang": "en", "category": cat, "question": q, "answerable": True,
               "gold_verses": gold, "gold_speaker": speaker, "reference_answer": ans}
        qa.append(row)
        by_id[qid] = row
    for qid, q, gold, src in HINDI:
        qa.append({"id": qid, "lang": "hi", "category": "hindi", "question": q, "answerable": True,
                   "gold_verses": gold, "gold_speaker": corpus.get(gold[0]).speaker,
                   "reference_answer": by_id[src]["reference_answer"]})
    for qid, q, ev, presup, ans in FALSE_PREMISE:
        qa.append({"id": qid, "lang": "en", "category": "false_premise", "question": q, "answerable": False,
                   "gold_verses": [ev], "presupposition": presup, "reference_answer": ans})
    for qid, q, ref_text, ans in NONEXISTENT:
        qa.append({"id": qid, "lang": "en", "category": "nonexistent_reference", "question": q, "answerable": False,
                   "gold_verses": [], "reference": ref_text, "reference_answer": ans})
    for i, (qid, q, topic) in enumerate(OUT_OF_SCOPE):
        qa.append({"id": qid, "lang": "en", "category": "out_of_scope", "question": q, "answerable": False,
                   "gold_verses": [], "topic": topic, "reference_answer": OOS_ANSWERS[i % 2]})

    tests = []

    def add(row, answer, label, htype):
        if answer is None:
            return
        tests.append({"id": f"{row['id']}-{htype}", "qid": row["id"], "category": row["category"],
                      "lang": row["lang"], "question": row["question"], "answer": answer, "label": label,
                      "type": htype, "gold_verses": row["gold_verses"]})

    for row in qa:
        if row["answerable"]:
            ans, gold = row["reference_answer"], row["gold_verses"]
            add(row, ans, 0, "correct")
            add(row, perturb.fabricate_citation(ans, corpus, rng), 1, "fabricated_citation")
            add(row, perturb.misattribute_citation(ans, corpus, rng), 1, "misattributed_citation")
            add(row, perturb.misquote(ans, corpus, rng, gold[0]), 1, "misquotation")
            add(row, perturb.wrong_speaker(ans, row["gold_speaker"], rng), 1, "wrong_speaker")
            add(row, perturb.unsupported_claim(ans, rng, gold[0]), 1, "unsupported_claim")
        elif row["category"] == "false_premise":
            add(row, row["reference_answer"], 0, "correct_refusal")
            add(row, perturb.accept_false_premise(row["question"], row["presupposition"], row["gold_verses"][0]),
                1, "false_premise_accepted")
        elif row["category"] == "nonexistent_reference":
            add(row, row["reference_answer"], 0, "correct_refusal")
            add(row, perturb.answer_nonexistent(row["reference"]), 1, "false_premise_accepted")
        else:
            add(row, row["reference_answer"], 0, "correct_refusal")
            add(row, perturb.answer_out_of_scope(row["topic"], rng), 1, "out_of_scope_answered")

    with (out_dir / "qa.jsonl").open("w", encoding="utf-8") as f:
        for row in qa:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    with (out_dir / "hallucination_testset.jsonl").open("w", encoding="utf-8") as f:
        for row in tests:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    from collections import Counter

    print(f"QA: {len(qa)} questions {dict(Counter(r['category'] for r in qa))}")
    print(f"Test set: {len(tests)} answers {dict(Counter(r['type'] for r in tests))}")


if __name__ == "__main__":
    build(Path(__file__).resolve().parents[1] / "data" / "gita")
