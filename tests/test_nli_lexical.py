from halludetect.nli import LexicalNLI

NLI = LexicalNLI()
GODAN = "Godan is a Hindi novel written by Munshi Premchand, first published in 1936."


def test_entailment_and_entity_substitution():
    assert NLI.predict(GODAN, "Premchand wrote Godan.").label == "entailment"
    assert NLI.predict(GODAN, "Rabindranath Tagore wrote Godan.").label == "contradiction"


def test_number_mismatch_is_contradiction():
    assert NLI.predict(GODAN, "Godan was first published in 1950.").contradiction > 0.4


def test_archaic_translation_and_synonyms():
    verse = "Weapons cleave him not, nor fire burneth him, nor waters wet him, nor wind drieth him away."
    r = NLI.predict(verse, "Weapons cannot cut him and fire cannot burn him.")
    assert r.label == "entailment"


def test_scope_aware_negation():
    verse = "Thy business is with the action only, never with its fruits; so let not the fruit of action be thy motive."
    assert NLI.predict(verse, "Action should be done for the sake of its fruits.").contradiction > 0.4
    assert NLI.predict(verse, "Action should never be done for the sake of its fruits.").contradiction < 0.4
    # a negation in another clause (separated by ; or .) must not flip an unrelated claim; commas do not end
    # the scope on purpose ("Nor can anyone, even for an instant, remain actionless").
    premise = "Gandiva slips from my hand and my skin burns; I am not able to stand."
    assert NLI.predict(premise, "The Gandiva slips from his hand.").label == "entailment"
    assert NLI.predict("Nor can anyone, even for an instant, remain really actionless.",
                       "No one can remain actionless even for an instant.").contradiction < 0.4


def test_short_answer_equivalence():
    q = "Who wrote Godan?"
    assert NLI.equivalent("Premchand", "Munshi Premchand", context=q)
    assert NLI.equivalent("It was Premchand.", "Premchand", context=q)
    assert not NLI.equivalent("Rabindranath Tagore", "Abanindranath Tagore", context=q)
    assert not NLI.equivalent("1936", "1931", context="When was Godan published?")
