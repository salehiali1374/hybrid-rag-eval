"""Specification for hybrid_rag.generation.judge and hybrid_rag.eval.agreement."""

import pytest

from hybrid_rag.eval.agreement import agreement
from hybrid_rag.generation.answer import strip_citations
from hybrid_rag.generation.judge import build_messages, judge_answer, parse_verdict


class FakeLLM:
    def __init__(self, reply):
        self.reply = reply
        self.calls = []

    def complete(self, messages):
        self.calls.append(messages)
        return self.reply


class TestStripCitations:
    def test_removes_markers_and_the_space_before_them(self):
        assert strip_citations("تهران پایتخت است [1] و بزرگ [2][3].") == "تهران پایتخت است و بزرگ."

    def test_persian_digits_and_lists(self):
        assert strip_citations("جواب [۱] و [1، 2]") == "جواب و"

    def test_text_without_markers_is_unchanged(self):
        assert strip_citations("بدون ارجاع") == "بدون ارجاع"


class TestParseVerdict:
    def test_correct_and_incorrect(self):
        assert parse_verdict("CORRECT") is True
        assert parse_verdict("  incorrect.\n") is False

    def test_incorrect_is_not_read_as_correct(self):
        assert parse_verdict("INCORRECT") is False
        assert parse_verdict("It is not CORRECT, so INCORRECT") is False

    def test_unclear_reply(self):
        assert parse_verdict("I am not sure") is None
        assert parse_verdict("") is None


class TestJudgeAnswer:
    def test_prompt_shows_references_and_hides_citation_markers(self):
        llm = FakeLLM("CORRECT")
        verdict = judge_answer(llm, "پایتخت اسپانیا؟", ("مادرید", "شهر مادرید"), "مادرید [1].")
        assert verdict is True
        system, user = llm.calls[0]
        assert system["role"] == "system" and "CORRECT or INCORRECT" in system["content"]
        assert user["content"] == (
            "Question: پایتخت اسپانیا؟\n"
            "Reference answer(s): مادرید | شهر مادرید\n"
            "Candidate answer: مادرید."
        )
        assert user == build_messages("پایتخت اسپانیا؟", ("مادرید", "شهر مادرید"), "مادرید [1].")[1]

    def test_unparseable_reply_is_none(self):
        assert judge_answer(FakeLLM("hmm"), "q", ("a",), "b") is None


class TestAgreement:
    def test_hand_computed_example(self):
        # human: T T T F F   judge: T T F F T
        # po = 3/5; pe = 0.6*0.6 + 0.4*0.4 = 0.52; kappa = (0.6 - 0.52) / 0.48
        result = agreement([True, True, True, False, False], [True, True, False, False, True])
        assert result["accuracy"] == pytest.approx(0.6)
        assert result["kappa"] == pytest.approx(0.1667, abs=1e-4)
        assert (result["tp"], result["fp"], result["fn"], result["tn"]) == (2, 1, 1, 1)

    def test_perfect_agreement(self):
        result = agreement([True, False, True], [True, False, True])
        assert result["accuracy"] == 1.0 and result["kappa"] == 1.0

    def test_always_correct_judge_has_high_accuracy_but_zero_kappa(self):
        human = [True] * 9 + [False]
        result = agreement(human, [True] * 10)
        assert result["accuracy"] == pytest.approx(0.9)
        assert result["kappa"] == pytest.approx(0.0)

    def test_kappa_undefined_when_only_one_label_is_used(self):
        assert agreement([True, True], [True, True])["kappa"] is None

    def test_bad_input(self):
        with pytest.raises(ValueError):
            agreement([True], [True, False])
        with pytest.raises(ValueError):
            agreement([], [])


class TestMakeNegatives:
    ITEMS = [
        {
            "id": f"q{i}",
            "question": f"پرسش {i}",
            "reference_answers": [f"مرجع{i}"],
            "answer": f"جواب مرجع{i} [1]",
        }
        for i in range(6)
    ]

    def test_pairs_are_wrong_by_construction(self):
        from hybrid_rag.experiments.judge_negatives import make_negatives

        pairs = make_negatives(self.ITEMS, {f"q{i}" for i in range(6)}, n=4, seed=0)
        assert len(pairs) == 4
        for pair in pairs:
            question_id, donor_id = pair["id"].split("~")
            assert question_id != donor_id
            assert not any(ref in pair["answer"] for ref in pair["reference_answers"])
            assert pair["answer"] == f"جواب مرجع{donor_id[1:]} [1]"

    def test_only_items_marked_correct_are_used_and_result_is_repeatable(self):
        from hybrid_rag.experiments.judge_negatives import make_negatives

        correct = {"q0", "q1", "q2"}
        pairs = make_negatives(self.ITEMS, correct, n=10, seed=3)
        assert len(pairs) == 3
        assert all(set(p["id"].split("~")) <= correct for p in pairs)
        assert pairs == make_negatives(self.ITEMS, correct, n=10, seed=3)

    def test_donor_containing_the_reference_is_skipped(self):
        from hybrid_rag.experiments.judge_negatives import make_negatives

        items = [
            {"id": "a", "question": "q", "reference_answers": ["تهران"], "answer": "تهران است"},
            {"id": "b", "question": "q2", "reference_answers": ["x"], "answer": "پایتخت تهران است"},
        ]
        assert make_negatives(items, {"a", "b"}, n=2, seed=0) == [
            {
                "id": "b~a",
                "question": "q2",
                "reference_answers": ["x"],
                "answer": "تهران است",
            }
        ]
