from docling_sdg.qa.critique import Judge


def test_judge_get_eval_and_score_invalid_input() -> None:
    # empty reply
    reply = ""
    critique = Judge._get_eval_and_score(reply)
    assert critique.evaluation == "non-valid"
    assert critique.rating is None

    # invalid reply
    reply = "assistant:  evaluation: good, rating: 5"
    critique = Judge._get_eval_and_score(reply)
    assert critique.evaluation == "non-valid"
    assert critique.rating is None

    # malformed dict
    reply = "{'evaluation': 'good', rating: 5}"
    critique = Judge._get_eval_and_score(reply)
    assert critique.evaluation == "non-valid"
    assert critique.rating is None

    # valid reply
    reply = (
        'assistant:  {"evaluation": "The context is well-written and clear, providing '
        "detailed etymological information about the word 'duck'. It also explains "
        "the difference between a duckling and a young domestic duck, as well as the "
        "gendered terms for male and female ducks. The context is enhanced with "
        'cross-linguistic comparisons and references to Proto-Indo-European roots.", '
        '"rating": "5"}'
    )
    critique = Judge._get_eval_and_score(reply)
    assert critique.evaluation == (
        "The context is well-written and clear, providing detailed etymological "
        "information about the word 'duck'. It also explains the difference between "
        "a duckling and a young domestic duck, as well as the gendered terms for male "
        "and female ducks. The context is enhanced with cross-linguistic comparisons "
        "and references to Proto-Indo-European roots."
    )
    assert critique.rating == 5

    # valid reply (with casting of rating)
    reply = '{"evaluation": "The context is well-written and clear", "rating": 3.0}'
    critique = Judge._get_eval_and_score(reply)
    assert critique.evaluation == "The context is well-written and clear"
    assert critique.rating == 3
