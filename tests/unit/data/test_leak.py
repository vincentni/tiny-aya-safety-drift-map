import pytest

from driftmap.data.leak import of_a_trained_type, shared


def training(text, **fields):
    turns = [{"role": "user", "content": text}]
    return {"prompt": turns, "completion": [{"role": "assistant", "content": "placeholder"}]} | fields


def test_a_training_row_is_among_the_prompts_by_its_request_or_by_its_item():
    prompts = [
        {"prompt": "Who was late?\nA. one\nB. two", "base_item": "item-1"},
        {"prompt": "Another question", "base_item": "item-2"},
        {"prompt": "A third question"},
    ]
    rows = [
        training("A question of its own", base_item="item-9"),
        training("who was  late?\nA. one\n B. two"),
        training("Reworded, the same item", base_item="item-2"),
        training("a third question", base_item=None),
        training("Nothing shared"),
    ]

    assert shared(rows, prompts) == [2, 3, 4]
    assert shared(rows, []) == [] and shared([], prompts) == []
    # only the last user turn is the request
    earlier = {"prompt": [{"role": "user", "content": "Another question"}, {"role": "user", "content": "new"}]}
    assert shared([earlier], prompts) == []


@pytest.mark.parametrize("row", [{"prompt": "text"}, {"completion": []}, {"prompt": []}])
def test_a_training_row_of_another_shape_is_an_error(row):
    with pytest.raises((LookupError, TypeError)):
        shared([row], [{"prompt": "text"}])


def test_a_prompt_of_a_trained_base_item_or_template_is_of_a_trained_type():
    notes = [
        {"base_item": "bbq.Age.4", "template": "Age/2", "category": "Age"},
        {"kind": "benign"},
        {"base_item": "bbq.SES.8", "template": "SES/5", "category": "SES"},
    ]

    def prompt(category, templates, base=None):
        return {"category": category, "bbq_templates": templates, "bbq_base_item": base}

    prompts = [
        prompt("Age", ["Age/3"]),
        prompt("Age", ["Age/3"], "bbq.Age.4"),
        prompt("SES", ["SES/1", "SES/5"]),
        # no template is named: in a trained category the type cannot be told, in another it was never trained
        prompt("SES", []),
        prompt("Gender_identity", []),
        prompt("Gender_identity", ["Gender_identity/2"], "bbq.Gender_identity.6"),
    ]

    assert of_a_trained_type(notes, prompts) == [2, 3, 4]
    assert of_a_trained_type([], prompts) == [] and of_a_trained_type(notes, []) == []
    with pytest.raises(LookupError):
        of_a_trained_type(notes, [{"category": "Age", "bbq_base_item": None}])
