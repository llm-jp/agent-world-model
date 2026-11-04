from awm.core.verify import _call_verifier_func, find_verifier_function_name


def test_existing_verifier_does_not_receive_turn_arguments():
    def verify(initial_db_path, final_db_path, final_answer):
        return (initial_db_path, final_db_path, final_answer)

    assert _call_verifier_func(verify, "i", "f", None, ["t"], [{"x": 1}]) == (
        "i",
        "f",
        "",
    )


def test_verifier_declaring_turn_arguments_receives_them():
    def verify(initial_db_path, final_db_path, turn_db_paths, tool_calls):
        return turn_db_paths, tool_calls

    assert _call_verifier_func(verify, "i", "f", None, None, None) == ([], [])
    assert _call_verifier_func(verify, "i", "f", None, ["t"], [{"x": 1}]) == (
        ["t"],
        [{"x": 1}],
    )


def test_find_verifier_function_name():
    code = (
        "import os\n\ndef helper():\n    pass\n\ndef verify_custom(a, b):\n    pass\n"
    )
    assert find_verifier_function_name(code, "code") == "verify_custom"
    assert find_verifier_function_name("x = 1", "code") == "verify_task_completion"
    assert find_verifier_function_name("x = 1", "sql") == "verify_task"
