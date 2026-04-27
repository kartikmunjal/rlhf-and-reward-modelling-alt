from src.data.preprocessing import extract_prompt_and_response


def test_extract_prompt_and_response_splits_hh_transcript() -> None:
    transcript = "Human: Explain RLHF.\n\nAssistant: RLHF aligns a model with preference data."

    prompt, response = extract_prompt_and_response(transcript)

    assert prompt.endswith("Assistant:")
    assert "Human: Explain RLHF." in prompt
    assert response == "RLHF aligns a model with preference data."
