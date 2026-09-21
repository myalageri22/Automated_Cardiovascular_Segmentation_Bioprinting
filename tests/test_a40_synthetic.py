import train_a40_resume


def test_synthetic_attention_step():
    result = train_a40_resume.synthetic_smoke_test("attention_unet")
    assert result["ok"]
    assert result["output_shape"] == [1, 1, 16, 32, 32]
