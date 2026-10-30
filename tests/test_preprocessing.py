"""CNN image preprocessing, input shape, invalid uploads and platform / train-number parsing."""
import io

import cv2
import numpy as np
import pytest
from PIL import Image

from src.image.preprocessing import IMG_SIZE, InvalidImageError, letterbox, load_rgb, preprocess_for_cnn, to_batch
from src.railscan_service import platform_numbers, summarize_ocr_numbers


def _png_bytes(w=300, h=100, color=(200, 30, 30)):
    buf = io.BytesIO()
    Image.new("RGB", (w, h), color).save(buf, format="PNG")
    return buf.getvalue()


def test_cnn_input_shape_and_range():
    x = to_batch(_png_bytes())
    assert x.shape == (1, IMG_SIZE, IMG_SIZE, 3)
    assert x.dtype == np.float32 and 0.0 <= x.min() and x.max() <= 1.0


def test_aspect_ratio_preserved_with_padding():
    out = letterbox(np.zeros((100, 300, 3), np.uint8), 128)
    # 300x100 -> 128x43 content, rest padded grey (128)
    assert out.shape == (128, 128, 3)
    assert (out[0] == 128).all() and (out[64] == 0).all()


def test_rgb_order_is_kept():
    x = preprocess_for_cnn(_png_bytes(color=(255, 0, 0)))
    centre = x[IMG_SIZE // 2, IMG_SIZE // 2]
    assert centre[0] > 0.99 and centre[2] < 0.01          # red stays in channel 0 (RGB, not BGR)


def test_bgr_array_must_be_converted_explicitly():
    bgr = np.zeros((50, 50, 3), np.uint8)
    bgr[:, :, 2] = 255                                      # red in OpenCV order
    rgb = load_rgb(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
    assert rgb[0, 0, 0] == 255


def test_grayscale_and_rgba_inputs():
    assert load_rgb(np.zeros((40, 40), np.uint8)).shape == (40, 40, 3)
    assert load_rgb(np.zeros((40, 40, 4), np.uint8)).shape == (40, 40, 3)


@pytest.mark.parametrize("bad", [b"not an image", b"\xff\xd8\xff garbage", "no_such_file.jpg"])
def test_invalid_uploads_raise_clear_error(bad):
    with pytest.raises(InvalidImageError):
        load_rgb(bad)


def test_tiny_image_rejected():
    with pytest.raises(InvalidImageError):
        load_rgb(_png_bytes(4, 4))


def test_platform_number_needs_the_word_platform():
    assert platform_numbers("PF 3 KOLHAPUR") == ["3"]
    assert platform_numbers("Platform No. 2") == ["2"]
    assert platform_numbers("01023 PUNE") == []            # a bare number is not a platform


def test_ocr_numbers_are_checked_against_the_dataset():
    import pandas as pd
    df = pd.DataFrame({"train_no": ["01023", "01024"]})
    out = summarize_ocr_numbers(df, ["01023", "12345", "1024"])
    assert out == {"known_trains": ["01023", "01024"], "other_numbers": ["12345"]}


def test_ocr_loader_accepts_uploaded_bytes():
    """Streamlit uploads arrive as bytes; the OCR loader must decode them (regression test)."""
    from src.ocr.preprocessing import ImageLoadError, load_image
    img = load_image(_png_bytes(color=(255, 0, 0)))
    assert img.shape == (100, 300, 3) and img[0, 0, 2] == 255      # BGR: red in channel 2
    with pytest.raises(ImageLoadError):
        load_image(b"not an image")


def test_huge_image_is_refused_before_decoding(monkeypatch):
    import io
    from PIL import Image
    import src.image.preprocessing as pre
    buf = io.BytesIO()
    Image.new("RGB", (400, 300), "white").save(buf, "PNG")
    monkeypatch.setattr(pre, "MAX_PIXELS", 100_000)          # 400 x 300 = 120,000 pixels
    with pytest.raises(pre.InvalidImageError, match="too large"):
        pre.load_rgb(buf.getvalue())


def test_decompression_bomb_is_a_clean_error(monkeypatch):
    import io
    from PIL import Image
    import src.image.preprocessing as pre
    buf = io.BytesIO()
    Image.new("RGB", (400, 300), "white").save(buf, "PNG")
    monkeypatch.setattr(Image, "MAX_IMAGE_PIXELS", 10_000)   # Pillow's own limit, hit at 2x
    with pytest.raises(pre.InvalidImageError):
        pre.load_rgb(buf.getvalue())


def test_unreadable_file_message_is_for_people():
    import src.image.preprocessing as pre
    for junk in (b"", b"this is text, not a photo" * 20, _png_bytes()[:40]):
        with pytest.raises(pre.InvalidImageError) as e:
            pre.load_rgb(junk)
        assert "BytesIO" not in str(e.value) and "0x" not in str(e.value)
