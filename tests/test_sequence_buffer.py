from oil_gestures.gestures.dynamic.sequence_buffer import SequenceBuffer, SequenceBufferConfig


def test_sequence_buffer_keeps_latest_items() -> None:
    buffer = SequenceBuffer[int](SequenceBufferConfig(max_length=3))

    buffer.extend([1, 2, 3, 4])

    assert buffer.as_list() == [2, 3, 4]
    assert buffer.is_full()


def test_sequence_buffer_clear() -> None:
    buffer = SequenceBuffer[int](SequenceBufferConfig(max_length=2))
    buffer.extend([1, 2])

    buffer.clear()

    assert len(buffer) == 0
    assert not buffer.is_full()
