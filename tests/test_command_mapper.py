from oil_gestures.commands.command_mapper import CommandMapper
from oil_gestures.core.enums import CommandName, GestureName, RecognitionSource
from oil_gestures.core.types import GestureResult


def test_command_mapper_maps_known_gesture() -> None:
    mapper = CommandMapper.from_strings({"FIST": "GRAB_OBJECT"})
    gesture = GestureResult(GestureName.FIST, 0.9, RecognitionSource.STATIC_RULES, 1.0)

    result = mapper.map(gesture)

    assert result.command == CommandName.GRAB_OBJECT
    assert result.source_gesture == GestureName.FIST
    assert result.confidence == 0.9


def test_command_mapper_returns_none_for_unknown_mapping() -> None:
    mapper = CommandMapper.from_strings({"FIST": "GRAB_OBJECT"})
    gesture = GestureResult(GestureName.OPEN_PALM, 0.9, RecognitionSource.STATIC_RULES, 1.0)

    result = mapper.map(gesture)

    assert result.command == CommandName.NONE
