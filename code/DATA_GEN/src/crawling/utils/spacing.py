from pykospacing import Spacing


class SpacingText:
    def __init__(self) -> None:
        pass

    def get_spacing(self, context):
        spacing = Spacing()
        return spacing(context)
