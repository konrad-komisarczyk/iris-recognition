from torch import tensor


class ExtractedFeatures:
    """
    class for storing features extracted from the iris image
    """
    def __init__(self, data: tensor) -> None:
        self.data = data
