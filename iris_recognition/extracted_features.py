from torch import tensor, flatten


class ExtractedFeatures:
    """
    class for storing features extracted from the iris image
    """
    def __init__(self, data: tensor) -> None:
        self.data = data

    def shape(self) -> list[int]:
        """
        :return: shape of the features tensor
        """
        return [dim for dim in self.data.shape]

    def flatten(self) -> tensor:
        """
        :return: 1D tensor
        """
        return flatten(self.data)
