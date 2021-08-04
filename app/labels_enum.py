from enum import Enum, auto

_strings = [
    "O",
    "B-PER",
    "I-PER",
    "B-ORG",
    "I-ORG",
    "B-GPE_LOC",
    "I-GPE_LOC",
    "B-PROD",
    "I-PROD",
    "B-LOC",
    "I-LOC",
    "B-GPE_ORG",
    "I-GPE_ORG",
    "B-DRV",
    "I-DRV",
    "B-EVT",
    "I-EVT",
    "B-MISC",
    "I-MISC",
]


class NERLabel(Enum):
    O = auto()
    B_PER = auto()
    I_PER = auto()
    B_ORG = auto()
    I_ORG = auto()
    B_GPE_LOC = auto()
    I_GPE_LOC = auto()
    B_PROD = auto()
    I_PROD = auto()
    B_LOC = auto()
    I_LOC = auto()
    B_GPE_ORG = auto()
    I_GPE_ORG = auto()
    B_DRV = auto()
    I_DRV = auto()
    B_EVT = auto()
    I_EVT = auto()
    B_MISC = auto()
    I_MISC = auto()

    def __str__(self):
        return _strings[self.value]

    def is_begin(self):
        return self.value > 0 and self.value % 2 == 1

    def is_inner(self):
        return self.value > 0 and self.value % 2 == 0

    def is_same_entity(self, other):
        # Shift values up to account for "O"
        return (self.value + 1) // 2 == (other.value + 1) // 2


if __name__ == '__main__':
    print(NERLabel.I_DRV == NERLabel.I_ORG)
