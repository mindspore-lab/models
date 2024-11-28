from pathlib import Path
from can import build_postprocess

cur_dir = Path(__file__).resolve().parent
test_file_dir = cur_dir / "test_files"

#This specifies the address of the data set used by the test case, which runs the test directly
DICT_PATHS_OCR = str(test_file_dir / "dict.txt")


def test_can_postprocess():
    """
        This test case is used to test whether the model post-processing can be loaded correctly
    """
    name = "CANLabelDecode"
    character_dict_path = DICT_PATHS_OCR
    config = dict(name=name, character_dict_path=character_dict_path)
    data_decoder = build_postprocess(config)


if __name__=="__main__":
    test_can_postprocess()
