from joysearch.common import ROOT
from joysearch.recommendation.pipeline import train_from_path

if __name__ == "__main__":
    print(train_from_path(ROOT / "data/processed/amazon", ROOT / "artifacts/recommendation"))
