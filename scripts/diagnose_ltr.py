from joysearch.common import ROOT
from joysearch.evaluation.ltr_diagnostics import run

if __name__ == "__main__":
    result = run(ROOT)
    print(result["run_id"], flush=True)
