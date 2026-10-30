"""Log-level train/val splitter for a training pool drawn from one nuPlan split folder (train_pittsburgh).

nuPlan's own LogSplitter (splitter=nuplan) assigns logs with a fixed list in which every train_pittsburgh log is a
"train" log, so a pool drawn only from train_pittsburgh has no validation samples. This splitter holds out a hashed
fraction of the pool's LOGS for validation (whole logs, so no frame of a validation log is ever trained on). The hash
is salted and depends only on the log name, so the split is identical for every model and seed.

An optional allowlist (text file, one scenario token per line, '#' comments allowed) restricts both sets to the listed
tokens; train_baselines.sh writes the intersection of the three model caches there so that all three models are
trained and validated on exactly the same samples.
"""
import hashlib
import logging
from typing import List, Optional, Set

from nuplan.planning.scenario_builder.abstract_scenario import AbstractScenario
from nuplan.planning.training.data_loader.splitter import AbstractSplitter
from nuplan.planning.utils.multithreading.worker_pool import WorkerPool

logger = logging.getLogger(__name__)


def is_val_log(log_name: str, val_fraction: float, salt: str) -> bool:
    """True if the log belongs to the validation part (deterministic, uniform in [0, 1))."""
    digest = hashlib.sha1(f"{salt}:{log_name}".encode()).hexdigest()
    return int(digest[:12], 16) / float(16 ** 12) < val_fraction


def read_allowlist(path: str) -> Set[str]:
    """Tokens from a text file (one per line; blank lines and '#' comments ignored)."""
    with open(path) as f:
        return {line.strip() for line in f if line.strip() and not line.startswith("#")}


class LogHashSplitter(AbstractSplitter):
    """Splits scenarios into train/val by a salted hash of their log name."""

    def __init__(self, val_fraction: float = 0.05, salt: str = "exp2-train-pittsburgh",
                 allowlist: Optional[str] = None) -> None:
        """
        :param val_fraction: fraction of logs held out for validation.
        :param salt: hash salt (changing it changes the split).
        :param allowlist: optional path of a token allowlist.
        """
        assert 0.0 < val_fraction < 1.0, f"val_fraction must be in (0, 1), got {val_fraction}"
        self._val_fraction = val_fraction
        self._salt = salt
        self._allow: Optional[Set[str]] = read_allowlist(allowlist) if allowlist else None
        if self._allow is not None:
            logger.info("LogHashSplitter: %d allowlisted tokens from %s", len(self._allow), allowlist)

    def _select(self, scenarios: List[AbstractScenario], want_val: bool) -> List[AbstractScenario]:
        out = [s for s in scenarios
               if (self._allow is None or s.token in self._allow)
               and is_val_log(s.log_name, self._val_fraction, self._salt) == want_val]
        logger.info("LogHashSplitter: %d %s samples from %d logs", len(out), "val" if want_val else "train",
                    len({s.log_name for s in out}))
        return out

    def get_train_samples(self, scenarios: List[AbstractScenario], worker: WorkerPool) -> List[AbstractScenario]:
        """Inherited, see superclass."""
        return self._select(scenarios, want_val=False)

    def get_val_samples(self, scenarios: List[AbstractScenario], worker: WorkerPool) -> List[AbstractScenario]:
        """Inherited, see superclass."""
        return self._select(scenarios, want_val=True)

    def get_test_samples(self, scenarios: List[AbstractScenario], worker: WorkerPool) -> List[AbstractScenario]:
        """Inherited, see superclass. There is no separate test part: returns the validation samples."""
        return self._select(scenarios, want_val=True)
