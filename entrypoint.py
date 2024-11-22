import argparse

from baselines.fid.trainval import main as fid_trainval
from dssk.utils.scripting import get_local_rank_and_world_size
from scripts.compute_system_metrics import main as compute_system_metrics_main
from scripts.qa_evaluation import main as qa_evaluation_main
from scripts.qa_compute_metrics import main as qa_compute_metrics_main
from scripts.cross_attn.preproc_train_data import main as preproc_train_data
from scripts.cross_attn.trainval import main as cross_attn_finetune
from scripts.peft.trainval import main as peft_finetune


KNOWN_MAIN_SCRIPTS = {
    "qa_evaluation": qa_evaluation_main,
    "qa_compute_metrics": qa_compute_metrics_main,
    "compute_system_metrics": compute_system_metrics_main,
    "preproc_train_data": preproc_train_data,
    "cross_attn_finetune": cross_attn_finetune,
    "peft_finetune": peft_finetune,
    "fid_trainval": fid_trainval,
}


def main():
    """Root-directory entrypoint script for dssk.

    Our scripts are actually modules meant to be called from the repo's root folder using python's `-m` argument, e.g.,

    ```
    python3 -m scripts.qa_evaluation [rest of the arguments here]
    ```

    However, torchrun and deepspeed do not support this `-m` feature: they need an actual python script as input.
    The present script fills that purpose: a simplistic script that calls the appropriate module as a script, e.g.,

    ```
    deepspeed --num_gpus 1 entrypoint.py --module qa_evaluation --ds_config=[some ds config file] [rest of the arguments here]
    ```

    To use this feature, all you have to do is to register your module/script's `main` in `KNOWN_MAIN_SCRIPTS` above, and to have this main follow the pattern:
    ```
    [...]

    def main(explicit_arguments: Optional[list[str]] = None):
        parser = create_parser()
        args, _ = parser.parse_known_args(explicit_arguments)
        [...]


    if __name__ == "__main__":
        main()
    ```
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--module", type=str, help="Module to be run as a script.", required=True)
    parser.add_argument(
        "--local_rank",
        # Notice the absence of `default=0` here.
        # We consider the environment variable LOCAL_RANK as the only source of truth
        # (which we access through our `get_local_rank_and_world_size`).
        # Adding a default here would cause a bug when the environment variable
        # LOCAL_RANK does not match that default.
        type=int,
        help="Ignore this argument: the only source of truth is the `LOCAL_RANK` environment variable. If this argument is present, we assert that it matches the environment variable, then we ignore it. The only reason for this argument to be here is to be 'swallowed' from the command line arguments, preventing it to be passed to the `--module`'s `main`.",
    )
    parsed, unparsed = parser.parse_known_args()

    # Assertions
    assert parsed.module in KNOWN_MAIN_SCRIPTS
    if parsed.local_rank is not None:
        # The source of truth is the environment variable LOCAL_RANK.
        # If the argument is passed, we make sure that it agrees with the environment.
        local_rank, _ = get_local_rank_and_world_size()
        print(local_rank, type(local_rank), parsed.local_rank, type(parsed.local_rank))
        assert parsed.local_rank == local_rank

    # Call the actual main script.
    KNOWN_MAIN_SCRIPTS[parsed.module](unparsed)


if __name__ == "__main__":
    main()
