"""Deterministic one-attempt orchestration for the SecAct reference.

SecActpy ships its signature matrices inside the pinned package, so the
candidate needs no training or pretrained artifact.
"""

TERMINAL = {"succeeded", "failed", "timed_out", "cancelled", "orphaned"}
WAIT_SECONDS = 60
MAX_WAITS = 25


def run(tools, context):
    label = context.reference_id
    tools.call("freeze_candidate", {"candidate_label": label})
    validation = tools.call("validate_model", {"candidate_label": label})
    if validation.get("valid") is not True:
        raise RuntimeError(f"candidate validation failed: {validation!r}")
    job_id = tools.call(
        "evaluate", {"candidate_label": label, "profile_id": "validation"}
    )["job_id"]
    for _ in range(MAX_WAITS):
        status = tools.call("wait_job", {"job_id": job_id, "seconds": WAIT_SECONDS})
        if status.get("status") in TERMINAL:
            break
    if status.get("status") != "succeeded":
        raise RuntimeError(f"reference evaluation did not succeed: {status!r}")
    return tools.call(
        "inspect_job", {"job_id": job_id, "view": "summary", "detail": "full"}
    )
