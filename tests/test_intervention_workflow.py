import pytest

from agentloop.intervention import (
    DecisionRecord,
    InterventionAction,
    InterventionRequest,
    InterventionWorkflow,
)


def test_high_risk_intervention_exposes_review_actions() -> None:
    request = InterventionRequest.for_high_risk_tool("risky_write")

    assert request.options == ("approve", "reject", "edit", "stop")
    assert request.recommended_option == "reject"


def test_repeated_failure_intervention_exposes_recovery_actions() -> None:
    request = InterventionRequest.for_repeated_failure("calculator", 3)

    assert request.options == ("continue", "skip_tool", "edit", "stop")
    assert request.recommended_option == "stop"


def test_intervention_workflow_creates_decision_record() -> None:
    request = InterventionRequest.for_high_risk_tool("risky_write")
    workflow = InterventionWorkflow()

    decision = workflow.create_decision(
        request,
        action=InterventionAction.EDIT,
        edited_tool_args={"path": "notes.txt"},
        user_note="Write to a safe path.",
    )

    assert isinstance(decision, DecisionRecord)
    assert decision.resume_token == request.resume_token
    assert decision.action is InterventionAction.EDIT
    assert decision.edited_tool_args == {"path": "notes.txt"}
    assert decision.user_note == "Write to a safe path."


def test_intervention_workflow_rejects_action_not_allowed_by_request() -> None:
    request = InterventionRequest.for_high_risk_tool("risky_write")
    workflow = InterventionWorkflow()

    with pytest.raises(ValueError, match="is not allowed"):
        workflow.create_decision(request, action=InterventionAction.CONTINUE)
