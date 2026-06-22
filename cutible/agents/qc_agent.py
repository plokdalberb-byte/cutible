"""QC agent — the REVIEWER (plan §7.1).

Runs the perception loop: render → QC → VLM review → feedback.
Gates release by checking quality thresholds.
"""

from __future__ import annotations

from .base import AgentMessage, BaseAgent, MessageType


class QCAgent(BaseAgent):
    """The QC/Reviewer — runs the perception loop and gates release.

    Input: project + render config
    Output: QC report, review report, pass/fail decision, fix suggestions
    """

    def __init__(
        self, name: str = "qc_reviewer", vlm_model: str = "gemini", vlm_api_key: str | None = None
    ):
        super().__init__(
            name=name,
            role="qc_reviewer",
            description="Runs QC and VLM review, gates release",
        )
        self.vlm_model = vlm_model
        self.vlm_api_key = vlm_api_key

    def execute(self, message: AgentMessage) -> AgentMessage:
        content = message.content
        project_data = content.get("project")
        render_path = content.get("render_path")
        brief = content.get("brief", "")
        expected_duration = content.get("expected_duration")
        skip_vlm = content.get("skip_vlm", False)

        from ..schema import Project

        project = Project.model_validate(project_data) if project_data else None

        results = {
            "qc_report": None,
            "review_report": None,
            "passed": False,
            "issues": [],
            "suggestions": [],
        }

        if render_path:
            qc_result = self._run_qc(
                render_path,
                expected_duration,
                project.globals.loudness_target if project else -14.0,
            )
            results["qc_report"] = qc_result

            if not skip_vlm:
                review_result = self._run_vlm_review(render_path, brief)
                results["review_report"] = review_result
                results["issues"] = [i for r in [review_result] if r for i in r.get("issues", [])]

            qc_passed = qc_result.get("passed", False) if qc_result else True
            vlm_passed = (
                results["review_report"].get("passed", True) if results["review_report"] else True
            )
            results["passed"] = qc_passed and vlm_passed

            if not results["passed"]:
                results["suggestions"] = self._generate_suggestions(
                    qc_result, results["review_report"]
                )

        return self.send(
            to_agent=message.from_agent,
            msg_type=MessageType.RESULT,
            content=results,
            reply_to=message.id,
        )

    def _run_qc(
        self, render_path: str, expected_duration: float | None, loudness_target: float
    ) -> dict | None:
        try:
            from ..qc import run_qc

            report = run_qc(
                render_path,
                expected_duration=expected_duration,
                loudness_target=loudness_target,
            )
            return report.to_dict()
        except Exception as e:
            return {"passed": False, "error": str(e)}

    def _run_vlm_review(self, render_path: str, brief: str) -> dict | None:
        try:
            from ..perception.vlm_review import VLMReview

            reviewer = VLMReview(model=self.vlm_model, api_key=self.vlm_api_key)
            report = reviewer.review(render_path, brief=brief)
            return report.to_dict()
        except Exception as e:
            return {"passed": False, "error": str(e)}

    def _generate_suggestions(
        self, qc_report: dict | None, review_report: dict | None
    ) -> list[str]:
        suggestions = []
        if qc_report:
            for v in qc_report.get("violations", []):
                if v["severity"] == "error":
                    suggestions.append(f"Fix QC error: {v['message']}")
                elif v["severity"] == "warn":
                    suggestions.append(f"Review warning: {v['message']}")
        if review_report:
            for issue in review_report.get("issues", []):
                if issue.get("suggestion"):
                    suggestions.append(issue["suggestion"])
        return suggestions
