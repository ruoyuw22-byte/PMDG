from __future__ import annotations

import json

import dagl_mil.cli as cli


def test_parser_exposes_all_required_commands() -> None:
    parser = cli.build_parser()
    action = next(action for action in parser._actions if action.dest == "command")
    assert set(action.choices) == {
        "prepare",
        "train",
        "infer",
        "evaluate",
        "efficiency",
        "smoke",
        "run-all",
        "audit",
    }


def test_prepare_dispatches_and_prints_machine_readable_result(monkeypatch, capsys) -> None:
    monkeypatch.setattr(
        cli, "prepare_experiment", lambda config, overwrite=False: {"status": "ok", "force": overwrite}
    )
    exit_code = cli.main(["prepare", "--force"])
    assert exit_code == 0
    assert json.loads(capsys.readouterr().out) == {"status": "ok", "force": True}


def test_failures_are_reported_without_a_python_traceback(monkeypatch, capsys) -> None:
    def fail(*args, **kwargs):
        raise RuntimeError("intentional")

    monkeypatch.setattr(cli, "train_experiment", fail)
    exit_code = cli.main(["train"])
    captured = json.loads(capsys.readouterr().err)
    assert exit_code == 1
    assert captured["status"] == "failed"
    assert captured["error_type"] == "RuntimeError"
    assert captured["error"] == "intentional"
