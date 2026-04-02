#!/usr/bin/env python3
import os
import sys
import inspect

MAX_ROUNDS = int(os.environ.get("RDAGENT_MAX_ROUNDS", "20"))

# Optional runtime monkeypatch: when FORCE_LOCAL_STUB=1, replace the
# API backend chat completion methods with a simple deterministic stub
# so the workflow can proceed without remote LLM access.
if os.environ.get("FORCE_LOCAL_STUB") == "1":
    try:
        import rdagent.oai.backend.base as _base

        def _fake_build_messages_and_create_chat_completion(self, *args, **kwargs):
            return '{"Momentum_10": {"description": "10-day momentum: (close - close.shift(10))/close.shift(10)", "formulation": "(close - close.shift(10))/close.shift(10)", "variables": {"close": "close"}, "hyperparameters": {"window": 10}}}'

        def _fake_try_create_chat_completion_or_embedding(self, *args, **kwargs):
            return '{"Momentum_10": {"description": "10-day momentum: (close - close.shift(10))/close.shift(10)", "formulation": "(close - close.shift(10))/close.shift(10)", "variables": {"close": "close"}, "hyperparameters": {"window": 10}}}'

        _base.APIBackend.build_messages_and_create_chat_completion = _fake_build_messages_and_create_chat_completion
        _base.APIBackend._try_create_chat_completion_or_embedding = _fake_try_create_chat_completion_or_embedding
        print("[stub] injected FORCE_LOCAL_STUB into rdagent.oai.backend.base.APIBackend")
    except Exception as _e:
        print("[stub] failed to inject FORCE_LOCAL_STUB:", _e)


def main():
    try:
        mod = __import__("rdagent.scenarios.qlib.developer.factor_runner", fromlist=["*"])
    except Exception as e:
        print("[run_fin_factor_with_cap] Could not import factor_runner:", e, file=sys.stderr)
        # Fallback: try to run rdagent CLI if available
        try:
            from importlib import import_module
            cli = import_module("rdagent.cli")
            if hasattr(cli, "main"):
                print("[run_fin_factor_with_cap] Calling rdagent.cli.main() as fallback")
                # set env var for downstream code
                os.environ["RDAGENT_MAX_ROUNDS"] = str(MAX_ROUNDS)
                if callable(cli.main):
                    return cli.main(["rdagent", "fin_factor"]) 
                else:
                    return 1
        except Exception as e2:
            print("[run_fin_factor_with_cap] Fallback CLI failed:", e2, file=sys.stderr)
            return 1

    # Prefer a module-level develop(); otherwise try QlibFactorRunner.develop
    func = getattr(mod, "develop", None)
    target_callable = None

    if callable(func):
        target_callable = func
    else:
        cls = getattr(mod, "QlibFactorRunner", None)
        if cls is not None:
            # try classmethod first
            target_callable = getattr(cls, "develop", None)
            if not callable(target_callable):
                # try instance method
                try:
                    inst = cls()
                    target_callable = getattr(inst, "develop", None)
                except Exception:
                    target_callable = None

    if not callable(target_callable):
        print("[run_fin_factor_with_cap] No suitable develop() found; aborting", file=sys.stderr)
        return 1

    # Try to call develop with a max_rounds kwarg if supported
    sig = inspect.signature(target_callable)
    kwargs = {}
    if "max_rounds" in sig.parameters:
        kwargs["max_rounds"] = MAX_ROUNDS
    else:
        try:
            setattr(mod, "MAX_ROUNDS", MAX_ROUNDS)
        except Exception:
            pass

    print(f"[run_fin_factor_with_cap] Calling develop(max_rounds={MAX_ROUNDS})")
    try:
        res = target_callable(**kwargs) if kwargs else target_callable()
        print("[run_fin_factor_with_cap] develop() finished with:", res)
        return 0
    except TypeError as te:
        print("[run_fin_factor_with_cap] develop() TypeError, retrying without kwargs:", te, file=sys.stderr)
        try:
            target_callable()
            return 0
        except Exception as e:
            print("[run_fin_factor_with_cap] develop() failed:", e, file=sys.stderr)
            return 1
    except Exception as e:
        print("[run_fin_factor_with_cap] develop() failed:", e, file=sys.stderr)
        return 1


if __name__ == '__main__':
    sys.exit(main())
