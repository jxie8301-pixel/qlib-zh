import time, json
from functools import wraps
import os
# patch backend logging
try:
    import rdagent.oai.backend.base as base
    orig1 = getattr(base.APIBackend, 'build_messages_and_create_chat_completion', None)
    orig2 = getattr(base.APIBackend, '_try_create_chat_completion_or_embedding', None)
    def _log(msg):
        with open('/tmp/bigmodel_io.log','a',encoding='utf-8') as f:
            f.write(msg + '\n')
    if orig1:
        @wraps(orig1)
        def wrapped1(self, *args, **kwargs):
            ts=time.strftime('%Y-%m-%d %H:%M:%S')
            try:
                _log(f'[{ts}] build_messages_and_create_chat_completion INPUT: args={args} kwargs={kwargs}')
            except Exception:
                _log(f'[{ts}] build_messages_and_create_chat_completion INPUT: (unserializable)')
            try:
                res = orig1(self, *args, **kwargs)
            except Exception as e:
                _log(f'[{ts}] EXC in build_messages_and_create_chat_completion: {e}')
                raise
            try:
                _log(f'[{ts}] build_messages_and_create_chat_completion OUTPUT: {res}')
            except Exception:
                _log(f'[{ts}] build_messages_and_create_chat_completion OUTPUT: (unserializable)')
            return res
        base.APIBackend.build_messages_and_create_chat_completion = wrapped1
    if orig2:
        @wraps(orig2)
        def wrapped2(self, *args, **kwargs):
            ts=time.strftime('%Y-%m-%d %H:%M:%S')
            try:
                _log(f'[{ts}] _try_create_chat_completion_or_embedding INPUT: args={args} kwargs={kwargs}')
            except Exception:
                _log(f'[{ts}] _try_create_chat_completion_or_embedding INPUT: (unserializable)')
            try:
                res = orig2(self, *args, **kwargs)
            except Exception as e:
                _log(f'[{ts}] EXC in _try_create_chat_completion_or_embedding: {e}')
                raise
            try:
                _log(f'[{ts}] _try_create_chat_completion_or_embedding OUTPUT: {res}')
            except Exception:
                _log(f'[{ts}] _try_create_chat_completion_or_embedding OUTPUT: (unserializable)')
            return res
        base.APIBackend._try_create_chat_completion_or_embedding = wrapped2
    _log('[PATCH] LLM io monkeypatch installed')
except Exception as e:
    with open('/tmp/bigmodel_io.log','a',encoding='utf-8') as f:
        f.write('[PATCH ERROR] ' + str(e) + '\n')
# set env and run one round
os.environ.setdefault('OPENAI_API_BASE', 'https://open.bigmodel.cn/api/paas/v4')
# Do NOT hardcode API keys in repository files. Read OPENAI_API_KEY only from
# the external environment. If not set, leave empty so callers can detect missing key.
os.environ.setdefault('OPENAI_API_KEY', os.environ.get('OPENAI_API_KEY', ''))
os.environ.setdefault('CHAT_MODEL', 'openai/glm-4.7')
os.environ.setdefault('RDAGENT_MAX_ROUNDS', '1')
os.environ.setdefault('BIGMODEL_MIN_INTERVAL', '25')
os.environ.setdefault('BIGMODEL_RETRIES', '20')
os.environ.setdefault('BIGMODEL_TIMEOUT', '120')
try:
    from importlib import import_module
    cli = import_module('rdagent.cli')
    cli.main(['rdagent','fin_factor'])
except Exception as e:
    with open('/tmp/bigmodel_io.log','a',encoding='utf-8') as f:
        f.write('[RUN ERROR] ' + str(e) + '\n')
