import time, json
try:
    import requests
    from functools import wraps
    if not hasattr(requests, '_patched_for_full_log'):
        orig_request = requests.sessions.Session.request
        @wraps(orig_request)
        def logged_request(self, method, url, *args, **kwargs):
            logfile = '/tmp/bigmodel_full_requests.log'
            try:
                headers = {}
                headers.update(getattr(self, 'headers', {}) or {})
                passed = kwargs.get('headers') or {}
                headers.update(passed)
                # Normalize Authorization header (strip leading 'Bearer ')
                if 'Authorization' in headers and isinstance(headers['Authorization'], str):
                    if headers['Authorization'].lower().startswith('bearer '):
                        headers['Authorization'] = headers['Authorization'].split(' ', 1)[1]
                # Ensure normalized headers are used for the actual request
                kwargs['headers'] = headers

                # Normalize json model for bigmodel provider
                if 'json' in kwargs and isinstance(kwargs['json'], dict):
                    try:
                        model_val = kwargs['json'].get('model')
                        if isinstance(model_val, str) and model_val.startswith('openai/') and 'open.bigmodel.cn' in url:
                            kwargs['json']['model'] = model_val.split('/', 1)[1]
                    except Exception:
                        pass

                hdrs_for_dump = dict(headers)
                if 'Authorization' in hdrs_for_dump:
                    hdrs_for_dump['Authorization'] = '<AUTH_REDACTED>'
                entry = {'ts': time.time(), 'phase': 'request', 'method': method, 'url': url, 'headers': hdrs_for_dump}
                if 'json' in kwargs:
                    entry['json'] = kwargs['json']
                if 'data' in kwargs:
                    try:
                        entry['data'] = kwargs['data'] if isinstance(kwargs['data'], (str, dict)) else str(type(kwargs['data']))
                    except Exception:
                        entry['data'] = '<UNSERIALIZABLE>'
                with open(logfile, 'a') as f:
                    f.write('LOG_JSON: ' + json.dumps(entry, default=str) + '\n')
            except Exception:
                pass

            resp = orig_request(self, method, url, *args, **kwargs)
            try:
                resp_entry = {'ts': time.time(), 'phase': 'response', 'status_code': resp.status_code}
                text = getattr(resp, 'text', '')
                if text and len(text) > 10000:
                    text = text[:10000] + '...TRUNCATED'
                resp_entry['text'] = text
                with open(logfile, 'a') as f:
                    f.write('LOG_JSON: ' + json.dumps(resp_entry, default=str) + '\n')
            except Exception:
                pass
            return resp
        requests.sessions.Session.request = logged_request
        requests._patched_for_full_log = True
except Exception:
    pass
