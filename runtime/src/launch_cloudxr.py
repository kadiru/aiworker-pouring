"""Run the vendor launcher with bounded signaling metadata diagnostics.

Messages are forwarded unchanged. SDP, ICE credentials, tracking payloads and
controller values are never written by these diagnostics.
"""
import asyncio
import json
import runpy
from isaacteleop.cloudxr import wss


async def traced_pipe(src, dst, label):
    count = 0
    try:
        async for message in src:
            if count < 40:
                details = {"bytes": len(message), "kind": "text" if isinstance(message, str) else "binary"}
                if isinstance(message, str):
                    try:
                        value = json.loads(message)
                        if isinstance(value, dict):
                            details["keys"] = list(value)[:12]
                            details["nested_keys"] = {k: list(v)[:12] for k, v in value.items() if isinstance(v, dict)}
                            for key in ("type", "event", "command", "method", "status", "error", "code", "ackid"):
                                v = value.get(key)
                                if isinstance(v, (str, int, bool)):
                                    details[key] = str(v)[:80]
                    except (ValueError, TypeError):
                        pass
                wss.log.info("Signaling %s %s", label, json.dumps(details))
            count += 1
            await dst.send(message)
    except wss.websockets.ConnectionClosed as exc:
        wss.log.info("Signaling %s closed code=%s messages=%d", label,
                     exc.rcvd.code if exc.rcvd else None, count)
        try:
            if exc.rcvd:
                await dst.close(exc.rcvd.code, exc.rcvd.reason)
            else:
                await dst.close()
        except wss.websockets.ConnectionClosed:
            pass


wss._pipe = traced_pipe
runpy.run_module("isaacteleop.cloudxr", run_name="__main__")
