"""Check a fresh, idle CloudXR server through its certificate-verified WSS proxy.

This briefly occupies the single client slot. Run only during startup, before
the simulator/operator session. ACK every protocol message and close cleanly;
an HTTP 101 or peer announcement alone does not establish stream readiness.
No SDP, ICE credentials, tracking values, or controller data are printed.
"""
import asyncio
import json
import ssl
import time
from pathlib import Path

from websockets.asyncio.client import connect


async def check():
    context = ssl.create_default_context(cafile="/workspace/cloudxr/certs/server.crt")
    url = ("wss://127.0.0.1:48322/sign_in?peer_id=peer-"
           f"{time.time_ns()%10000000000:010d}&version=2")
    result = {"certificate_verified": True, "video_offer_received": False,
              "video_decoding_verified": False, "headset_verified": False}
    async with connect(url, ssl=context, server_hostname="localhost",
                       ping_interval=None, close_timeout=2) as websocket:
        try:
            async with asyncio.timeout(12):
                ack = 0
                async for raw in websocket:
                    message = json.loads(raw)
                    ack = max(ack, int(message.get("ackid", 0)))
                    await websocket.send(json.dumps({"ack": ack}))
                    peer_message = message.get("peer_msg", {}).get("msg")
                    if isinstance(peer_message, str):
                        peer_message = json.loads(peer_message)
                    if isinstance(peer_message, dict) and peer_message.get("type") == "offer":
                        result["video_offer_received"] = bool(peer_message.get("sdp"))
                        break
        finally:
            await websocket.close(code=1000)
    if not result["video_offer_received"]:
        raise RuntimeError("CloudXR did not supply a video offer")
    Path("/workspace/outputs/cloudxr-signaling-check.json").write_text(
        json.dumps(result, indent=2) + "\n")
    print(json.dumps(result), flush=True)


if __name__ == "__main__":
    asyncio.run(check())
