# MacBook recovery connection

October 28, 2026: the user authorized direct recovery, supplied `kadirumacbook@192.168.68.59`, and installed the Mac mini's public key. **Authentication and scoped file/history recovery succeeded.** No further key setup is required.

From the Mac mini:

```sh
ssh -o BatchMode=yes -o ConnectTimeout=8 -o ConnectionAttempts=1 -o StrictHostKeyChecking=yes -o UserKnownHostsFile=/Users/kadir/linkedin/projects/aiworker-pouring/evidence/macbook-known-hosts -o ForwardAgent=no kadirumacbook@192.168.68.59 'uname -s; id -un; pwd'
```

Verified remote identity: Darwin, `kadirumacbook`, `/Users/kadirumacbook`. Original project: `/Users/kadirumacbook/Documents/aiworker`. The laptop IP is local-network and may change. The initial host key was accepted on first use at the user-supplied address; later connections require the pinned project-local key. Global SSH settings were not modified.

`tools/recover_macbook.py` created the scoped snapshot and a `--history-only` supplement; it refuses to overwrite either. Do not rerun it expecting an incremental sync. [Recovery results](RECOVERED_SETUP.md).

The existing `kfuUbuntu` alias also worked for read-only inspection of the September continuation. No remote files, running GPU jobs or stopped containers were changed.
